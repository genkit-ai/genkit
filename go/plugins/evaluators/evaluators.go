// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0

// Package evaluators defines a set of Genkit Evaluators for popular use-cases
package evaluators

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"regexp"
	"sync"

	jsonata "github.com/blues/jsonata-go"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/logger"
)

const provider = "genkitEval"

// EvaluatorType is an enum used to indicate the type of evaluator being
// configured for use
type EvaluatorType int

const (
	EvaluatorDeepEqual EvaluatorType = iota
	EvaluatorRegex
	EvaluatorJsonata
)

var evaluatorTypeName = map[EvaluatorType]string{
	EvaluatorDeepEqual: "DEEP_EQUAL",
	EvaluatorRegex:     "REGEX",
	EvaluatorJsonata:   "JSONATA",
}

func (ss EvaluatorType) String() string {
	return evaluatorTypeName[ss]
}

// MetricConfig provides configuration options for a specific metric. More
// Params (judge LLMs, etc.) could be configured by extending this struct
type MetricConfig struct {
	MetricType EvaluatorType
}

// GenkitEval is a Genkit plugin that provides evaluators
type GenkitEval struct {
	Metrics []MetricConfig // Configs for individual metrics
	initted bool           // Whether the plugin has been initialized
	mu      sync.Mutex     // Mutex to manage locks
}

func (ge *GenkitEval) Name() string {
	return provider
}

// Init initializes the plugin.
func (ge *GenkitEval) Init(ctx context.Context) []api.Action {
	if ge == nil {
		ge = &GenkitEval{}
	}
	ge.mu.Lock()
	defer ge.mu.Unlock()
	if ge.initted {
		panic("genkitEval.Init already called")
	}
	if ge == nil || len(ge.Metrics) == 0 {
		panic("genkitEval: need to configure at least one metric")
	}
	ge.initted = true

	var actions []api.Action
	for _, metric := range ge.Metrics {
		actions = append(actions, ConfigureMetric(metric).(api.Action))
	}
	return actions
}

func ConfigureMetric(metric MetricConfig) ai.Evaluator {
	switch metric.MetricType {
	case EvaluatorDeepEqual:
		return configureDeepEqualEvaluator()
	case EvaluatorJsonata:
		return configureJsonataEvaluator()
	case EvaluatorRegex:
		return configureRegexEvaluator()
	default:
		panic(fmt.Sprintf("Unsupported genkitEval metric type: %s", metric.MetricType.String()))
	}
}

func configureRegexEvaluator() ai.Evaluator {
	evalOptions := ai.EvaluatorOptions{
		DisplayName: "RegExp",
		Definition:  "Tests output against the regexp provided as reference",
		IsBilled:    false,
	}
	return ai.NewEvaluator(api.NewName(provider, "regex"), &evalOptions, func(ctx context.Context, req *ai.EvaluatorCallbackRequest) (*ai.EvaluatorCallbackResponse, error) {
		dataPoint := req.Input
		var score ai.Score
		if dataPoint.Output == nil {
			return nil, errors.New("output was not provided")
		}
		if dataPoint.Reference == nil {
			return nil, errors.New("reference was not provided")
		}
		if reflect.TypeOf(dataPoint.Reference).String() != "string" {
			return nil, errors.New("reference must be a string (regex)")
		}
		if reflect.TypeOf(dataPoint.Output).String() == "string" {
			// Test against provided regexp
			match, _ := regexp.MatchString((dataPoint.Reference).(string), (dataPoint.Output).(string))
			status := ai.ScoreStatusUnknown
			if match {
				status = ai.ScoreStatusPass
			} else {
				status = ai.ScoreStatusFail
			}
			score = ai.Score{
				Score:  match,
				Status: status.String(),
			}
		} else {
			// Mark as failed if output is not string type
			logger.FromContext(ctx).Debug("genkitEval",
				"regex", fmt.Sprintf("Failed regex evaluation, as output is not string api. TestCaseId: %s", dataPoint.TestCaseId))
			score = ai.Score{
				Score:  false,
				Status: ai.ScoreStatusFail.String(),
			}
		}
		callbackResponse := ai.EvaluatorCallbackResponse{
			TestCaseId: req.Input.TestCaseId,
			Evaluation: []ai.Score{score},
		}
		return &callbackResponse, nil
	})
}

func configureDeepEqualEvaluator() ai.Evaluator {
	evalOptions := ai.EvaluatorOptions{
		DisplayName: "Deep Equal",
		Definition:  "Tests equality of output against the provided reference",
		IsBilled:    false,
	}
	return ai.NewEvaluator(api.NewName(provider, "deep_equal"), &evalOptions, func(ctx context.Context, req *ai.EvaluatorCallbackRequest) (*ai.EvaluatorCallbackResponse, error) {
		dataPoint := req.Input
		var score ai.Score
		if dataPoint.Output == nil {
			return nil, errors.New("output was not provided")
		}
		if dataPoint.Reference == nil {
			return nil, errors.New("reference was not provided")
		}
		deepEqual := reflect.DeepEqual(dataPoint.Reference, dataPoint.Output)
		status := ai.ScoreStatusUnknown
		if deepEqual {
			status = ai.ScoreStatusPass
		} else {
			status = ai.ScoreStatusFail
		}
		score = ai.Score{
			Score:  deepEqual,
			Status: status.String(),
		}

		callbackResponse := ai.EvaluatorCallbackResponse{
			TestCaseId: req.Input.TestCaseId,
			Evaluation: []ai.Score{score},
		}
		return &callbackResponse, nil
	})
}

func configureJsonataEvaluator() ai.Evaluator {
	evalOptions := ai.EvaluatorOptions{
		DisplayName: "JSONata",
		Definition:  "Tests JSONata expression (provided in reference) against output",
		IsBilled:    false,
	}
	return ai.NewEvaluator(api.NewName(provider, "jsonata"), &evalOptions, func(ctx context.Context, req *ai.EvaluatorCallbackRequest) (*ai.EvaluatorCallbackResponse, error) {
		dataPoint := req.Input
		var score ai.Score
		if dataPoint.Output == nil {
			return nil, errors.New("output was not provided")
		}
		if dataPoint.Reference == nil {
			return nil, errors.New("reference was not provided")
		}
		if reflect.TypeOf(dataPoint.Reference).String() != "string" {
			return nil, errors.New("reference must be a string (jsonata)")
		}
		// Test against provided jsonata
		exp := jsonata.MustCompile((dataPoint.Reference).(string))
		res, err := exp.Eval(dataPoint.Output)
		if err != nil {
			return nil, err
		}
		status := ai.ScoreStatusUnknown
		if res == false || res == "" || res == nil {
			status = ai.ScoreStatusFail
		} else {
			status = ai.ScoreStatusPass
		}
		score = ai.Score{
			Score:  res,
			Status: status.String(),
		}

		callbackResponse := ai.EvaluatorCallbackResponse{
			TestCaseId: req.Input.TestCaseId,
			Evaluation: []ai.Score{score},
		}
		return &callbackResponse, nil
	})
}
