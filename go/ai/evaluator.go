// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

package ai

import (
	"context"
	"fmt"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/core/tracing"
	"github.com/google/uuid"
	"go.opentelemetry.io/otel/trace"
)

// EvaluatorFunc is the function type for evaluator implementations.
// Config is the evaluator's typed configuration: the framework deserializes
// the request's raw config into it before calling the function (see [DefineEvaluator]).
type EvaluatorFunc[Config any] = func(context.Context, *EvaluatorCallbackRequest, Config) (*EvaluatorCallbackResponse, error)

// BatchEvaluatorFunc is the function type for batch evaluator implementations.
// Config is the evaluator's typed configuration: the framework deserializes
// the request's raw config into it before calling the function (see [DefineBatchEvaluator]).
type BatchEvaluatorFunc[Config any] = func(context.Context, *EvaluatorRequest, Config) (*EvaluatorResponse, error)

// Evaluator is a dataset evaluator backed by a registry action. Create one
// with [DefineEvaluator], [DefineBatchEvaluator], or their New* equivalents,
// or fetch a registered one with [LookupEvaluator]. Pass it to
// [WithEvaluator] to use it with [Evaluate].
type Evaluator struct {
	action[*EvaluatorRequest, *EvaluatorResponse, struct{}]
}

var (
	_ api.Action   = (*Evaluator)(nil)
	_ EvaluatorArg = (*Evaluator)(nil)
)

// Example is a single example that requires evaluation
type Example struct {
	TestCaseID string   `json:"testCaseId,omitempty"`
	Input      any      `json:"input"`
	Output     any      `json:"output,omitempty"`
	Context    []any    `json:"context,omitempty"`
	Reference  any      `json:"reference,omitempty"`
	TraceIDs   []string `json:"traceIds,omitempty"`
}

// EvaluatorRequest is the data we pass to evaluate a dataset.
// The Config field is specific to the actual evaluator implementation.
type EvaluatorRequest struct {
	Dataset      []*Example `json:"dataset"`
	EvaluationID string     `json:"evalRunId"`
	Config       any        `json:"options,omitempty"`
}

// ScoreStatus is an enum used to indicate if a Score has passed or failed. This
// drives additional features in tooling / the Dev UI.
type ScoreStatus int

const (
	ScoreStatusUnknown ScoreStatus = iota
	ScoreStatusFail
	ScoreStatusPass
)

var statusName = map[ScoreStatus]string{
	ScoreStatusUnknown: "UNKNOWN",
	ScoreStatusFail:    "FAIL",
	ScoreStatusPass:    "PASS",
}

func (ss ScoreStatus) String() string {
	return statusName[ss]
}

// Score is the evaluation score that represents the result of an evaluator.
// This struct includes information such as the score (numeric, string or other
// types), the reasoning provided for this score (if any), the score status (if
// any) and other details.
type Score struct {
	ID      string         `json:"id,omitempty"`
	Score   any            `json:"score,omitempty"`
	Status  string         `json:"status,omitempty" jsonschema:"enum=UNKNOWN,enum=FAIL,enum=PASS"`
	Error   string         `json:"error,omitempty"`
	Details map[string]any `json:"details,omitempty"`
}

// EvaluationResult is the result of running the evaluator on a single Example.
// An evaluator may provide multiple scores simultaneously (e.g. if they are using
// an API to score on multiple criteria)
type EvaluationResult struct {
	TestCaseID string  `json:"testCaseId"`
	TraceID    string  `json:"traceId,omitempty"`
	SpanID     string  `json:"spanId,omitempty"`
	Evaluation []Score `json:"evaluation"`
}

// EvaluatorResponse is a collection of [EvaluationResult] structs, it
// represents the result on the entire input dataset.
type EvaluatorResponse = []EvaluationResult

type EvaluatorOptions struct {
	// ConfigSchema is the JSON schema for the evaluator's config.
	ConfigSchema map[string]any `json:"configSchema,omitempty"`
	// DisplayName is the name of the evaluator as it appears in the UI.
	DisplayName string `json:"displayName"`
	// Definition is the definition of the evaluator.
	Definition string `json:"definition"`
	// IsBilled is a flag indicating if the evaluator is billed.
	IsBilled bool `json:"isBilled,omitempty"`
}

// EvaluatorCallbackRequest is the data we pass to the callback function
// provided in defineEvaluator. The Config field is specific to the actual
// evaluator implementation.
type EvaluatorCallbackRequest struct {
	Input  Example `json:"input"`
	Config any     `json:"options,omitempty"`
}

// EvaluatorCallbackResponse is the result on evaluating a single [Example]
type EvaluatorCallbackResponse = EvaluationResult

// NewEvaluator creates a new unregistered [Evaluator].
// This method processes the input dataset one-by-one.
//
// Config is the evaluator's typed configuration; it is usually inferred from
// fn's signature. The framework deserializes the request's raw config into
// Config before calling fn: the exact Config type (or a pointer to it) and
// map[string]any (from the Dev UI and other JSON callers) are accepted, and
// mismatched types are rejected. The config's JSON schema is inferred from
// Config unless [EvaluatorOptions.ConfigSchema] overrides it.
func NewEvaluator[Config any](name string, opts *EvaluatorOptions, fn EvaluatorFunc[Config]) *Evaluator {
	if name == "" {
		panic("ai.NewEvaluator: evaluator name is required")
	}

	if opts == nil {
		opts = &EvaluatorOptions{}
	}

	// TODO(ssbushi): Set this on `evaluator` key on action metadata
	metadata := map[string]any{
		"type": api.ActionTypeEvaluator,
		"evaluator": map[string]any{
			"evaluatorIsBilled":    opts.IsBilled,
			"evaluatorDisplayName": opts.DisplayName,
			"evaluatorDefinition":  opts.Definition,
		},
	}

	_, inputSchema := actionConfigSchemas[Config](opts.ConfigSchema, EvaluatorRequest{}, "options")

	return &Evaluator{
		action: *core.NewAction(api.ActionTypeEvaluator, name, &core.ActionOptions{
			Metadata:    metadata,
			InputSchema: inputSchema,
		}, func(ctx context.Context, req *EvaluatorRequest) (output *EvaluatorResponse, err error) {
			cfg, err := resolveConfig[Config](req.Config)
			if err != nil {
				return nil, err
			}
			// Normalize the request so its type-erased Config always carries
			// the same converted value the typed parameter does.
			req.Config = cfg

			var results []EvaluationResult
			for _, datapoint := range req.Dataset {
				if datapoint.TestCaseID == "" {
					datapoint.TestCaseID = uuid.New().String()
				}
				spanMetadata := &tracing.SpanMetadata{
					Name:    fmt.Sprintf("TestCase %s", datapoint.TestCaseID),
					Type:    "evaluator",
					Subtype: "evaluator",
				}
				_, err := tracing.RunInNewSpan(ctx, spanMetadata, datapoint,
					func(ctx context.Context, input *Example) (*EvaluatorCallbackResponse, error) {
						traceId := trace.SpanContextFromContext(ctx).TraceID().String()
						spanId := trace.SpanContextFromContext(ctx).SpanID().String()

						callbackRequest := EvaluatorCallbackRequest{
							Input:  *input,
							Config: cfg,
						}

						result, err := fn(ctx, &callbackRequest, cfg)
						if err != nil {
							failedScore := Score{
								Status: ScoreStatusFail.String(),
								Error:  fmt.Sprintf("Evaluation of test case %s failed: \n %s", input.TestCaseID, err.Error()),
							}
							failedResult := EvaluationResult{
								TestCaseID: input.TestCaseID,
								Evaluation: []Score{failedScore},
								TraceID:    traceId,
								SpanID:     spanId,
							}
							results = append(results, failedResult)

							return nil, err
						}

						result.TraceID = traceId
						result.SpanID = spanId

						results = append(results, *result)

						return result, nil
					})
				if err != nil {
					logger.FromContext(ctx).Debug("EvaluatorAction", "err", err)
					continue
				}
			}
			return &results, nil
		}),
	}
}

// DefineEvaluator creates a new [Evaluator] and registers it.
// This method processes the input dataset one-by-one.
//
// Config is the evaluator's typed configuration; it is usually inferred from
// fn's signature. See [NewEvaluator] for how the request's config is deserialized.
func DefineEvaluator[Config any](r api.Registry, name string, opts *EvaluatorOptions, fn EvaluatorFunc[Config]) *Evaluator {
	e := NewEvaluator(name, opts, fn)
	e.Register(r)
	return e
}

// NewBatchEvaluator creates a new unregistered [Evaluator].
// This method provides the full [EvaluatorRequest] to the callback function,
// giving more flexibility to the user for processing the data, such as batching or parallelization.
//
// Config is the evaluator's typed configuration; it is usually inferred from
// fn's signature. See [NewEvaluator] for how the request's config is deserialized.
func NewBatchEvaluator[Config any](name string, opts *EvaluatorOptions, fn BatchEvaluatorFunc[Config]) *Evaluator {
	if name == "" {
		panic("ai.NewBatchEvaluator: batch evaluator name is required")
	}

	if opts == nil {
		opts = &EvaluatorOptions{}
	}

	metadata := map[string]any{
		"type": api.ActionTypeEvaluator,
		"evaluator": map[string]any{
			"evaluatorIsBilled":    opts.IsBilled,
			"evaluatorDisplayName": opts.DisplayName,
			"evaluatorDefinition":  opts.Definition,
		},
	}

	_, inputSchema := actionConfigSchemas[Config](opts.ConfigSchema, EvaluatorRequest{}, "options")

	rawFn := func(ctx context.Context, req *EvaluatorRequest) (*EvaluatorResponse, error) {
		cfg, err := resolveConfig[Config](req.Config)
		if err != nil {
			return nil, err
		}
		// Normalize the request so its type-erased Config always carries the
		// same converted value the typed parameter does.
		req.Config = cfg
		return fn(ctx, req, cfg)
	}

	return &Evaluator{
		action: *core.NewAction(api.ActionTypeEvaluator, name, &core.ActionOptions{
			Metadata:    metadata,
			InputSchema: inputSchema,
		}, rawFn),
	}
}

// DefineBatchEvaluator creates a new [Evaluator] and registers it.
// This method provides the full [EvaluatorRequest] to the callback function,
// giving more flexibility to the user for processing the data, such as batching or parallelization.
//
// Config is the evaluator's typed configuration; it is usually inferred from
// fn's signature. See [NewEvaluator] for how the request's config is deserialized.
func DefineBatchEvaluator[Config any](r api.Registry, name string, opts *EvaluatorOptions, fn BatchEvaluatorFunc[Config]) *Evaluator {
	e := NewBatchEvaluator(name, opts, fn)
	e.Register(r)
	return e
}

// LookupEvaluator looks up an [Evaluator] registered by [DefineEvaluator].
// It returns nil if the evaluator was not defined.
func LookupEvaluator(r api.Registry, name string) *Evaluator {
	action := core.ResolveActionFor[*EvaluatorRequest, *EvaluatorResponse, struct{}](r, api.ActionTypeEvaluator, name)
	if action == nil {
		return nil
	}
	return &Evaluator{
		action: *action,
	}
}

// Name returns the registry name of the evaluator, or the empty string if the
// evaluator is nil (e.g. from a failed lookup).
func (e *Evaluator) Name() string {
	if e == nil {
		return ""
	}
	return e.action.Name()
}

func (e *Evaluator) evaluatorArg() {}

// Evaluate runs the given [Evaluator].
func (e *Evaluator) Evaluate(ctx context.Context, req *EvaluatorRequest) (*EvaluatorResponse, error) {
	if e == nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "Evaluator.Evaluate: evaluator called on a nil evaluator; check that all evaluators are defined")
	}

	return e.Run(ctx, req, nil)
}

// Evaluate calls the retrivers with provided options.
func Evaluate(ctx context.Context, r api.Registry, opts ...EvaluatorOption) (*EvaluatorResponse, error) {
	evalOpts := &evaluatorOptions{}
	for _, opt := range opts {
		if err := opt.applyEvaluator(evalOpts); err != nil {
			return nil, err
		}
	}

	if evalOpts.Evaluator == nil {
		return nil, fmt.Errorf("ai.Evaluate: evaluator must be set")
	}
	e, ok := evalOpts.Evaluator.(*Evaluator)
	if !ok {
		e = LookupEvaluator(r, evalOpts.Evaluator.Name())
	}
	if e == nil {
		return nil, fmt.Errorf("ai.Evaluate: evaluator not found: %s", evalOpts.Evaluator.Name())
	}

	if ref, ok := evalOpts.Evaluator.(EvaluatorRef); ok && evalOpts.Config == nil {
		evalOpts.Config = ref.Config()
	}

	req := &EvaluatorRequest{
		Dataset:      evalOpts.Dataset,
		EvaluationID: evalOpts.ID,
		Config:       evalOpts.Config,
	}

	return e.Evaluate(ctx, req)
}
