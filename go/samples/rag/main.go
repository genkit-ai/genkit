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

// This program can be manually tested like so:
//
// In development mode (with the environment variable GENKIT_ENV="dev"):
// Start the server listening on port 3100:
//
//	go run . &
//
// Tell it to run a flow:
//
//	curl -d '{"key":"/flow/simpleQaFlow/simpleQaFlow", "input":{"start": {"input":{"question": "What is the capital of UK?"}}}}' http://localhost:3100/api/runAction
//
// In production mode (GENKIT_ENV missing or set to "prod"):
// Start the server listening on port 3400:
//
//	go run . &
//
// Tell it to run a flow:
//
//   curl -d '{"question": "What is the capital of UK?"}' http://localhost:3400/simpleQaFlow

package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/evaluators"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/localvec"
)

const simpleQaPromptTemplate = `
You're a helpful agent that answers the user's common questions based on the context provided.

Here is the user's query: {{query}}

Here is the context you should use: {{context}}

Please provide the best answer you can.
`

type simpleQaInput struct {
	Question string `json:"question"`
}

type simpleQaPromptInput struct {
	Query   string `json:"query"`
	Context string `json:"context"`
}

func main() {
	ctx := context.Background()
	metrics := []evaluators.MetricConfig{
		{
			MetricType: evaluators.EvaluatorDeepEqual,
		},
		{
			MetricType: evaluators.EvaluatorRegex,
		},
		{
			MetricType: evaluators.EvaluatorJsonata,
		},
	}
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}, &evaluators.GenkitEval{Metrics: metrics}))

	embedder := googlegenai.GoogleAIEmbedder(g, "embedding-001")
	if embedder == nil {
		log.Fatal("embedder is not defined")
	}

	if err := localvec.Init(); err != nil {
		log.Fatal(err)
	}
	retOpts := &ai.RetrieverOptions{
		ConfigSchema: core.InferSchemaMap(localvec.RetrieverOptions{}),
		Label:        "simpleQa",
		Supports: &ai.RetrieverSupports{
			Media: false,
		},
	}
	docStore, retriever, err := localvec.DefineRetriever(g, "simpleQa", localvec.Config{Embedder: embedder}, retOpts)
	if err != nil {
		log.Fatal(err)
	}

	simpleQaPrompt := genkit.DefinePrompt(g, "simpleQaPrompt",
		ai.WithModelName("googleai/gemini-2.5-flash"),
		ai.WithPrompt(simpleQaPromptTemplate),
		ai.WithInputType(simpleQaPromptInput{}),
		ai.WithOutputFormat(ai.OutputFormatText),
	)

	// Dummy evaluator for testing
	evalOptions := ai.EvaluatorOptions{
		DisplayName: "Simple Evaluator",
		Definition:  "Just says true or false randomly",
		IsBilled:    false,
	}

	genkit.DefineEvaluator(g, api.NewName("custom", "simpleEvaluator"), &evalOptions, func(ctx context.Context, req *ai.EvaluatorCallbackRequest) (*ai.EvaluatorCallbackResponse, error) {
		m := make(map[string]any)
		m["reasoning"] = "No good reason"
		score := ai.Score{
			Id:      "testScore",
			Score:   1,
			Status:  ai.ScoreStatusPass.String(),
			Details: m,
		}
		callbackResponse := ai.EvaluatorCallbackResponse{
			TestCaseId: req.Input.TestCaseId,
			Evaluation: []ai.Score{score},
		}
		return &callbackResponse, nil
	})

	genkit.DefineBatchEvaluator(g, api.NewName("custom", "simpleBatchEvaluator"), &evalOptions, func(ctx context.Context, req *ai.EvaluatorRequest) (*ai.EvaluatorResponse, error) {
		var evalResponses []ai.EvaluationResult
		for _, datapoint := range req.Dataset {
			m := make(map[string]any)
			m["reasoning"] = fmt.Sprintf("batch of cookies, %s", datapoint.Input)
			score := ai.Score{
				Id:      "testScore",
				Score:   true,
				Status:  ai.ScoreStatusPass.String(),
				Details: m,
			}
			callbackResponse := ai.EvaluationResult{
				TestCaseId: datapoint.TestCaseId,
				Evaluation: []ai.Score{score},
			}
			evalResponses = append(evalResponses, callbackResponse)
		}
		return &evalResponses, nil
	})

	genkit.DefineFlow(g, "simpleQaFlow", func(ctx context.Context, input *simpleQaInput) (string, error) {
		d1 := ai.DocumentFromText("Paris is the capital of France", nil)
		d4 := ai.DocumentFromText("India will become a new capital of France", nil)
		d2 := ai.DocumentFromText("USA is the largest importer of coffee", nil)
		d3 := ai.DocumentFromText("Water exists in 3 states - solid, liquid and gas", nil)

		err := localvec.Index(ctx, []*ai.Document{d1, d2, d3, d4}, docStore)
		if err != nil {
			return "", err
		}

		dRequest := ai.DocumentFromText(input.Question, nil)
		response, err := genkit.Retrieve(ctx, g,
			ai.WithRetriever(retriever),
			ai.WithDocs(dRequest),
			ai.WithConfig(&localvec.RetrieverOptions{K: 2}))
		if err != nil {
			return "", err
		}

		var sb strings.Builder
		for _, d := range response.Documents {
			sb.WriteString(d.Content[0].Text)
			sb.WriteByte('\n')
		}

		promptInput := &simpleQaPromptInput{
			Query:   input.Question,
			Context: sb.String(),
		}

		resp, err := simpleQaPrompt.Execute(ctx, ai.WithInput(promptInput))
		if err != nil {
			return "", err
		}
		return resp.Text(), nil
	})

	genkit.DefineFlow(g, "echoFlow", func(ctx context.Context, input string) (string, error) {
		return input, nil
	})

	<-ctx.Done()
}
