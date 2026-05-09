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

package evaluators_test

import (
	"context"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/evaluators"
)

func TestEvaluators(t *testing.T) {
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
	g := genkit.Init(ctx,
		genkit.WithPlugins(&evaluators.GenkitEval{Metrics: metrics}))

	t.Run("deep equal", func(t *testing.T) {
		var dataset = []*ai.Example{
			{
				Input:     "sample",
				Reference: "hello world",
				Output:    "hello world",
			},
			{
				Input:     "sample",
				Output:    "Foo bar",
				Reference: "gablorken",
			},
			{
				Input:  "sample",
				Output: "Foo bar",
			},
		}
		var testRequest = ai.EvaluatorRequest{
			Dataset:      dataset,
			EvaluationId: "testrun",
		}

		evalAction := genkit.LookupEvaluator(g, "genkitEval/deep_equal")
		if evalAction == nil {
			t.Fatal("evalAction is nil")
		}
		resp, err := evalAction.Evaluate(ctx, &testRequest)
		if err != nil {
			t.Fatal(err)
		}
		if got, want := (*resp)[0].Evaluation[0].Score, true; got != want {
			t.Errorf("got %v, want %v", got, want)
		}
		if got, want := (*resp)[1].Evaluation[0].Score, false; got != want {
			t.Errorf("got %v, want %v", got, want)
		}
		if got := (*resp)[2].Evaluation[0].Error; got == "" {
			t.Errorf("got %v, want error", got)
		}
	})

	t.Run("regex", func(t *testing.T) {
		var dataset = []*ai.Example{
			{
				Input:     "sample",
				Reference: "ba?a?a",
				Output:    "banana",
			},
			{
				Input:     "sample",
				Reference: "ba?a?a",
				Output:    "apple",
			},
			{
				Input:     "sample",
				Reference: 12345,
				Output:    "apple",
			},
		}
		var testRequest = ai.EvaluatorRequest{
			Dataset:      dataset,
			EvaluationId: "testrun",
		}

		evalAction := genkit.LookupEvaluator(g, "genkitEval/regex")
		resp, err := evalAction.Evaluate(ctx, &testRequest)
		if err != nil {
			t.Fatal(err)
		}
		if got, want := (*resp)[0].Evaluation[0].Score, true; got != want {
			t.Errorf("got %v, want %v", got, want)
		}
		if got, want := (*resp)[1].Evaluation[0].Score, false; got != want {
			t.Errorf("got %v, want %v", got, want)
		}
		if got := (*resp)[2].Evaluation[0].Error; got == "" {
			t.Errorf("got %v, want error", got)
		}
	})

	t.Run("jsonata", func(t *testing.T) {
		var dataset = []*ai.Example{
			{
				Input:     "sample",
				Reference: "age=33",
				Output: map[string]any{
					"name": "Bob",
					"age":  33,
				},
			},
			{
				Input:     "sample",
				Reference: "age=31",
				Output: map[string]any{
					"name": "Bob",
					"age":  33,
				},
			},
			{
				Input:     "sample",
				Reference: 123456,
				Output: map[string]any{
					"name": "Bob",
					"age":  33,
				},
			},
		}
		var testRequest = ai.EvaluatorRequest{
			Dataset:      dataset,
			EvaluationId: "testrun",
		}

		evalAction := genkit.LookupEvaluator(g, "genkitEval/jsonata")
		resp, err := evalAction.Evaluate(ctx, &testRequest)
		if err != nil {
			t.Fatal(err)
		}
		if got, want := (*resp)[0].Evaluation[0].Score, true; got != want {
			t.Errorf("got %v, want %v", got, want)
		}
		if got, want := (*resp)[1].Evaluation[0].Score, false; got != want {
			t.Errorf("got %v, want %v", got, want)
		}
		if got := (*resp)[2].Evaluation[0].Error; got == "" {
			t.Errorf("got %v, want error", got)
		}
	})
}
