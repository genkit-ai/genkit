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

package checks

import (
	"context"
	"fmt"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"golang.org/x/sync/errgroup"
)

// evaluatorConcurrency bounds parallel classifyContent calls; Checks rate-limits
// per project.
const evaluatorConcurrency = 4

// newEvaluator builds the checks/guardrails batch evaluator, which scores each
// datapoint's output against all configured policies in one API call.
func newEvaluator(c *Checks) ai.Evaluator {
	g := newGuardrails(c.ts, c.projectID, c.classifyURL())

	opts := &ai.EvaluatorOptions{
		DisplayName: "checks/guardrails",
		Definition:  "Evaluates input text against the configured Checks AI-safety policies.",
		IsBilled:    true,
	}

	fn := func(ctx context.Context, req *ai.EvaluatorRequest) (*ai.EvaluatorResponse, error) {
		results := make([]ai.EvaluationResult, len(req.Dataset))

		eg, egctx := errgroup.WithContext(ctx)
		eg.SetLimit(evaluatorConcurrency)
		for i, dp := range req.Dataset {
			eg.Go(func() error {
				results[i] = evaluateOne(egctx, g, c.Metrics, dp)
				return nil
			})
		}
		if err := eg.Wait(); err != nil {
			return nil, err
		}

		resp := ai.EvaluatorResponse(results)
		return &resp, nil
	}

	return ai.NewBatchEvaluator(api.NewName(provider, "guardrails"), opts, fn)
}

// evaluateOne classifies a single datapoint. A failed call is isolated to that
// datapoint's result (as a Score with Error) rather than failing the whole run.
func evaluateOne(ctx context.Context, g *Guardrails, metrics []ChecksMetricConfig, dp *ai.Example) ai.EvaluationResult {
	res := ai.EvaluationResult{TestCaseId: dp.TestCaseId}

	resp, err := g.ClassifyContent(ctx, outputText(dp.Output), metrics)
	if err != nil {
		res.Evaluation = []ai.Score{{Error: err.Error()}}
		return res
	}

	scores := make([]ai.Score, 0, len(resp.PolicyResults))
	for _, pr := range resp.PolicyResults {
		s := ai.Score{
			Id:      pr.PolicyType,
			Details: map[string]any{"reasoning": "Status " + pr.ViolationResult},
		}
		// Score is optional in the API; only set it when present so an absent
		// score is omitted rather than serialized as null.
		if pr.Score != nil {
			s.Score = *pr.Score
		}
		scores = append(scores, s)
	}
	res.Evaluation = scores
	return res
}

// outputText extracts the text to classify from an Example.Output (the field
// the JS evaluator uses).
func outputText(v any) string {
	switch t := v.(type) {
	case nil:
		return ""
	case string:
		return t
	default:
		return fmt.Sprint(t)
	}
}
