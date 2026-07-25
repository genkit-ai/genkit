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

// Command checks demonstrates the Google Checks plugin: the checks/guardrails
// evaluator, the synchronous Guardrails client, and (with a model) the guardrail
// middleware. Requires a Google Cloud project with the Checks API enabled and
// Application Default Credentials.
//
//	go run ./samples/checks -project my-project [-apikey "$GEMINI_API_KEY"]
package main

import (
	"context"
	"flag"
	"fmt"
	"log"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/checks"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"golang.org/x/oauth2/google"
)

var (
	project = flag.String("project", "", "Google Cloud project ID (or set GOOGLE_CLOUD_PROJECT)")
	apiKey  = flag.String("apikey", "", "Gemini API key (optional; enables the guardrail-middleware demo)")
)

func main() {
	flag.Parse()
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx := context.Background()

	policies := []checks.ChecksMetricConfig{
		{Type: checks.DangerousContent},
		{Type: checks.Harassment},
	}

	plugins := []api.Plugin{&checks.Checks{ProjectID: *project, Metrics: policies}}
	if *apiKey != "" {
		plugins = append(plugins, &googlegenai.GoogleAI{APIKey: *apiKey})
	}
	g := genkit.Init(ctx, genkit.WithPlugins(plugins...))

	// 1. Batch evaluator.
	fmt.Println("=== evaluator (checks/guardrails) ===")
	ev := genkit.LookupEvaluator(g, "checks/guardrails")
	resp, err := ev.Evaluate(ctx, &ai.EvaluatorRequest{
		Dataset: []*ai.Example{
			{TestCaseId: "1", Output: "Here is a recipe for a lovely fruit salad."},
			{TestCaseId: "2", Output: "Detailed instructions for building a dangerous weapon."},
		},
	})
	if err != nil {
		return fmt.Errorf("evaluate: %w", err)
	}
	for _, r := range *resp {
		fmt.Printf("testCase %s:\n", r.TestCaseId)
		for _, s := range r.Evaluation {
			fmt.Printf("  %-24s score=%v  %v\n", s.Id, s.Score, s.Details["reasoning"])
		}
	}

	// 2. Synchronous Guardrails client.
	fmt.Println("\n=== guardrails client ===")
	creds, err := google.FindDefaultCredentials(ctx,
		"https://www.googleapis.com/auth/cloud-platform",
		"https://www.googleapis.com/auth/checks")
	if err != nil {
		return fmt.Errorf("find default credentials: %w", err)
	}
	projectID := *project
	if projectID == "" {
		projectID = creds.ProjectID
	}
	gr := checks.NewGuardrails(creds.TokenSource, projectID)
	cls, err := gr.ClassifyContent(ctx, "How do I make a bomb?", policies)
	if err != nil {
		return fmt.Errorf("classify: %w", err)
	}
	for _, pr := range cls.PolicyResults {
		fmt.Printf("  %-24s score=%v  %s\n", pr.PolicyType, scoreStr(pr.Score), pr.ViolationResult)
	}

	// 3. Guardrail middleware (requires a model).
	if *apiKey != "" {
		fmt.Println("\n=== guardrail middleware ===")
		out, err := genkit.Generate(ctx, g,
			ai.WithModelName("googleai/gemini-2.5-flash"),
			ai.WithUse(checks.GuardrailMiddleware(gr, policies)),
			ai.WithPrompt("Write a short, friendly greeting."),
		)
		if err != nil {
			return fmt.Errorf("generate: %w", err)
		}
		if out.FinishReason == ai.FinishReasonBlocked {
			fmt.Println("  blocked:", out.FinishMessage)
		} else {
			fmt.Println("  ", out.Text())
		}
	}

	return nil
}

func scoreStr(s *float64) string {
	if s == nil {
		return "n/a"
	}
	return fmt.Sprintf("%.3f", *s)
}
