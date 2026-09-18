// Copyright 2026 Google LLC
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

package exp

import (
	"math"
	"os"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/internal/base"
)

// The live checks run jev through OpenRouter, the gateway an ordinary key
// can reach today; TypeSafe's own API is behind a waitlist.
func TestOpenRouterLive(t *testing.T) {
	if os.Getenv("OPENROUTER_API_KEY") == "" {
		t.Skip("OPENROUTER_API_KEY is not set")
	}
	g := genkit.Init(t.Context(), genkit.WithPlugins(&TypeSafe{Endpoint: OpenRouter()}))
	model := Model(g, "jev-latest")

	t.Run("decision", func(t *testing.T) {
		// The documented call: the model, the state, and the type.
		out, resp, err := genkit.GenerateData[triage](t.Context(), g,
			ai.WithModel(model),
			ai.WithPromptParts(ai.NewDataPart(map[string]any{
				"ticket":       "I was charged twice for one order and I need the duplicate refunded today.",
				"account_tier": "business",
			})))
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("answers: %s", base.JSONString(out))
		t.Logf("custom: %s", base.JSONString(resp.Custom))

		if _, ok := out.Department.Choice.Criteria()[out.Department.Choice]; !ok {
			t.Errorf("choice %q is outside the criteria", out.Department.Choice)
		}
		if out.Department.Choice != "billing" {
			t.Errorf("department = %q, want billing for a double charge", out.Department.Choice)
		}
		var mass float64
		for _, p := range out.Department.Probabilities {
			mass += p
		}
		if math.Abs(mass-1) > 0.05 {
			t.Errorf("probabilities sum to %v, want about 1: %v", mass, out.Department.Probabilities)
		}
		if out.Department.Confidence <= 0 || out.Department.Confidence > 1 {
			t.Errorf("confidence = %v, want within (0, 1]", out.Department.Confidence)
		}
		if out.IsUrgent.Probability < 0.5 {
			t.Errorf("is_urgent = %v, want over 0.5 for \"today\"", out.IsUrgent.Probability)
		}
		if out.Frustration.Score < 0 || out.Frustration.Score > 2 || len(out.Frustration.Legend) != 3 {
			t.Errorf("frustration = %+v", out.Frustration)
		}
		if resp.Usage == nil || resp.Usage.InputTokens == 0 {
			t.Errorf("usage = %+v", resp.Usage)
		}
		custom, _ := resp.Custom.(map[string]any)
		if model, _ := custom["model"].(string); model == "" {
			t.Errorf("no resolved model version on the response: %v", resp.Custom)
		}
	})

	t.Run("decision with preamble", func(t *testing.T) {
		out, _, err := genkit.GenerateData[triage](t.Context(), g,
			ai.WithModel(model),
			ai.WithSystem("The state is a support ticket from a business customer of an online store."),
			ai.WithPromptParts(ai.NewDataPart(map[string]any{
				"ticket": "I was charged twice for one order and I need the duplicate refunded today.",
			})))
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("answers with preamble: %s", base.JSONString(out))
		if out.Department.Choice != "billing" || out.IsUrgent.Probability < 0.5 {
			t.Errorf("with a preamble: department = %q, is_urgent = %v", out.Department.Choice, out.IsUrgent.Probability)
		}
	})

	t.Run("enum", func(t *testing.T) {
		resp, err := genkit.Generate(t.Context(), g,
			ai.WithModel(model),
			ai.WithSystem("Which team should handle this ticket?"),
			ai.WithOutputEnums("billing", "technical", "sales"),
			ai.WithPrompt("The API returns 500 errors since this morning's deploy."))
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("enum: %q custom: %s", resp.Text(), base.JSONString(resp.Custom))
		if resp.Text() != "technical" {
			t.Errorf("text = %q, want technical", resp.Text())
		}
	})

	t.Run("history as state", func(t *testing.T) {
		type handoff struct {
			WantsHuman Noul `json:"wants_human" jsonschema_description:"Does the user ask to talk to a human?"`
		}
		out, _, err := genkit.GenerateData[handoff](t.Context(), g,
			ai.WithModel(model),
			ai.WithMessages(
				ai.NewUserMessage(ai.NewTextPart("Hi, I cannot log in.")),
				ai.NewModelMessage(ai.NewTextPart("Let me help. Have you tried resetting your password?")),
				ai.NewUserMessage(ai.NewTextPart("Yes, three times. Can I please just talk to a person?")),
			))
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("wants_human: %v", out.WantsHuman.Probability)
		if out.WantsHuman.Probability < 0.5 {
			t.Errorf("wants_human = %v, want over 0.5", out.WantsHuman.Probability)
		}
	})
}
