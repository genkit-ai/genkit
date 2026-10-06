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
	"errors"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/status"
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
	const model = "typesafe/jev-latest"

	t.Run("decision", func(t *testing.T) {
		// The documented call: the model, the state, and the type.
		out, resp, err := genkit.GenerateData[triage](t.Context(), g,
			ai.WithModelName(model),
			ai.WithPromptParts(ai.NewDataPart(map[string]any{
				"ticket":       "I was charged twice for one order and I need the duplicate refunded today.",
				"account_tier": "business",
			})))
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("answers: %s", base.JSONString(out))
		t.Logf("info: %s, usage: %s", base.JSONString(ResponseInfo(resp)), base.JSONString(resp.Usage))

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
		if ResponseInfo(resp).Model == "" {
			t.Errorf("no resolved model version on the response: %v", resp.Raw)
		}
		if resp.Usage.Custom["cost"] <= 0 {
			t.Errorf("usage custom = %v, want the gateway's cost", resp.Usage.Custom)
		}
	})

	t.Run("decision with preamble", func(t *testing.T) {
		out, _, err := genkit.GenerateData[triage](t.Context(), g,
			ai.WithModelName(model),
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

	t.Run("guidance", func(t *testing.T) {
		// Object criteria on an option, a level, and a side: the gateway
		// takes them, and the legend keeps the rubric's strings.
		out, resp, err := genkit.GenerateData[guidedTriage](t.Context(), g,
			ai.WithModelName(model),
			ai.WithPromptParts(ai.NewDataPart(map[string]any{
				"ticket": "THIS IS THE THIRD TIME. I was charged twice and I want my money back TODAY.",
			})))
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("answers: %s", base.JSONString(out))
		t.Logf("info: %s, usage: %s", base.JSONString(ResponseInfo(resp)), base.JSONString(resp.Usage))
		if out.Department.Choice != "billing" {
			t.Errorf("department = %q, want billing", out.Department.Choice)
		}
		if out.Frustration.Legend["2"] != "Very angry" {
			t.Errorf("legend = %v, want the rubric strings", out.Frustration.Legend)
		}
	})

	t.Run("enum", func(t *testing.T) {
		resp, err := genkit.Generate(t.Context(), g,
			ai.WithModelName(model),
			ai.WithSystem("Which team should handle this ticket?"),
			ai.WithOutputEnums("billing", "technical", "sales"),
			ai.WithPrompt("The API returns 500 errors since this morning's deploy."))
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("enum: %q info: %s", resp.Text(), base.JSONString(ResponseInfo(resp)))
		if resp.Text() != "technical" {
			t.Errorf("text = %q, want technical", resp.Text())
		}
	})

	t.Run("history as state", func(t *testing.T) {
		type handoff struct {
			WantsHuman Noul `json:"wants_human" jsonschema_description:"Does the user ask to talk to a human?"`
		}
		out, _, err := genkit.GenerateData[handoff](t.Context(), g,
			ai.WithModelName(model),
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

	t.Run("pinned minor version", func(t *testing.T) {
		_, resp, err := genkit.GenerateData[triage](t.Context(), g,
			ai.WithModelName("typesafe/jev-1.13"),
			ai.WithPrompt("I was charged twice for one order."))
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("jev-1.13 resolved to %v", ResponseInfo(resp).Model)
	})

	t.Run("patch version refused", func(t *testing.T) {
		_, _, err := genkit.GenerateData[triage](t.Context(), g,
			ai.WithModelName("typesafe/jev-1.13.0"),
			ai.WithPrompt("I was charged twice for one order."))
		if !errors.Is(err, status.ErrInvalidArgument) {
			t.Errorf("error = %v, want invalid argument", err)
		}
	})

	t.Run("preview alias", func(t *testing.T) {
		// The alias is forwarded rather than served as latest, so the
		// gateway says whether it has a preview channel.
		_, resp, err := genkit.GenerateData[triage](t.Context(), g,
			ai.WithModelName("typesafe/jev-preview"),
			ai.WithPrompt("I was charged twice for one order."))
		if err != nil {
			if !errors.Is(err, status.ErrInvalidArgument) || !strings.Contains(err.Error(), "jev-preview") {
				t.Errorf("error = %v, want the gateway's refusal of jev-preview", err)
			}
			t.Logf("no preview channel: %v", err)
			return
		}
		t.Logf("jev-preview resolved to %v", ResponseInfo(resp).Model)
	})

	t.Run("runtime questions", func(t *testing.T) {
		// Options from data, in the order given, with guidance on one;
		// structured instructions beside a system message.
		resp, err := genkit.Generate(t.Context(), g,
			ai.WithModelName(model),
			ai.WithSystem("The state is a request a user made to an assistant."),
			ai.WithOutputSchema(Schema(map[string]Question{
				"tool": ChoiceQuestion{
					Instructions: "Which tool serves the request?",
					Options: []ChoiceOption{
						{Name: "web_search", Criteria: "Look up facts, news, or prices on the web"},
						{Name: "calendar", Criteria: "Read or change the user's own calendar", Guidance: map[string]any{
							"examples": []string{"What meetings do I have on Friday?", "Move my 3pm to 4pm."},
						}},
						{Name: "none", Criteria: "No tool fits the request"},
					},
				},
				"effort": ScoreQuestion{
					Instructions: map[string]any{
						"question": "How much work does fulfilling the request take?",
						"field":    map[string]any{"name": "request", "description": "What the user asked for"},
					},
					Levels: []string{"A single lookup", "A few steps", "A multi-step project"},
				},
				"personal": NoulQuestion{
					Instructions: "Does the request involve the user's own data?",
					Yes:          "Names the user's files, mail, or calendar",
					No:           "Asks about the world at large",
				},
			})),
			ai.WithPromptParts(ai.NewDataPart(map[string]any{"request": "What is on my calendar tomorrow morning?"})))
		if err != nil {
			t.Fatal(err)
		}
		var answers map[string]Answer
		if err := resp.Output(&answers); err != nil {
			t.Fatal(err)
		}
		t.Logf("answers: %s", base.JSONString(answers))
		if a := answers["tool"]; a.Choice != "calendar" {
			t.Errorf("tool = %+v, want calendar", a)
		}
		if a := answers["personal"]; a.Probability < 0.5 {
			t.Errorf("personal = %+v, want over 0.5", a)
		}
		if a := answers["effort"]; a.Score < 0 || a.Score > 2 || a.Legend["0"] != "A single lookup" {
			t.Errorf("effort = %+v", a)
		}
	})

	t.Run("stateJSON with a large number", func(t *testing.T) {
		out, _, err := genkit.GenerateData[triage](t.Context(), g,
			ai.WithModelName(model),
			ai.WithConfig(&Config{StateJSON: true}),
			ai.WithPrompt(`{"order_id": 9007199254740993, "ticket": "I was charged twice for this order."}`))
		if err != nil {
			t.Fatal(err)
		}
		if out.Department.Choice != "billing" {
			t.Errorf("department = %q, want billing", out.Department.Choice)
		}
	})

	t.Run("document with data", func(t *testing.T) {
		type stock struct {
			InStock Noul `json:"in_stock" jsonschema_description:"Does the context show the item the user asks about as in stock?"`
		}
		out, _, err := genkit.GenerateData[stock](t.Context(), g,
			ai.WithModelName(model),
			ai.WithPrompt("Is the blue kettle available?"),
			ai.WithDocs(&ai.Document{Content: []*ai.Part{ai.NewDataPart(map[string]any{"item": "blue kettle", "stock": 12})}}))
		if err != nil {
			t.Fatal(err)
		}
		t.Logf("in_stock: %v", out.InStock.Probability)
		if out.InStock.Probability < 0.5 {
			t.Errorf("in_stock = %v, want over 0.5: the data document did not reach the state", out.InStock.Probability)
		}
	})
}
