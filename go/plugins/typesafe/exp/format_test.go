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
	"encoding/json"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
)

func newHandler(t *testing.T) *decisionHandler {
	t.Helper()
	h, err := DecisionFormat{}.Handler(triageSchema(t))
	if err != nil {
		t.Fatal(err)
	}
	return h.(*decisionHandler)
}

func TestDecisionFormatRejectsBeforeTheModel(t *testing.T) {
	if _, err := (DecisionFormat{}).Handler(nil); err == nil {
		t.Error("a nil schema was accepted")
	}
	schema := map[string]any{"type": "object", "properties": map[string]any{"mood": map[string]any{"type": "string"}}}
	if _, err := (DecisionFormat{}).Handler(schema); err == nil || !strings.Contains(err.Error(), "not a question") {
		t.Errorf("plain field: error = %v", err)
	}
}

func TestDecisionFormatConfig(t *testing.T) {
	h := newHandler(t)
	cfg := h.Config()
	if cfg.Format != OutputFormatDecision || !cfg.Constrained || cfg.ContentType != "application/json" || cfg.Schema == nil {
		t.Errorf("config = %+v", cfg)
	}
}

func TestDecisionFormatInstructions(t *testing.T) {
	text := newHandler(t).Instructions()
	for _, want := range []string{
		"department: Which team should handle this?",
		`"billing" (Payments, invoicing, refunds)`,
		"frustration: How frustrated is the customer?",
		"0 = Calm, 1 = Concerned but civil, 2 = Very angry",
		`Answer {"noul": 1} for yes or {"noul": 0} for no. Yes means: Explicitly time-sensitive. No means: No urgency expressed.`,
	} {
		if !strings.Contains(text, want) {
			t.Errorf("instructions lack %q:\n%s", want, text)
		}
	}
	if idx := strings.Index(text, "department:"); idx > strings.Index(text, "frustration:") || strings.Index(text, "frustration:") > strings.Index(text, "is_urgent:") {
		t.Errorf("questions are not in ID order:\n%s", text)
	}
}

func calibratedMessage(text string) *ai.Message {
	part := ai.NewTextPart(text)
	part.Metadata = map[string]any{"typesafe": map[string]any{"model": "jev-1.13.0", calibratedKey: true}}
	return ai.NewModelMessage(part)
}

func TestParseKeepsCalibratedNumbers(t *testing.T) {
	h := newHandler(t)
	resp := &response{}
	if err := json.Unmarshal([]byte(cannedReply), resp); err != nil {
		t.Fatal(err)
	}
	text, err := answersText(resp, triageQuestions, false)
	if err != nil {
		t.Fatal(err)
	}

	parsed, err := h.ParseMessage(calibratedMessage(text))
	if err != nil {
		t.Fatal(err)
	}
	if len(parsed.Content) != 1 || !calibrated(parsed.Content[0].Metadata) {
		t.Fatalf("ParseMessage dropped the calibration mark: %+v", parsed.Content)
	}
	out, err := h.ParseOutput(parsed)
	if err != nil {
		t.Fatal(err)
	}
	answers := out.(map[string]any)
	department := answers["department"].(map[string]any)
	if department["confidence"] != 0.6 || department["probabilities"] == nil {
		t.Errorf("calibrated department was hardened: %v", department)
	}
	if answers["frustration"].(map[string]any)["score"] != 1.3 {
		t.Errorf("calibrated score was rounded: %v", answers["frustration"])
	}
}

func TestParseHardensUncalibratedAnswers(t *testing.T) {
	h := newHandler(t)
	// What a language model tends to send: fenced, bare values in places,
	// and a confidence it made up.
	llm := "Here you go:\n```json\n" + `{
		"department": {"choice": "billing", "confidence": 0.97, "probabilities": {"billing": 0.97}},
		"is_urgent": true,
		"frustration": 1.6
	}` + "\n```"

	parsed, err := h.ParseMessage(ai.NewModelMessage(ai.NewTextPart(llm)))
	if err != nil {
		t.Fatal(err)
	}
	var out triage
	if err := json.Unmarshal([]byte(parsed.Text()), &out); err != nil {
		t.Fatal(err)
	}
	if out.Department.Choice != "billing" {
		t.Errorf("choice = %q", out.Department.Choice)
	}
	if out.Department.Confidence != 0 || out.Department.Probabilities != nil {
		t.Errorf("an invented confidence survived: %+v", out.Department)
	}
	if out.IsUrgent.Probability != 1 {
		t.Errorf("noul from a bool = %v, want 1", out.IsUrgent.Probability)
	}
	if out.Frustration.Score != 2 || out.Frustration.Legend != nil {
		t.Errorf("score = %+v, want the whole level 2 with no legend", out.Frustration)
	}

	// An option outside the criteria still fails validation.
	bad := `{"department": "legal", "is_urgent": 0, "frustration": 0}`
	if _, err := h.ParseMessage(ai.NewModelMessage(ai.NewTextPart(bad))); err == nil {
		t.Error("an option outside the criteria was accepted")
	}
	if _, err := h.ParseMessage(ai.NewModelMessage(ai.NewTextPart("no json here"))); err == nil {
		t.Error("text without JSON was accepted")
	}
}

func TestParseChunk(t *testing.T) {
	h := newHandler(t)
	chunk := func(index int, text string) *ai.ModelResponseChunk {
		return &ai.ModelResponseChunk{Index: index, Content: []*ai.Part{ai.NewTextPart(text)}}
	}
	if out, err := h.ParseChunk(chunk(0, `{"department": {"choice": "bil`)); err != nil || out != nil {
		t.Errorf("partial chunk = %v, %v; want nothing yet", out, err)
	}
	out, err := h.ParseChunk(chunk(0, `ling", "confidence": 0.9}, "is_urgent": 0.7, "frustration": 0.4}`))
	if err != nil {
		t.Fatal(err)
	}
	answers := out.(map[string]any)
	if answers["department"].(map[string]any)["confidence"] != nil {
		t.Errorf("streamed confidence survived: %v", answers)
	}
	if answers["is_urgent"].(map[string]any)["noul"] != 1.0 {
		t.Errorf("streamed noul = %v, want 1", answers["is_urgent"])
	}
	// A new turn starts over.
	if out, _ := h.ParseChunk(chunk(1, `{"is_urgent"`)); out != nil {
		t.Errorf("a new index did not reset the text: %v", out)
	}
}
