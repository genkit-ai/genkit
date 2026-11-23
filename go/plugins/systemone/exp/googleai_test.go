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
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/internal/systemone/systemonetest"
)

// newGoogleAI starts a fake endpoint that answers as the Decisions API
// does, and a Genkit with the plugin pointed at it. The key comes from
// the environment, as it does in an app.
func newGoogleAI(t *testing.T, fake *systemonetest.Server) *genkit.Genkit {
	t.Helper()
	fake.Usage = map[string]any{
		"total_input_tokens":       300,
		"input_tokens_by_modality": []any{map[string]any{"modality": "TEXT", "tokens": 44}, map[string]any{"modality": "IMAGE", "tokens": 256}},
		"total_output_tokens":      0,
	}
	if fake.Wrap == nil {
		fake.Wrap = func(reply map[string]any) any {
			for _, a := range reply["answers"].(map[string]any) {
				if a := a.(map[string]any); a["type"] == "noul" {
					a["confidence"] = 0.93
				}
			}
			delete(reply, "model")
			return reply
		}
	}
	srv := httptest.NewServer(fake)
	t.Cleanup(srv.Close)
	t.Setenv("GEMINI_API_KEY", "gemini-key")
	t.Setenv("GOOGLE_API_KEY", "google-key")
	p := GoogleAI()
	p.BaseURL = srv.URL
	return genkit.Init(t.Context(), genkit.WithPlugins(p))
}

func TestGoogleAI(t *testing.T) {
	fake := &systemonetest.Server{}
	g := newGoogleAI(t, fake)
	out, resp, err := genkit.GenerateData[triage](t.Context(), g,
		ai.WithModelName("googleai-decisions/gemini-decision-1"),
		ai.WithConfig(&Config{MediaResolution: "low"}),
		ai.WithMessages(ai.NewUserMessage(
			ai.NewTextPart("Arrived like this."),
			ai.NewMediaPart("image/png", "https://example.com/box.png"),
			ai.NewMediaPart("application/pdf", "data:application/pdf;base64,JVBERi0="))))
	if err != nil {
		t.Fatal(err)
	}
	req, body := fake.Last(t)
	if req.URL.Path != "/v1beta/decisions" || body["model"] != "gemini-decision-1" {
		t.Errorf("path = %q, model = %v", req.URL.Path, body["model"])
	}
	if got := req.Header.Get("x-goog-api-key"); got != "gemini-key" {
		t.Errorf("key = %q, want GEMINI_API_KEY ahead of GOOGLE_API_KEY", got)
	}
	if got := req.Header.Get("x-goog-api-client"); !strings.HasPrefix(got, "genkit-go/") {
		t.Errorf("x-goog-api-client = %q", got)
	}
	if input, _ := body["input"].([]any); len(input) != 3 {
		t.Errorf("input = %v, want the image by URL, the PDF, and the text", body["input"])
	}
	if out.IsUrgent.Probability != 0.93 || out.IsUrgent.Confidence != 0.93 {
		t.Errorf("is_urgent = %+v, want the noul's confidence kept", out.IsUrgent)
	}
	if resp.Usage.InputTokens != 300 || resp.Usage.Custom["inputImageTokens"] != 256 || resp.Usage.Custom["inputTextTokens"] != 44 {
		t.Errorf("usage = %+v", resp.Usage)
	}
	if ResponseInfo(resp).Model != "gemini-decision-1" {
		t.Errorf("model = %q, want the model the request named", ResponseInfo(resp).Model)
	}
}

func TestGoogleAIBlocked(t *testing.T) {
	fake := &systemonetest.Server{Wrap: func(map[string]any) any {
		return map[string]any{"prompt_feedback": map[string]any{"block_reason": "safety", "block_reason_message": "Input blocked."}}
	}}
	g := newGoogleAI(t, fake)
	resp, err := genkit.Generate(t.Context(), g,
		ai.WithModelName("googleai-decisions/gemini-decision-1"),
		ai.WithOutputType(triage{}),
		ai.WithPrompt("something unsafe"))
	if err != nil {
		t.Fatal(err)
	}
	if resp.FinishReason != ai.FinishReasonBlocked || resp.FinishMessage != "Input blocked." {
		t.Errorf("finish = %q %q, want blocked with the API's message", resp.FinishReason, resp.FinishMessage)
	}
}

func TestMediaResolutionRefusedElsewhere(t *testing.T) {
	fake := &systemonetest.Server{}
	g := newGenkit(t, fake)
	_, err := genkit.Generate(t.Context(), g,
		ai.WithModelName("typesafe/jev-latest"),
		ai.WithOutputType(triage{}),
		ai.WithConfig(&Config{MediaResolution: "low"}),
		ai.WithPrompt("hi"))
	if err == nil || !strings.Contains(err.Error(), "media resolution") || fake.Calls() != 0 {
		t.Errorf("error = %v, calls = %d, want a refusal before the request", err, fake.Calls())
	}
}
