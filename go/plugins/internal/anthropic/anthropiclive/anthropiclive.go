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

// Package anthropiclive is the live checklist tier for the plugins built on
// the shared Anthropic Messages API code: the anthropic plugin and the Model
// Garden Claude models. It runs the shared [livetest] checklist and then the
// checks the shared code owes, and is where a gap or a check that code has
// whatever the plugin is recorded once for both plugins.
package anthropiclive

import (
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/internal/livetest"
)

// Run walks the plugin registered on g through the shared checklist and then
// the Anthropic one, under an "anthropic" subtest. See [livetest.Run] for what
// it defines on g.
func Run(t *testing.T, g *genkit.Genkit, s livetest.Suite) {
	t.Helper()
	livetest.Run(t, g, s, livetest.Group{Name: "anthropic", Cases: []livetest.Case{
		{
			// The SDK refuses a non-streaming request whose max_tokens it
			// expects to take over ten minutes, about 21,333 tokens, so the
			// plugin streams it and returns one response. Billing counts the
			// tokens written, not max_tokens, so this costs a short answer.
			Name: "large max_tokens without streaming",
			Run: func(t *testing.T) {
				resp, err := genkit.Generate(t.Context(), g,
					ai.WithModel(s.Model),
					ai.WithConfig(&anthropic.MessageNewParams{MaxTokens: 32000}),
					ai.WithPrompt("What is the capital of France? Reply with just the city name."),
				)
				if err != nil {
					t.Fatalf("Generate() error = %v", err)
				}
				if !strings.Contains(strings.ToLower(resp.Text()), "paris") {
					t.Errorf("Text() = %q, want it to contain %q", resp.Text(), "Paris")
				}
				if resp.FinishReason != ai.FinishReasonStop {
					t.Errorf("FinishReason = %q, want %q", resp.FinishReason, ai.FinishReasonStop)
				}
				if u := resp.Usage; u == nil || u.InputTokens == 0 || u.OutputTokens == 0 {
					t.Errorf("Usage = %+v, want input and output tokens", u)
				}
			},
		},
	}})
}
