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

package xai_test

import (
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/xai"
	"github.com/firebase/genkit/go/plugins/internal/livetest"
	"github.com/firebase/genkit/go/plugins/internal/oailive"
)

func TestPluginLive(t *testing.T) {
	livetest.Env(t, "XAI_API_KEY")
	g := livetest.Init(t, &xai.XAI{})

	oailive.Run(t, g, oailive.Suite{
		Suite: livetest.Suite{
			Model: xai.ModelRef("grok-4.5", nil),
			// Grok takes the effort knob but keeps the reasoning content
			// server-side.
			ReasoningModel: xai.ModelRef("grok-4.3", &xai.ChatConfig{
				MaxOutputTokens: 512,
				ReasoningEffort: xai.ReasoningEffortLow,
			}),
			LimitConfig:  &xai.ChatConfig{MaxOutputTokens: 16},
			BadKeyPlugin: &xai.XAI{APIKey: "invalid"},
			Skip:         map[string]string{},
		},
		ExtraConfig: map[string]any{
			"extra": map[string]any{"user": "genkit-livetest"},
		},
	})

	// grok-4.6 is the one model xAI documents the extra-high level for.
	t.Run("reasoning effort xhigh", func(t *testing.T) {
		resp, err := genkit.Generate(t.Context(), g,
			ai.WithModel(xai.ModelRef("grok-4.6", &xai.ChatConfig{
				MaxOutputTokens: 2048,
				ReasoningEffort: xai.ReasoningEffortXHigh,
			})),
			ai.WithPrompt("What is 27 * 43? Answer with just the number."),
		)
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if !strings.Contains(resp.Text(), "1161") {
			t.Errorf("Text() = %q, want it to contain 1161", resp.Text())
		}
	})
}
