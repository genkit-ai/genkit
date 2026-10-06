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

package anthropic_test

import (
	"testing"

	"github.com/firebase/genkit/go/plugins/compat_oai/anthropic"
	"github.com/firebase/genkit/go/plugins/compat_oai/internal/oailive"
	"github.com/firebase/genkit/go/plugins/internal/livetest"
)

func TestPluginLive(t *testing.T) {
	livetest.Env(t, "ANTHROPIC_API_KEY")
	g := livetest.Init(t, &anthropic.Anthropic{})

	const model = "claude-haiku-4-5-20251001"
	oailive.Run(t, g, oailive.Suite{
		Suite: livetest.Suite{
			Model: anthropic.ModelRef(model, nil),
			// The OpenAI-compatible endpoint takes the thinking knob but
			// never returns the thinking content itself.
			ReasoningModel: anthropic.ModelRef(model, &anthropic.ChatConfig{
				MaxOutputTokens: 4096,
				Thinking:        &anthropic.ThinkingConfig{Type: "enabled", BudgetTokens: 2048},
			}),
			LimitConfig:  &anthropic.ChatConfig{MaxOutputTokens: 16},
			BadKeyPlugin: &anthropic.Anthropic{APIKey: "invalid"},
		},
		ExtraConfig: map[string]any{
			"extra": map[string]any{"thinking": map[string]any{"type": "disabled"}},
		},
	})
}
