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

package deepseek_test

import (
	"testing"

	"github.com/firebase/genkit/go/plugins/compat_oai/deepseek"
	"github.com/firebase/genkit/go/plugins/internal/livetest"
	"github.com/firebase/genkit/go/plugins/internal/oailive"
)

func TestPluginLive(t *testing.T) {
	livetest.Env(t, "DEEPSEEK_API_KEY")
	g := livetest.Init(t, &deepseek.DeepSeek{})

	// Thinking is on by default, so the cheap checks turn it off and the
	// reasoning checks turn it back on.
	noThinking := &deepseek.ThinkingConfig{Type: deepseek.ThinkingTypeDisabled}
	oailive.Run(t, g, oailive.Suite{
		Suite: livetest.Suite{
			Model: deepseek.ModelRef("deepseek-v4-flash", &deepseek.ChatConfig{Thinking: noThinking}),
			ReasoningModel: deepseek.ModelRef("deepseek-v4-flash", &deepseek.ChatConfig{
				ReasoningEffort: deepseek.ReasoningEffortLow,
				Thinking:        &deepseek.ThinkingConfig{Type: deepseek.ThinkingTypeEnabled},
			}),
			ReasoningContent: true,
			LimitConfig:      &deepseek.ChatConfig{MaxOutputTokens: 16, Thinking: noThinking},
			BadKeyPlugin:     &deepseek.DeepSeek{APIKey: "invalid"},
			Skip:             map[string]string{},
		},
		ExtraConfig: map[string]any{
			"thinking": map[string]any{"type": "disabled"},
			"extra":    map[string]any{"logprobs": true},
		},
	})
}
