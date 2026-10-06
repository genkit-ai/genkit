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

package zai_test

import (
	"testing"

	"github.com/firebase/genkit/go/plugins/compat_oai/internal/oailive"
	"github.com/firebase/genkit/go/plugins/compat_oai/zai"
	"github.com/firebase/genkit/go/plugins/internal/livetest"
)

func TestPluginLive(t *testing.T) {
	livetest.Env(t, "ZAI_API_KEY")
	g := livetest.Init(t, &zai.ZAI{})

	// Thinking is on by default, so the cheap checks turn it off and the
	// reasoning checks turn it back on.
	noThinking := &zai.ThinkingConfig{Type: zai.ThinkingTypeDisabled}
	oailive.Run(t, g, oailive.Suite{
		Suite: livetest.Suite{
			Model: zai.ModelRef("glm-5.1", &zai.ChatConfig{Thinking: noThinking}),
			ReasoningModel: zai.ModelRef("glm-5.1", &zai.ChatConfig{
				Thinking: &zai.ThinkingConfig{Type: zai.ThinkingTypeEnabled},
			}),
			ReasoningContent: true,
			VisionModel:      zai.ModelRef("glm-5v-turbo", nil),
			LimitConfig:      &zai.ChatConfig{MaxOutputTokens: 16, Thinking: noThinking},
			BadKeyPlugin:     &zai.ZAI{APIKey: "invalid"},
			Skip: map[string]string{
				"generate/unknown model": "Z.ai answers an unknown model with 400 (code 1211), not 404",
			},
		},
		ExtraConfig: map[string]any{
			"thinking": map[string]any{"type": "disabled"},
			"extra":    map[string]any{"user_id": "genkit-livetest"},
		},
	})
}
