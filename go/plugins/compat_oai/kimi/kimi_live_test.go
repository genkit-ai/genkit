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

package kimi_test

import (
	"testing"

	"github.com/firebase/genkit/go/plugins/compat_oai/kimi"
	"github.com/firebase/genkit/go/plugins/internal/livetest"
	"github.com/firebase/genkit/go/plugins/internal/oailive"
)

func TestPluginLive(t *testing.T) {
	livetest.Env(t, "KIMI_API_KEY", "MOONSHOT_API_KEY")
	g := livetest.Init(t, &kimi.Kimi{})

	oailive.Run(t, g, oailive.Suite{
		Suite: livetest.Suite{
			Model:            kimi.ModelRef("kimi-k3", nil),
			ReasoningModel:   kimi.ModelRef("kimi-k2.6", nil),
			ReasoningContent: true,
			LimitConfig:      &kimi.ChatConfig{MaxOutputTokens: 16},
			BadKeyPlugin:     &kimi.Kimi{APIKey: "invalid"},
			Skip: map[string]string{
				// The catalog claims constrained output alongside tools, but
				// with a response format set the model answers in JSON at
				// once and never calls the tool.
				"generate/tools then structured output": "kimi-k3 skips tools under a response format",
			},
		},
		ExtraConfig: map[string]any{
			"extra": map[string]any{"thinking": map[string]any{"type": "disabled"}},
		},
	})
}
