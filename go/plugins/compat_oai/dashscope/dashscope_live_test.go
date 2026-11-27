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

package dashscope_test

import (
	"testing"

	"github.com/firebase/genkit/go/plugins/compat_oai/dashscope"
	"github.com/firebase/genkit/go/plugins/internal/livetest"
	"github.com/firebase/genkit/go/plugins/internal/oailive"
)

func TestPluginLive(t *testing.T) {
	livetest.Env(t, "DASHSCOPE_API_KEY")
	g := livetest.Init(t, &dashscope.DashScope{})

	thinking := true
	oailive.Run(t, g, oailive.Suite{
		Suite: livetest.Suite{
			Model: dashscope.ModelRef("qwen-plus", nil),
			ReasoningModel: dashscope.ModelRef("qwen-plus", &dashscope.ChatConfig{
				EnableThinking: &thinking,
			}),
			ReasoningContent: true,
			// DashScope only serves thinking on streaming calls.
			StreamOnlyReasoning: true,
			VisionModel:         dashscope.ModelRef("qwen3-vl-plus", nil),
			LimitConfig:         &dashscope.ChatConfig{MaxOutputTokens: 16},
			BadKeyPlugin:        &dashscope.DashScope{APIKey: "invalid"},
		},
		ExtraConfig: map[string]any{
			"extra": map[string]any{"enable_thinking": false},
		},
	})
}
