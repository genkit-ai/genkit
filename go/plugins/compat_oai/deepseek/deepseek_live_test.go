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
	"context"
	"os"
	"testing"

	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/deepseek"
	"github.com/firebase/genkit/go/plugins/compat_oai/internal/livetest"
)

func TestPluginLive(t *testing.T) {
	if os.Getenv("DEEPSEEK_API_KEY") == "" {
		t.Skip("DEEPSEEK_API_KEY is not set")
	}

	ctx := context.Background()
	g := genkit.Init(ctx,
		genkit.WithPlugins(&deepseek.DeepSeek{}),
		genkit.WithDefaultModel("deepseek/deepseek-v4-flash"),
	)

	// Thinking is on by default, so the cheap checks turn it off and the
	// reasoning checks turn it back on.
	livetest.Run(t, g, livetest.Suite{
		Model: deepseek.ModelRef("deepseek-v4-flash", &deepseek.ChatConfig{
			Thinking: &deepseek.ThinkingConfig{Type: deepseek.ThinkingTypeDisabled},
		}),
		ReasoningModel: deepseek.ModelRef("deepseek-v4-flash", &deepseek.ChatConfig{
			ReasoningEffort: deepseek.ReasoningEffortLow,
			Thinking:        &deepseek.ThinkingConfig{Type: deepseek.ThinkingTypeEnabled},
		}),
		ReasoningContent: true,
		ToolChoice:       true,
		ExtraConfig: map[string]any{
			"thinking": map[string]any{"type": "disabled"},
			"extra":    map[string]any{"logprobs": true},
		},
	})
}
