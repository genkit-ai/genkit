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
	"context"
	"os"
	"testing"

	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/anthropic"
	"github.com/firebase/genkit/go/plugins/compat_oai/internal/livetest"
)

func TestPluginLive(t *testing.T) {
	if os.Getenv("ANTHROPIC_API_KEY") == "" {
		t.Skip("ANTHROPIC_API_KEY is not set")
	}

	ctx := context.Background()
	g := genkit.Init(ctx,
		genkit.WithPlugins(&anthropic.Anthropic{}),
		genkit.WithDefaultModel("anthropic/claude-haiku-4-5-20251001"),
	)

	livetest.Run(t, g, livetest.Suite{
		Model: anthropic.ModelRef("claude-haiku-4-5-20251001", nil),
		// The OpenAI-compatible endpoint takes the thinking knob but never
		// returns the thinking content itself.
		ReasoningModel: anthropic.ModelRef("claude-haiku-4-5-20251001", &anthropic.ChatConfig{
			MaxOutputTokens: 4096,
			Thinking:        &anthropic.ThinkingConfig{Type: "enabled", BudgetTokens: 2048},
		}),
		VisionModel: anthropic.ModelRef("claude-haiku-4-5-20251001", nil),
		ToolChoice:  true,
		ExtraConfig: map[string]any{
			"extra": map[string]any{"thinking": map[string]any{"type": "disabled"}},
		},
	})
}
