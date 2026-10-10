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
//
// SPDX-License-Identifier: Apache-2.0

package anthropic_test

import (
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	anthropicPlugin "github.com/firebase/genkit/go/plugins/anthropic"
	"github.com/firebase/genkit/go/plugins/internal/anthropic/anthropiclive"
	"github.com/firebase/genkit/go/plugins/internal/livetest"
)

func TestPluginLive(t *testing.T) {
	livetest.Env(t, "ANTHROPIC_API_KEY")
	g := livetest.Init(t, &anthropicPlugin.Anthropic{})

	const model = "claude-haiku-4-5"
	reasoningModel := anthropicPlugin.ModelRef(model, &anthropic.MessageNewParams{
		MaxTokens: 4096,
		Thinking: anthropic.ThinkingConfigParamUnion{
			OfEnabled: &anthropic.ThinkingConfigEnabledParam{BudgetTokens: 2048},
		},
	})
	anthropiclive.Run(t, g, livetest.Suite{
		Model:             anthropicPlugin.ModelRef(model, nil),
		ReasoningModel:    reasoningModel,
		ReasoningContent:  true,
		LimitConfig:       &anthropic.MessageNewParams{MaxTokens: 16},
		ToolResponseMedia: true,
		BadKeyPlugin:      &anthropicPlugin.Anthropic{APIKey: "invalid"},
	})

	// The API counts thinking inside output_tokens and reports it apart
	// only in a field the SDK does not model, which the plugin reads to keep
	// the two disjoint.
	t.Run("usage with thinking", func(t *testing.T) {
		resp, err := genkit.Generate(t.Context(), g,
			ai.WithModel(reasoningModel),
			ai.WithPrompt("A bat and a ball cost 1.10 in total. The bat costs 1.00 more than the ball. How much is the ball?"))
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		u := resp.Usage
		if u == nil || u.InputTokens == 0 || u.OutputTokens == 0 || u.ThoughtsTokens == 0 {
			t.Fatalf("Usage = %+v, want input, output, and thoughts tokens", u)
		}
		if sum := u.InputTokens + u.OutputTokens + u.ThoughtsTokens; u.TotalTokens != sum {
			t.Errorf("TotalTokens = %d, want input + output + thoughts = %d", u.TotalTokens, sum)
		}
	})
}
