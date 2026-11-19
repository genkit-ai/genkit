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

package mistral_test

import (
	"testing"

	"github.com/firebase/genkit/go/plugins/compat_oai/mistral"
	"github.com/firebase/genkit/go/plugins/internal/livetest"
	"github.com/firebase/genkit/go/plugins/internal/oailive"
)

func TestPluginLive(t *testing.T) {
	livetest.Env(t, "MISTRAL_API_KEY")
	g := livetest.Init(t, &mistral.Mistral{})

	// Mistral Small reasons only when asked, so the cheap checks leave the
	// effort unset and the reasoning checks ask for high, one of the two
	// levels it takes.
	oailive.Run(t, g, oailive.Suite{
		Suite: livetest.Suite{
			Model: mistral.ModelRef("mistral-small-latest", nil),
			ReasoningModel: mistral.ModelRef("mistral-small-latest", &mistral.ChatConfig{
				ReasoningEffort: mistral.ReasoningEffortHigh,
			}),
			ReasoningContent: true,
			LimitConfig:      &mistral.ChatConfig{MaxOutputTokens: 16},
			BadKeyPlugin:     &mistral.Mistral{APIKey: "invalid"},
			// Each of these fails on Mistral Small's own behavior, with the
			// request and response on the wire checked to be correct.
			Skip: map[string]string{
				"generate/tool choice required":        "Mistral Small ignores a required tool choice on an unrelated prompt and generates until max_tokens, so the request does not return",
				"generate/sequential tool calls":       "Mistral Small issues both steps in parallel, guessing the second step's input, and answers with the wrong number",
				"agent/media across turns":             "Mistral Small answers the gablorken question after an image turn without calling the tool",
				"agent/structured output across turns": "Mistral Small, asked for the capital of the first country named, intermittently fills the city field with the country",
			},
		},
		ExtraConfig: map[string]any{
			"extra": map[string]any{"random_seed": 42},
		},
	})

	livetest.RunEmbedder(t, g, livetest.EmbedderSuite{
		Embedder:   mistral.EmbedderRef("codestral-embed", &mistral.EmbedConfig{Dimensions: 256}),
		Dimensions: 256,
	})
}
