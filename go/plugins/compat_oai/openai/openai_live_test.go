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

package openai_test

import (
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/openai"
	"github.com/firebase/genkit/go/plugins/internal/livetest"
	"github.com/firebase/genkit/go/plugins/internal/oailive"
	openaiGo "github.com/openai/openai-go"
	"github.com/openai/openai-go/shared"
)

func TestPluginLive(t *testing.T) {
	apiKey := livetest.Env(t, "OPENAI_API_KEY")
	oai := &openai.OpenAI{APIKey: apiKey}
	g := livetest.Init(t, oai)

	oailive.Run(t, g, oailive.Suite{
		Suite: livetest.Suite{
			Model: openai.ModelRef("gpt-4o-mini", nil),
			// The chat completions API takes the effort knob but keeps the
			// reasoning content server-side.
			ReasoningModel: openai.ModelRef("gpt-5-nano", &openaiGo.ChatCompletionNewParams{
				ReasoningEffort: shared.ReasoningEffortLow,
			}),
			VisionModel: openai.ModelRef("gpt-4.1-nano", nil),
			// Chat completions read input_audio on the audio models only.
			AudioModel:    openai.ModelRef("gpt-audio-1.5", nil),
			DocumentModel: openai.ModelRef("gpt-4o-mini", nil),
			LimitConfig: &openaiGo.ChatCompletionNewParams{
				MaxCompletionTokens: openaiGo.Int(16),
			},
			BadKeyPlugin: &openai.OpenAI{APIKey: "invalid"},
		},
		// No ExtraConfig: this plugin speaks the SDK's own request type,
		// which has no extra passthrough.
	})

	livetest.RunEmbedder(t, g, livetest.EmbedderSuite{
		Embedder:   oai.Embedder(g, "text-embedding-3-small"),
		Dimensions: 1536,
		Normalized: true,
	})

	// The SDK's own request type carries the sampling fields.
	t.Run("sdk config", func(t *testing.T) {
		resp, err := genkit.Generate(t.Context(), g,
			ai.WithModel(openai.ModelRef("gpt-4o-mini", &openaiGo.ChatCompletionNewParams{
				Temperature:         openaiGo.Float(0.2),
				MaxCompletionTokens: openaiGo.Int(50),
				TopP:                openaiGo.Float(0.5),
				Stop: openaiGo.ChatCompletionNewParamsStopUnion{
					OfStringArray: []string{".", "!", "?"},
				},
			})),
			ai.WithPrompt("Write a short sentence about artificial intelligence."),
		)
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if resp.Text() == "" {
			t.Error("Text() is empty")
		}
	})
}
