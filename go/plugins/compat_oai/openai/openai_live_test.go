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
	"context"
	"os"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/internal/livetest"
	"github.com/firebase/genkit/go/plugins/compat_oai/openai"
	openaiGo "github.com/openai/openai-go"
	"github.com/openai/openai-go/shared"
)

func TestPluginLive(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("Skipping test: OPENAI_API_KEY environment variable not set")
	}

	ctx := context.Background()
	oai := &openai.OpenAI{APIKey: apiKey}
	g := genkit.Init(ctx,
		genkit.WithDefaultModel("openai/gpt-4o-mini"),
		genkit.WithPlugins(oai),
	)

	livetest.Run(t, g, livetest.Suite{
		Model: openai.ModelRef("gpt-4o-mini", nil),
		// The chat completions API takes the effort knob but keeps the
		// reasoning content server-side.
		ReasoningModel: openai.ModelRef("gpt-5-nano", &openaiGo.ChatCompletionNewParams{
			ReasoningEffort: shared.ReasoningEffortLow,
		}),
		VisionModel: openai.ModelRef("gpt-4.1-nano", nil),
		ToolChoice:  true,
		// No ExtraConfig: this plugin speaks the SDK's own request type,
		// which has no extra passthrough.
	})

	t.Run("embedder", func(t *testing.T) {
		embedder := oai.Embedder(g, "text-embedding-3-small")
		res, err := genkit.Embed(ctx, g, ai.WithEmbedder(embedder), ai.WithTextDocs("yellow banana"))
		if err != nil {
			t.Fatal(err)
		}
		out := res.Embeddings[0].Embedding
		// There's not a whole lot we can test about the result.
		// Just do a few sanity checks.
		if len(out) < 100 {
			t.Errorf("embedding vector looks too short: len(out)=%d", len(out))
		}
		var normSquared float32
		for _, x := range out {
			normSquared += x * x
		}
		if normSquared < 0.9 || normSquared > 1.1 {
			t.Errorf("embedding vector not unit length: %f", normSquared)
		}
	})

	t.Run("sdk config", func(t *testing.T) {
		config := &openaiGo.ChatCompletionNewParams{
			Temperature:         openaiGo.Float(0.2),
			MaxCompletionTokens: openaiGo.Int(50),
			TopP:                openaiGo.Float(0.5),
			Stop: openaiGo.ChatCompletionNewParamsStopUnion{
				OfStringArray: []string{".", "!", "?"},
			},
		}
		resp, err := genkit.Generate(ctx, g,
			ai.WithPrompt("Write a short sentence about artificial intelligence."),
			ai.WithConfig(config),
		)
		if err != nil {
			t.Fatal(err)
		}
		if resp.Text() == "" {
			t.Error("Text() is empty")
		}
	})
}
