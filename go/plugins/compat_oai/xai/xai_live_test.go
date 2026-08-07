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

package xai_test

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/xai"
	"github.com/openai/openai-go"
)

func TestPluginLive(t *testing.T) {
	if os.Getenv("XAI_API_KEY") == "" {
		t.Skip("XAI_API_KEY is not set")
	}

	ctx := context.Background()
	plugin := &xai.XAI{}
	g := genkit.Init(
		ctx,
		genkit.WithPlugins(plugin),
		genkit.WithDefaultModel("xai/grok-4.5"),
	)

	t.Run("complete", func(t *testing.T) {
		resp, err := genkit.Generate(ctx, g,
			ai.WithPrompt("What is the capital of France? Answer with the city only."),
		)
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if !strings.Contains(strings.ToLower(resp.Text()), "paris") {
			t.Fatalf("Text() = %q, want Paris", resp.Text())
		}
	})

	// Exercises the fields this plugin maps by hand: maxOutputTokens must
	// reach xAI as max_completion_tokens, and reasoningEffort as
	// reasoning_effort.
	t.Run("reasoning effort", func(t *testing.T) {
		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(xai.ModelRef("grok-4.3", &xai.ChatConfig{
				MaxOutputTokens: 256,
				ReasoningEffort: "low",
			})),
			ai.WithPrompt("Name one prime number between 10 and 20."),
		)
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if resp.Text() == "" {
			t.Fatal("Text() is empty")
		}
	})

	// Exercises the search_parameters shape against the live API.
	t.Run("live search", func(t *testing.T) {
		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(xai.ModelRef("grok-4.5", &xai.ChatConfig{
				SearchParameters: &xai.SearchParameters{
					Mode:             "on",
					ReturnCitations:  openai.Ptr(true),
					MaxSearchResults: openai.Ptr(3),
				},
			})),
			ai.WithPrompt("In one sentence, what happened in AI research this week?"),
		)
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if resp.Text() == "" {
			t.Fatal("Text() is empty")
		}
	})

	t.Run("streaming", func(t *testing.T) {
		var text strings.Builder
		resp, err := genkit.Generate(ctx, g,
			ai.WithPrompt("Explain briefly why the sky appears blue."),
			ai.WithStreaming(func(_ context.Context, chunk *ai.ModelResponseChunk) error {
				text.WriteString(chunk.Text())
				return nil
			}),
		)
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if text.String() == "" {
			t.Fatal("streamed text is empty")
		}
		if resp.Text() != text.String() {
			t.Fatalf("final text = %q, want streamed %q", resp.Text(), text.String())
		}
	})
}
