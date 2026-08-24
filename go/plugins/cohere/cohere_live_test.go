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

package cohere_test

import (
	"context"
	"os"
	"strings"
	"testing"

	cohere "github.com/cohere-ai/cohere-go/v2"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	coheregenkit "github.com/firebase/genkit/go/plugins/cohere"
)

func requireEnv(key string) (string, bool) {
	value, ok := os.LookupEnv(key)
	if !ok || value == "" {
		return "", false
	}
	return value, true
}

func TestCohereLive(t *testing.T) {
	if _, ok := requireEnv("COHERE_API_KEY"); !ok {
		t.Skip("COHERE_API_KEY not found in the environment")
	}

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&coheregenkit.Cohere{}))

	maxTokens := 256
	temperature := 0.3
	config := &cohere.V2ChatRequest{MaxTokens: &maxTokens, Temperature: &temperature}

	t.Run("generate", func(t *testing.T) {
		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(coheregenkit.Model(g, "command-r")),
			ai.WithConfig(config),
			ai.WithSystem("be very terse"),
			ai.WithPrompt("what is the capital of France? answer with one word"),
		)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(strings.ToLower(resp.Text()), "paris") {
			t.Fatalf("want Paris, got: %s", resp.Text())
		}
	})

	t.Run("streaming", func(t *testing.T) {
		var out strings.Builder
		var chunks int
		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(coheregenkit.Model(g, "command-r")),
			ai.WithConfig(config),
			ai.WithPrompt("count from 1 to 5"),
			ai.WithStreaming(func(ctx context.Context, c *ai.ModelResponseChunk) error {
				for _, p := range c.Content {
					if p.IsText() {
						out.WriteString(p.Text)
					}
				}
				chunks++
				return nil
			}),
		)
		if err != nil {
			t.Fatal(err)
		}
		if chunks == 0 {
			t.Fatal("expected at least one streamed chunk")
		}
		if out.String() != resp.Text() {
			t.Fatalf("streamed text %q != final text %q", out.String(), resp.Text())
		}
		if resp.Text() == "" {
			t.Fatal("empty streamed response")
		}
	})

	t.Run("tools", func(t *testing.T) {
		addTool := genkit.DefineTool(g, "add", "adds two integers",
			func(toolCtx *ai.ToolContext, input struct {
				A int `json:"a"`
				B int `json:"b"`
			}) (int, error) {
				return input.A + input.B, nil
			},
		)
		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(coheregenkit.Model(g, "command-r-plus")),
			ai.WithConfig(config),
			ai.WithPrompt("use the add tool to compute 17 plus 25, then state only the result"),
			ai.WithTools(addTool),
		)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(resp.Text(), "42") {
			t.Fatalf("want 42 in answer, got: %s", resp.Text())
		}
	})

	t.Run("embed", func(t *testing.T) {
		resp, err := genkit.Embed(ctx, g,
			ai.WithEmbedderName("cohere/embed-v4.0"),
			ai.WithTextDocs("the quick brown fox"),
		)
		if err != nil {
			t.Fatal(err)
		}
		if len(resp.Embeddings) != 1 {
			t.Fatalf("expected 1 embedding, got %d", len(resp.Embeddings))
		}
		if len(resp.Embeddings[0].Embedding) == 0 {
			t.Fatal("expected a non-empty embedding vector")
		}
	})

	t.Run("embed with input type", func(t *testing.T) {
		resp, err := genkit.Embed(ctx, g,
			ai.WithEmbedderName("cohere/embed-v4.0"),
			ai.WithConfig(&coheregenkit.EmbedOptions{InputType: "search_query"}),
			ai.WithTextDocs("what is the capital of France"),
		)
		if err != nil {
			t.Fatal(err)
		}
		if len(resp.Embeddings) != 1 || len(resp.Embeddings[0].Embedding) == 0 {
			t.Fatalf("expected one non-empty embedding, got %+v", resp.Embeddings)
		}
	})
}
