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
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/anthropic"
	"github.com/openai/openai-go/option"
)

func TestPlugin(t *testing.T) {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		t.Skip("Skipping test: ANTHROPIC_API_KEY environment variable not set")
	}

	ctx := context.Background()

	// Initialize genkit with claude-4-5-sonnet as default model
	g := genkit.Init(
		ctx,
		genkit.WithDefaultModel("anthropic/claude-sonnet-4-5-20250929"),
		genkit.WithPlugins(&anthropic.Anthropic{
			Opts: []option.RequestOption{
				option.WithAPIKey(apiKey),
			},
		}),
	)
	t.Log("genkit initialized")

	t.Run("basic completion", func(t *testing.T) {
		t.Log("generating basic completion response")
		resp, err := genkit.Generate(ctx, g,
			ai.WithPrompt("What is the capital of France?"),
		)
		if err != nil {
			t.Fatal("error generating basic completion response: ", err)
		}
		t.Logf("basic completion response: %+v", resp)

		out := resp.Message.Content[0].Text
		if !strings.Contains(strings.ToLower(out), "paris") {
			t.Errorf("got %q, expecting it to contain 'Paris'", out)
		}

		// Verify usage statistics are present
		if resp.Usage == nil || resp.Usage.TotalTokens == 0 {
			t.Error("Expected non-zero usage statistics")
		}
	})

	t.Run("streaming", func(t *testing.T) {
		var streamedOutput string
		chunks := 0

		final, err := genkit.Generate(ctx, g,
			ai.WithPrompt("Write a short paragraph about artificial intelligence."),
			ai.WithStreaming(func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
				chunks++
				for _, content := range chunk.Content {
					streamedOutput += content.Text
				}
				return nil
			}))
		if err != nil {
			t.Fatal(err)
		}

		// Verify streaming worked
		if chunks <= 1 {
			t.Error("Expected multiple chunks for streaming")
		}

		// Verify final output matches streamed content
		finalOutput := ""
		for _, content := range final.Message.Content {
			finalOutput += content.Text
		}
		if streamedOutput != finalOutput {
			t.Errorf("Streaming output doesn't match final output\nStreamed: %s\nFinal: %s",
				streamedOutput, finalOutput)
		}

		t.Logf("streaming response: %+v", finalOutput)
	})

	t.Run("system message", func(t *testing.T) {
		resp, err := genkit.Generate(ctx, g,
			ai.WithPrompt("What are you?"),
			ai.WithSystem("You are a helpful math tutor who loves numbers."),
		)
		if err != nil {
			t.Fatal(err)
		}

		out := resp.Message.Content[0].Text
		if !strings.Contains(strings.ToLower(out), "math") {
			t.Errorf("got %q, expecting response to mention being a math tutor", out)
		}

		t.Logf("system message response: %+v", out)
	})
}
