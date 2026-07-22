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

package xai

import (
	"context"
	"math"
	"os"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/openai/openai-go/option"
)

func TestPlugin(t *testing.T) {
	apiKey := os.Getenv("XAI_API_KEY")
	if apiKey == "" {
		t.Skip("Skipping test: XAI_API_KEY environment variable not set")
	}

	ctx := context.Background()

	x := &XAi{
		Opts: []option.RequestOption{
			option.WithAPIKey(apiKey),
		},
	}

	g := genkit.Init(ctx,
		genkit.WithDefaultModel("xai/grok-3-mini"),
		genkit.WithPlugins(x))

	gablorkenTool := genkit.DefineTool(g, "gablorken", "use when need to calculate a gablorken",
		func(ctx *ai.ToolContext, input struct {
			Value float64
			Over  float64
		},
		) (float64, error) {
			return math.Pow(input.Value, input.Over), nil
		},
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

		// Verify the final output matches streamed content
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

	t.Run("media part", func(t *testing.T) {
		image := "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAHIAAABUAQMAAABk5vEVAAAABlBMVEX///8AAABVwtN+" +
			"AAAAI0lEQVR4nGNgGHaA/z8UHIDwOWASDqP8Uf7w56On/1FAQwAAVM0exw1hqwkAAAAASUVORK5CYII="

		resp, err := genkit.Generate(ctx, g,
			ai.WithModelName("xai/grok-4-fast-non-reasoning"),
			ai.WithMessages(
				ai.NewUserMessage(
					ai.NewMediaPart("image/png", image),
					ai.NewTextPart("Is there a rectangle in the picture? Yes or not."),
				),
			),
		)

		if err != nil {
			t.Fatal(err)
		}

		text := resp.Message.Content[0].Text
		if !strings.Contains(strings.ToLower(text), "yes") {
			t.Errorf("got %q, expecting it to contain 'yes'", text)
		}
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

	t.Run("tool usage with basic completion", func(t *testing.T) {
		resp, err := genkit.Generate(ctx, g,
			ai.WithPrompt("what is a gablorken of 2 over 3?"),
			ai.WithTools(gablorkenTool))
		if err != nil {
			t.Fatal(err)
		}

		out := resp.Message.Content[0].Text
		const want = "8"
		if !strings.Contains(out, want) {
			t.Errorf("got %q, expecting it to contain %q", out, want)
		}

		t.Logf("tool usage with basic completion response: %+v", out)
	})

	t.Run("tool usage with streaming", func(t *testing.T) {
		var streamedOutput string
		chunks := 0

		final, err := genkit.Generate(ctx, g,
			ai.WithPrompt("what is a gablorken of 2 over 3?"),
			ai.WithTools(gablorkenTool),
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

		// Verify the final output matches streamed content
		finalOutput := ""
		for _, content := range final.Message.Content {
			finalOutput += content.Text
		}
		if streamedOutput != finalOutput {
			t.Errorf("Streaming output doesn't match final output\nStreamed: %s\nFinal: %s",
				streamedOutput, finalOutput)
		}

		const want = "8"
		if !strings.Contains(finalOutput, want) {
			t.Errorf("got %q, expecting it to contain %q", finalOutput, want)
		}

		t.Logf("tool usage with streaming response: %+v", finalOutput)
	})

	t.Run("invalid config type", func(t *testing.T) {
		// Try to use a string as config instead of *openai.ChatCompletionNewParams
		config := "not a config"

		_, err := genkit.Generate(ctx, g,
			ai.WithPrompt("Write a short sentence about artificial intelligence."),
			ai.WithConfig(config),
		)
		if err == nil {
			t.Fatal("expected error for invalid config type")
		}
		if !strings.Contains(err.Error(), "unexpected config type: string") {
			t.Errorf("got error %q, want error containing 'unexpected config type: string'", err.Error())
		}
		t.Logf("invalid config type error: %v", err)
	})
}
