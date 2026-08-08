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
//
// SPDX-License-Identifier: Apache-2.0

package openai_test

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/openai/openai-go"
	oresp "github.com/openai/openai-go/responses"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	openaiPlugin "github.com/firebase/genkit/go/plugins/openai"
)

func TestOpenAILive(t *testing.T) {
	if _, ok := requireEnv("OPENAI_API_KEY"); !ok {
		t.Skip("OPENAI_API_KEY not found in the environment")
	}

	type outFormat struct {
		Country string `json:"country"`
	}

	ctx := context.Background()
	plugin := &openaiPlugin.OpenAI{}
	if baseURL, ok := os.LookupEnv("OPENAI_BASE_URL"); ok {
		plugin.BaseURL = baseURL
	}
	g := genkit.Init(ctx, genkit.WithPlugins(plugin))

	modelName := requireEnvValue("OPENAI_MODEL", "gpt-4.1")

	t.Run("text generation", func(t *testing.T) {
		m := openaiPlugin.Model(g, modelName)
		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithMessages(ai.NewUserMessage(ai.NewTextPart("Say ARR exactly once and be short."))),
		)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(resp.Text(), "ARR") {
			t.Fatalf("expected ARR in response, got: %s", resp.Text())
		}
	})

	t.Run("streaming", func(t *testing.T) {
		m := openaiPlugin.Model(g, modelName)
		streamed := ""
		final, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithPrompt("Tell me a short story about a frog."),
			ai.WithStreaming(func(ctx context.Context, c *ai.ModelResponseChunk) error {
				streamed += c.Text()
				return nil
			}),
		)
		if err != nil {
			t.Fatal(err)
		}
		if streamed == "" {
			t.Fatal("expected streamed content")
		}
		if final.Text() == "" {
			t.Fatal("expected final text")
		}
	})

	t.Run("structured output", func(t *testing.T) {
		m := openaiPlugin.Model(g, modelName)
		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithPrompt("Which country was Napoleon emperor of?"),
			ai.WithOutputType(outFormat{}),
		)
		if err != nil {
			t.Fatal(err)
		}
		var ans outFormat
		if err := resp.Output(&ans); err != nil {
			t.Fatal(err)
		}
		if ans.Country == "" {
			t.Fatal("expected structured output country")
		}
	})

	t.Run("tools", func(t *testing.T) {
		m := openaiPlugin.Model(g, modelName)
		weatherTool := genkit.DefineTool(
			g,
			"weather",
			"Returns the weather for the given location",
			func(ctx *ai.ToolContext, input struct {
				Location string `json:"location"`
			}) (string, error) {
				return "sunny", nil
			},
		)

		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithPrompt("What is the weather in San Francisco?"),
			ai.WithTools(weatherTool),
		)
		if err != nil {
			t.Fatal(err)
		}
		if resp.Text() == "" {
			t.Fatal("expected tool-assisted response")
		}
	})

	t.Run("previous response id", func(t *testing.T) {
		m := openaiPlugin.Model(g, modelName)
		first, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithPrompt("Reply with exactly: first"),
		)
		if err != nil {
			t.Fatal(err)
		}

		raw, ok := first.Raw.(string)
		if !ok || raw == "" {
			t.Skip("raw response not stored as string; skipping previous_response_id test")
		}

		req := oresp.Response{}
		if err := req.UnmarshalJSON([]byte(raw)); err != nil {
			t.Skipf("unable to decode raw response: %v", err)
		}
		if req.ID == "" {
			t.Skip("response id unavailable")
		}

		second, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithConfig(&oresp.ResponseNewParams{
				PreviousResponseID: openai.String(req.ID),
			}),
			ai.WithPrompt("Now reply with exactly: second"),
		)
		if err != nil {
			t.Fatal(err)
		}
		if second.Text() == "" {
			t.Fatal("expected continuation response")
		}
	})
}

func requireEnv(key string) (string, bool) {
	val, ok := os.LookupEnv(key)
	return val, ok
}

func requireEnvValue(key, fallback string) string {
	if val, ok := os.LookupEnv(key); ok && val != "" {
		return val
	}
	return fallback
}
