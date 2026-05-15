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
	"context"
	"encoding/base64"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	anthropicPlugin "github.com/firebase/genkit/go/plugins/anthropic"
)

func TestAnthropicLive(t *testing.T) {
	if _, ok := requireEnv("ANTHROPIC_API_KEY"); !ok {
		t.Skip("ANTHROPIC_API_KEY not found in the environment")
	}

	type outFormat struct {
		Country string `json:"country"`
	}

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&anthropicPlugin.Anthropic{}))
	t.Run("model version ok", func(t *testing.T) {
		m := anthropicPlugin.Model(g, "claude-sonnet-4-5-20250929")
		resp, err := genkit.Generate(ctx, g,
			ai.WithConfig(&anthropic.MessageNewParams{
				Temperature: anthropic.Float(1),
				MaxTokens:   1024,
			}),
			ai.WithModel(m),
			ai.WithSystem("talk to me like an evil pirate and say ARR several times but be very short"),
			ai.WithMessages(ai.NewUserMessage(ai.NewTextPart("I'm a fish"))),
		)
		if err != nil {
			t.Fatal(err)
		}

		if !strings.Contains(resp.Text(), "ARR") {
			t.Fatalf("not a pirate :( :%s", resp.Text())
		}
	})
	t.Run("model version not ok", func(t *testing.T) {
		m := anthropicPlugin.Model(g, "claude-sonnet-4-5-20250929")
		_, err := genkit.Generate(ctx, g,
			ai.WithConfig(&anthropic.MessageNewParams{
				Temperature: anthropic.Float(1),
				MaxTokens:   1024,
			}),
			ai.WithModel(m),
		)
		if err == nil {
			t.Fatal("should have failed due wrong model version")
		}
	})
	t.Run("media content", func(t *testing.T) {
		i, err := fetchImgAsBase64()
		if err != nil {
			t.Fatal(err)
		}
		m := anthropicPlugin.Model(g, "claude-sonnet-4-5-20250929")
		resp, err := genkit.Generate(ctx, g,
			ai.WithSystem("You are a professional image detective that talks like an evil pirate that loves animals, your task is to tell the name of the animal in the image but be very short"),
			ai.WithModel(m),
			ai.WithConfig(&anthropic.MessageNewParams{
				Temperature: anthropic.Float(1),
				MaxTokens:   1024,
			}),
			ai.WithMessages(
				ai.NewUserMessage(
					ai.NewTextPart("do you know which animal is in the image?"),
					ai.NewMediaPart("", "data:image/jpeg;base64,"+i))))
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(strings.ToLower(resp.Text()), "cat") {
			t.Fatalf("want: cat, got: %s", resp.Text())
		}
	})
	t.Run("media content stream", func(t *testing.T) {
		i, err := fetchImgAsBase64()
		if err != nil {
			t.Fatal(err)
		}
		out := ""
		m := anthropicPlugin.Model(g, "claude-sonnet-4-5-20250929")
		resp, err := genkit.Generate(ctx, g,
			ai.WithSystem("You are a professional image detective that talks like an evil pirate that loves animals, your task is to tell the name of the animal in the image but be very short"),
			ai.WithModel(m),
			ai.WithConfig(&anthropic.MessageNewParams{
				Temperature: anthropic.Float(1),
				MaxTokens:   1024,
			}),
			ai.WithStreaming(func(ctx context.Context, c *ai.ModelResponseChunk) error {
				out += c.Content[0].Text
				return nil
			}),
			ai.WithMessages(
				ai.NewUserMessage(
					ai.NewTextPart("do you know which animal is in the image?"),
					ai.NewMediaPart("", "data:image/jpeg;base64,"+i))))
		if err != nil {
			t.Fatal(err)
		}
		if out != resp.Text() {
			t.Fatalf("want: %s, got: %s", resp.Text(), out)
		}
		if !strings.Contains(strings.ToLower(resp.Text()), "cat") {
			t.Fatalf("want: cat, got: %s", resp.Text())
		}
	})
	t.Run("media content stream with thinking", func(t *testing.T) {
		i, err := fetchImgAsBase64()
		if err != nil {
			t.Fatal(err)
		}
		out := ""
		m := anthropicPlugin.Model(g, "claude-sonnet-4-5-20250929")
		resp, err := genkit.Generate(ctx, g,
			ai.WithSystem(`You are a professional image detective that
			talks like an evil pirate that loves animals, your task is to tell the name
			of the animal in the image but be very short`),
			ai.WithModel(m),
			ai.WithConfig(&anthropic.MessageNewParams{
				Temperature: anthropic.Float(1),
				MaxTokens:   2048,
				Thinking: anthropic.ThinkingConfigParamUnion{
					OfEnabled: &anthropic.ThinkingConfigEnabledParam{
						BudgetTokens: 1024,
					},
				},
			}),
			ai.WithStreaming(func(ctx context.Context, c *ai.ModelResponseChunk) error {
				for _, p := range c.Content {
					if p.IsText() {
						out += c.Content[0].Text
					}
				}
				return nil
			}),
			ai.WithMessages(
				ai.NewUserMessage(
					ai.NewTextPart("do you know which animal is in the image?"),
					ai.NewMediaPart("", "data:image/jpeg;base64,"+i))))
		if err != nil {
			t.Fatal(err)
		}
		if out != resp.Text() {
			t.Fatalf("want: %s, got: %s", resp.Text(), out)
		}
		if !strings.Contains(strings.ToLower(resp.Text()), "cat") {
			t.Fatalf("want: cat, got: %s", resp.Text())
		}
		if resp.Reasoning() == "" {
			t.Fatalf("empty reasoning found")
		}
	})
	t.Run("tools", func(t *testing.T) {
		m := anthropicPlugin.Model(g, "claude-sonnet-4-5-20250929")
		myJokeTool := genkit.DefineTool(
			g,
			"myJoke",
			"When the user asks for a joke, this tool must be used to generate a joke, try to come up with a joke that uses the output of the tool",
			func(ctx *ai.ToolContext, input *any) (string, error) {
				return "why did the chicken cross the road?", nil
			},
		)
		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithConfig(&anthropic.MessageNewParams{
				Temperature: anthropic.Float(1),
				MaxTokens:   1024,
			}),
			ai.WithPrompt("tell me a joke"),
			ai.WithTools(myJokeTool))
		if err != nil {
			t.Fatal(err)
		}

		if len(resp.Text()) == 0 {
			t.Fatal("expected a response but nothing was returned")
		}
	})
	t.Run("tools with schema", func(t *testing.T) {
		m := anthropicPlugin.Model(g, "claude-sonnet-4-5-20250929")

		type WeatherInput struct {
			Location string `json:"location"`
		}

		weatherTool := genkit.DefineTool(
			g,
			"weather",
			"Returns the weather for the given location",
			func(ctx *ai.ToolContext, input *WeatherInput) (string, error) {
				return "sunny", nil
			},
		)

		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithConfig(&anthropic.MessageNewParams{
				Temperature: anthropic.Float(1),
				MaxTokens:   1024,
			}),
			ai.WithPrompt("what is the weather in San Francisco?"),
			ai.WithTools(weatherTool))
		if err != nil {
			t.Fatal(err)
		}

		if len(resp.Text()) == 0 {
			t.Fatal("expected a response but nothing was returned")
		}
	})
	t.Run("streaming", func(t *testing.T) {
		m := anthropicPlugin.Model(g, "claude-sonnet-4-5-20250929")
		out := ""

		final, err := genkit.Generate(ctx, g,
			ai.WithPrompt("Tell me a short story about a frog and a princess"),
			ai.WithConfig(&anthropic.MessageNewParams{
				Temperature: anthropic.Float(1),
				MaxTokens:   1024,
			}),
			ai.WithModel(m),
			ai.WithStreaming(func(ctx context.Context, c *ai.ModelResponseChunk) error {
				out += c.Content[0].Text
				return nil
			}),
		)
		if err != nil {
			t.Fatal(err)
		}

		out2 := ""
		for _, p := range final.Message.Content {
			out2 += p.Text
		}

		if out != out2 {
			t.Fatalf("streaming and final should contain the same text.\nstreaming: %s\nfinal:%s\n", out, out2)
		}
		if final.Usage.InputTokens == 0 || final.Usage.OutputTokens == 0 {
			t.Fatalf("empty usage stats: %#v", *final.Usage)
		}
	})
	t.Run("streaming with thinking", func(t *testing.T) {
		m := anthropicPlugin.Model(g, "claude-sonnet-4-5-20250929")
		out := ""
		reasoningStream := ""

		final, err := genkit.Generate(ctx, g,
			ai.WithPrompt("Tell me a short story about a frog and a princess"),
			ai.WithConfig(&anthropic.MessageNewParams{
				Temperature: anthropic.Float(1),
				Thinking: anthropic.ThinkingConfigParamUnion{
					OfEnabled: &anthropic.ThinkingConfigEnabledParam{
						BudgetTokens: 1024,
					},
				},
				MaxTokens: 2048,
			}),
			ai.WithModel(m),
			ai.WithStreaming(func(ctx context.Context, c *ai.ModelResponseChunk) error {
				for _, p := range c.Content {
					if p.IsText() {
						out += p.Text
					}
					if p.IsReasoning() {
						reasoningStream += p.Text
					}
				}
				return nil
			}),
		)
		if err != nil {
			t.Fatal(err)
		}

		out2 := ""
		for _, p := range final.Message.Content {
			if p.IsText() {
				out2 += p.Text
			}
		}
		if out != out2 {
			t.Fatalf("streaming and final should contain the same text.\n\nstreaming: %s\n\nfinal: %s\n\n", out, out2)
		}
		if final.Reasoning() == "" {
			t.Fatal("empty reasoning found")
		}
		if final.Reasoning() != reasoningStream {
			t.Fatalf("mismatch reasoning, got: %s, want: %s", reasoningStream, final.Reasoning())
		}
		if final.Usage.InputTokens == 0 || final.Usage.OutputTokens == 0 {
			t.Fatalf("empty usage stats: %#v", *final.Usage)
		}
	})
	t.Run("tools streaming", func(t *testing.T) {
		m := anthropicPlugin.Model(g, "claude-sonnet-4-5-20250929")
		out := ""

		myStoryTool := genkit.DefineTool(
			g,
			"myStory",
			"When the user asks for a story, create a story about a frog and a fox that are good friends",
			func(ctx *ai.ToolContext, input *any) (string, error) {
				return "the fox is named Goph and the frog is called Fred", nil
			},
		)

		final, err := genkit.Generate(ctx, g,
			ai.WithPrompt("Tell me a short story about a frog and a fox, do no mention anything else, only the short story"),
			ai.WithModel(m),
			ai.WithConfig(&anthropic.MessageNewParams{
				Temperature: anthropic.Float(1),
				MaxTokens:   1024,
			}),
			ai.WithTools(myStoryTool),
			ai.WithStreaming(func(ctx context.Context, c *ai.ModelResponseChunk) error {
				out += c.Content[0].Text
				return nil
			}),
		)
		if err != nil {
			t.Fatal(err)
		}

		out2 := ""
		for _, p := range final.Message.Content {
			if p.IsText() {
				out2 += p.Text
			}
		}

		if out != out2 {
			t.Fatalf("streaming and final should contain the same text\n\nstreaming: %s\n\nfinal: %s\n\n", out, out2)
		}
		if final.Usage.InputTokens == 0 || final.Usage.OutputTokens == 0 {
			t.Fatalf("empty usage stats: %#v", *final.Usage)
		}
	})
	t.Run("constrained generation", func(t *testing.T) {
		m := anthropicPlugin.Model(g, "claude-sonnet-4-5-20250929")
		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithConfig(&anthropic.MessageNewParams{MaxTokens: 1024}),
			ai.WithPrompt("Which country was Napoleon the emperor of?"),
			ai.WithOutputType(outFormat{}),
		)
		if err != nil {
			t.Fatal(err)
		}

		var ans outFormat
		if err := resp.Output(&ans); err != nil {
			t.Fatal(err)
		}
		const want = "France"
		if ans.Country != want {
			t.Errorf("got %q, expecting %q", ans.Country, want)
		}
	})
	t.Run("streaming constrained generation", func(t *testing.T) {
		m := anthropicPlugin.Model(g, "claude-sonnet-4-5-20250929")

		var streamedContent strings.Builder
		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithConfig(&anthropic.MessageNewParams{MaxTokens: 1024}),
			ai.WithPrompt("Which country was Napoleon the emperor of?"),
			ai.WithOutputType(outFormat{}),
			ai.WithStreaming(func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
				streamedContent.WriteString(chunk.Text())
				return nil
			}),
		)
		if err != nil {
			t.Fatal(err)
		}

		var ans outFormat
		if err := resp.Output(&ans); err != nil {
			t.Fatal(err)
		}
		const want = "France"
		if ans.Country != want {
			t.Errorf("got %q, expecting %q", ans.Country, want)
		}

		if streamedContent.Len() == 0 {
			t.Error("expected streamed content, got empty")
		}
	})
	t.Run("tools streaming with constrained gen", func(t *testing.T) {
		m := anthropicPlugin.Model(g, "claude-opus-4-6")
		answerOfEverythingTool := genkit.DefineTool(
			g,
			"answerOfEverythingTool",
			"use this tool when the user asks for the answer of everything or the universe",
			func(ctx *ai.ToolContext, input *any) (int, error) {
				return 42, nil
			},
		)
		type Output struct {
			AnswerOfEverything int `json:"answer_of_everything"`
		}

		resp, err := genkit.Generate(ctx, g,
			ai.WithPrompt("what's the answer of everything?"),
			ai.WithModel(m),
			ai.WithConfig(&anthropic.MessageNewParams{
				Temperature: anthropic.Float(1),
				MaxTokens:   2048,
			}),
			ai.WithOutputType(Output{}),
			ai.WithStreaming(func(ctx context.Context, c *ai.ModelResponseChunk) error {
				return nil
			}),

			ai.WithTools(answerOfEverythingTool))
		if err != nil {
			t.Fatal(err)
		}

		var out Output
		err = resp.Output(&out)
		if err != nil {
			t.Fatal(err)
		}
		if out.AnswerOfEverything != 42 {
			t.Fatalf("constrained generation failed, want: 42, got: %d", out.AnswerOfEverything)
		}
	})
}

func fetchImgAsBase64() (string, error) {
	// CC0 license image
	imgURL := "https://pd.w.org/2025/07/896686fbbcd9990c9.84605288-2048x1365.jpg"
	resp, err := http.Get(imgURL)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", err
	}

	imageBytes, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	base64string := base64.StdEncoding.EncodeToString(imageBytes)
	return base64string, nil
}

func requireEnv(key string) (string, bool) {
	value, ok := os.LookupEnv(key)
	if !ok || value == "" {
		return "", false
	}

	return value, true
}
