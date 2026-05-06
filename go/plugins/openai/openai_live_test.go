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

package openai_test

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	oai "github.com/firebase/genkit/go/plugins/openai"
	openai "github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"
	"github.com/openai/openai-go/v3/shared"
)

func TestOpenAILive(t *testing.T) {
	if _, ok := requireEnv("OPENAI_API_KEY"); !ok {
		t.Skip("OPENAI_API_KEY not found in the environment")
	}

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&oai.OpenAI{}))

	myJokeTool := genkit.DefineTool(
		g,
		"myJoke",
		"When the user asks for a joke, this tool must be used to generate a joke, try to come up with a joke that uses the output of the tool",
		func(ctx *ai.ToolContext, input *any) (string, error) {
			return "why did the chicken cross the road?", nil
		},
	)

	myStoryTool := genkit.DefineTool(
		g,
		"myStory",
		"When the user asks for a story, create a story about a frog and a fox that are good friends",
		func(ctx *ai.ToolContext, input *any) (string, error) {
			return "the fox is named Goph and the frog is called Fred", nil
		},
	)

	type WeatherInput struct {
		Location string `json:"location"`
	}

	weatherTool := genkit.DefineTool(
		g,
		"weather",
		"Returns the weather for the given location",
		func(ctx *ai.ToolContext, input *WeatherInput) (string, error) {
			report := fmt.Sprintf("The weather in %s is sunny", input.Location)
			return report, nil
		},
	)

	gablorkenDefinitionTool := genkit.DefineTool(
		g,
		"gablorkenDefinitionTool",
		"Custom tool that must be used when the user asks for the definition of a gablorken",
		func(ctx *ai.ToolContext, input *any) (string, error) {
			return "A gablorken is a interstellar currency for the Andromeda Galaxy. It is equivalent to 0.4 USD per Gablorken (GAB)", nil
		},
	)

	t.Run("system messages", func(t *testing.T) {
		instruction := "Talk to me as an evil pirate and say \"Arr\" several times. Be very short in your response."
		m := oai.Model(g, "gpt-4o")
		resp, err := genkit.Generate(ctx, g,
			ai.WithConfig(&responses.ResponseNewParams{
				Temperature:     openai.Float(1),
				MaxOutputTokens: openai.Int(1024),
			}),
			ai.WithModel(m),
			ai.WithSystem(instruction),
			ai.WithMessages(ai.NewUserMessage(ai.NewTextPart("I'm a fish"))),
		)
		if err != nil {
			t.Fatal(err)
		}
		if resp.Raw == nil {
			t.Error("expecting Raw contents to be populated, got nil")
		}

		count := 0
		for _, m := range resp.Request.Messages {
			count += 1
			if m.Role == ai.RoleSystem {
				if m.Text() != instruction {
					t.Errorf("invalid system message found, got: %s, want: %s", m.Text(), instruction)
				}
			}
		}
		if count != 2 {
			t.Errorf("Expecting exactly 2 messages, got: %d", count)
		}

		if !strings.Contains(resp.Text(), "Arr") {
			t.Errorf("not a pirate, got: %s", resp.Text())
		}
	})

	t.Run("model version not ok", func(t *testing.T) {
		m := oai.Model(g, "non-existent-model")
		_, err := genkit.Generate(ctx, g,
			ai.WithConfig(&responses.ResponseNewParams{
				Temperature:     openai.Float(1),
				MaxOutputTokens: openai.Int(1024),
			}),
			ai.WithModel(m),
		)
		if err == nil {
			t.Fatal("should have failed due wrong model version")
		}
	})

	t.Run("model ref", func(t *testing.T) {
		config := &responses.ResponseNewParams{
			Temperature:     openai.Float(1),
			MaxOutputTokens: openai.Int(1024),
			// Add built-in tool via config
		}
		mr := oai.ModelRef("gpt-4o", config)

		resp, err := genkit.Generate(ctx, g,
			ai.WithPrompt("tell me a fact about Golang"),
			ai.WithModel(mr),
		)
		if err != nil {
			t.Fatal(err)
		}
		if resp.Text() == "" {
			t.Error("expected to have a response, got empty")
		}
		if resp.Request.Config == nil {
			t.Fatal("expected a not nil configuration, got empty")
		}
		if cfg, ok := resp.Request.Config.(*responses.ResponseNewParams); ok {
			if cfg.MaxOutputTokens != openai.Int(1024) {
				t.Errorf("wrong MaxOutputTokens value, got: %d, want: 1024", cfg.MaxOutputTokens.Value)
			}
			if cfg.Temperature != openai.Float(1) {
				t.Errorf("wrongTemperature value, got: %f, want: 1", cfg.Temperature.Value)
			}
		} else {
			t.Fatalf("unexpected config, got: %T, want: %T", cfg, config)
		}
	})

	t.Run("media content", func(t *testing.T) {
		i, err := fetchImgAsBase64()
		if err != nil {
			t.Fatal(err)
		}
		m := oai.Model(g, "gpt-4o")
		resp, err := genkit.Generate(ctx, g,
			ai.WithSystem("You are a professional image detective that talks like an evil pirate that loves animals, your task is to tell the name of the animal in the image but be very short"),
			ai.WithModel(m),
			ai.WithConfig(&responses.ResponseNewParams{
				Temperature:     openai.Float(1),
				MaxOutputTokens: openai.Int(1024),
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
		m := oai.Model(g, "gpt-4o")
		resp, err := genkit.Generate(ctx, g,
			ai.WithSystem("You are a professional image detective that talks like an evil pirate that loves animals, your task is to tell the name of the animal in the image but be very short"),
			ai.WithModel(m),
			ai.WithConfig(&responses.ResponseNewParams{
				Temperature:     openai.Float(1),
				MaxOutputTokens: openai.Int(1024),
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
		m := oai.Model(g, "gpt-5")
		resp, err := genkit.Generate(ctx, g,
			ai.WithSystem(`You are a professional image detective that
				talks like an evil pirate that loves animals, your task is to tell the name
				of the animal in the image but be very short`),
			ai.WithModel(m),
			ai.WithConfig(&responses.ResponseNewParams{
				Reasoning: shared.ReasoningParam{
					Effort: shared.ReasoningEffortMedium,
				},
			}),
			ai.WithStreaming(func(ctx context.Context, c *ai.ModelResponseChunk) error {
				for _, p := range c.Content {
					if p.IsText() {
						out += p.Text
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
		if resp.Usage.ThoughtsTokens == 0 {
			t.Log("No reasoning tokens found in usage (expected for reasoning models)")
		}
	})

	t.Run("tools", func(t *testing.T) {
		m := oai.Model(g, "gpt-4o")
		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithConfig(&responses.ResponseNewParams{
				Temperature:     openai.Float(1),
				MaxOutputTokens: openai.Int(1024),
			}),
			ai.WithPrompt("tell me the definition of a gablorken"),
			ai.WithTools(gablorkenDefinitionTool))
		if err != nil {
			t.Fatal(err)
		}

		if len(resp.Text()) == 0 {
			t.Fatal("expected a response but nothing was returned")
		}
	})

	t.Run("tools with schema", func(t *testing.T) {
		m := oai.Model(g, "gpt-4o")
		type weather struct {
			Report string `json:"report"`
		}

		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithConfig(&responses.ResponseNewParams{
				Temperature:     openai.Float(1),
				MaxOutputTokens: openai.Int(1024),
			}),
			ai.WithPrompt("what is the weather in San Francisco?"),
			ai.WithOutputType(weather{}),
			ai.WithTools(weatherTool))
		if err != nil {
			t.Fatal(err)
		}

		var w weather
		if err = resp.Output(&w); err != nil {
			t.Fatal(err)
		}
		if w.Report == "" {
			t.Fatal("empty weather report, tool should have provided an output")
		}
	})

	t.Run("streaming", func(t *testing.T) {
		m := oai.Model(g, "gpt-4o")
		out := ""

		final, err := genkit.Generate(ctx, g,
			ai.WithPrompt("Tell me a short story about a frog and a princess"),
			ai.WithConfig(&responses.ResponseNewParams{
				Temperature:     openai.Float(1),
				MaxOutputTokens: openai.Int(1024),
			}),
			ai.WithModel(m),
			ai.WithStreaming(func(ctx context.Context, c *ai.ModelResponseChunk) error {
				for _, p := range c.Content {
					if p.IsText() {
						out += p.Text
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
		m := oai.Model(g, "gpt-4o")
		out := ""

		final, err := genkit.Generate(ctx, g,
			ai.WithPrompt("Sing me a song about metaphysics"),
			ai.WithConfig(&responses.ResponseNewParams{}),
			ai.WithModel(m),
			ai.WithStreaming(func(ctx context.Context, c *ai.ModelResponseChunk) error {
				for _, p := range c.Content {
					if p.IsText() {
						out += p.Text
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

		if final.Usage.ThoughtsTokens > 0 {
			t.Logf("Reasoning tokens: %d", final.Usage.ThoughtsTokens)
		} else {
			// this might happen if the model decides not to reason much or if stats are missing.
			t.Log("No reasoning tokens reported.")
		}
	})

	t.Run("tools streaming", func(t *testing.T) {
		m := oai.Model(g, "gpt-4o")
		out := ""

		final, err := genkit.Generate(ctx, g,
			ai.WithPrompt("Tell me a short story about a frog and a fox, do no mention anything else, only the short story"),
			ai.WithModel(m),
			ai.WithConfig(&responses.ResponseNewParams{
				Temperature:     openai.Float(1),
				MaxOutputTokens: openai.Int(1024),
			}),
			ai.WithTools(myStoryTool),
			ai.WithStreaming(func(ctx context.Context, c *ai.ModelResponseChunk) error {
				for _, p := range c.Content {
					if p.IsText() {
						out += p.Text
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
			t.Fatalf("streaming and final should contain the same text\n\nstreaming: %s\n\nfinal: %s\n\n", out, out2)
		}
		if final.Usage.InputTokens == 0 || final.Usage.OutputTokens == 0 {
			t.Fatalf("empty usage stats: %#v", *final.Usage)
		}
	})

	t.Run("built-in tools", func(t *testing.T) {
		m := oai.Model(g, "gpt-4o")

		webSearchTool := responses.ToolParamOfWebSearch(responses.WebSearchToolTypeWebSearch)
		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithConfig(&responses.ResponseNewParams{
				Temperature:     openai.Float(1),
				MaxOutputTokens: openai.Int(1024),
				// Add built-in tool via config
				Tools: []responses.ToolUnionParam{webSearchTool},
			}),
			ai.WithPrompt("What's the current weather in SFO?"),
		)
		if err != nil {
			t.Fatal(err)
		}

		if len(resp.Text()) == 0 {
			t.Fatal("expected a response but nothing was returned")
		}
	})

	t.Run("mixed tools", func(t *testing.T) {
		m := oai.Model(g, "gpt-4o")

		webSearchTool := responses.ToolParamOfWebSearch(responses.WebSearchToolTypeWebSearch)

		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithConfig(&responses.ResponseNewParams{
				Temperature:       openai.Float(1),
				MaxOutputTokens:   openai.Int(1024),
				ParallelToolCalls: openai.Bool(true),
				// Add built-in tool via config
				Tools: []responses.ToolUnionParam{webSearchTool},
			}),
			ai.WithPrompt("I'd would like to ask you two things: What's the current weather in SFO? What's the meaning of gablorken?. Use the web search tool to get the weather in SFO and use the gablorken definition tool to give me its definition. Make sure to include the response for both questions in your answer"),
			ai.WithTools(gablorkenDefinitionTool),
		)
		if err != nil {
			t.Fatal(err)
		}

		if len(resp.Text()) == 0 {
			t.Fatal("expected a response but nothing was returned")
		}
	})

	t.Run("structured output", func(t *testing.T) {
		m := oai.Model(g, "gpt-4o")

		type MovieReview struct {
			Title  string `json:"title"`
			Rating int    `json:"rating"`
			Reason string `json:"reason"`
		}

		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithPrompt("Review the movie 'Inception'"),
			ai.WithOutputType(MovieReview{}),
		)
		if err != nil {
			t.Fatal(err)
		}
		var out MovieReview
		if err := resp.Output(&out); err != nil {
			t.Errorf("expected a movie review, got: %v", err)
		}
		if out.Title == "" || out.Rating == 0 || out.Reason == "" {
			t.Fatalf("expected a movie review, got %#v", out)
		}

		review, _, err := genkit.GenerateData[MovieReview](ctx, g,
			ai.WithModel(m),
			ai.WithPrompt("Review the movie 'Signs'"),
		)
		if err != nil {
			t.Fatal(err)
		}

		if review.Title == "" || review.Rating == 0 || review.Reason == "" {
			t.Fatalf("expected a movie review, got %#v", review)
		}
	})

	t.Run("streaming using GenerateDataStream", func(t *testing.T) {
		m := oai.Model(g, "gpt-4o")

		type answerChunk struct {
			Text string `json:"text"`
		}

		chunksCount := 0
		var finalAnswer answerChunk
		for val, err := range genkit.GenerateDataStream[answerChunk](ctx, g,
			ai.WithModel(m),
			ai.WithPrompt("Tell me how's a black hole created in 2 sentences."),
		) {
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if val.Done {
				finalAnswer = val.Output
			} else {
				chunksCount++
			}
		}

		if chunksCount == 0 {
			t.Errorf("expected to receive some chunks, got 0")
		}
		if finalAnswer.Text == "" {
			t.Errorf("expected final answer, got empty")
		}
	})

	t.Run("GenerateDataStream with custom tools", func(t *testing.T) {
		m := oai.Model(g, "gpt-4o")

		type JokeResponse struct {
			Setup     string `json:"setup"`
			Punchline string `json:"punchline"`
		}

		chunksCount := 0
		var finalJoke JokeResponse

		for val, err := range genkit.GenerateDataStream[JokeResponse](ctx, g,
			ai.WithModel(m),
			ai.WithPrompt("Tell me a joke about a chicken crossing the road. Use the myJoke tool to get the punchline."),
			ai.WithTools(myJokeTool),
		) {
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if val.Done {
				finalJoke = val.Output
			} else {
				chunksCount++
			}
		}

		if chunksCount == 0 {
			t.Errorf("expected to receive some chunks, got 0")
		}
		if finalJoke.Setup == "" || finalJoke.Punchline == "" {
			t.Errorf("expected final joke setup and punchline to be populated, got %+v", finalJoke)
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
