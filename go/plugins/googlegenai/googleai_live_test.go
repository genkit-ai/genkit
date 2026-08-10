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

package googlegenai_test

import (
	"context"
	"encoding/base64"
	"flag"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"google.golang.org/genai"
)

// To run this test suite: go test -v -run TestGoogleAI

var cache = flag.String("cache", "", "Local file to cache (large text document)")

func requireEnv(key string) (string, bool) {
	value, ok := os.LookupEnv(key)
	if !ok || value == "" {
		return "", false
	}

	return value, true
}

// We can't test the DefineAll functions along with the other tests because
// we get duplicate definitions of models.
var testAll = flag.Bool("all", false, "test DefineAllXXX functions")

func TestGoogleAILive(t *testing.T) {
	apiKey, ok := requireEnv("GEMINI_API_KEY")
	if !ok {
		apiKey, ok = requireEnv("GOOGLE_API_KEY")
		if !ok {
			t.Skip("no gemini api key provided, set either GEMINI_API_KEY or GOOGLE_API_KEY in environment")
		}
	}
	if *testAll {
		t.Skip("-all provided")
	}

	ctx := context.Background()

	g := genkit.Init(ctx,
		genkit.WithDefaultModel("googleai/gemini-2.5-flash"),
		genkit.WithPlugins(&googlegenai.GoogleAI{APIKey: apiKey}),
	)

	gablorkenTool := genkit.DefineTool(g, "gablorken", "use this tool when the user asks to calculate a gablorken, carefuly inspect the user input to determine which value from the prompt corresponds to the input structure",
		func(ctx *ai.ToolContext, input struct {
			Value int
			Over  float64
		},
		) (float64, error) {
			return math.Pow(float64(input.Value), input.Over), nil
		},
	)

	answerOfEverythingTool := genkit.DefineTool(g, "answerOfEverything", "use this tool when the user asks for the answer of life, the universe and everything",
		func(ctx *ai.ToolContext, input any) (int, error) {
			return 42, nil
		},
	)

	t.Run("embedder", func(t *testing.T) {
		res, err := genkit.Embed(ctx, g, ai.WithEmbedderName("googleai/gemini-embedding-001"), ai.WithTextDocs("yellow banana"))
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

	t.Run("generate", func(t *testing.T) {
		resp, err := genkit.Generate(ctx, g,
			ai.WithPrompt("Which country was Napoleon the emperor of? Name the country, nothing else"),
		)
		if err != nil {
			t.Fatal(err)
		}

		out := strings.ReplaceAll(resp.Message.Content[0].Text, "\n", "")
		const want = "France"
		if out != want {
			t.Errorf("got %q, expecting %q", out, want)
		}
		if resp.Request == nil {
			t.Error("Request field not set properly")
		}
		if resp.Usage.InputTokens == 0 || resp.Usage.OutputTokens == 0 || resp.Usage.TotalTokens == 0 {
			t.Errorf("Empty usage stats %#v", *resp.Usage)
		}
	})

	t.Run("streaming", func(t *testing.T) {
		out := ""
		parts := 0
		final, err := genkit.Generate(ctx, g,
			ai.WithPrompt("Write one paragraph about the North Pole."),
			ai.WithStreaming(func(ctx context.Context, c *ai.ModelResponseChunk) error {
				parts++
				out += c.Content[0].Text
				return nil
			}))
		if err != nil {
			t.Fatal(err)
		}
		out2 := ""
		for _, p := range final.Message.Content {
			out2 += p.Text
		}
		if out != out2 {
			t.Errorf("streaming and final should contain the same text.\nstreaming:%s\nfinal:%s", out, out2)
		}
		const want = "North"
		if !strings.Contains(out, want) {
			t.Errorf("got %q, expecting it to contain %q", out, want)
		}
		if parts == 1 {
			// Check if streaming actually occurred.
			t.Errorf("expecting more than one part")
		}
		if final.Usage.InputTokens == 0 || final.Usage.OutputTokens == 0 || final.Usage.TotalTokens == 0 {
			t.Errorf("Empty usage stats %#v", *final.Usage)
		}
	})

	t.Run("tool", func(t *testing.T) {
		resp, err := genkit.Generate(ctx, g,
			ai.WithPrompt("what is a gablorken of 2 over 3.5?"),
			ai.WithTools(gablorkenTool))
		if err != nil {
			t.Fatal(err)
		}

		out := resp.Text()
		const want = "11.31"
		if !strings.Contains(out, want) {
			t.Errorf("got %q, expecting it to contain %q", out, want)
		}
	})
	t.Run("tool stream", func(t *testing.T) {
		parts := 0
		out := ""
		final, err := genkit.Generate(ctx, g,
			ai.WithPrompt("what is a gablorken of 2 over 3.5?"),
			ai.WithTools(gablorkenTool),
			ai.WithStreaming(func(ctx context.Context, c *ai.ModelResponseChunk) error {
				parts++
				out += c.Content[0].Text
				return nil
			}))
		if err != nil {
			t.Fatal(err)
		}
		out2 := ""
		for _, p := range final.Message.Content {
			out2 += p.Text
		}
		if out != out2 {
			t.Errorf("streaming and final should contain the same text.\nstreaming:%s\nfinal:%s", out, out2)
		}

		const want = "11.31"
		if !strings.Contains(final.Text(), want) {
			t.Errorf("got %q, expecting it to contain %q", out, want)
		}
	})

	t.Run("tool with thinking", func(t *testing.T) {
		m := googlegenai.GoogleAIModel(g, "gemini-2.5-flash")
		resp, err := genkit.Generate(ctx, g,
			ai.WithConfig(&genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{
					ThinkingBudget: genai.Ptr[int32](1024),
				},
			}),
			ai.WithModel(m),
			ai.WithPrompt("what is a gablorken of value 2 over 3.5?"),
			ai.WithTools(gablorkenTool))
		if err != nil {
			t.Fatal(err)
		}

		out := resp.Text()
		const want = "11.31"
		if !strings.Contains(out, want) {
			t.Errorf("got %q, expecting it to contain %q", out, want)
		}
	})
	t.Run("api side tools", func(t *testing.T) {
		m := googlegenai.GoogleAIModel(g, "gemini-2.5-flash")
		_, err := genkit.Generate(ctx, g,
			ai.WithConfig(&genai.GenerateContentConfig{
				Tools: []*genai.Tool{
					{GoogleSearch: &genai.GoogleSearch{}},
					{CodeExecution: &genai.ToolCodeExecution{}},
				},
			}),
			ai.WithModel(m),
			ai.WithPrompt("When is the next lunar eclipse in US?"))
		if err != nil {
			t.Fatal(err)
		}
	})
	t.Run("api and custom tools with GoogleSearch", func(t *testing.T) {
		// Note: The Gemini API does not support combining GoogleSearch with function calling.
		// This test verifies that tools are properly merged (not silently dropped),
		// even though the API will reject this specific combination.
		// See: https://github.com/google/adk-python/issues/53
		m := googlegenai.GoogleAIModel(g, "gemini-2.5-flash")
		_, err := genkit.Generate(ctx, g,
			ai.WithConfig(&genai.GenerateContentConfig{
				Tools: []*genai.Tool{
					{GoogleSearch: &genai.GoogleSearch{}},
				},
			}),
			ai.WithModel(m),
			ai.WithTools(gablorkenTool, answerOfEverythingTool),
			ai.WithPrompt("What is the answer of life?"))
		// Expect API error because GoogleSearch + function calling is unsupported
		if err == nil {
			t.Fatal("expected error combining GoogleSearch with function calling, but got none")
		}
		if !strings.Contains(err.Error(), "Tool use with function calling is unsupported") &&
			!strings.Contains(err.Error(), "INVALID_ARGUMENT") {
			t.Fatalf("unexpected error: %v", err)
		}
	})
	t.Run("tool with json output", func(t *testing.T) {
		type weatherQuery struct {
			Location string `json:"location"`
		}

		type weather struct {
			Report string `json:"report"`
		}

		weatherTool := genkit.DefineTool(g, "weatherTool",
			"Use this tool to get the weather report for a specific location",
			func(ctx *ai.ToolContext, input weatherQuery) (string, error) {
				report := fmt.Sprintf("The weather in %s is sunny and 70 degrees today.", input.Location)
				return report, nil
			},
		)

		resp, err := genkit.Generate(ctx, g,
			ai.WithTools(weatherTool),
			ai.WithPrompt("what's the weather in San Francisco?"),
			ai.WithOutputType(weather{}),
		)
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
	t.Run("avoid tool", func(t *testing.T) {
		resp, err := genkit.Generate(ctx, g,
			ai.WithPrompt("what is a gablorken of value 2 over 3.5?"),
			ai.WithTools(gablorkenTool),
			ai.WithToolChoice(ai.ToolChoiceNone),
		)
		if err != nil {
			t.Fatal(err)
		}

		out := resp.Text()
		const doNotWant = "11.31"
		if strings.Contains(out, doNotWant) {
			t.Errorf("got %q, expecting it NOT to contain %q", out, doNotWant)
		}
	})
	t.Run("cache", func(t *testing.T) {
		if *cache == "" {
			t.Skip("no cache contents provided, use -cache flag")
		}

		textContent, err := os.ReadFile(*cache)
		if err != nil {
			t.Fatal(err)
		}

		resp, err := genkit.Generate(ctx, g,
			ai.WithMessages(
				ai.NewUserTextMessage(string(textContent)).WithCacheTTL(360),
			),
			ai.WithPrompt("write a summary of the content"))
		if err != nil {
			t.Fatal(err)
		}

		// inspect metadata just to make sure the cache was created
		m := resp.Message.Metadata
		cacheName := ""
		if cache, ok := m["cache"].(map[string]any); ok {
			if n, ok := cache["name"].(string); ok {
				if n == "" {
					t.Fatal("expecting a cache name, but got nothing")
				}
				cacheName = n
			} else {
				t.Fatalf("cache name should be a string but got %T", n)
			}
		} else {
			t.Fatalf("cache name should be a map but got %T", cache)
		}

		resp, err = genkit.Generate(ctx, g,
			ai.WithMessages(resp.History()...),
			ai.WithPrompt("rewrite the previous summary but now talking like a pirate, say Ahoy a lot of times"),
		)
		if err != nil {
			t.Fatal(err)
		}
		if resp.Usage.CachedContentTokens == 0 {
			t.Fatal("expecting cached content tokens but got empty")
		}
		text := resp.Text()
		if !strings.Contains(text, "Ahoy") {
			t.Fatalf("expecting a response as a pirate but got %v", text)
		}

		// cache metadata should have not changed...
		if cache, ok := m["cache"].(map[string]any); ok {
			if n, ok := cache["name"].(string); ok {
				if n == "" {
					t.Fatal("expecting a cache name, but got nothing")
				}
				if cacheName != n {
					t.Fatalf("cache name mismatch, want: %s, got: %s", cacheName, n)
				}
			} else {
				t.Fatalf("cache name should be a string but got %T", n)
			}
		} else {
			t.Fatalf("cache name should be a map but got %T", cache)
		}
	})
	t.Run("media content (inline data)", func(t *testing.T) {
		i, err := fetchImgAsBase64()
		if err != nil {
			t.Fatal(err)
		}
		resp, err := genkit.Generate(ctx, g,
			ai.WithSystem("You are a pirate expert in animals, your response should include the name of the animal in the provided image"),
			ai.WithMessages(
				ai.NewUserMessage(
					ai.NewTextPart("do you what animal is in the image?"),
					ai.NewMediaPart("image/jpeg", "data:image/jpeg;base64,"+i),
				),
			),
		)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(strings.ToLower(resp.Text()), "cat") {
			t.Fatalf("image detection failed, want: cat, got: %s", resp.Text())
		}
	})
	t.Run("media content", func(t *testing.T) {
		resp, err := genkit.Generate(ctx, g,
			ai.WithMessages(
				ai.NewUserMessage(
					ai.NewTextPart("do you know what's the video about?"),
					ai.NewMediaPart("video/mp4", `https://www.youtube.com/watch?v=_6FYhqGgel8`),
				),
			),
		)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(resp.Text(), "Mario Kart") {
			t.Fatalf("image detection failed, want: Mario Kart, got: %s", resp.Text())
		}
	})
	t.Run("data content (inline data)", func(t *testing.T) {
		i, err := fetchImgAsBase64()
		if err != nil {
			t.Fatal(err)
		}
		resp, err := genkit.Generate(ctx, g,
			ai.WithSystem("You are an excellent animal detector, the user will provide you a request with an image, identify which animal is in there"),
			ai.WithMessages(
				ai.NewUserMessage(
					ai.NewTextPart("do you know who's in the image?"),
					ai.NewDataPart("data:image/jpeg;base64,"+i),
				),
			),
		)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(strings.ToLower(resp.Text()), "cat") {
			t.Fatalf("image detection failed, want: cat, got: %s", resp.Text())
		}
	})
	t.Run("image generation", func(t *testing.T) {
		m := googlegenai.GoogleAIModel(g, "gemini-2.5-flash-image")
		resp, err := genkit.Generate(ctx, g,
			ai.WithConfig(genai.GenerateContentConfig{
				ResponseModalities: []string{"IMAGE", "TEXT"},
			}),
			ai.WithMessages(
				ai.NewUserTextMessage("generate an image of a dog wearing a black tejana while playing the accordion"),
			),
			ai.WithModel(m),
		)
		if err != nil {
			t.Fatal(err)
		}
		if len(resp.Message.Content) == 0 {
			t.Fatal("empty response")
		}
		foundMediaPart := false
		for _, part := range resp.Message.Content {
			if part.ContentType == "image/png" {
				foundMediaPart = true
				if part.Kind != ai.PartMedia {
					t.Errorf("expecting part to be Media type but got: %q", part.Kind)
				}
				if part.Text == "" {
					t.Error("empty response")
				}
			}
		}

		if !foundMediaPart {
			t.Error("no media found in the response message")
		}
	})
	t.Run("constrained generation", func(t *testing.T) {
		type outFormat struct {
			Country string
		}
		resp, err := genkit.Generate(ctx, g,
			ai.WithPrompt("Which country was Napoleon the emperor of?"),
			ai.WithOutputType(outFormat{}),
		)
		if err != nil {
			t.Fatal(err)
		}

		var ans outFormat
		err = resp.Output(&ans)
		if err != nil {
			t.Fatal(err)
		}
		const want = "France"
		if ans.Country != want {
			t.Errorf("got %q, expecting %q", ans.Country, want)
		}
		if resp.Request == nil {
			t.Error("Request field not set properly")
		}
		if resp.Usage.InputTokens == 0 || resp.Usage.OutputTokens == 0 || resp.Usage.TotalTokens == 0 {
			t.Errorf("Empty usage stats %#v", *resp.Usage)
		}
	})
	t.Run("thinking", func(t *testing.T) {
		m := googlegenai.GoogleAIModel(g, "gemini-2.5-flash")
		resp, err := genkit.Generate(ctx, g,
			ai.WithConfig(genai.GenerateContentConfig{
				Temperature: genai.Ptr[float32](0.4),
				ThinkingConfig: &genai.ThinkingConfig{
					IncludeThoughts: true,
					ThinkingBudget:  genai.Ptr[int32](1024),
				},
			}),
			ai.WithModel(m),
			ai.WithPrompt("Analogize photosynthesis and growing up."))
		if err != nil {
			t.Fatal(err)
		}
		if resp == nil {
			t.Fatal("nil response obtanied")
		}
		if resp.Usage.ThoughtsTokens == 0 || resp.Usage.ThoughtsTokens > 1024 {
			t.Fatalf("thoughts tokens should not be zero or greater than 100, got: %d", resp.Usage.ThoughtsTokens)
		}
	})
	t.Run("thinking stream with structured output", func(t *testing.T) {
		type Output struct {
			Text string `json:"text"`
		}

		m := googlegenai.GoogleAIModel(g, "gemini-2.5-flash")
		resp, err := genkit.Generate(ctx, g,
			ai.WithConfig(genai.GenerateContentConfig{
				Temperature: genai.Ptr[float32](0.4),
				ThinkingConfig: &genai.ThinkingConfig{
					IncludeThoughts: true,
					ThinkingBudget:  genai.Ptr[int32](1024),
				},
			}),
			ai.WithModel(m),
			ai.WithOutputType(Output{}),
			ai.WithPrompt("Analogize photosynthesis and growing up."),
			ai.WithStreaming(func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
				return nil
			}),
		)
		if err != nil {
			t.Fatal(err)
		}
		if resp == nil {
			t.Fatal("nil response obtanied")
		}
		var out Output
		err = resp.Output(&out)
		if err != nil {
			t.Fatalf("unable to unmarshal response: %v", err)
		}
		if resp.Usage.ThoughtsTokens == 0 || resp.Usage.ThoughtsTokens > 1024 {
			t.Fatalf("thoughts tokens should not be zero or greater than 1024, got: %d", resp.Usage.ThoughtsTokens)
		}
		if resp.Reasoning() == "" {
			t.Fatalf("no reasoning found")
		}
	})
	t.Run("thinking disabled", func(t *testing.T) {
		m := googlegenai.GoogleAIModel(g, "gemini-2.5-flash")
		resp, err := genkit.Generate(ctx, g,
			ai.WithConfig(genai.GenerateContentConfig{
				Temperature: genai.Ptr[float32](0.4),
				ThinkingConfig: &genai.ThinkingConfig{
					IncludeThoughts: false,
					ThinkingBudget:  genai.Ptr[int32](0),
				},
			}),
			ai.WithModel(m),
			ai.WithPrompt("Analogize photosynthesis and growing up."))
		if err != nil {
			t.Fatal(err)
		}
		if resp == nil {
			t.Fatal("nil response obtanied")
		}
		if resp.Usage.ThoughtsTokens > 0 {
			t.Fatal("thoughts tokens should be zero")
		}
	})
	t.Run("multipart tool", func(t *testing.T) {
		m := googlegenai.GoogleAIModel(g, "gemini-3-pro-preview")
		img64, err := fetchImgAsBase64()
		if err != nil {
			t.Fatal(err)
		}

		tool := genkit.DefineMultipartTool(g, "getImage", "returns a misterious image",
			func(ctx *ai.ToolContext, input any) (*ai.MultipartToolResponse, error) {
				return &ai.MultipartToolResponse{
					Output: map[string]any{"status": "success"},
					Content: []*ai.Part{
						ai.NewMediaPart("image/jpeg", "data:image/jpeg;base64,"+img64),
					},
				}, nil
			},
		)

		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithTools(tool),
			ai.WithPrompt("get an image and tell me what is in it"),
		)
		if err != nil {
			t.Fatal(err)
		}

		if !strings.Contains(strings.ToLower(resp.Text()), "cat") {
			t.Errorf("expected response to contain 'cat', got: %s", resp.Text())
		}
	})
}

func TestCacheHelper(t *testing.T) {
	t.Run("cache metadata", func(t *testing.T) {
		req := ai.ModelRequest{
			Messages: []*ai.Message{
				ai.NewUserMessage(
					ai.NewTextPart(("this is just a test")),
				),
				ai.NewModelMessage(
					ai.NewTextPart("oh really? is it?")).WithCacheTTL(100),
			},
		}

		for _, m := range req.Messages {
			if m.Role == ai.RoleModel {
				metadata := m.Metadata
				if len(metadata) == 0 {
					t.Fatal("expected metadata with contents, got empty")
				}
				cache, ok := metadata["cache"].(map[string]any)
				if !ok {
					t.Fatalf("cache should be a map, got: %T", cache)
				}
				if cache["ttlSeconds"] != 100 {
					t.Fatalf("expecting ttlSeconds to be 100s, got: %q", cache["ttlSeconds"])
				}
			}
		}
	})
	t.Run("cache metadata overwrite", func(t *testing.T) {
		m := ai.NewModelMessage(ai.NewTextPart("foo bar")).WithCacheTTL(100)
		metadata := m.Metadata
		if len(metadata) == 0 {
			t.Fatal("expected metadata with contents, got empty")
		}
		cache, ok := metadata["cache"].(map[string]any)
		if !ok {
			t.Fatalf("cache should be a map, got: %T", cache)
		}
		if cache["ttlSeconds"] != 100 {
			t.Fatalf("expecting ttlSeconds to be 100s, got: %q", cache["ttlSeconds"])
		}

		m.Metadata["foo"] = "bar"
		m.WithCacheTTL(50)

		metadata = m.Metadata
		cache, ok = metadata["cache"].(map[string]any)
		if !ok {
			t.Fatalf("cache should be a map, got: %T", cache)
		}
		if cache["ttlSeconds"] != 50 {
			t.Fatalf("expecting ttlSeconds to be 50s, got: %d", cache["ttlSeconds"])
		}
		_, ok = metadata["foo"]
		if !ok {
			t.Fatal("metadata contents were altered, expecting foo key")
		}
		bar, ok := metadata["foo"].(string)
		if !ok {
			t.Fatalf(`metadata["foo"] contents got altered, expecting string, got: %T`, bar)
		}
		if bar != "bar" {
			t.Fatalf("expecting to be bar but got: %q", bar)
		}

		m.WithCacheName("dummy-name")
		metadata = m.Metadata
		cache, ok = metadata["cache"].(map[string]any)
		if !ok {
			t.Fatalf("cache should be a map, got: %T", cache)
		}
		ttl, ok := cache["ttlSeconds"].(int)
		if ok {
			t.Fatalf("cache should have been overwriten, expecting cache name, not ttl: %d", ttl)
		}
		name, ok := cache["name"].(string)
		if !ok {
			t.Fatalf("cache should have been overwriten, expecting cache name, got: %v", name)
		}
		if name != "dummy-name" {
			t.Fatalf("cache name mismatch, want dummy-name, got: %s", name)
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
