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

// Package livetest drives an OpenAI-compatible plugin through the core
// Genkit generate features against the provider's real API: generation,
// history, system prompts, streaming, tool calling, structured output,
// reasoning, vision, and the extra config passthrough.
//
// Each plugin's live test builds a [Suite] naming the models to spend and
// the capabilities the provider claims, then calls [Run] after its API key
// check. The checklist is identical across providers, so one green run per
// plugin is the release sanity check; provider-specific behavior stays in
// the plugin's own subtests.
package livetest

import (
	"bytes"
	"context"
	"encoding/base64"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"math"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

// Suite describes how to drive one provider through the shared checklist.
type Suite struct {
	// Model answers the cheap text prompts. Every check that needs no
	// special capability runs against it. Required.
	Model ai.ModelArg
	// ReasoningModel emits thinking, usually a [ai.ModelRef] carrying the
	// plugin's thinking config. Nil skips the reasoning checks.
	ReasoningModel ai.ModelArg
	// ReasoningContent reports whether the provider returns the thinking
	// content itself. Endpoints that accept the knob but keep the content
	// server-side leave this false and still get the no-error checks.
	ReasoningContent bool
	// StreamOnlyReasoning skips the non-streaming reasoning check, for
	// providers that only think on streaming calls.
	StreamOnlyReasoning bool
	// VisionModel accepts inline images. Nil skips the vision check.
	VisionModel ai.ModelArg
	// ToolChoice runs the tool-choice checks, for providers whose catalog
	// claims the capability.
	ToolChoice bool
	// ExtraConfig, when set, is the whole request config for the
	// passthrough check and must route at least one field through the
	// config's extra map.
	ExtraConfig map[string]any
}

// capitalFacts is the structured output target; asking about France keeps
// every assertion a stable substring check.
type capitalFacts struct {
	City    string `json:"city"`
	Country string `json:"country"`
}

// Run walks the plugin registered on g through the shared checklist. It
// defines a tool named "gablorken" on g, so call it once per Genkit
// instance and keep that name free in the plugin's own subtests.
func Run(t *testing.T, g *genkit.Genkit, s Suite) {
	ctx := context.Background()
	if s.Model == nil {
		t.Fatal("livetest: Suite.Model is required")
	}

	gablorken := genkit.DefineTool(g, "gablorken", "use when you need to calculate a gablorken",
		func(ctx *ai.ToolContext, input struct {
			Value float64
			Over  float64
		}) (float64, error) {
			return math.Pow(input.Value, input.Over), nil
		})

	t.Run("generate", func(t *testing.T) {
		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(s.Model),
			ai.WithPrompt("What is the capital of France? Reply with just the city name."),
		)
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if !strings.Contains(strings.ToLower(resp.Text()), "paris") {
			t.Errorf("Text() = %q, want it to contain %q", resp.Text(), "Paris")
		}
		if resp.Usage == nil || resp.Usage.TotalTokens == 0 {
			t.Error("Usage.TotalTokens = 0, want the request's token counts")
		}
		if resp.FinishReason != ai.FinishReasonStop {
			t.Errorf("FinishReason = %q, want %q", resp.FinishReason, ai.FinishReasonStop)
		}
	})

	t.Run("history", func(t *testing.T) {
		first, err := genkit.Generate(ctx, g,
			ai.WithModel(s.Model),
			ai.WithPrompt("My name is Zebulon Quixote. Greet me in one short sentence."),
		)
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(s.Model),
			ai.WithMessages(first.History()...),
			ai.WithPrompt("What is my name? Reply with just the name."),
		)
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if !strings.Contains(strings.ToLower(resp.Text()), "zebulon") {
			t.Errorf("Text() = %q, want the name from the first turn", resp.Text())
		}
	})

	t.Run("system prompt", func(t *testing.T) {
		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(s.Model),
			ai.WithSystem("You are named Quixotebot. When asked your name, reply with exactly that name."),
			ai.WithPrompt("What is your name?"),
		)
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if !strings.Contains(strings.ToLower(resp.Text()), "quixotebot") {
			t.Errorf("Text() = %q, want the name the system prompt assigns", resp.Text())
		}
	})

	t.Run("streaming", func(t *testing.T) {
		var streamed strings.Builder
		chunks := 0
		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(s.Model),
			ai.WithPrompt("Write one short paragraph about the ocean."),
			ai.WithStreaming(func(_ context.Context, chunk *ai.ModelResponseChunk) error {
				chunks++
				streamed.WriteString(chunk.Text())
				return nil
			}),
		)
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if chunks <= 1 {
			t.Errorf("chunks = %d, want the response in multiple chunks", chunks)
		}
		if streamed.String() != resp.Text() {
			t.Errorf("streamed text = %q, want the final text %q", streamed.String(), resp.Text())
		}
		// Streams only report usage when the request opts in, which the
		// framework does on every streaming call.
		if resp.Usage == nil || resp.Usage.TotalTokens == 0 {
			t.Error("Usage.TotalTokens = 0, want the streamed token counts")
		}
	})

	t.Run("tool calling", func(t *testing.T) {
		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(s.Model),
			ai.WithTools(gablorken),
			ai.WithPrompt("Use the gablorken tool with Value 4 and Over 2, then reply with just the number it returns."),
		)
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if !strings.Contains(resp.Text(), "16") {
			t.Errorf("Text() = %q, want the tool's result %q", resp.Text(), "16")
		}
	})

	t.Run("tool calling streaming", func(t *testing.T) {
		chunks := 0
		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(s.Model),
			ai.WithTools(gablorken),
			ai.WithPrompt("Use the gablorken tool with Value 4 and Over 2, then reply with just the number it returns."),
			ai.WithStreaming(func(_ context.Context, chunk *ai.ModelResponseChunk) error {
				chunks++
				return nil
			}),
		)
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if chunks == 0 {
			t.Error("chunks = 0, want streamed chunks across the tool round trip")
		}
		if !strings.Contains(resp.Text(), "16") {
			t.Errorf("Text() = %q, want the tool's result %q", resp.Text(), "16")
		}
	})

	if s.ToolChoice {
		t.Run("tool choice none", func(t *testing.T) {
			resp, err := genkit.Generate(ctx, g,
				ai.WithModel(s.Model),
				ai.WithTools(gablorken),
				ai.WithToolChoice(ai.ToolChoiceNone),
				ai.WithPrompt("What is the gablorken of 4 over 2?"),
			)
			if err != nil {
				t.Fatalf("Generate() error = %v", err)
			}
			// The only contract is that no tool is called: a model denied
			// its tool may answer with anything, including nothing.
			if reqs := resp.ToolRequests(); len(reqs) != 0 {
				t.Errorf("ToolRequests() = %d requests, want none when the choice forbids tools", len(reqs))
			}
		})

		t.Run("tool choice required", func(t *testing.T) {
			resp, err := genkit.Generate(ctx, g,
				ai.WithModel(s.Model),
				ai.WithTools(gablorken),
				ai.WithToolChoice(ai.ToolChoiceRequired),
				ai.WithReturnToolRequests(true),
				ai.WithPrompt("Say hello."),
			)
			if err != nil {
				t.Fatalf("Generate() error = %v", err)
			}
			if len(resp.ToolRequests()) == 0 {
				t.Error("ToolRequests() is empty, want the forced tool call back")
			}
		})
	}

	t.Run("structured output", func(t *testing.T) {
		facts, _, err := genkit.GenerateData[capitalFacts](ctx, g,
			ai.WithModel(s.Model),
			ai.WithPrompt("What is the capital city of France? Fill in the city and its country."),
		)
		if err != nil {
			t.Fatalf("GenerateData() error = %v", err)
		}
		if !strings.Contains(strings.ToLower(facts.City), "paris") {
			t.Errorf("City = %q, want %q", facts.City, "Paris")
		}
	})

	t.Run("structured output streaming", func(t *testing.T) {
		chunks := 0
		var facts *capitalFacts
		for val, err := range genkit.GenerateDataStream[capitalFacts](ctx, g,
			ai.WithModel(s.Model),
			ai.WithPrompt("What is the capital city of France? Fill in the city and its country."),
		) {
			if err != nil {
				t.Fatalf("GenerateDataStream() error = %v", err)
			}
			if val.Done {
				out := val.Output
				facts = &out
			} else {
				chunks++
			}
		}
		if chunks == 0 {
			t.Error("chunks = 0, want the structured output streamed")
		}
		if facts == nil {
			t.Fatal("the stream never delivered the final value")
		}
		if !strings.Contains(strings.ToLower(facts.City), "paris") {
			t.Errorf("City = %q, want %q", facts.City, "Paris")
		}
	})

	t.Run("json mode", func(t *testing.T) {
		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(s.Model),
			ai.WithPrompt("Name the capital city of France."),
			ai.WithOutputFormat(ai.OutputFormatJSON),
			ai.WithOutputInstructions("Reply with a JSON object holding a single string field named city."),
		)
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		var out map[string]any
		if err := resp.Output(&out); err != nil {
			t.Fatalf("Output() error = %v on %q", err, resp.Text())
		}
		city, _ := out["city"].(string)
		if !strings.Contains(strings.ToLower(city), "paris") {
			t.Errorf("Output()[city] = %q, want %q", city, "Paris")
		}
	})

	if s.ReasoningModel != nil {
		if !s.StreamOnlyReasoning {
			t.Run("reasoning", func(t *testing.T) {
				resp, err := genkit.Generate(ctx, g,
					ai.WithModel(s.ReasoningModel),
					ai.WithPrompt("Is 91 a prime number? Answer yes or no."),
				)
				if err != nil {
					t.Fatalf("Generate() error = %v", err)
				}
				if resp.Text() == "" {
					t.Error("Text() is empty")
				}
				if s.ReasoningContent && resp.Reasoning() == "" {
					t.Error("Reasoning() is empty, want the thinking content")
				}
			})
		}

		t.Run("reasoning streaming", func(t *testing.T) {
			var reasoning, text strings.Builder
			resp, err := genkit.Generate(ctx, g,
				ai.WithModel(s.ReasoningModel),
				ai.WithPrompt("Is 91 a prime number? Answer yes or no."),
				ai.WithStreaming(func(_ context.Context, chunk *ai.ModelResponseChunk) error {
					reasoning.WriteString(chunk.Reasoning())
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
				t.Errorf("final text = %q, want the streamed %q", resp.Text(), text.String())
			}
			if s.ReasoningContent {
				if reasoning.String() == "" {
					t.Fatal("streamed reasoning is empty, want the thinking content")
				}
				if resp.Reasoning() != reasoning.String() {
					t.Errorf("final reasoning = %q, want the streamed %q", resp.Reasoning(), reasoning.String())
				}
			}
		})
	}

	if s.VisionModel != nil {
		t.Run("vision", func(t *testing.T) {
			resp, err := genkit.Generate(ctx, g,
				ai.WithModel(s.VisionModel),
				ai.WithMessages(ai.NewUserMessage(
					ai.NewMediaPart("image/png", redImageDataURL(t)),
					ai.NewTextPart("What is the dominant color of this image? Reply with one word."),
				)),
			)
			if err != nil {
				t.Fatalf("Generate() error = %v", err)
			}
			if !strings.Contains(strings.ToLower(resp.Text()), "red") {
				t.Errorf("Text() = %q, want it to name the color red", resp.Text())
			}
		})
	}

	if s.ExtraConfig != nil {
		t.Run("extra config passthrough", func(t *testing.T) {
			resp, err := genkit.Generate(ctx, g,
				ai.WithModel(s.Model),
				ai.WithConfig(s.ExtraConfig),
				ai.WithPrompt("What is the capital of France? Reply with just the city name."),
			)
			if err != nil {
				t.Fatalf("Generate() error = %v", err)
			}
			if !strings.Contains(strings.ToLower(resp.Text()), "paris") {
				t.Errorf("Text() = %q, want it to contain %q", resp.Text(), "Paris")
			}
		})
	}
}

// redImageDataURL returns a solid red 100x100 PNG as a data URL, big enough
// that every provider's minimum-size check accepts it.
func redImageDataURL(t *testing.T) string {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 100, 100))
	draw.Draw(img, img.Bounds(), &image.Uniform{C: color.RGBA{R: 255, A: 255}}, image.Point{}, draw.Src)
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatalf("png.Encode() error = %v", err)
	}
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
}
