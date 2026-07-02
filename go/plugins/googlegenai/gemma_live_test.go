// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai_test

import (
	"bytes"
	"context"
	"encoding/base64"
	"image"
	"image/color"
	"image/png"
	"os"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
)

// To run against the live Google AI API:
//
//	GEMINI_API_KEY=... go test ./plugins/googlegenai/ -run TestGemmaLive -v
//
// Skipped automatically when no key is present.
func TestGemmaLive(t *testing.T) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("GOOGLE_API_KEY")
	}
	if apiKey == "" {
		t.Skip("no gemini api key provided, set GEMINI_API_KEY or GOOGLE_API_KEY in environment")
	}

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{APIKey: apiKey}))

	// The two known Gemma models registered for Google AI.
	for _, name := range []string{"gemma-4-31b-it", "gemma-4-26b-a4b-it"} {
		t.Run(name, func(t *testing.T) {
			m := googlegenai.GoogleAIModel(g, name)

			resp, err := genkit.Generate(ctx, g,
				ai.WithModel(m),
				ai.WithPrompt("Reply with exactly the word: pong"),
			)
			if err != nil {
				t.Fatalf("generate: %v", err)
			}
			out := resp.Text()
			t.Logf("%s response: %q", name, out)
			if strings.TrimSpace(out) == "" {
				t.Fatal("empty response text")
			}
			if resp.Usage != nil {
				t.Logf("%s usage: input=%d output=%d total=%d",
					name, resp.Usage.InputTokens, resp.Usage.OutputTokens, resp.Usage.TotalTokens)
			}
		})
	}

	// Multi-turn: a prior assistant turn that carries reasoning content must not
	// break Gemma, because reasoning is stripped before the request is sent.
	t.Run("multi-turn strips prior reasoning", func(t *testing.T) {
		m := googlegenai.GoogleAIModel(g, "gemma-4-31b-it")

		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithMessages(
				ai.NewUserMessage(ai.NewTextPart("My favourite colour is teal. Remember it.")),
				ai.NewModelMessage(
					ai.NewReasoningPart("The user told me their favourite colour is teal.", []byte("sig")),
					ai.NewTextPart("Got it."),
				),
				ai.NewUserMessage(ai.NewTextPart("What is my favourite colour? One word.")),
			),
		)
		if err != nil {
			t.Fatalf("multi-turn generate (reasoning should have been stripped): %v", err)
		}
		t.Logf("multi-turn response: %q", resp.Text())
	})

	// Exercises the tools capability we now advertise (Gemma 4 supports function
	// calling per its model card). A made-up return value the model cannot guess
	// proves the tool was actually invoked through Go's path.
	t.Run("tool calling", func(t *testing.T) {
		m := googlegenai.GoogleAIModel(g, "gemma-4-31b-it")

		secretTool := genkit.DefineTool(g, "gemmaSecretNumber",
			"returns the secret number associated with a given name",
			func(ctx *ai.ToolContext, input struct {
				Name string `json:"name"`
			}) (int, error) {
				return 4242, nil
			},
		)

		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithTools(secretTool),
			ai.WithPrompt("Use the tool to get the secret number for the name 'genkit', then reply with just that number."),
		)
		if err != nil {
			t.Fatalf("tool-calling generate: %v", err)
		}
		out := resp.Text()
		t.Logf("tool-calling response: %q", out)
		if !strings.Contains(out, "4242") {
			t.Errorf("expected the tool's value 4242 in the response, got %q (tool may not have been invoked)", out)
		}
	})

	// Exercises the systemRole capability we now advertise. With systemRole=true
	// the system message flows natively (no simulateSystemPrompt rewrite).
	t.Run("native system prompt", func(t *testing.T) {
		m := googlegenai.GoogleAIModel(g, "gemma-4-31b-it")

		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithSystem("Respond with only the single word BANANA and nothing else, regardless of the question."),
			ai.WithPrompt("Say hello."),
		)
		if err != nil {
			t.Fatalf("system-prompt generate: %v", err)
		}
		out := resp.Text()
		t.Logf("system-prompt response: %q", out)
		if !strings.Contains(strings.ToUpper(out), "BANANA") {
			t.Errorf("system instruction not honoured: response %q does not contain BANANA", out)
		}
	})

	// Exercises the constrained-output capability (Constrained: All). JS advertises
	// constrained 'all' for Gemma; this confirms native structured output works.
	t.Run("constrained json output", func(t *testing.T) {
		m := googlegenai.GoogleAIModel(g, "gemma-4-31b-it")

		type capital struct {
			Country string `json:"country"`
			City    string `json:"city"`
		}

		out, _, err := genkit.GenerateData[capital](ctx, g,
			ai.WithModel(m),
			ai.WithPrompt("What is the capital of Japan? Respond as JSON with fields country and city."),
		)
		if err != nil {
			t.Fatalf("constrained generate: %v", err)
		}
		t.Logf("constrained output: %+v", out)
		if !strings.EqualFold(out.City, "Tokyo") {
			t.Errorf("city = %q, want Tokyo", out.City)
		}
	})

	// Exercises the media-input capability (Media: true). A generated solid-red
	// image should be identified, confirming Gemma accepts image input via Go.
	t.Run("media input", func(t *testing.T) {
		m := googlegenai.GoogleAIModel(g, "gemma-4-31b-it")

		img := image.NewRGBA(image.Rect(0, 0, 16, 16))
		for y := 0; y < 16; y++ {
			for x := 0; x < 16; x++ {
				img.Set(x, y, color.RGBA{R: 255, G: 0, B: 0, A: 255})
			}
		}
		var buf bytes.Buffer
		if err := png.Encode(&buf, img); err != nil {
			t.Fatalf("encode png: %v", err)
		}
		dataURI := "data:image/png;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())

		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithMessages(ai.NewUserMessage(
				ai.NewTextPart("What single colour dominates this image? Reply with just the colour name."),
				ai.NewMediaPart("image/png", dataURI),
			)),
		)
		if err != nil {
			t.Fatalf("media generate: %v", err)
		}
		out := resp.Text()
		t.Logf("media response: %q", out)
		if !strings.Contains(strings.ToLower(out), "red") {
			t.Errorf("expected the model to identify red, got %q", out)
		}
	})
}
