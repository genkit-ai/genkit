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
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/internal/livetest"
)

// gemmaLiveModels are the Gemma models registered for Google AI.
var gemmaLiveModels = []string{"gemma-4-31b-it", "gemma-4-26b-a4b-it"}

// To run against the live Google AI API:
//
//	GENKIT_LIVE=1 GEMINI_API_KEY=... go test ./plugins/googlegenai/ -run TestGemmaLive -v
//
// Skipped without GENKIT_LIVE and a key (see livetest.Env).
func TestGemmaLive(t *testing.T) {
	apiKey := livetest.Env(t, "GEMINI_API_KEY", "GOOGLE_API_KEY")
	g := livetest.Init(t, &googlegenai.GoogleAI{APIKey: apiKey})

	gen := func(t *testing.T, opts ...ai.GenerateOption) *ai.ModelResponse {
		t.Helper()
		resp, err := genkit.Generate(t.Context(), g, opts...)
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		return resp
	}
	// perModel runs fn once per Gemma model, as subtests of one case.
	perModel := func(fn func(t *testing.T, m ai.ModelArg)) func(t *testing.T) {
		return func(t *testing.T) {
			for _, name := range gemmaLiveModels {
				t.Run(name, func(t *testing.T) { fn(t, googlegenai.GoogleAIModelRef(name, nil)) })
			}
		}
	}

	secretTool := genkit.DefineTool(g, "gemmaSecretNumber",
		"returns the secret number associated with a given name",
		func(ctx *ai.ToolContext, input struct {
			Name string `json:"name"`
		}) (int, error) {
			return 4242, nil
		},
	)

	// The shared livetest.Run checklist is not used: against Gemma, cases such
	// as array output hang for minutes on the service side (checked live), so
	// the Gemma-specific cases below are run directly. The setup, gating and
	// Init come from livetest.
	cases := []livetest.Case{
		{Name: "plain generation", Run: perModel(func(t *testing.T, m ai.ModelArg) {
			resp := gen(t, ai.WithModel(m), ai.WithPrompt("Reply with exactly the word: pong"))
			if !strings.Contains(strings.ToLower(resp.Text()), "pong") {
				t.Errorf("Text() = %q, want pong", resp.Text())
			}
		})},

		// A prior model turn that carries reasoning (and a signed text part, as
		// Gemini replies do) must not break Gemma, since both are stripped
		// before the request is sent. The answer has to come from the earlier
		// turn, so the history really reached the model.
		{Name: "multi-turn strips prior reasoning", Run: perModel(func(t *testing.T, m ai.ModelArg) {
			history := func() []*ai.Message {
				reply := ai.NewTextPart("Got it, teal.")
				reply.Metadata = map[string]any{"signature": []byte("sig")}
				return []*ai.Message{
					ai.NewUserTextMessage("My favourite colour is teal. Remember it."),
					ai.NewModelMessage(ai.NewReasoningPart("The user told me their favourite colour is teal.", []byte("sig")), reply),
					ai.NewUserTextMessage("What is my favourite colour? One word."),
				}
			}
			resp := gen(t, ai.WithModel(m), ai.WithMessages(history()...))
			if !strings.Contains(strings.ToLower(resp.Text()), "teal") {
				t.Errorf("Text() = %q, want the colour from the first turn (teal)", resp.Text())
			}
			// And on the streaming path.
			var streamed strings.Builder
			resp = gen(t, ai.WithModel(m), ai.WithMessages(history()...),
				ai.WithStreaming(func(_ context.Context, c *ai.ModelResponseChunk) error {
					streamed.WriteString(c.Text())
					return nil
				}))
			if !strings.Contains(strings.ToLower(resp.Text()), "teal") {
				t.Errorf("streamed Text() = %q, want teal", resp.Text())
			}
		})},

		// A tool call followed by its result, then a further turn: the history
		// the model produced for the tool call goes back on the next request.
		{Name: "tool calling", Run: perModel(func(t *testing.T, m ai.ModelArg) {
			resp := gen(t, ai.WithModel(m), ai.WithTools(secretTool),
				ai.WithPrompt("Use the tool to get the secret number for the name 'genkit', then reply with just that number."))
			if !strings.Contains(resp.Text(), "4242") {
				t.Errorf("Text() = %q, want the tool's value 4242", resp.Text())
			}
			resp = gen(t, ai.WithModel(m), ai.WithTools(secretTool),
				ai.WithMessages(resp.History()...),
				ai.WithPrompt("Repeat the number you just found, as a word count of its digits: how many digits does it have? Reply with just the digit count."))
			if !strings.Contains(resp.Text(), "4") {
				t.Errorf("follow-up Text() = %q, want 4", resp.Text())
			}
		})},

		{Name: "native system prompt", Run: perModel(func(t *testing.T, m ai.ModelArg) {
			resp := gen(t, ai.WithModel(m),
				ai.WithSystem("Respond with only the single word BANANA and nothing else, regardless of the question."),
				ai.WithPrompt("Say hello."))
			if !strings.Contains(strings.ToUpper(resp.Text()), "BANANA") {
				t.Errorf("Text() = %q, want BANANA", resp.Text())
			}
		})},

		{Name: "constrained json output", Run: perModel(func(t *testing.T, m ai.ModelArg) {
			type capital struct {
				Country string `json:"country"`
				City    string `json:"city"`
			}
			out, _, err := genkit.GenerateData[capital](t.Context(), g, ai.WithModel(m),
				ai.WithPrompt("What is the capital of Japan? Respond as JSON with fields country and city."))
			if err != nil {
				t.Fatalf("GenerateData() error = %v", err)
			}
			if !strings.EqualFold(out.City, "Tokyo") {
				t.Errorf("city = %q, want Tokyo", out.City)
			}
		})},

		{Name: "media input", Run: perModel(func(t *testing.T, m ai.ModelArg) {
			img := image.NewRGBA(image.Rect(0, 0, 16, 16))
			for y := 0; y < 16; y++ {
				for x := 0; x < 16; x++ {
					img.Set(x, y, color.RGBA{R: 255, A: 255})
				}
			}
			var buf bytes.Buffer
			if err := png.Encode(&buf, img); err != nil {
				t.Fatalf("encode png: %v", err)
			}
			dataURI := "data:image/png;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
			resp := gen(t, ai.WithModel(m), ai.WithMessages(ai.NewUserMessage(
				ai.NewTextPart("What single colour dominates this image? Reply with just the colour name."),
				ai.NewMediaPart("image/png", dataURI),
			)))
			if !strings.Contains(strings.ToLower(resp.Text()), "red") {
				t.Errorf("Text() = %q, want red", resp.Text())
			}
		})},
	}
	for _, c := range cases {
		t.Run(c.Name, c.Run)
	}
}
