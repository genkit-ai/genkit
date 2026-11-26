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

// antigravityLiveModel is the Antigravity model the live test exercises.
const antigravityLiveModel = "antigravity-preview-09-2026"

// To run against the live Google AI interactions endpoint:
//
//	GENKIT_LIVE=1 GEMINI_API_KEY=... go test ./plugins/googlegenai/ -run TestAntigravityLive -v
//
// Skipped without GENKIT_LIVE and a key (see livetest.Env).
//
// The shared livetest.Run checklist is not used: Antigravity is an agent that
// runs a sandbox (every call carries thousands of prompt tokens and takes
// seconds to minutes) and has no constrained output, so the checks below cover
// what is specific to it. Setup, gating and Init come from livetest.
func TestAntigravityLive(t *testing.T) {
	apiKey := livetest.Env(t, "GEMINI_API_KEY", "GOOGLE_API_KEY")
	g := livetest.Init(t, &googlegenai.GoogleAI{APIKey: apiKey})
	m := googlegenai.GoogleAIModelRef(antigravityLiveModel, nil)

	gen := func(t *testing.T, opts ...ai.GenerateOption) *ai.ModelResponse {
		t.Helper()
		resp, err := genkit.Generate(t.Context(), g, append([]ai.GenerateOption{ai.WithModel(m)}, opts...)...)
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		return resp
	}

	t.Run("basic generation, metadata and usage", func(t *testing.T) {
		resp := gen(t, ai.WithPrompt("Reply with exactly the word: pong"))
		if !strings.Contains(strings.ToLower(resp.Text()), "pong") {
			t.Errorf("Text() = %q, want pong", resp.Text())
		}
		if resp.FinishReason != ai.FinishReasonStop {
			t.Errorf("FinishReason = %q, want stop", resp.FinishReason)
		}
		// The interaction and environment IDs are what multi-turn continues
		// from; their presence also confirms the interactions path was taken.
		for _, k := range []string{"interactionId", "environmentId"} {
			if v, _ := resp.Message.Metadata[k].(string); v == "" {
				t.Errorf("message metadata %q missing: %v", k, resp.Message.Metadata)
			}
		}
		u := resp.Usage
		if u == nil || u.InputTokens <= 0 || u.OutputTokens <= 0 || u.TotalTokens < u.InputTokens+u.OutputTokens {
			t.Fatalf("usage = %+v, want positive token counts with total >= input+output", u)
		}
		if u.Custom["inputTextTokens"] <= 0 {
			t.Errorf("usage.Custom = %v, want inputTextTokens", u.Custom)
		}
		// The framework counts these from the messages, not from tokens.
		if u.InputCharacters != len("Reply with exactly the word: pong") {
			t.Errorf("InputCharacters = %d, want the prompt's length", u.InputCharacters)
		}
	})

	t.Run("system prompt", func(t *testing.T) {
		resp := gen(t,
			ai.WithSystem("Respond with only the single word BANANA and nothing else, regardless of the question."),
			ai.WithPrompt("Say hello."))
		if !strings.Contains(strings.ToUpper(resp.Text()), "BANANA") {
			t.Errorf("Text() = %q, want BANANA", resp.Text())
		}
	})

	t.Run("multi-turn continues the interaction", func(t *testing.T) {
		first := gen(t, ai.WithPrompt("The codeword is PINEAPPLE. Remember it and reply with just: OK"))
		id, _ := first.Message.Metadata["interactionId"].(string)
		if id == "" {
			t.Fatalf("first turn returned no interactionId: %v", first.Message.Metadata)
		}
		second := gen(t,
			ai.WithMessages(first.History()...),
			ai.WithPrompt("What was the codeword? Reply with just the word."))
		if !strings.Contains(strings.ToUpper(second.Text()), "PINEAPPLE") {
			t.Errorf("second answer = %q, want it to recall PINEAPPLE from the first turn", second.Text())
		}
		// The continuation runs in the first turn's sandbox.
		if first.Message.Metadata["environmentId"] != second.Message.Metadata["environmentId"] {
			t.Errorf("environment changed across turns: %v -> %v", first.Message.Metadata["environmentId"], second.Message.Metadata["environmentId"])
		}
	})

	t.Run("config passthrough and built-in tool", func(t *testing.T) {
		resp := gen(t,
			ai.WithConfig(map[string]any{
				"tools": []any{map[string]any{"type": "code_execution"}},
				"model": "gemini-3.5-flash-lite",
			}),
			ai.WithPrompt("Use code to compute 123456 * 654321 and reply with just the number."))
		if !strings.Contains(resp.Text(), "80779853376") {
			t.Errorf("Text() = %q, want 80779853376", resp.Text())
		}
	})

	t.Run("function tool loop", func(t *testing.T) {
		secret := genkit.DefineTool(g, "antigravitySecretNumber",
			"returns the secret number associated with a given name",
			func(ctx *ai.ToolContext, input struct {
				Name string `json:"name"`
			}) (int, error) {
				return 4242, nil
			})
		resp := gen(t, ai.WithTools(secret),
			ai.WithPrompt("Use the antigravitySecretNumber tool for the name 'genkit', then reply with just that number."))
		if !strings.Contains(resp.Text(), "4242") {
			t.Errorf("Text() = %q, want the tool's value 4242", resp.Text())
		}
	})

	t.Run("media input", func(t *testing.T) {
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
		resp := gen(t, ai.WithMessages(ai.NewUserMessage(
			ai.NewTextPart("What single colour dominates this image? Reply with just the colour name."),
			ai.NewMediaPart("image/png", dataURI),
		)))
		if !strings.Contains(strings.ToLower(resp.Text()), "red") {
			t.Errorf("Text() = %q, want red", resp.Text())
		}
	})

	t.Run("streaming emits the response", func(t *testing.T) {
		var chunks int
		var streamed strings.Builder
		resp := gen(t, ai.WithPrompt("Reply with exactly the word: pong"),
			ai.WithStreaming(func(_ context.Context, c *ai.ModelResponseChunk) error {
				chunks++
				streamed.WriteString(c.Text())
				return nil
			}))
		// The endpoint is called without server-sent streaming, so the whole
		// response arrives as one chunk.
		if chunks != 1 || !strings.Contains(strings.ToLower(streamed.String()), "pong") {
			t.Errorf("chunks = %d, streamed = %q", chunks, streamed.String())
		}
		if !strings.Contains(strings.ToLower(resp.Text()), "pong") {
			t.Errorf("Text() = %q", resp.Text())
		}
	})
}
