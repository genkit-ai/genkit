// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai_test

import (
	"context"
	"os"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
)

// To run this against the real interactions endpoint:
//
//	GEMINI_API_KEY=... go test ./plugins/googlegenai/ -run TestAntigravityLive -v
//
// It is skipped automatically when no key is present.
func TestAntigravityLive(t *testing.T) {
	apiKey := os.Getenv("GEMINI_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("GOOGLE_API_KEY")
	}
	if apiKey == "" {
		t.Skip("no gemini api key provided, set GEMINI_API_KEY or GOOGLE_API_KEY in environment")
	}

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{APIKey: apiKey}))
	m := googlegenai.GoogleAIModel(g, "antigravity-preview-05-2026")

	t.Run("basic generation", func(t *testing.T) {
		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithPrompt("Reply with exactly the word: pong"),
		)
		if err != nil {
			t.Fatalf("generate: %v", err)
		}

		out := resp.Text()
		t.Logf("response text: %q", out)
		if strings.TrimSpace(out) == "" {
			t.Fatal("empty response text")
		}

		// The interactions endpoint returns an interaction id that we surface as
		// message metadata; its presence confirms we hit the interactions path
		// rather than generateContent.
		if resp.Message == nil || resp.Message.Metadata["interactionId"] == nil {
			t.Errorf("expected interactionId in message metadata, got %+v", resp.Message)
		} else {
			t.Logf("interactionId: %v", resp.Message.Metadata["interactionId"])
		}

		if resp.Usage != nil {
			t.Logf("usage: input=%d output=%d total=%d",
				resp.Usage.InputTokens, resp.Usage.OutputTokens, resp.Usage.TotalTokens)
		}
	})

	t.Run("multi-turn via previousInteractionId", func(t *testing.T) {
		first, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithPrompt("My favourite colour is teal. Acknowledge in one word."),
		)
		if err != nil {
			t.Fatalf("first turn: %v", err)
		}
		id, _ := first.Message.Metadata["interactionId"].(string)
		if id == "" {
			t.Skip("no interactionId returned; cannot exercise multi-turn")
		}
		// Reuse the same sandbox by passing the returned environmentId, matching
		// the JS testapp's multi-turn flow.
		envID, _ := first.Message.Metadata["environmentId"].(string)
		t.Logf("first interactionId: %s environmentId: %s", id, envID)

		cfg := &googlegenai.AntigravityConfig{PreviousInteractionID: id}
		if envID != "" {
			cfg.Environment = envID
		}
		second, err := genkit.Generate(ctx, g,
			ai.WithModel(m),
			ai.WithConfig(cfg),
			ai.WithPrompt("What is my favourite colour? Answer in one word."),
		)
		if err != nil {
			t.Fatalf("second turn: %v", err)
		}
		t.Logf("second turn response: %q", second.Text())
	})
}
