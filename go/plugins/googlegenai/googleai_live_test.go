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
	"flag"
	"strings"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/internal/livetest"
	"google.golang.org/genai"
)

// Deep Research runs are long (minutes) and consume preview-tier quota, so they
// are gated behind an explicit flag in addition to the API key.
var deepResearch = flag.Bool("deepresearch", false, "run live Deep Research tests (slow, consumes preview quota)")

func TestGoogleAILive(t *testing.T) {
	apiKey := livetest.Env(t, "GEMINI_API_KEY", "GOOGLE_API_KEY")
	g := livetest.Init(t, &googlegenai.GoogleAI{APIKey: apiKey})

	const flash = "gemini-3.5-flash"
	runGenAI(t, g, genaiSuite{
		Suite: livetest.Suite{
			Model: googlegenai.GoogleAIModelRef(flash, nil),
			ReasoningModel: googlegenai.GoogleAIModelRef(flash, &genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{IncludeThoughts: true},
			}),
			// Gemini thinks on every request but returns its thought
			// summaries only on some, so the content is not checked; the
			// thought signatures still have to survive every turn.
			LimitConfig:       &genai.GenerateContentConfig{MaxOutputTokens: 16},
			ToolResponseMedia: true,
			BadKeyPlugin:      &googlegenai.GoogleAI{APIKey: "invalid"},
		},
		ref:         googlegenai.GoogleAIModelRef,
		flash:       flash,
		imageModel:  "gemini-3.1-flash-image",
		speechModel: "gemini-2.5-flash-preview-tts",
		videoModel:  "veo-3.1-lite-generate-preview",
		videoRef:    func(id string) ai.ModelRef { return googlegenai.VideoModelRef("googleai/"+id, nil) },
	})

	livetest.RunEmbedder(t, g, livetest.EmbedderSuite{
		Embedder:   googlegenai.GoogleAIEmbedder(g, "gemini-embedding-001"),
		Dimensions: 3072,
		Normalized: true,
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

		// A name and a ttl say different things: which cache an earlier turn
		// built, and how long to keep the one this turn builds. Setting one
		// leaves the other in place.
		m.WithCacheName("dummy-name")
		metadata = m.Metadata
		cache, ok = metadata["cache"].(map[string]any)
		if !ok {
			t.Fatalf("cache should be a map, got: %T", cache)
		}
		ttl, ok := cache["ttlSeconds"].(int)
		if !ok || ttl != 50 {
			t.Fatalf("ttlSeconds should have survived setting the cache name, got: %v", cache["ttlSeconds"])
		}
		name, ok := cache["name"].(string)
		if !ok {
			t.Fatalf("expecting a cache name, got: %v", cache["name"])
		}
		if name != "dummy-name" {
			t.Fatalf("cache name mismatch, want dummy-name, got: %s", name)
		}
	})
}

// TestGoogleAIDeepResearchLive exercises the Deep Research background model
// against the real Gemini API: it starts an interaction, polls it to
// completion and checks the text and usage.
//
// It is gated behind both an API key and the -deepresearch flag because a
// single run takes minutes and consumes preview-tier quota. Run it with:
//
//	GEMINI_API_KEY=... go test ./plugins/googlegenai/ -v -run TestGoogleAIDeepResearchLive -deepresearch -timeout 30m
func TestGoogleAIDeepResearchLive(t *testing.T) {
	apiKey := livetest.Env(t, "GEMINI_API_KEY", "GOOGLE_API_KEY")
	if !*deepResearch {
		t.Skip("skipping live Deep Research test; pass -deepresearch to run (slow, consumes preview quota)")
	}
	g := livetest.Init(t, &googlegenai.GoogleAI{APIKey: apiKey})
	ctx := t.Context()

	op, err := genkit.GenerateOperation(ctx, g,
		ai.WithModelName("googleai/deep-research-preview-04-2026"),
		ai.WithMessages(ai.NewUserTextMessage(
			"Briefly research the history of the Go programming language and summarize it in three bullet points.")),
		ai.WithConfig(&googlegenai.DeepResearchConfig{GoogleSearch: true}),
	)
	if err != nil {
		t.Fatalf("GenerateOperation failed to start: %v", err)
	}
	if op.ID == "" {
		t.Fatal("started operation has no ID")
	}
	t.Logf("started Deep Research operation %s", op.ID)

	// Runs take several minutes; poll until done or the test timeout fires.
	ticker := time.NewTicker(15 * time.Second)
	defer ticker.Stop()
	for !op.Done {
		select {
		case <-ctx.Done():
			t.Fatalf("context cancelled while polling: %v", ctx.Err())
		case <-ticker.C:
			op, err = genkit.CheckModelOperation(ctx, g, op)
			if err != nil {
				t.Fatalf("CheckModelOperation failed: %v", err)
			}
			t.Logf("operation %s done=%t", op.ID, op.Done)
		}
	}

	if op.Error != nil {
		t.Fatalf("operation finished with error: %v", op.Error)
	}
	if op.Output == nil || op.Output.Message == nil {
		t.Fatal("completed operation has no output message")
	}

	var text strings.Builder
	for _, part := range op.Output.Message.Content {
		if part.IsText() {
			text.WriteString(part.Text)
		}
	}
	if strings.TrimSpace(text.String()) == "" {
		t.Errorf("expected non-empty text output, got %#v", op.Output.Message.Content)
	}
	t.Logf("Deep Research output:\n%s", text.String())

	if op.Output.Usage == nil || op.Output.Usage.TotalTokens == 0 {
		t.Errorf("expected non-zero usage, got %#v", op.Output.Usage)
	}
}
