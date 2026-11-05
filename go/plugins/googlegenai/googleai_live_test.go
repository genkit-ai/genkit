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
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/internal/livetest"
	"google.golang.org/genai"
)

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
