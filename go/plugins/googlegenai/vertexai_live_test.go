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
	"os"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/internal/livetest"
	"google.golang.org/genai"
)

func TestVertexAILive(t *testing.T) {
	// Authentication is ambient (Application Default Credentials).
	projectID := livetest.Env(t, "GOOGLE_CLOUD_PROJECT")
	location := os.Getenv("GOOGLE_CLOUD_LOCATION")
	if location == "" {
		location = "global"
	}
	g := livetest.Init(t, &googlegenai.VertexAI{ProjectID: projectID, Location: location})

	const flash = "gemini-3.5-flash"
	s := genaiSuite{
		Suite: livetest.Suite{
			Model: googlegenai.VertexAIModelRef(flash, nil),
			ReasoningModel: googlegenai.VertexAIModelRef(flash, &genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{IncludeThoughts: true},
			}),
			// Gemini thinks on every request but returns its thought
			// summaries only on some, so the content is not checked; the
			// thought signatures still have to survive every turn.
			LimitConfig:       &genai.GenerateContentConfig{MaxOutputTokens: 16},
			ToolResponseMedia: true,
		},
		ref:         googlegenai.VertexAIModelRef,
		flash:       flash,
		speechModel: "gemini-2.5-flash-tts",
		imagenModel: "imagen-4.0-fast-generate-001",
		videoModel:  "veo-3.1-lite-generate-001",
		imagenRef:   func(id string) ai.ModelRef { return googlegenai.ImageModelRef("vertexai/"+id, nil) },
		videoRef:    func(id string) ai.ModelRef { return googlegenai.VideoModelRef("vertexai/"+id, nil) },
	}
	// Gemini image output is served only from the global location.
	if location == "global" {
		s.imageModel = "gemini-3.1-flash-image"
	}
	runGenAI(t, g, s)

	livetest.RunEmbedder(t, g, livetest.EmbedderSuite{
		Embedder:   googlegenai.VertexAIEmbedder(g, "gemini-embedding-001"),
		Dimensions: 3072,
		Normalized: true,
	})

	// sayHello generates on a fresh Genkit instance with plugin, which these
	// checks configure differently from the one above.
	sayHello := func(t *testing.T, plugin *googlegenai.VertexAI) {
		t.Helper()
		gAlt := genkit.Init(t.Context(), genkit.WithPlugins(plugin))
		resp, err := genkit.Generate(t.Context(), gAlt,
			ai.WithModel(googlegenai.VertexAIModelRef(flash, nil)),
			ai.WithPrompt("Say hello in one short sentence."))
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if strings.TrimSpace(resp.Text()) == "" {
			t.Error("Text() is empty")
		}
	}

	t.Run("tuned gemini endpoint", func(t *testing.T) {
		endpointID := os.Getenv("GENKIT_VERTEX_TUNED_ENDPOINT")
		if endpointID == "" {
			t.Skip("GENKIT_VERTEX_TUNED_ENDPOINT is not set")
		}
		name := endpointID
		if !strings.HasPrefix(name, "endpoints/") && !strings.HasPrefix(name, "projects/") {
			name = "endpoints/" + name
		}
		// The tuned model is defined on the plugin before generation, so
		// it needs an instance of its own.
		plugin := &googlegenai.VertexAI{ProjectID: projectID, Location: location}
		gTuned := genkit.Init(t.Context(), genkit.WithPlugins(plugin))
		m, err := plugin.DefineModel(gTuned, name, nil)
		if err != nil {
			t.Fatalf("DefineModel(%q) error = %v", name, err)
		}
		resp, err := genkit.Generate(t.Context(), gTuned,
			ai.WithModel(m),
			ai.WithPrompt("Say hello in one short sentence."))
		if err != nil {
			t.Fatalf("Generate() error = %v", err)
		}
		if strings.TrimSpace(resp.Text()) == "" {
			t.Error("Text() is empty")
		}
	})

	// Multi-region ("us"/"eu") endpoints are not enabled on every project,
	// so the check is opt-in.
	t.Run("multi-region location", func(t *testing.T) {
		multiRegion := os.Getenv("GENKIT_VERTEX_MULTIREGION_LOCATION")
		if multiRegion == "" {
			t.Skip("GENKIT_VERTEX_MULTIREGION_LOCATION is not set")
		}
		sayHello(t, &googlegenai.VertexAI{ProjectID: projectID, Location: multiRegion})
	})

	t.Run("plugin-level apiVersion override", func(t *testing.T) {
		sayHello(t, &googlegenai.VertexAI{ProjectID: projectID, Location: location, APIVersion: "v1"})
	})
}
