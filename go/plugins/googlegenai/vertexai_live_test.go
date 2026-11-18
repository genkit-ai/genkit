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
	"encoding/base64"
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
		videoModel:  "veo-3.1-lite-generate-001",
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

	t.Run("virtual try-on registration", func(t *testing.T) {
		m := googlegenai.VertexAIModel(g, "virtual-try-on-001")
		if m == nil {
			t.Fatal("virtual-try-on-001 model was not registered")
		}
	})

	t.Run("virtual try-on generation", func(t *testing.T) {
		personPart, productPart := vtoPartsFromEnv(t)

		m := googlegenai.VertexAIModel(g, "virtual-try-on-001")
		resp, err := genkit.Generate(t.Context(), g,
			ai.WithModel(m),
			ai.WithMessages(ai.NewUserMessage(personPart, productPart)),
			ai.WithConfig(&genai.RecontextImageConfig{NumberOfImages: genai.Ptr[int32](1)}),
		)
		if err != nil {
			t.Fatal(err)
		}
		assertTryOnImages(t, resp, 1)
	})

	// Two config fields the service does not default to, so a config that
	// never reached the API fails here rather than passing on the defaults.
	// The MIME type is the sturdier of the two: it holds even if the model
	// only ever returns a single image.
	t.Run("virtual try-on honors config", func(t *testing.T) {
		personPart, productPart := vtoPartsFromEnv(t)

		m := googlegenai.VertexAIModel(g, "virtual-try-on-001")
		resp, err := genkit.Generate(t.Context(), g,
			ai.WithModel(m),
			ai.WithMessages(ai.NewUserMessage(personPart, productPart)),
			ai.WithConfig(&genai.RecontextImageConfig{
				NumberOfImages: genai.Ptr[int32](2),
				OutputMIMEType: "image/jpeg",
			}),
		)
		if err != nil {
			t.Fatal(err)
		}
		assertTryOnImages(t, resp, 2)
		for _, part := range resp.Message.Content {
			if part.Kind == ai.PartMedia && part.ContentType != "image/jpeg" {
				t.Errorf("content type = %q, want image/jpeg (outputMimeType was not applied)", part.ContentType)
			}
		}
	})

	// gs:// inputs take a different branch than inline bytes and are only
	// resolvable by the service, so they need a live call to verify. These are
	// the public sample images from the generative-ai docs bucket.
	t.Run("virtual try-on from gcs uris", func(t *testing.T) {
		personPart := ai.NewMediaPart("image/png", "gs://cloud-samples-data/generative-ai/image/person.png")
		personPart.Metadata = map[string]any{"type": googlegenai.PartMetadataTypePersonImage}
		productPart := ai.NewMediaPart("image/jpeg", "gs://cloud-samples-data/generative-ai/image/shirt.jpg")
		productPart.Metadata = map[string]any{"type": googlegenai.PartMetadataTypeProductImage}

		m := googlegenai.VertexAIModel(g, "virtual-try-on-001")
		resp, err := genkit.Generate(t.Context(), g,
			ai.WithModel(m),
			ai.WithMessages(ai.NewUserMessage(personPart, productPart)),
			ai.WithConfig(&genai.RecontextImageConfig{NumberOfImages: genai.Ptr[int32](1)}),
		)
		if err != nil {
			t.Fatal(err)
		}
		assertTryOnImages(t, resp, 1)
	})
}

// vtoPartsFromEnv builds the person and product parts the virtual try-on
// models are addressed with from local files, skipping the test when they are
// not configured.
func vtoPartsFromEnv(t *testing.T) (person, product *ai.Part) {
	t.Helper()

	personPath := os.Getenv("GENKIT_VERTEX_VTO_PERSON_IMAGE")
	productPath := os.Getenv("GENKIT_VERTEX_VTO_PRODUCT_IMAGE")
	if personPath == "" || productPath == "" {
		t.Skip("GENKIT_VERTEX_VTO_PERSON_IMAGE and GENKIT_VERTEX_VTO_PRODUCT_IMAGE must point to local JPEG/PNG files")
	}

	build := func(path, typ string) *ai.Part {
		data, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		mime := "image/jpeg"
		if strings.HasSuffix(strings.ToLower(path), ".png") {
			mime = "image/png"
		}
		p := ai.NewMediaPart(mime, "data:"+mime+";base64,"+base64.StdEncoding.EncodeToString(data))
		p.Metadata = map[string]any{"type": typ}
		return p
	}

	return build(personPath, googlegenai.PartMetadataTypePersonImage),
		build(productPath, googlegenai.PartMetadataTypeProductImage)
}

// assertTryOnImages checks the response carries the expected number of image
// parts and that each one has a payload. A part whose data URL decodes to
// nothing counts as a failure: an empty media part is the shape a dropped
// image takes, and it is indistinguishable from a real one by content type
// alone.
func assertTryOnImages(t *testing.T, resp *ai.ModelResponse, want int) {
	t.Helper()

	if resp.FinishReason != ai.FinishReasonStop {
		t.Errorf("finish reason = %s, want %s (%s)", resp.FinishReason, ai.FinishReasonStop, resp.FinishMessage)
	}

	got := 0
	for _, part := range resp.Message.Content {
		if part.Kind != ai.PartMedia || !strings.HasPrefix(part.ContentType, "image/") {
			continue
		}
		got++

		_, payload, found := strings.Cut(part.Text, ";base64,")
		if !found {
			t.Errorf("media part is not a base64 data URL: %.60q", part.Text)
			continue
		}
		data, err := base64.StdEncoding.DecodeString(payload)
		if err != nil {
			t.Errorf("media part payload is not valid base64: %v", err)
			continue
		}
		if len(data) == 0 {
			t.Error("media part decoded to zero bytes")
		}
	}

	if got != want {
		t.Errorf("image parts = %d, want %d", got, want)
	}
}
