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

package googlegenai

import (
	"context"
	"encoding/base64"
	"errors"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"google.golang.org/genai"
)

func TestClassifyModel_VirtualTryOn(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name string
		want ModelType
	}{
		{"virtual-try-on-001", ModelTypeVirtualTryOn},
		{"virtual-try-on-preview", ModelTypeVirtualTryOn},
		{"imagen-3.0", ModelTypeImagen},
		{"gemini-2.0-flash", ModelTypeGemini},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ClassifyModel(tc.name); got != tc.want {
				t.Errorf("ClassifyModel(%q) = %v, want %v", tc.name, got, tc.want)
			}
		})
	}
}

func TestVertexPredictURL(t *testing.T) {
	t.Parallel()
	cases := []struct {
		name             string
		location, model  string
		wantHost         string
		wantPathContains string
	}{
		{
			name:             "regional",
			location:         "us-central1",
			model:            "virtual-try-on-001",
			wantHost:         "us-central1-aiplatform.googleapis.com",
			wantPathContains: "/locations/us-central1/publishers/google/models/virtual-try-on-001:predict",
		},
		{
			name:             "global",
			location:         "global",
			model:            "virtual-try-on-001",
			wantHost:         "aiplatform.googleapis.com",
			wantPathContains: "/locations/global/publishers/google/models/virtual-try-on-001:predict",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := vertexPredictURL(tc.location, "proj-1", tc.model)
			wantPrefix := "https://" + tc.wantHost + "/"
			if !strings.HasPrefix(got, wantPrefix) {
				t.Errorf("url = %q, want host %q", got, tc.wantHost)
			}
			if !strings.Contains(got, tc.wantPathContains) {
				t.Errorf("url = %q, want path containing %q", got, tc.wantPathContains)
			}
			// Sanity: regional must NOT contain the bare-host form.
			if tc.location != "global" && strings.Contains(got, "//aiplatform.googleapis.com") {
				t.Errorf("regional url leaked bare host: %q", got)
			}
		})
	}
}

func TestVirtualTryOnConfigFromRequest(t *testing.T) {
	t.Parallel()
	t.Run("value", func(t *testing.T) {
		cfg, err := virtualTryOnConfigFromRequest(&ai.ModelRequest{
			Config: VirtualTryOnConfig{SampleCount: 3},
		})
		if err != nil || cfg.SampleCount != 3 {
			t.Fatalf("got cfg=%+v err=%v", cfg, err)
		}
	})
	t.Run("pointer", func(t *testing.T) {
		cfg, err := virtualTryOnConfigFromRequest(&ai.ModelRequest{
			Config: &VirtualTryOnConfig{SampleCount: 2},
		})
		if err != nil || cfg.SampleCount != 2 {
			t.Fatalf("got cfg=%+v err=%v", cfg, err)
		}
	})
	t.Run("nil pointer", func(t *testing.T) {
		var p *VirtualTryOnConfig
		cfg, err := virtualTryOnConfigFromRequest(&ai.ModelRequest{Config: p})
		if err != nil || cfg.SampleCount != 0 {
			t.Fatalf("got cfg=%+v err=%v", cfg, err)
		}
	})
	t.Run("map", func(t *testing.T) {
		cfg, err := virtualTryOnConfigFromRequest(&ai.ModelRequest{
			Config: map[string]any{"sampleCount": 4, "baseSteps": 32},
		})
		if err != nil || cfg.SampleCount != 4 || cfg.BaseSteps != 32 {
			t.Fatalf("got cfg=%+v err=%v", cfg, err)
		}
	})
	t.Run("nil config", func(t *testing.T) {
		cfg, err := virtualTryOnConfigFromRequest(&ai.ModelRequest{Config: nil})
		if err != nil || cfg == nil {
			t.Fatalf("got cfg=%+v err=%v", cfg, err)
		}
	})
	t.Run("invalid type", func(t *testing.T) {
		_, err := virtualTryOnConfigFromRequest(&ai.ModelRequest{Config: 42})
		var ufe *core.UserFacingError
		if !errors.As(err, &ufe) || ufe.Status != core.INVALID_ARGUMENT {
			t.Fatalf("want INVALID_ARGUMENT, got err=%v", err)
		}
	})
}

func TestExtractMediaByType(t *testing.T) {
	t.Parallel()
	imgBytes := []byte{0x89, 0x50, 0x4e, 0x47}
	dataURI := "data:image/png;base64," + base64.StdEncoding.EncodeToString(imgBytes)

	personMeta := map[string]any{"type": PartMetadataTypePersonImage}
	productMeta := map[string]any{"type": PartMetadataTypeProductImage}

	req := &ai.ModelRequest{
		Messages: []*ai.Message{
			{
				Role: ai.RoleUser,
				Content: []*ai.Part{
					{Kind: ai.PartMedia, Text: dataURI, Metadata: personMeta, ContentType: "image/png"},
					{Kind: ai.PartMedia, Text: "gs://bucket/product.png", Metadata: productMeta, ContentType: "image/png"},
					{Kind: ai.PartMedia, Text: "data:image/png;base64,!!!malformed", Metadata: productMeta, ContentType: "image/png"},
					{Kind: ai.PartText, Text: "ignored"},
					{Kind: ai.PartMedia, Text: dataURI, Metadata: map[string]any{"type": "other"}, ContentType: "image/png"},
				},
			},
		},
	}

	persons := extractMediaByType(req, PartMetadataTypePersonImage)
	if len(persons) != 1 {
		t.Fatalf("persons = %d, want 1", len(persons))
	}
	if persons[0].BytesBase64Encoded == "" || persons[0].GCSURI != "" {
		t.Errorf("person not base64: %+v", persons[0])
	}

	products := extractMediaByType(req, PartMetadataTypeProductImage)
	// One valid gs://, one malformed data URI is skipped — expect exactly 1.
	if len(products) != 1 {
		t.Fatalf("products = %d, want 1 (malformed should be skipped)", len(products))
	}
	if products[0].GCSURI != "gs://bucket/product.png" {
		t.Errorf("product gs uri not preserved: %+v", products[0])
	}
}

func TestToVirtualTryOnRequest(t *testing.T) {
	t.Parallel()

	person := &ai.Part{
		Kind:        ai.PartMedia,
		Text:        "gs://bucket/person.png",
		Metadata:    map[string]any{"type": PartMetadataTypePersonImage},
		ContentType: "image/png",
	}
	product := &ai.Part{
		Kind:        ai.PartMedia,
		Text:        "gs://bucket/product.png",
		Metadata:    map[string]any{"type": PartMetadataTypeProductImage},
		ContentType: "image/png",
	}

	cfg := &VirtualTryOnConfig{SampleCount: 1}

	t.Run("no person", func(t *testing.T) {
		req := &ai.ModelRequest{Messages: []*ai.Message{{Role: ai.RoleUser, Content: []*ai.Part{product}}}}
		_, err := toVirtualTryOnRequest(req, cfg)
		var ufe *core.UserFacingError
		if !errors.As(err, &ufe) || ufe.Status != core.INVALID_ARGUMENT {
			t.Fatalf("want INVALID_ARGUMENT, got err=%v", err)
		}
	})

	t.Run("no product", func(t *testing.T) {
		req := &ai.ModelRequest{Messages: []*ai.Message{{Role: ai.RoleUser, Content: []*ai.Part{person}}}}
		_, err := toVirtualTryOnRequest(req, cfg)
		var ufe *core.UserFacingError
		if !errors.As(err, &ufe) || ufe.Status != core.INVALID_ARGUMENT {
			t.Fatalf("want INVALID_ARGUMENT, got err=%v", err)
		}
	})

	t.Run("happy path", func(t *testing.T) {
		req := &ai.ModelRequest{
			Messages: []*ai.Message{{Role: ai.RoleUser, Content: []*ai.Part{person, product}}},
		}
		got, err := toVirtualTryOnRequest(req, cfg)
		if err != nil {
			t.Fatalf("err = %v", err)
		}
		if len(got.Instances) != 1 {
			t.Fatalf("instances = %d, want 1", len(got.Instances))
		}
		inst := got.Instances[0]
		if inst.PersonImage == nil || inst.PersonImage.Image.GCSURI != "gs://bucket/person.png" {
			t.Errorf("person not preserved: %+v", inst.PersonImage)
		}
		if len(inst.ProductImages) != 1 || inst.ProductImages[0].Image.GCSURI != "gs://bucket/product.png" {
			t.Errorf("products wrong: %+v", inst.ProductImages)
		}
		if got.Parameters.SampleCount != 1 {
			t.Errorf("parameters.SampleCount = %d, want 1", got.Parameters.SampleCount)
		}
	})
}

func TestTranslateVirtualTryOnResponse(t *testing.T) {
	t.Parallel()
	resp := &virtualTryOnPredictResponse{
		Predictions: []virtualTryOnPrediction{
			{BytesBase64Encoded: "abc", MimeType: "image/png"},
			{BytesBase64Encoded: "def", MimeType: "image/jpeg"},
		},
	}
	out := translateVirtualTryOnResponse(resp, &ai.ModelRequest{})
	if out.FinishReason != ai.FinishReasonStop {
		t.Errorf("finishReason = %v, want stop", out.FinishReason)
	}
	if len(out.Message.Content) != 2 {
		t.Fatalf("content parts = %d, want 2", len(out.Message.Content))
	}
	if !strings.HasPrefix(out.Message.Content[0].Text, "data:image/png;base64,abc") {
		t.Errorf("first part data url wrong: %q", out.Message.Content[0].Text)
	}
	if !strings.HasPrefix(out.Message.Content[1].Text, "data:image/jpeg;base64,def") {
		t.Errorf("second part data url wrong: %q", out.Message.Content[1].Text)
	}
	if out.Message.Content[0].ContentType != "image/png" {
		t.Errorf("first part ContentType = %q, want image/png", out.Message.Content[0].ContentType)
	}
	if out.Message.Content[1].ContentType != "image/jpeg" {
		t.Errorf("second part ContentType = %q, want image/jpeg", out.Message.Content[1].ContentType)
	}
}

func TestGenerateVirtualTryOn_StreamingRejected(t *testing.T) {
	t.Parallel()
	// cb != nil short-circuits before client is dereferenced, so we can pass
	// nil for the client without panicking.
	_, err := generateVirtualTryOn(context.Background(), nil, "virtual-try-on-001", &ai.ModelRequest{},
		func(context.Context, *ai.ModelResponseChunk) error { return nil })
	var ufe *core.UserFacingError
	if !errors.As(err, &ufe) || ufe.Status != core.INVALID_ARGUMENT {
		t.Fatalf("want INVALID_ARGUMENT, got err=%v", err)
	}
	if !strings.Contains(ufe.Message, "streaming") {
		t.Errorf("message = %q, want one mentioning streaming", ufe.Message)
	}
}

func TestGenerateVirtualTryOn_NonVertexBackend(t *testing.T) {
	t.Parallel()
	ctx := context.Background()
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		Backend: genai.BackendGeminiAPI,
		APIKey:  "fake",
	})
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	_, err = generateVirtualTryOn(ctx, client, "virtual-try-on-001", &ai.ModelRequest{}, nil)
	var ufe *core.UserFacingError
	if !errors.As(err, &ufe) || ufe.Status != core.INVALID_ARGUMENT {
		t.Fatalf("want INVALID_ARGUMENT, got err=%v", err)
	}
	if !strings.Contains(ufe.Message, "Vertex") {
		t.Errorf("message = %q, want one mentioning Vertex", ufe.Message)
	}
}
