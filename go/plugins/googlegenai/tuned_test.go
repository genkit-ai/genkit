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
	"net/http"
	"testing"

	"google.golang.org/genai"
)

func TestClassifyModelTunedEndpoint(t *testing.T) {
	cases := []struct {
		name string
		want ModelType
	}{
		{"endpoints/1234567890", ModelTypeGemini},
		{"projects/my-proj/locations/us-central1/endpoints/1234567890", ModelTypeGemini},
		{"gemini-2.5-flash", ModelTypeGemini},
		{"imagen-3.0-generate-001", ModelTypeImagen},
		{"veo-3.0-generate-001", ModelTypeVeo},
		{"text-embedding-004", ModelTypeEmbedder},
		{"random-name-with-no-prefix", ModelTypeUnknown},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ClassifyModel(tc.name); got != tc.want {
				t.Fatalf("ClassifyModel(%q) = %v, want %v", tc.name, got, tc.want)
			}
		})
	}
}

func TestResolveVertexModelName(t *testing.T) {
	ctx := context.Background()

	vertex, err := genai.NewClient(ctx, &genai.ClientConfig{
		Backend:    genai.BackendVertexAI,
		Project:    "test-project",
		Location:   "us-central1",
		HTTPClient: &http.Client{},
	})
	if err != nil {
		t.Fatalf("genai.NewClient (vertex): %v", err)
	}

	geminiAPI, err := genai.NewClient(ctx, &genai.ClientConfig{
		Backend: genai.BackendGeminiAPI,
		APIKey:  "test-key",
	})
	if err != nil {
		t.Fatalf("genai.NewClient (gemini): %v", err)
	}

	cases := []struct {
		name   string
		client *genai.Client
		in     string
		want   string
	}{
		{
			name:   "short form on Vertex expands",
			client: vertex,
			in:     "endpoints/1234567890",
			want:   "projects/test-project/locations/us-central1/endpoints/1234567890",
		},
		{
			name:   "fully qualified path is unchanged",
			client: vertex,
			in:     "projects/my-proj/locations/us-central1/endpoints/999",
			want:   "projects/my-proj/locations/us-central1/endpoints/999",
		},
		{
			name:   "non-tuned name is unchanged",
			client: vertex,
			in:     "gemini-2.5-flash",
			want:   "gemini-2.5-flash",
		},
		{
			name:   "short form on Gemini API backend is unchanged",
			client: geminiAPI,
			in:     "endpoints/1234567890",
			want:   "endpoints/1234567890",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := resolveVertexModelName(tc.client, tc.in); got != tc.want {
				t.Errorf("resolveVertexModelName(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

func TestIsTunedGeminiName(t *testing.T) {
	cases := []struct {
		name string
		want bool
	}{
		{"endpoints/1234567890", true},
		{"projects/p/locations/us-central1/endpoints/999", true},
		{"projects/p/endpoints/999", false},
		{"gemini-2.5-flash", false},
		{"imagen-3.0-generate-001", false},
		{"projects/p/locations/us-central1/publishers/google/models/gemini-2.5-flash", false},
		{"", false},
	}
	for _, tc := range cases {
		if got := isTunedGeminiName(tc.name); got != tc.want {
			t.Errorf("isTunedGeminiName(%q) = %v, want %v", tc.name, got, tc.want)
		}
	}
}
