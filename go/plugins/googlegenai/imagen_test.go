// Copyright 2026 Google LLC
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
	"testing"

	"github.com/firebase/genkit/go/ai"
	"google.golang.org/genai"
)

func TestImagenConfigFromRequest(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		request     *ai.ModelRequest
		expectError bool
	}{
		{
			name: "valid config struct pointer",
			request: &ai.ModelRequest{
				Config: &genai.GenerateImagesConfig{
					NumberOfImages: 2,
				},
			},
			expectError: false,
		},
		{
			name: "valid config struct value",
			request: &ai.ModelRequest{
				Config: genai.GenerateImagesConfig{
					NumberOfImages: 1,
				},
			},
			expectError: false,
		},
		{
			name: "valid map config",
			request: &ai.ModelRequest{
				Config: map[string]any{
					"numberOfImages": 4,
				},
			},
			expectError: false,
		},
		{
			name: "nil config",
			request: &ai.ModelRequest{
				Config: nil,
			},
			expectError: false,
		},
		{
			name: "invalid config type",
			request: &ai.ModelRequest{
				Config: &genai.GenerateContentConfig{},
			},
			expectError: true,
		},
		{
			name: "invalid map values",
			request: &ai.ModelRequest{
				Config: map[string]any{
					"numberOfImages": "not-a-number",
				},
			},
			expectError: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := imagenConfigFromRequest(tt.request)
			if (err != nil) != tt.expectError {
				t.Errorf("imagenConfigFromRequest() error = %v, expectError %v", err, tt.expectError)
			}
		})
	}
}

func TestTranslateImagenResponse(t *testing.T) {
	t.Parallel()

	resp := &genai.GenerateImagesResponse{
		GeneratedImages: []*genai.GeneratedImage{
			{
				Image: &genai.Image{
					MIMEType:   "image/png",
					ImageBytes: []byte("fake-image-data"),
				},
			},
		},
	}

	res := translateImagenResponse(resp)
	if res.FinishReason != ai.FinishReasonStop {
		t.Errorf("expected finish reason %s, got %s", ai.FinishReasonStop, res.FinishReason)
	}
	if len(res.Message.Content) != 1 {
		t.Fatalf("expected 1 content part, got %d", len(res.Message.Content))
	}
	if res.Message.Content[0].ContentType != "image/png" {
		t.Errorf("expected content type image/png, got %s", res.Message.Content[0].ContentType)
	}
}
