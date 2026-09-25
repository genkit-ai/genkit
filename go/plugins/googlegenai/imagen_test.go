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
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"google.golang.org/genai"
)

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

// TestTranslateImagenCandidatesPartialResults covers the candidates that carry
// no inline bytes: an entry filtered by Responsible AI has a nil Image, and a
// request that wrote to Cloud Storage gets a URI instead.
func TestTranslateImagenCandidatesPartialResults(t *testing.T) {
	t.Parallel()

	res := translateImagenCandidates([]*genai.GeneratedImage{
		{RAIFilteredReason: "filtered for safety"},
		nil,
		{Image: &genai.Image{MIMEType: "image/png", GCSURI: "gs://bucket/out-0.png"}},
		{Image: &genai.Image{MIMEType: "image/png", ImageBytes: []byte("fake-image-data")}},
	})

	if res.FinishReason != ai.FinishReasonStop {
		t.Errorf("finish reason = %s, want %s", res.FinishReason, ai.FinishReasonStop)
	}
	if len(res.Message.Content) != 2 {
		t.Fatalf("content parts = %d, want 2", len(res.Message.Content))
	}
	if got, want := res.Message.Content[0].Text, "gs://bucket/out-0.png"; got != want {
		t.Errorf("gcs part url = %q, want %q", got, want)
	}
	if !strings.HasPrefix(res.Message.Content[1].Text, "data:image/png;base64,") {
		t.Errorf("inline part = %q, want a data URL", res.Message.Content[1].Text)
	}
}

// TestTranslateImagenCandidatesAllFiltered checks that a response whose every
// candidate was filtered comes back blocked, carrying the reasons, rather than
// as an empty but successful response.
//
// The Vertex converter always populates image, so a filtered candidate arrives
// with an empty Image rather than a nil one; the nil case is what the Gemini
// API path can produce. Both have to count as filtered.
func TestTranslateImagenCandidatesAllFiltered(t *testing.T) {
	t.Parallel()

	res := translateImagenCandidates([]*genai.GeneratedImage{
		{Image: &genai.Image{}, RAIFilteredReason: "reason one"},
		{RAIFilteredReason: "reason two"},
	})

	if res.FinishReason != ai.FinishReasonBlocked {
		t.Errorf("finish reason = %s, want %s", res.FinishReason, ai.FinishReasonBlocked)
	}
	if len(res.Message.Content) != 0 {
		t.Errorf("content parts = %d, want 0", len(res.Message.Content))
	}
	for _, want := range []string{"reason one", "reason two"} {
		if !strings.Contains(res.FinishMessage, want) {
			t.Errorf("finish message = %q, want it to mention %q", res.FinishMessage, want)
		}
	}
}

// TestTranslateImagenCandidatesPartiallyFiltered checks that dropping some of
// the requested images is reported: the surviving images still make it a
// successful response, but the reasons are not thrown away.
func TestTranslateImagenCandidatesPartiallyFiltered(t *testing.T) {
	t.Parallel()

	res := translateImagenCandidates([]*genai.GeneratedImage{
		{Image: &genai.Image{}, RAIFilteredReason: "one was filtered"},
		{Image: &genai.Image{MIMEType: "image/png", ImageBytes: []byte("fake-image-data")}},
	})

	if res.FinishReason != ai.FinishReasonStop {
		t.Errorf("finish reason = %s, want %s", res.FinishReason, ai.FinishReasonStop)
	}
	if len(res.Message.Content) != 1 {
		t.Fatalf("content parts = %d, want 1", len(res.Message.Content))
	}
	if !strings.Contains(res.FinishMessage, "one was filtered") {
		t.Errorf("finish message = %q, want it to report the filtered image", res.FinishMessage)
	}
}
