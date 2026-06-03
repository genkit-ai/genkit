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

package bedrock

import (
	"encoding/base64"
	"encoding/json"
	"testing"

	"github.com/firebase/genkit/go/ai"
)

func TestIsNovaMultimodalEmbedder(t *testing.T) {
	for _, tc := range []struct {
		modelID string
		want    bool
	}{
		{"amazon.nova-2-multimodal-embeddings-v1:0", true},
		{"us.amazon.nova-2-multimodal-embeddings-v1:0", true},
		{"amazon.nova-pro-v1:0", false},
		{"amazon.titan-embed-text-v2:0", false},
		{"cohere.embed-english-v3", false},
	} {
		if got := isNovaMultimodalEmbedder(tc.modelID); got != tc.want {
			t.Errorf("isNovaMultimodalEmbedder(%q) = %v, want %v", tc.modelID, got, tc.want)
		}
	}
}

func TestEmbedderAcceptsImage(t *testing.T) {
	for _, tc := range []struct {
		name string
		want bool
	}{
		{"amazon.titan-embed-image-v1", true},
		{"amazon.nova-2-multimodal-embeddings-v1:0", true},
		{"cohere.embed-v4:0", true},
		{"amazon.titan-embed-text-v2:0", false},
		{"cohere.embed-english-v3", false},
	} {
		if got := embedderAcceptsImage(tc.name); got != tc.want {
			t.Errorf("embedderAcceptsImage(%q) = %v, want %v", tc.name, got, tc.want)
		}
	}
}

func TestImageFormatString(t *testing.T) {
	for _, tc := range []struct {
		mime string
		want string
	}{
		{"image/png", "png"},
		{"image/jpeg", "jpeg"},
		{"image/jpg", "jpeg"},
		{"image/gif", "gif"},
		{"image/webp", "webp"},
		{"image/tiff", ""},
		{"application/pdf", ""},
	} {
		if got := imageFormatString(tc.mime); got != tc.want {
			t.Errorf("imageFormatString(%q) = %q, want %q", tc.mime, got, tc.want)
		}
	}
}

func TestNovaEmbedPayload_Text(t *testing.T) {
	body, err := novaEmbedPayload(ai.DocumentFromText("hello", nil))
	if err != nil {
		t.Fatal(err)
	}
	if body.SchemaVersion != novaEmbedSchemaVersion || body.TaskType != novaEmbedTaskType {
		t.Errorf("schema/task = %q/%q", body.SchemaVersion, body.TaskType)
	}
	if body.SingleEmbeddingParams.EmbeddingPurpose != novaEmbeddingPurposeDefault {
		t.Errorf("purpose = %q, want %q", body.SingleEmbeddingParams.EmbeddingPurpose, novaEmbeddingPurposeDefault)
	}
	if body.SingleEmbeddingParams.Text == nil || body.SingleEmbeddingParams.Text.Value != "hello" {
		t.Errorf("Text = %+v, want value=hello", body.SingleEmbeddingParams.Text)
	}
	if body.SingleEmbeddingParams.Image != nil {
		t.Error("Image should be nil for text input")
	}
}

func TestNovaEmbedPayload_ImageTakesPrecedence(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString([]byte("fake png"))
	body, err := novaEmbedPayload(&ai.Document{
		Content: []*ai.Part{
			ai.NewTextPart("caption"),
			ai.NewMediaPart("image/png", "data:image/png;base64,"+encoded),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if body.SingleEmbeddingParams.Text != nil {
		t.Error("Text should be nil when image present (image takes precedence)")
	}
	img := body.SingleEmbeddingParams.Image
	if img == nil {
		t.Fatal("Image is nil")
	}
	if img.Format != "png" {
		t.Errorf("Format = %q, want png", img.Format)
	}
	if img.Source.Bytes != encoded {
		t.Errorf("Source.Bytes = %q, want %q", img.Source.Bytes, encoded)
	}
}

func TestNovaEmbedPayload_Errors(t *testing.T) {
	if _, err := novaEmbedPayload(ai.DocumentFromText("", nil)); err == nil {
		t.Error("expected error for empty document")
	}
	if _, err := novaEmbedPayload(&ai.Document{
		Content: []*ai.Part{ai.NewMediaPart("image/tiff", "data:image/tiff;base64,aGVsbG8=")},
	}); err == nil {
		t.Error("expected error for unsupported image format")
	}
}

func TestCohereEmbedPayload_TextAndImage(t *testing.T) {
	text, err := cohereEmbedPayload(ai.DocumentFromText("hello", nil))
	if err != nil {
		t.Fatal(err)
	}
	if text.InputType != cohereInputTypeDefault || len(text.Texts) != 1 || text.Texts[0] != "hello" {
		t.Errorf("text payload = %+v", text)
	}
	if len(text.Images) != 0 {
		t.Error("text payload should not carry images")
	}

	encoded := base64.StdEncoding.EncodeToString([]byte("fake png"))
	img, err := cohereEmbedPayload(&ai.Document{
		Content: []*ai.Part{
			ai.NewTextPart("caption"),
			ai.NewMediaPart("image/png", "data:image/png;base64,"+encoded),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if img.InputType != cohereInputTypeImage {
		t.Errorf("input_type = %q, want %q", img.InputType, cohereInputTypeImage)
	}
	if len(img.EmbeddingTypes) != 2 || img.EmbeddingTypes[0] != cohereEmbeddingTypeInt8 || img.EmbeddingTypes[1] != cohereEmbeddingTypeFloat {
		t.Errorf("embedding_types = %v, want [%q %q]", img.EmbeddingTypes, cohereEmbeddingTypeInt8, cohereEmbeddingTypeFloat)
	}
	wantURI := "data:image/png;base64," + encoded
	if len(img.Images) != 1 || img.Images[0] != wantURI {
		t.Errorf("Images = %v, want [%q]", img.Images, wantURI)
	}
	if len(img.Texts) != 0 {
		t.Error("image payload should not carry texts")
	}
}

func TestCohereEmbeddings_UnmarshalBothShapes(t *testing.T) {
	// Legacy array shape.
	var legacy cohereEmbedResp
	if err := json.Unmarshal([]byte(`{"embeddings":[[0.1,0.2],[0.3,0.4]]}`), &legacy); err != nil {
		t.Fatal(err)
	}
	if len(legacy.Embeddings) != 2 || legacy.Embeddings[0][1] != 0.2 {
		t.Errorf("legacy parse = %v", legacy.Embeddings)
	}

	// Typed object shape (embedding_types negotiated, common for image input).
	var typed cohereEmbedResp
	if err := json.Unmarshal([]byte(`{"embeddings":{"float":[[1.5,2.5]]}}`), &typed); err != nil {
		t.Fatal(err)
	}
	if len(typed.Embeddings) != 1 || typed.Embeddings[0][0] != 1.5 {
		t.Errorf("typed parse = %v", typed.Embeddings)
	}
}

func TestRerankOptions(t *testing.T) {
	if got := rerankOptions(&RerankOptions{TopN: 3}); got == nil || got.TopN != 3 {
		t.Errorf("pointer opts = %+v", got)
	}
	if got := rerankOptions(RerankOptions{TopN: 5}); got == nil || got.TopN != 5 {
		t.Errorf("value opts = %+v", got)
	}
	if got := rerankOptions(nil); got != nil {
		t.Errorf("nil opts = %+v, want nil", got)
	}
	if got := rerankOptions("nonsense"); got != nil {
		t.Errorf("wrong-type opts = %+v, want nil", got)
	}
}

func TestBuildRerankResponse(t *testing.T) {
	docs := []*ai.Document{
		ai.DocumentFromText("first", nil),
		ai.DocumentFromText("second", nil),
		ai.DocumentFromText("third", nil),
	}
	var resp cohereRerankResp
	if err := json.Unmarshal([]byte(`{"results":[{"index":2,"relevance_score":0.9},{"index":0,"relevance_score":0.4}]}`), &resp); err != nil {
		t.Fatal(err)
	}
	out, err := buildRerankResponse(resp, docs)
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Documents) != 2 {
		t.Fatalf("got %d documents, want 2", len(out.Documents))
	}
	// Order must follow Cohere's ranking: index 2 ("third") first.
	if got := out.Documents[0].Content[0].Text; got != "third" {
		t.Errorf("first ranked = %q, want third", got)
	}
	if got := out.Documents[0].Metadata.Score; got != 0.9 {
		t.Errorf("first score = %v, want 0.9", got)
	}
	if got := out.Documents[1].Content[0].Text; got != "first" {
		t.Errorf("second ranked = %q, want first", got)
	}
}

func TestBuildRerankResponse_IndexOutOfRange(t *testing.T) {
	docs := []*ai.Document{ai.DocumentFromText("only", nil)}
	resp := cohereRerankResp{}
	resp.Results = append(resp.Results, struct {
		Index          int     `json:"index"`
		RelevanceScore float64 `json:"relevance_score"`
	}{Index: 5, RelevanceScore: 0.1})
	if _, err := buildRerankResponse(resp, docs); err == nil {
		t.Error("expected out-of-range index to error")
	}
}
