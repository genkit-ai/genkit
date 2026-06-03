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
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"

	"github.com/firebase/genkit/go/ai"
)

// embedderFunc dispatches to the Titan, Cohere, or Nova request shape based on
// the model-ID prefix. Bedrock embedders are InvokeModel-based; the SDK's body
// is opaque []byte and the JSON shape is provider-specific.
func embedderFunc(client *bedrockruntime.Client, modelID string) ai.EmbedderFunc {
	return func(ctx context.Context, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
		if req == nil {
			return nil, errors.New("bedrock: embed request required")
		}
		if len(req.Input) == 0 {
			return &ai.EmbedResponse{}, nil
		}
		switch {
		case isNovaMultimodalEmbedder(modelID):
			return embedNova(ctx, client, modelID, req)
		case strings.HasPrefix(modelID, "amazon.titan-embed-"):
			return embedTitan(ctx, client, modelID, req)
		case strings.HasPrefix(modelID, "cohere.embed-"):
			return embedCohere(ctx, client, modelID, req)
		default:
			return nil, fmt.Errorf("bedrock: unrecognised embedder model %q (expected amazon.titan-embed-*, cohere.embed-*, or amazon.nova-*-multimodal-embeddings-*)", modelID)
		}
	}
}

// embedTitan submits one InvokeModel call per input document. Titan text
// embedders accept text; Titan multimodal embedders accept text, image, or
// both, and all return a single vector.
type titanEmbedReq struct {
	InputText       string           `json:"inputText,omitempty"`
	InputImage      string           `json:"inputImage,omitempty"`
	EmbeddingConfig map[string]int32 `json:"embeddingConfig,omitempty"`
}

type titanEmbedResp struct {
	Embedding []float32 `json:"embedding"`
	Message   string    `json:"message,omitempty"`
}

func embedTitan(ctx context.Context, client *bedrockruntime.Client, modelID string, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
	out := &ai.EmbedResponse{Embeddings: make([]*ai.Embedding, 0, len(req.Input))}
	for i, doc := range req.Input {
		body, err := titanEmbedPayload(modelID, doc)
		if err != nil {
			return nil, fmt.Errorf("bedrock: titan embedder: document %d: %w", i, err)
		}
		var resp titanEmbedResp
		if err := invokeJSON(ctx, client, modelID, body, &resp); err != nil {
			return nil, err
		}
		if resp.Message != "" {
			return nil, fmt.Errorf("bedrock: titan embedder: %s", resp.Message)
		}
		out.Embeddings = append(out.Embeddings, &ai.Embedding{Embedding: resp.Embedding})
	}
	return out, nil
}

func titanEmbedPayload(modelID string, doc *ai.Document) (titanEmbedReq, error) {
	text := docText(doc)
	if strings.Contains(modelID, "titan-embed-image") {
		image, err := docImageBase64(doc)
		if err != nil {
			return titanEmbedReq{}, err
		}
		if text == "" && image == "" {
			return titanEmbedReq{}, errors.New("no text or image content")
		}
		return titanEmbedReq{InputText: text, InputImage: image}, nil
	}
	if text == "" {
		return titanEmbedReq{}, errors.New("no text content")
	}
	return titanEmbedReq{InputText: text}, nil
}

func docImageBase64(d *ai.Document) (string, error) {
	p := imagePart(d)
	if p == nil {
		return "", nil
	}
	b, err := decodeMediaPayload(p.Text)
	if err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(b), nil
}

// imagePart returns the first image media part of d, or nil if none.
func imagePart(d *ai.Document) *ai.Part {
	if d == nil {
		return nil
	}
	for _, p := range d.Content {
		if p != nil && p.IsMedia() && strings.HasPrefix(strings.ToLower(p.ContentType), "image/") {
			return p
		}
	}
	return nil
}

// docHasImage reports whether d carries at least one image media part.
func docHasImage(d *ai.Document) bool { return imagePart(d) != nil }

// docImageDataURI returns the first image of d as a "data:<mime>;base64,..."
// URL — the form Cohere's multimodal embedders expect.
func docImageDataURI(d *ai.Document) (string, error) {
	p := imagePart(d)
	if p == nil {
		return "", errors.New("no image content")
	}
	b, err := decodeMediaPayload(p.Text)
	if err != nil {
		return "", err
	}
	return "data:" + p.ContentType + ";base64," + base64.StdEncoding.EncodeToString(b), nil
}

// imageFormatString maps an image MIME type to the bare format token
// ("png"/"jpeg"/"gif"/"webp") used by Nova's source schema. Returns "" for
// unsupported types.
func imageFormatString(contentType string) string {
	switch strings.ToLower(contentType) {
	case "image/png":
		return "png"
	case "image/jpeg", "image/jpg":
		return "jpeg"
	case "image/gif":
		return "gif"
	case "image/webp":
		return "webp"
	}
	return ""
}

// embedCohere submits documents to a Cohere embedder. Text-only requests are
// batched into a single InvokeModel call (Cohere returns parallel embeddings);
// when any document carries an image, the request falls back to one call per
// document because Cohere accepts only a single input type (texts OR images)
// per call.
type cohereEmbedReq struct {
	Texts          []string `json:"texts,omitempty"`
	Images         []string `json:"images,omitempty"`
	InputType      string   `json:"input_type"`
	EmbeddingTypes []string `json:"embedding_types,omitempty"`
}

// cohereEmbeddings tolerates both Cohere response shapes: the legacy
// "embeddings": [[...]] array and the typed "embeddings": {"float": [[...]]}
// object returned when embedding_types is negotiated (common for image input).
type cohereEmbeddings [][]float32

func (c *cohereEmbeddings) UnmarshalJSON(b []byte) error {
	var arr [][]float32
	if err := json.Unmarshal(b, &arr); err == nil {
		*c = arr
		return nil
	}
	var typed struct {
		Float [][]float32 `json:"float"`
		Int8  [][]float32 `json:"int8"`
	}
	if err := json.Unmarshal(b, &typed); err != nil {
		return fmt.Errorf("bedrock: cohere embedder: unexpected embeddings shape: %w", err)
	}
	switch {
	case len(typed.Float) > 0:
		*c = typed.Float
	case len(typed.Int8) > 0:
		*c = typed.Int8
	default:
		*c = nil
	}
	return nil
}

type cohereEmbedResp struct {
	Embeddings cohereEmbeddings `json:"embeddings"`
}

const cohereInputTypeDefault = "search_document"
const cohereInputTypeImage = "image"
const cohereEmbeddingTypeInt8 = "int8"
const cohereEmbeddingTypeFloat = "float"

func embedCohere(ctx context.Context, client *bedrockruntime.Client, modelID string, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
	if cohereHasImage(req) {
		return embedCoherePerDoc(ctx, client, modelID, req)
	}
	texts := make([]string, 0, len(req.Input))
	for i, doc := range req.Input {
		t := docText(doc)
		if t == "" {
			return nil, fmt.Errorf("bedrock: cohere embedder: document %d has no text", i)
		}
		texts = append(texts, t)
	}
	var resp cohereEmbedResp
	if err := invokeJSON(ctx, client, modelID, cohereEmbedReq{Texts: texts, InputType: cohereInputTypeDefault}, &resp); err != nil {
		return nil, err
	}
	if len(resp.Embeddings) != len(texts) {
		return nil, fmt.Errorf("bedrock: cohere embedder: got %d embeddings for %d texts", len(resp.Embeddings), len(texts))
	}
	out := &ai.EmbedResponse{Embeddings: make([]*ai.Embedding, 0, len(resp.Embeddings))}
	for _, e := range resp.Embeddings {
		out.Embeddings = append(out.Embeddings, &ai.Embedding{Embedding: e})
	}
	return out, nil
}

func cohereHasImage(req *ai.EmbedRequest) bool {
	for _, doc := range req.Input {
		if docHasImage(doc) {
			return true
		}
	}
	return false
}

// embedCoherePerDoc handles mixed text/image batches by issuing one
// InvokeModel call per document, preserving input order. Image documents take
// precedence over any accompanying text in the same document.
func embedCoherePerDoc(ctx context.Context, client *bedrockruntime.Client, modelID string, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
	out := &ai.EmbedResponse{Embeddings: make([]*ai.Embedding, 0, len(req.Input))}
	for i, doc := range req.Input {
		body, err := cohereEmbedPayload(doc)
		if err != nil {
			return nil, fmt.Errorf("bedrock: cohere embedder: document %d: %w", i, err)
		}
		var resp cohereEmbedResp
		if err := invokeJSON(ctx, client, modelID, body, &resp); err != nil {
			return nil, err
		}
		if len(resp.Embeddings) != 1 {
			return nil, fmt.Errorf("bedrock: cohere embedder: document %d: expected 1 embedding, got %d", i, len(resp.Embeddings))
		}
		out.Embeddings = append(out.Embeddings, &ai.Embedding{Embedding: resp.Embeddings[0]})
	}
	return out, nil
}

// cohereEmbedPayload builds a single-document Cohere request, choosing the
// image shape when the document carries image media and the text shape
// otherwise.
func cohereEmbedPayload(doc *ai.Document) (cohereEmbedReq, error) {
	if docHasImage(doc) {
		uri, err := docImageDataURI(doc)
		if err != nil {
			return cohereEmbedReq{}, err
		}
		return cohereEmbedReq{
			Images:         []string{uri},
			InputType:      cohereInputTypeImage,
			EmbeddingTypes: []string{cohereEmbeddingTypeInt8, cohereEmbeddingTypeFloat},
		}, nil
	}
	t := docText(doc)
	if t == "" {
		return cohereEmbedReq{}, errors.New("no text or image content")
	}
	return cohereEmbedReq{Texts: []string{t}, InputType: cohereInputTypeDefault}, nil
}

// isNovaMultimodalEmbedder reports whether modelID is an Amazon Nova
// multimodal embedding model (e.g. "amazon.nova-2-multimodal-embeddings-v1:0",
// with or without a cross-region inference-profile prefix).
func isNovaMultimodalEmbedder(modelID string) bool {
	base := stripInferenceProfilePrefix(modelID)
	return strings.HasPrefix(base, "amazon.nova-") && strings.Contains(base, "multimodal-embed")
}

// Nova multimodal embedding request shape (synchronous SINGLE_EMBEDDING).
// Exactly one of Text or Image is populated per call.
type novaEmbedReq struct {
	SchemaVersion         string           `json:"schemaVersion"`
	TaskType              string           `json:"taskType"`
	SingleEmbeddingParams novaSingleParams `json:"singleEmbeddingParams"`
}

type novaSingleParams struct {
	EmbeddingPurpose string     `json:"embeddingPurpose"`
	Text             *novaText  `json:"text,omitempty"`
	Image            *novaImage `json:"image,omitempty"`
}

type novaText struct {
	TruncationMode string `json:"truncationMode"`
	Value          string `json:"value"`
}

type novaImage struct {
	Format string     `json:"format"`
	Source novaSource `json:"source"`
}

type novaSource struct {
	Bytes string `json:"bytes"`
}

type novaEmbedResp struct {
	Embeddings []struct {
		Embedding     []float32 `json:"embedding"`
		EmbeddingType string    `json:"embeddingType"`
	} `json:"embeddings"`
}

const (
	novaEmbedSchemaVersion = "nova-multimodal-embed-v1"
	novaEmbedTaskType      = "SINGLE_EMBEDDING"
	// novaEmbeddingPurposeDefault optimises vectors for indexing into a vector
	// store, the most common embedder use case. Bedrock requires this field.
	novaEmbeddingPurposeDefault = "GENERIC_INDEX"
	novaTruncationModeDefault   = "NONE"
)

// embedNova submits one InvokeModel call per document. Each Nova multimodal
// request carries exactly one modality; image content takes precedence over
// any accompanying text in the same document.
func embedNova(ctx context.Context, client *bedrockruntime.Client, modelID string, req *ai.EmbedRequest) (*ai.EmbedResponse, error) {
	out := &ai.EmbedResponse{Embeddings: make([]*ai.Embedding, 0, len(req.Input))}
	for i, doc := range req.Input {
		body, err := novaEmbedPayload(doc)
		if err != nil {
			return nil, fmt.Errorf("bedrock: nova embedder: document %d: %w", i, err)
		}
		var resp novaEmbedResp
		if err := invokeJSON(ctx, client, modelID, body, &resp); err != nil {
			return nil, err
		}
		if len(resp.Embeddings) != 1 {
			return nil, fmt.Errorf("bedrock: nova embedder: document %d: expected 1 embedding, got %d", i, len(resp.Embeddings))
		}
		out.Embeddings = append(out.Embeddings, &ai.Embedding{Embedding: resp.Embeddings[0].Embedding})
	}
	return out, nil
}

func novaEmbedPayload(doc *ai.Document) (novaEmbedReq, error) {
	params := novaSingleParams{EmbeddingPurpose: novaEmbeddingPurposeDefault}
	if p := imagePart(doc); p != nil {
		format := imageFormatString(p.ContentType)
		if format == "" {
			return novaEmbedReq{}, fmt.Errorf("unsupported image type %q (want png/jpeg/gif/webp)", p.ContentType)
		}
		b, err := decodeMediaPayload(p.Text)
		if err != nil {
			return novaEmbedReq{}, err
		}
		params.Image = &novaImage{
			Format: format,
			Source: novaSource{Bytes: base64.StdEncoding.EncodeToString(b)},
		}
	} else {
		text := docText(doc)
		if text == "" {
			return novaEmbedReq{}, errors.New("no text or image content")
		}
		params.Text = &novaText{TruncationMode: novaTruncationModeDefault, Value: text}
	}
	return novaEmbedReq{
		SchemaVersion:         novaEmbedSchemaVersion,
		TaskType:              novaEmbedTaskType,
		SingleEmbeddingParams: params,
	}, nil
}

// docText concatenates the text parts of a document. Non-text parts (media)
// are skipped — embedders only see text content.
func docText(d *ai.Document) string {
	if d == nil {
		return ""
	}
	var sb strings.Builder
	for _, p := range d.Content {
		if p != nil && p.Kind == ai.PartText && p.Text != "" {
			sb.WriteString(p.Text)
		}
	}
	return sb.String()
}

// invokeJSON marshals in, calls InvokeModel, and decodes the response body
// into out. Used by both embedders and the image-gen helpers.
func invokeJSON(ctx context.Context, client *bedrockruntime.Client, modelID string, in any, out any) error {
	if client == nil {
		return errors.New("bedrock: client is nil")
	}
	body, err := json.Marshal(in)
	if err != nil {
		return fmt.Errorf("bedrock: marshal request: %w", err)
	}
	resp, err := client.InvokeModel(ctx, &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(modelID),
		Body:        body,
		ContentType: aws.String("application/json"),
		Accept:      aws.String("application/json"),
	})
	if err != nil {
		return fmt.Errorf("bedrock: InvokeModel(%s): %w", modelID, err)
	}
	if err := json.Unmarshal(resp.Body, out); err != nil {
		return fmt.Errorf("bedrock: decode response body: %w", err)
	}
	return nil
}
