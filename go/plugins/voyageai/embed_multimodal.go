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

package voyageai

import (
	"context"
	"errors"
	"fmt"
	"mime"
	"slices"
	"strings"

	"github.com/firebase/genkit/go/ai"
)

// MultimodalEmbedConfig is the per-request configuration of the multimodal
// embedders (voyage-multimodal-3.5 and voyage-multimodal-3). Every field is
// optional.
type MultimodalEmbedConfig struct {
	// InputType is "document" or "query"; leave it empty to embed the input
	// as is.
	InputType string `json:"inputType,omitempty" jsonschema:"enum=document,enum=query"`
	// OutputDimension asks for a vector of that size: 1024 (the default),
	// 256, 512, or 2048 for voyage-multimodal-3.5, and only 1024 for
	// voyage-multimodal-3.
	OutputDimension int `json:"outputDimension,omitempty"`
	// Truncation controls whether an input longer than the model's context
	// is truncated (the API default) or rejected.
	Truncation *bool `json:"truncation,omitempty"`
}

type multimodalRequest struct {
	Inputs          []multimodalInput `json:"inputs"`
	Model           string            `json:"model"`
	InputType       string            `json:"input_type,omitempty"`
	Truncation      *bool             `json:"truncation,omitempty"`
	OutputDimension int               `json:"output_dimension,omitempty"`
}

type multimodalInput struct {
	Content []multimodalContent `json:"content"`
}

// multimodalContent is one text or media item; Type selects which of the
// other fields is set.
type multimodalContent struct {
	Type        string `json:"type"`
	Text        string `json:"text,omitempty"`
	ImageURL    string `json:"image_url,omitempty"`
	ImageBase64 string `json:"image_base64,omitempty"`
	VideoURL    string `json:"video_url,omitempty"`
	VideoBase64 string `json:"video_base64,omitempty"`
}

type multimodalResponse struct {
	Data []embeddingItem `json:"data"`
}

// Media types the multimodal endpoint accepts.
var (
	imageTypes = []string{"image/png", "image/jpeg", "image/webp", "image/gif"}
	videoTypes = []string{"video/mp4"}
)

// embedMultimodal maps each document to one input holding its text and media
// parts in order.
func embedMultimodal(ctx context.Context, c *client, m *model, req *ai.EmbedRequest, cfg MultimodalEmbedConfig) (*ai.EmbedResponse, error) {
	if len(req.Input) == 0 {
		return emptyResponse(), nil
	}
	if err := checkInputCount(len(req.Input)); err != nil {
		return nil, err
	}
	if err := checkInputType(cfg.InputType, false); err != nil {
		return nil, err
	}
	if err := m.validateDimension(cfg.OutputDimension); err != nil {
		return nil, err
	}

	// A request must carry its media uniformly as URLs or uniformly as
	// base64; track which form the first media part used.
	var form mediaForm
	inputs := make([]multimodalInput, len(req.Input))
	for i, doc := range req.Input {
		if doc == nil {
			return nil, invalidArgument("input %d: document is nil", i)
		}
		for j, p := range doc.Content {
			if p == nil {
				return nil, invalidArgument("input %d, part %d: part is nil", i, j)
			}
			var item multimodalContent
			var err error
			switch {
			case p.IsText():
				if p.Text == "" {
					continue
				}
				item = multimodalContent{Type: "text", Text: p.Text}
			case p.IsMedia():
				item, err = mediaContent(m, p, &form)
			default:
				err = errors.New("only text and media parts are supported")
			}
			if err != nil {
				return nil, invalidArgument("input %d, part %d: %v", i, j, err)
			}
			inputs[i].Content = append(inputs[i].Content, item)
		}
		if len(inputs[i].Content) == 0 {
			return nil, invalidArgument("input %d: document has no text or media to embed", i)
		}
	}

	var resp multimodalResponse
	err := c.post(ctx, "/v1/multimodalembeddings", multimodalRequest{
		Inputs:          inputs,
		Model:           m.name,
		InputType:       cfg.InputType,
		Truncation:      cfg.Truncation,
		OutputDimension: cfg.OutputDimension,
	}, &resp)
	if err != nil {
		return nil, err
	}
	vectors, err := orderByIndex(resp.Data, len(inputs))
	if err != nil {
		return nil, err
	}
	out := &ai.EmbedResponse{Embeddings: make([]*ai.Embedding, len(vectors))}
	for i, v := range vectors {
		out.Embeddings[i] = &ai.Embedding{Embedding: v}
	}
	return out, nil
}

// mediaForm records how the media of a request has been given so far.
type mediaForm struct {
	url, base64 bool
}

// mediaContent converts a media part. A data: URL becomes a base64 item and an
// http(s) URL a URL item; the media type must be one the model accepts.
func mediaContent(m *model, p *ai.Part, form *mediaForm) (multimodalContent, error) {
	location := p.Text
	var mediaType string
	var isBase64 bool
	switch {
	case strings.HasPrefix(location, "data:"):
		header, payload, ok := strings.Cut(strings.TrimPrefix(location, "data:"), ",")
		if !ok {
			return multimodalContent{}, fmt.Errorf("malformed data URL: missing comma")
		}
		header, isBase64 = strings.CutSuffix(header, ";base64")
		if !isBase64 {
			return multimodalContent{}, fmt.Errorf("data URL must be base64 encoded")
		}
		mediaType = header
		location = payload
		isBase64 = true
	case strings.HasPrefix(location, "http://") || strings.HasPrefix(location, "https://"):
		mediaType = p.ContentType
		if mediaType == "" {
			return multimodalContent{}, fmt.Errorf("contentType is required for a media URL")
		}
	default:
		return multimodalContent{}, fmt.Errorf("media must be a base64 data: URL or an http(s) URL")
	}

	parsed, _, err := mime.ParseMediaType(mediaType)
	if err != nil {
		return multimodalContent{}, fmt.Errorf("invalid media type %q", mediaType)
	}
	if declared := p.ContentType; declared != "" && isBase64 {
		d, _, err := mime.ParseMediaType(declared)
		if err != nil || d != parsed {
			return multimodalContent{}, fmt.Errorf("contentType %q does not match the data URL's media type %q", declared, parsed)
		}
	}
	isImage := slices.Contains(imageTypes, parsed)
	isVideo := slices.Contains(videoTypes, parsed)
	switch {
	case isVideo && !m.supports("video"):
		return multimodalContent{}, fmt.Errorf("%s does not support video input, use voyage-multimodal-3.5", m.name)
	case !isImage && !isVideo:
		return multimodalContent{}, fmt.Errorf("media type %q is not supported, want one of %v for images or %v for video", parsed, imageTypes, videoTypes)
	}

	if isBase64 {
		form.base64 = true
	} else {
		form.url = true
	}
	if form.url && form.base64 {
		return multimodalContent{}, fmt.Errorf("a request cannot mix media given as URLs with media given as base64 data: use one form for all media")
	}

	switch {
	case isImage && isBase64:
		return multimodalContent{Type: "image_base64", ImageBase64: "data:" + parsed + ";base64," + location}, nil
	case isImage:
		return multimodalContent{Type: "image_url", ImageURL: location}, nil
	case isBase64:
		return multimodalContent{Type: "video_base64", VideoBase64: "data:" + parsed + ";base64," + location}, nil
	default:
		return multimodalContent{Type: "video_url", VideoURL: location}, nil
	}
}
