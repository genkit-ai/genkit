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
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/plugins/internal/uri"
	"google.golang.org/genai"
)

// Part metadata keys used to tag images passed to virtual try-on.
//
// Callers should attach `Metadata: map[string]any{"type": "personImage"}` to
// the single media part representing the person, and
// `Metadata: map[string]any{"type": "productImage"}` to each media part
// representing a garment/product image.
const (
	PartMetadataTypePersonImage  = "personImage"
	PartMetadataTypeProductImage = "productImage"
)

// extractMediaByType walks the request messages and collects the media and data
// parts whose metadata["type"] matches typ. Uses the shared uri.Data helper
// (the same one veo.go uses) so the parsing of data:<mime>[;base64],<data>
// URIs is consistent across plugins. gs:// URIs are passed through as-is.
//
// A part tagged with typ that cannot be parsed is an error rather than a skip:
// dropping it would either report the image as missing or quietly send fewer
// product images than the caller asked for.
func extractMediaByType(input *ai.ModelRequest, typ string) ([]*genai.Image, error) {
	var out []*genai.Image
	for _, msg := range input.Messages {
		for _, p := range msg.Content {
			if !p.IsMedia() && !p.IsData() {
				continue
			}
			if metaType, _ := p.Metadata["type"].(string); metaType != typ {
				continue
			}
			switch {
			case strings.HasPrefix(p.Text, "gs://"):
				// uri.Data enforces this for the paths it handles; the fast
				// path here has to do it itself.
				if p.ContentType == "" {
					return nil, status.Errorf(status.ErrInvalidArgument, "virtual try-on: %s part from a gs:// URI must set a content type", typ)
				}
				out = append(out, &genai.Image{GCSURI: p.Text, MIMEType: p.ContentType})
			case strings.HasPrefix(p.Text, "http://"), strings.HasPrefix(p.Text, "https://"):
				// This branch takes no download middleware, and uri.Data would
				// hand back the URL's own bytes as if they were image data.
				return nil, status.Errorf(status.ErrInvalidArgument, "virtual try-on: %s part must be inline data or a gs:// URI; http(s) URLs are not fetched for this model", typ)
			default:
				mimeType, data, err := uri.Data(p)
				if err != nil {
					return nil, status.Errorf(status.ErrInvalidArgument, "virtual try-on: unreadable %s part: %w", typ, err)
				}
				out = append(out, &genai.Image{ImageBytes: data, MIMEType: mimeType})
			}
		}
	}
	return out, nil
}

// toRecontextImageSource builds the SDK's source from the tagged parts of the
// request: one person image and one or more product images.
func toRecontextImageSource(input *ai.ModelRequest) (*genai.RecontextImageSource, error) {
	persons, err := extractMediaByType(input, PartMetadataTypePersonImage)
	if err != nil {
		return nil, err
	}
	products, err := extractMediaByType(input, PartMetadataTypeProductImage)
	if err != nil {
		return nil, err
	}
	if len(persons) == 0 {
		return nil, status.Errorf(status.ErrInvalidArgument, "virtual try-on requires a media part with metadata.type=%q", PartMetadataTypePersonImage)
	}
	if len(products) == 0 {
		return nil, status.Errorf(status.ErrInvalidArgument, "virtual try-on requires at least one media part with metadata.type=%q", PartMetadataTypeProductImage)
	}

	if len(persons) > 1 {
		return nil, status.Errorf(status.ErrInvalidArgument, "virtual try-on accepts a single %s part, got %d", PartMetadataTypePersonImage, len(persons))
	}

	source := &genai.RecontextImageSource{PersonImage: persons[0]}
	for _, img := range products {
		source.ProductImages = append(source.ProductImages, &genai.ProductImage{ProductImage: img})
	}
	return source, nil
}

// generateVirtualTryOn requests a recontextualization call to the specified
// virtual try-on model with the provided configuration. The model dresses the
// person image in the product images; there is no text prompt.
func generateVirtualTryOn(
	ctx context.Context,
	client *genai.Client,
	model string,
	input *ai.ModelRequest,
	config *genai.RecontextImageConfig,
	cb func(context.Context, *ai.ModelResponseChunk) error,
) (*ai.ModelResponse, error) {
	if cb != nil {
		return nil, status.Errorf(status.ErrUnimplemented, "streaming mode not supported for virtual try-on")
	}
	// The SDK refuses this call on the Gemini Developer API, but says so in
	// terms of its own backend names; fail earlier with the plugin's.
	if client.ClientConfig().Backend != genai.BackendVertexAI {
		return nil, status.Errorf(status.ErrUnimplemented, "virtual try-on is only available through the Vertex AI backend")
	}

	source, err := toRecontextImageSource(input)
	if err != nil {
		return nil, err
	}

	resp, err := client.Models.RecontextImage(ctx, model, source, config)
	if err != nil {
		return nil, wrapAPIError(err)
	}

	// Vertex returning no images for a well-formed request almost always means
	// safety filters blocked the output. Surface that as a blocked response so
	// callers can handle it idiomatically, mirroring veo.go.
	if len(resp.GeneratedImages) == 0 {
		return &ai.ModelResponse{
			Message:       &ai.Message{Role: ai.RoleModel},
			FinishReason:  ai.FinishReasonBlocked,
			FinishMessage: "virtual try-on: no images returned (likely content-filtered)",
			Request:       input,
		}, nil
	}

	r := translateImagenCandidates(resp.GeneratedImages)
	r.Request = input
	return r, nil
}
