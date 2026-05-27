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
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/firebase/genkit/go/plugins/internal/uri"
	"google.golang.org/genai"
)

// VirtualTryOnConfig mirrors the JS ImagenTryOnConfigSchema used for the
// virtual-try-on-001 model.
type VirtualTryOnConfig struct {
	SampleCount      int                     `json:"sampleCount,omitempty"`
	Seed             *int                    `json:"seed,omitempty"`
	BaseSteps        int                     `json:"baseSteps,omitempty"`
	PersonGeneration string                  `json:"personGeneration,omitempty"`
	SafetySetting    string                  `json:"safetySetting,omitempty"`
	StorageURI       string                  `json:"storageUri,omitempty"`
	AddWatermark     *bool                   `json:"addWatermark,omitempty"`
	EnhancePrompt    *bool                   `json:"enhancePrompt,omitempty"`
	OutputOptions    *VirtualTryOnOutputOpts `json:"outputOptions,omitempty"`
}

// VirtualTryOnOutputOpts controls the output format of generated try-on images.
type VirtualTryOnOutputOpts struct {
	MimeType           string `json:"mimeType,omitempty"`
	CompressionQuality int    `json:"compressionQuality,omitempty"`
}

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

type virtualTryOnImage struct {
	BytesBase64Encoded string `json:"bytesBase64Encoded,omitempty"`
	GCSURI             string `json:"gcsUri,omitempty"`
}

type virtualTryOnPersonImage struct {
	Image virtualTryOnImage `json:"image"`
}

type virtualTryOnProductImage struct {
	Image virtualTryOnImage `json:"image"`
}

type virtualTryOnInstance struct {
	PersonImage   *virtualTryOnPersonImage   `json:"personImage,omitempty"`
	ProductImages []virtualTryOnProductImage `json:"productImages,omitempty"`
}

type virtualTryOnPredictRequest struct {
	Instances  []virtualTryOnInstance `json:"instances"`
	Parameters VirtualTryOnConfig     `json:"parameters,omitempty"`
}

type virtualTryOnPrediction struct {
	BytesBase64Encoded string `json:"bytesBase64Encoded"`
	MimeType           string `json:"mimeType"`
}

type virtualTryOnPredictResponse struct {
	Predictions []virtualTryOnPrediction `json:"predictions"`
}

func virtualTryOnConfigFromRequest(input *ai.ModelRequest) (*VirtualTryOnConfig, error) {
	var result VirtualTryOnConfig
	switch config := input.Config.(type) {
	case VirtualTryOnConfig:
		result = config
	case *VirtualTryOnConfig:
		if config != nil {
			result = *config
		}
	case map[string]any:
		r, err := base.MapToStruct[VirtualTryOnConfig](config)
		if err != nil {
			return nil, core.NewPublicError(core.INVALID_ARGUMENT, fmt.Sprintf("The virtual try-on configuration settings are not in the correct format. Check that the names and values match what the model expects: %v", err), nil)
		}
		result = r
	case nil:
	default:
		return nil, core.NewPublicError(core.INVALID_ARGUMENT, fmt.Sprintf("Invalid virtual try-on configuration type: %T. Expected *googlegenai.VirtualTryOnConfig.", input.Config), nil)
	}
	return &result, nil
}

// extractMediaByType walks the request messages and collects the base64-encoded
// payloads of media and data parts whose metadata["type"] matches the provided
// typ. Uses the shared uri.Data helper (the same one veo.go uses) so the
// parsing logic for data:<mime>[;base64],<data> URIs is consistent across
// plugins. gs:// URIs are passed through to the API as-is.
func extractMediaByType(input *ai.ModelRequest, typ string) []virtualTryOnImage {
	var out []virtualTryOnImage
	for _, msg := range input.Messages {
		for _, p := range msg.Content {
			if !p.IsMedia() && !p.IsData() {
				continue
			}
			metaType, _ := p.Metadata["type"].(string)
			if metaType != typ {
				continue
			}
			if strings.HasPrefix(p.Text, "gs://") {
				out = append(out, virtualTryOnImage{GCSURI: p.Text})
				continue
			}
			_, data, err := uri.Data(p)
			if err != nil {
				continue
			}
			out = append(out, virtualTryOnImage{
				BytesBase64Encoded: base64.StdEncoding.EncodeToString(data),
			})
		}
	}
	return out
}

func toVirtualTryOnRequest(input *ai.ModelRequest, cfg *VirtualTryOnConfig) (*virtualTryOnPredictRequest, error) {
	persons := extractMediaByType(input, PartMetadataTypePersonImage)
	products := extractMediaByType(input, PartMetadataTypeProductImage)
	if len(persons) == 0 {
		return nil, fmt.Errorf("virtual try-on requires a media part with metadata.type=%q", PartMetadataTypePersonImage)
	}
	if len(products) == 0 {
		return nil, fmt.Errorf("virtual try-on requires at least one media part with metadata.type=%q", PartMetadataTypeProductImage)
	}

	instance := virtualTryOnInstance{
		PersonImage: &virtualTryOnPersonImage{Image: persons[0]},
	}
	for _, img := range products {
		instance.ProductImages = append(instance.ProductImages, virtualTryOnProductImage{Image: img})
	}

	return &virtualTryOnPredictRequest{
		Instances:  []virtualTryOnInstance{instance},
		Parameters: *cfg,
	}, nil
}

func translateVirtualTryOnResponse(resp *virtualTryOnPredictResponse, input *ai.ModelRequest) *ai.ModelResponse {
	msg := &ai.Message{Role: ai.RoleModel}
	for _, p := range resp.Predictions {
		url := fmt.Sprintf("data:%s;base64,%s", p.MimeType, p.BytesBase64Encoded)
		msg.Content = append(msg.Content, ai.NewMediaPart(p.MimeType, url))
	}
	return &ai.ModelResponse{
		FinishReason: ai.FinishReasonStop,
		Message:      msg,
		Request:      input,
	}
}

// generateVirtualTryOn issues the POST :predict call for the virtual try-on
// models. The google.golang.org/genai SDK has no matching method, so this
// function drives the authenticated *http.Client that the plugin already
// configured on the genai.Client (credentials, quota project header,
// OpenTelemetry tracing).
func generateVirtualTryOn(
	ctx context.Context,
	client *genai.Client,
	model string,
	input *ai.ModelRequest,
	cb func(context.Context, *ai.ModelResponseChunk) error,
) (*ai.ModelResponse, error) {
	if cb != nil {
		return nil, fmt.Errorf("streaming mode not supported for virtual try-on")
	}

	cc := client.ClientConfig()
	if cc.Backend != genai.BackendVertexAI {
		return nil, fmt.Errorf("virtual try-on is only available through the Vertex AI backend")
	}
	if cc.HTTPClient == nil {
		return nil, fmt.Errorf("virtual try-on: genai.Client has no HTTP client configured")
	}

	cfg, err := virtualTryOnConfigFromRequest(input)
	if err != nil {
		return nil, err
	}

	payload, err := toVirtualTryOnRequest(input, cfg)
	if err != nil {
		return nil, err
	}

	location := cc.Location
	if location == "" {
		return nil, fmt.Errorf("virtual try-on requires a Vertex AI location")
	}
	if cc.Project == "" {
		return nil, fmt.Errorf("virtual try-on requires a Vertex AI project id")
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("virtual try-on: marshaling request: %w", err)
	}

	// Vertex's global endpoint uses the bare aiplatform.googleapis.com host;
	// only regional locations get a <region>- prefix.
	host := location + "-aiplatform.googleapis.com"
	if location == "global" {
		host = "aiplatform.googleapis.com"
	}
	url := fmt.Sprintf(
		"https://%s/v1/projects/%s/locations/%s/publishers/google/models/%s:predict",
		host, cc.Project, location, model,
	)

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := cc.HTTPClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("virtual try-on: request failed: %w", err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("virtual try-on: %s: %s", resp.Status, string(data))
	}

	var vr virtualTryOnPredictResponse
	if err := json.Unmarshal(data, &vr); err != nil {
		return nil, fmt.Errorf("virtual try-on: unmarshaling response: %w", err)
	}
	if len(vr.Predictions) == 0 {
		// Vertex returning zero predictions for a well-formed request almost
		// always means safety filters blocked the output. Surface this as a
		// FinishReasonBlocked response so callers can handle it idiomatically,
		// mirroring the pattern in veo.go and compat_oai/generate.go.
		return &ai.ModelResponse{
			Message:       &ai.Message{Role: ai.RoleModel},
			FinishReason:  ai.FinishReasonBlocked,
			FinishMessage: "virtual try-on: no predictions returned (likely content-filtered)",
			Request:       input,
		}, nil
	}

	return translateVirtualTryOnResponse(&vr, input), nil
}
