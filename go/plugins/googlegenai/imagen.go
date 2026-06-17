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
	"fmt"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/internal/base"
	"google.golang.org/genai"
)

// imagenConfigFromRequest translates an [*ai.ModelRequest] configuration to [*genai.GenerateImagesConfig]
func imagenConfigFromRequest(input *ai.ModelRequest) (*genai.GenerateImagesConfig, error) {
	var result genai.GenerateImagesConfig

	switch config := input.Config.(type) {
	case genai.GenerateImagesConfig:
		result = config
	case *genai.GenerateImagesConfig:
		result = *config
	case map[string]any:
		var err error
		result, err = base.MapToStruct[genai.GenerateImagesConfig](config)
		if err != nil {
			return nil, core.NewPublicError(core.INVALID_ARGUMENT, fmt.Sprintf("The image configuration settings are not in the correct format. Check that the names and values match what the model expects: %v", err), nil)
		}
	case nil:
		// empty but valid config
	default:
		return nil, core.NewPublicError(core.INVALID_ARGUMENT, fmt.Sprintf("Invalid configuration type: %T. Expected *genai.GenerateImagesConfig. Ensure you are using the correct ModelRef helper (e.g., ImageModelRef) or passing the correct configuration struct.", input.Config), nil)
	}

	return &result, nil
}

// translateImagenCandidates translates the image generation response to [*ai.ModelResponse]
func translateImagenCandidates(images []*genai.GeneratedImage) *ai.ModelResponse {
	m := &ai.ModelResponse{}
	m.FinishReason = ai.FinishReasonStop

	msg := &ai.Message{}
	msg.Role = ai.RoleModel

	for _, img := range images {
		msg.Content = append(msg.Content, ai.NewMediaPart(img.Image.MIMEType, "data:"+img.Image.MIMEType+";base64,"+base64.StdEncoding.EncodeToString(img.Image.ImageBytes)))
	}

	m.Message = msg
	return m
}

// translateImagenResponse translates [*genai.GenerateImagesResponse] to an [*ai.ModelResponse]
func translateImagenResponse(resp *genai.GenerateImagesResponse) *ai.ModelResponse {
	return translateImagenCandidates(resp.GeneratedImages)
}

// generateImage requests a generate call to the specified imagen model with the
// provided configuration
func generateImage(
	ctx context.Context,
	client *genai.Client,
	model string,
	input *ai.ModelRequest,
	cb func(context.Context, *ai.ModelResponseChunk) error,
) (*ai.ModelResponse, error) {
	gic, err := imagenConfigFromRequest(input)
	if err != nil {
		return nil, err
	}

	var userPrompt string
	for _, m := range input.Messages {
		if m.Role == ai.RoleUser {
			userPrompt += m.Text()
		}
	}
	if userPrompt == "" {
		return nil, fmt.Errorf("error generating images: empty prompt detected")
	}

	if cb != nil {
		return nil, fmt.Errorf("streaming mode not supported for image generation")
	}

	resp, err := client.Models.GenerateImages(ctx, model, userPrompt, gic)
	if err != nil {
		return nil, err
	}

	r := translateImagenResponse(resp)
	r.Request = input
	return r, nil
}
