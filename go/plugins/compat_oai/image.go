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

package compat_oai

import (
	"context"
	"fmt"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"
)

// ImageGenerationConfig contains the OpenAI-compatible image generation
// options that may be supplied through a Genkit model request.
type ImageGenerationConfig struct {
	Version           string                                   `json:"version,omitempty"`
	Background        openai.ImageGenerateParamsBackground     `json:"background,omitempty"`
	Moderation        openai.ImageGenerateParamsModeration     `json:"moderation,omitempty"`
	N                 int64                                    `json:"n,omitempty"`
	OutputCompression int64                                    `json:"output_compression,omitempty"`
	OutputFormat      openai.ImageGenerateParamsOutputFormat   `json:"output_format,omitempty"`
	Quality           openai.ImageGenerateParamsQuality        `json:"quality,omitempty"`
	ResponseFormat    openai.ImageGenerateParamsResponseFormat `json:"response_format,omitempty"`
	Size              openai.ImageGenerateParamsSize           `json:"size,omitempty"`
	Style             openai.ImageGenerateParamsStyle          `json:"style,omitempty"`
	User              string                                   `json:"user,omitempty"`
}

func imageGenerateParams(modelName string, input *ai.ModelRequest) (openai.ImageGenerateParams, error) {
	if input == nil {
		return openai.ImageGenerateParams{}, fmt.Errorf("model request cannot be nil")
	}
	if len(input.Messages) == 0 || input.Messages[0] == nil {
		return openai.ImageGenerateParams{}, fmt.Errorf("image generation requires a prompt message")
	}

	var promptBuilder strings.Builder
	for _, part := range input.Messages[0].Content {
		if part != nil && part.IsText() {
			promptBuilder.WriteString(part.Text)
		}
	}
	prompt := promptBuilder.String()
	if prompt == "" {
		return openai.ImageGenerateParams{}, fmt.Errorf("image generation requires text in the first message")
	}

	params := openai.ImageGenerateParams{Model: modelName, Prompt: prompt}
	switch config := input.Config.(type) {
	case ImageGenerationConfig:
		applyImageGenerationConfig(&params, config)
	case *ImageGenerationConfig:
		if config != nil {
			applyImageGenerationConfig(&params, *config)
		}
	case openai.ImageGenerateParams:
		params = config
	case *openai.ImageGenerateParams:
		if config != nil {
			params = *config
		}
	case map[string]any:
		converted, err := base.MapToStruct[ImageGenerationConfig](config)
		if err != nil {
			return openai.ImageGenerateParams{}, fmt.Errorf("failed to convert image generation config: %w", err)
		}
		applyImageGenerationConfig(&params, converted)
	case nil:
		// An empty configuration is valid.
	default:
		return openai.ImageGenerateParams{}, fmt.Errorf("unexpected image generation config type: %T", input.Config)
	}

	params.Prompt = prompt
	if params.Model == "" {
		params.Model = modelName
	}
	// DALL-E defaults to short-lived URLs. Match the other Genkit runtimes by
	// requesting durable inline media unless the caller explicitly opts out.
	if params.ResponseFormat == "" && !strings.Contains(string(params.Model), "gpt-image") {
		params.ResponseFormat = openai.ImageGenerateParamsResponseFormatB64JSON
	}
	// GPT Image always returns base64 and rejects response_format.
	if strings.Contains(string(params.Model), "gpt-image") {
		params.ResponseFormat = ""
		// Style is a DALL-E 3 option and is rejected by GPT Image models.
		params.Style = ""
	}
	return params, nil
}

func applyImageGenerationConfig(params *openai.ImageGenerateParams, config ImageGenerationConfig) {
	if config.Version != "" {
		params.Model = config.Version
	}
	params.Background = config.Background
	params.Moderation = config.Moderation
	params.OutputFormat = config.OutputFormat
	params.Quality = config.Quality
	params.ResponseFormat = config.ResponseFormat
	params.Size = config.Size
	params.Style = config.Style
	if config.N != 0 {
		params.N = param.NewOpt(config.N)
	}
	if config.OutputCompression != 0 {
		params.OutputCompression = param.NewOpt(config.OutputCompression)
	}
	if config.User != "" {
		params.User = param.NewOpt(config.User)
	}
}

func imageResponse(
	result *openai.ImagesResponse,
	input *ai.ModelRequest,
	format openai.ImageGenerateParamsOutputFormat,
) *ai.ModelResponse {
	contentType := imageContentType(format)
	response := &ai.ModelResponse{
		FinishReason: ai.FinishReasonStop,
		Message:      &ai.Message{Role: ai.RoleModel},
		Raw:          result,
		Request:      input,
	}
	for _, image := range result.Data {
		url := image.URL
		if url == "" && image.B64JSON != "" {
			url = "data:" + contentType + ";base64," + image.B64JSON
		}
		if url != "" {
			response.Message.Content = append(response.Message.Content, ai.NewMediaPart(contentType, url))
		}
	}
	return response
}

func imageContentType(format openai.ImageGenerateParamsOutputFormat) string {
	switch format {
	case openai.ImageGenerateParamsOutputFormatJPEG:
		return "image/jpeg"
	case openai.ImageGenerateParamsOutputFormatWebP:
		return "image/webp"
	default:
		return "image/png"
	}
}

func generateImage(
	ctx context.Context,
	client *openai.Client,
	modelName string,
	input *ai.ModelRequest,
	cb func(context.Context, *ai.ModelResponseChunk) error,
) (*ai.ModelResponse, error) {
	if cb != nil {
		return nil, fmt.Errorf("streaming mode not supported for image generation")
	}
	if client == nil {
		return nil, fmt.Errorf("openai client is not initialized")
	}
	params, err := imageGenerateParams(modelName, input)
	if err != nil {
		return nil, err
	}
	result, err := client.Images.Generate(ctx, params)
	if err != nil {
		return nil, err
	}
	if result == nil {
		return nil, fmt.Errorf("received nil response from OpenAI Images API")
	}
	return imageResponse(result, input, params.OutputFormat), nil
}
