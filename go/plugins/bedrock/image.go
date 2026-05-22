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
	"errors"
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"

	"github.com/firebase/genkit/go/ai"
)

// imagerFunc dispatches to the Titan, Nova Canvas, or Stable Diffusion
// request shape based on the model-ID prefix. Output images are surfaced as
// "image/png" media parts carrying a base64 data URL.
func imagerFunc(client *bedrockruntime.Client, modelID string) ai.ModelFunc {
	return func(ctx context.Context, req *ai.ModelRequest, _ ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		prompt := promptOf(req)
		if prompt == "" {
			return nil, errors.New("bedrock: image generation: empty prompt")
		}
		var images []string
		var err error
		switch {
		case strings.HasPrefix(modelID, "amazon.titan-image-"), strings.HasPrefix(modelID, "amazon.nova-canvas-"):
			images, err = imageTitanLike(ctx, client, modelID, prompt)
		case strings.HasPrefix(modelID, "stability."), strings.HasPrefix(modelID, "stable-"):
			images, err = imageStableDiffusion(ctx, client, modelID, prompt)
		default:
			return nil, fmt.Errorf("bedrock: unrecognised image model %q", modelID)
		}
		if err != nil {
			return nil, err
		}
		parts := make([]*ai.Part, 0, len(images))
		for _, b64 := range images {
			parts = append(parts, ai.NewMediaPart("image/png", "data:image/png;base64,"+b64))
		}
		return &ai.ModelResponse{
			Message:      &ai.Message{Role: ai.RoleModel, Content: parts},
			FinishReason: ai.FinishReasonStop,
			Request:      req,
		}, nil
	}
}

// promptOf extracts the text prompt from the last user message in req.
func promptOf(req *ai.ModelRequest) string {
	if req == nil {
		return ""
	}
	for i := len(req.Messages) - 1; i >= 0; i-- {
		m := req.Messages[i]
		if m.Role != ai.RoleUser {
			continue
		}
		var sb strings.Builder
		for _, p := range m.Content {
			if p.Text != "" {
				sb.WriteString(p.Text)
			}
		}
		if sb.Len() > 0 {
			return sb.String()
		}
	}
	return ""
}

// Titan-family request shape (also used by Nova Canvas).
type titanImageReq struct {
	TaskType              string               `json:"taskType"`
	TextToImageParams     titanImageTextParams `json:"textToImageParams"`
	ImageGenerationConfig titanImageGenConfig  `json:"imageGenerationConfig"`
}

type titanImageTextParams struct {
	Text string `json:"text"`
}

type titanImageGenConfig struct {
	NumberOfImages int `json:"numberOfImages"`
	Width          int `json:"width"`
	Height         int `json:"height"`
}

type titanImageResp struct {
	Images []string `json:"images"`
	Error  string   `json:"error,omitempty"`
}

func imageTitanLike(ctx context.Context, client *bedrockruntime.Client, modelID, prompt string) ([]string, error) {
	in := titanImageReq{
		TaskType:          "TEXT_IMAGE",
		TextToImageParams: titanImageTextParams{Text: prompt},
		ImageGenerationConfig: titanImageGenConfig{
			NumberOfImages: 1,
			Width:          1024,
			Height:         1024,
		},
	}
	var resp titanImageResp
	if err := invokeJSON(ctx, client, modelID, in, &resp); err != nil {
		return nil, err
	}
	if resp.Error != "" {
		return nil, fmt.Errorf("bedrock: %s: %s", modelID, resp.Error)
	}
	if len(resp.Images) == 0 {
		return nil, fmt.Errorf("bedrock: %s returned no images", modelID)
	}
	return resp.Images, nil
}

// Stable Diffusion family request/response shape.
type sdReq struct {
	TextPrompts []sdTextPrompt `json:"text_prompts"`
	CfgScale    float32        `json:"cfg_scale,omitempty"`
	Steps       int            `json:"steps,omitempty"`
	Width       int            `json:"width,omitempty"`
	Height      int            `json:"height,omitempty"`
}

type sdTextPrompt struct {
	Text string `json:"text"`
}

type sdResp struct {
	Artifacts []sdArtifact `json:"artifacts"`
}

type sdArtifact struct {
	Base64       string `json:"base64"`
	FinishReason string `json:"finishReason"`
}

func imageStableDiffusion(ctx context.Context, client *bedrockruntime.Client, modelID, prompt string) ([]string, error) {
	in := sdReq{
		TextPrompts: []sdTextPrompt{{Text: prompt}},
		CfgScale:    7,
		Steps:       30,
		Width:       1024,
		Height:      1024,
	}
	var resp sdResp
	if err := invokeJSON(ctx, client, modelID, in, &resp); err != nil {
		return nil, err
	}
	if len(resp.Artifacts) == 0 {
		return nil, fmt.Errorf("bedrock: %s returned no artifacts", modelID)
	}
	images := make([]string, 0, len(resp.Artifacts))
	for _, a := range resp.Artifacts {
		if a.Base64 != "" {
			images = append(images, a.Base64)
		}
	}
	if len(images) == 0 {
		return nil, fmt.Errorf("bedrock: %s returned empty artifacts", modelID)
	}
	return images, nil
}
