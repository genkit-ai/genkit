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
	"strings"

	"google.golang.org/genai"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/status"
)

// translateImagenCandidates translates the image generation response to
// [*ai.ModelResponse].
//
// Not every candidate carries an image. A candidate filtered by Responsible AI
// comes back with a reason and no payload - on Vertex as an empty Image rather
// than a nil one, since the converter always populates the field - and a
// request that wrote its output to Cloud Storage (outputGcsUri) gets a URI
// instead of bytes. Filter reasons are always reported; they only turn the
// response blocked when nothing survived.
func translateImagenCandidates(images []*genai.GeneratedImage) *ai.ModelResponse {
	m := &ai.ModelResponse{}
	m.FinishReason = ai.FinishReasonStop

	msg := &ai.Message{}
	msg.Role = ai.RoleModel

	var filtered []string
	for _, img := range images {
		if img == nil {
			continue
		}
		if img.RAIFilteredReason != "" {
			filtered = append(filtered, img.RAIFilteredReason)
		}
		if img.Image == nil {
			continue
		}
		switch {
		case len(img.Image.ImageBytes) > 0:
			msg.Content = append(msg.Content, ai.NewMediaPart(img.Image.MIMEType, "data:"+img.Image.MIMEType+";base64,"+base64.StdEncoding.EncodeToString(img.Image.ImageBytes)))
		case img.Image.GCSURI != "":
			msg.Content = append(msg.Content, ai.NewMediaPart(img.Image.MIMEType, img.Image.GCSURI))
		}
	}

	if len(filtered) > 0 {
		m.FinishMessage = strings.Join(filtered, "; ")
		if len(msg.Content) == 0 {
			m.FinishReason = ai.FinishReasonBlocked
		}
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
	gic *genai.GenerateImagesConfig,
	cb func(context.Context, *ai.ModelResponseChunk) error,
) (*ai.ModelResponse, error) {
	var userPrompt string
	for _, m := range input.Messages {
		if m.Role == ai.RoleUser {
			userPrompt += m.Text()
		}
	}
	if userPrompt == "" {
		return nil, status.Errorf(status.ErrInvalidArgument, "empty prompt detected")
	}

	if cb != nil {
		return nil, status.Errorf(status.ErrUnimplemented, "streaming mode not supported for image generation")
	}

	resp, err := client.Models.GenerateImages(ctx, model, userPrompt, gic)
	if err != nil {
		return nil, wrapAPIError(err)
	}

	r := translateImagenResponse(resp)
	r.Request = input
	return r, nil
}
