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
	"strings"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/firebase/genkit/go/plugins/internal/uri"
	"google.golang.org/genai"
)

// defineVeoModels defines a new Veo background model for video generation.
// using Google's Veo API through the genai client.
func newVeoModel(
	client *genai.Client,
	name string,
	info ai.ModelOptions,
) ai.BackgroundModel {
	provider := googleAIProvider
	if client.ClientConfig().Backend == genai.BackendVertexAI {
		provider = vertexAIProvider
	}

	startFunc := func(ctx context.Context, req *ai.ModelRequest) (*ai.ModelOperation, error) {
		// Extract text prompt from the request
		prompt := extractTextFromRequest(req)
		if prompt == "" {
			return nil, fmt.Errorf("no text prompt found in request")
		}

		video := extractVeoVideoFromRequest(req)
		image := extractVeoImageFromRequest(req)
		videoConfig, err := toVeoParameters(req)
		if err != nil {
			return nil, err
		}

		// prevent SDK to pick a default number of video generation (usually 2)
		// if users do not provide this setting
		if videoConfig.NumberOfVideos == 0 {
			videoConfig.NumberOfVideos = 1
		}

		sourceConfig := &genai.GenerateVideosSource{
			Prompt: prompt,
			Image:  image,
			Video:  video,
		}

		operation, err := client.Models.GenerateVideosFromSource(
			ctx,
			name,
			sourceConfig,
			videoConfig,
		)
		if err != nil {
			return nil, fmt.Errorf("veo video generation failed: %w", err)
		}

		op := fromVeoOperation(operation)

		if op.Metadata == nil {
			op.Metadata = make(map[string]any)
		}
		op.Metadata["inputRequest"] = req
		op.Metadata["startTime"] = time.Now()

		return op, nil
	}

	checkFunc := func(ctx context.Context, op *ai.ModelOperation) (*ai.ModelOperation, error) {
		veoOp, err := checkVeoOperation(ctx, client, op)
		if err != nil {
			return nil, fmt.Errorf("veo operation status check failed: %w", err)
		}

		updatedOp := fromVeoOperation(veoOp)

		// Restore metadata from the original operation
		if op.Metadata != nil {
			if updatedOp.Metadata == nil {
				updatedOp.Metadata = make(map[string]any)
			}
			for k, v := range op.Metadata {
				updatedOp.Metadata[k] = v
			}
		}

		// Add telemetry metrics when operation completes
		if updatedOp.Done && updatedOp.Output != nil {
			if req, ok := updatedOp.Metadata["inputRequest"].(*ai.ModelRequest); ok {
				ai.CalculateInputOutputUsage(req, updatedOp.Output)
			} else {
				ai.CalculateInputOutputUsage(nil, updatedOp.Output)
			}

			// Calculate latency if startTime is available
			if startTime, ok := updatedOp.Metadata["startTime"].(time.Time); ok {
				latencyMs := float64(time.Since(startTime).Nanoseconds()) / 1e6
				if updatedOp.Output.LatencyMs == 0 {
					updatedOp.Output.LatencyMs = latencyMs
				}
			}
		}

		return updatedOp, nil
	}

	return ai.NewBackgroundModel(api.NewName(provider, name), &ai.BackgroundModelOptions{ModelOptions: info}, startFunc, checkFunc)
}

// extractTextFromRequest extracts the text prompt from a model request.
func extractTextFromRequest(request *ai.ModelRequest) string {
	if len(request.Messages) == 0 {
		return ""
	}
	for _, message := range request.Messages {
		for _, part := range message.Content {
			if part.Text != "" {
				return part.Text
			}
		}
	}
	return ""
}

// extractVeoImageFromRequest extracts image content from a model request for Veo.
func extractVeoImageFromRequest(request *ai.ModelRequest) *genai.Image {
	if len(request.Messages) == 0 {
		return nil
	}

	for _, message := range request.Messages {
		for _, part := range message.Content {
			if part.IsMedia() && !part.IsVideo() {
				_, data, err := uri.Data(part)
				if err != nil {
					return nil
				}
				return &genai.Image{
					ImageBytes: data,
					MIMEType:   part.ContentType,
				}
			}
		}
	}

	return nil
}

// extractVeoVideoFromRequest extracts video content from a model request for Veo.
func extractVeoVideoFromRequest(request *ai.ModelRequest) *genai.Video {
	if len(request.Messages) == 0 {
		return nil
	}

	for _, message := range request.Messages {
		for _, part := range message.Content {
			if !part.IsVideo() {
				continue
			}
			if strings.HasPrefix(part.Text, "data:") {
				contentType, data, err := uri.Data(part)
				if err != nil {
					return nil
				}
				return &genai.Video{
					VideoBytes: data,
					MIMEType:   contentType,
				}
			}
			return &genai.Video{
				URI: part.Text,
			}
		}
	}

	return nil
}

// toVeoParameters converts model request configuration to Veo video generation parameters.
func toVeoParameters(request *ai.ModelRequest) (*genai.GenerateVideosConfig, error) {
	if request.Config == nil {
		return &genai.GenerateVideosConfig{}, nil
	}

	switch config := request.Config.(type) {
	case *genai.GenerateVideosConfig:
		return config, nil
	case genai.GenerateVideosConfig:
		return &config, nil
	case map[string]any:
		var result genai.GenerateVideosConfig
		var err error
		result, err = base.MapToStruct[genai.GenerateVideosConfig](config)
		if err != nil {
			return nil, core.NewPublicError(core.INVALID_ARGUMENT, fmt.Sprintf("The video configuration settings are not in the correct format. Check that the names and values match what the model expects: %v", err), nil)
		}
		return &result, nil
	default:
		return nil, core.NewPublicError(core.INVALID_ARGUMENT, fmt.Sprintf("The configuration type %T is not supported. Use the correct configuration for this model (like VideoModelRef) or a configuration struct.", request.Config), nil)
	}
}

// fromVeoOperation converts a Veo API operation to a Genkit core operation.
func fromVeoOperation(veoOp *genai.GenerateVideosOperation) *ai.ModelOperation {
	operation := &ai.ModelOperation{
		ID:       veoOp.Name,
		Done:     veoOp.Done,
		Metadata: make(map[string]any),
	}

	// Copy any API-provided metadata (e.g. progress percentage, queue status)
	// so developers can use it for debugging or UI updates.
	if veoOp.Metadata != nil {
		for k, v := range veoOp.Metadata {
			operation.Metadata[k] = v
		}
	}

	// Handle error cases
	if veoOp.Error != nil {
		if errorMsg, ok := veoOp.Error["message"].(string); ok {
			operation.Error = fmt.Errorf("%s", errorMsg)
		} else {
			operation.Error = fmt.Errorf("operation error: %v", veoOp.Error)
		}
		return operation
	}

	// Handle in-progress operations
	if !veoOp.Done {
		operation.Output = &ai.ModelResponse{
			Message: &ai.Message{
				Role:    ai.RoleModel,
				Content: []*ai.Part{ai.NewTextPart("Video generation in progress...")},
			},
		}
		return operation
	}

	// Handle completed operations with response
	if veoOp.Done && veoOp.Response != nil && veoOp.Response.GeneratedVideos != nil && len(veoOp.Response.GeneratedVideos) > 0 {
		content := make([]*ai.Part, 0, len(veoOp.Response.GeneratedVideos))
		for _, sample := range veoOp.Response.GeneratedVideos {
			if sample.Video != nil {
				if sample.Video.URI != "" {
					content = append(content, ai.NewMediaPart("video/mp4", sample.Video.URI))
				} else if len(sample.Video.VideoBytes) > 0 {
					importBase64 := "data:video/mp4;base64," + base64.StdEncoding.EncodeToString(sample.Video.VideoBytes)
					content = append(content, ai.NewMediaPart("video/mp4", importBase64))
				}
			}
		}

		if len(content) > 0 {
			operation.Output = &ai.ModelResponse{
				Message: &ai.Message{
					Role:    ai.RoleModel,
					Content: content,
				},
				FinishReason: ai.FinishReasonStop,
				Raw:          veoOp.Response,
			}
			return operation
		}
	}

	// Handle Responsible AI (Safety) filtering
	if veoOp.Done && veoOp.Response != nil && veoOp.Response.RAIMediaFilteredCount > 0 {
		reasons := strings.Join(veoOp.Response.RAIMediaFilteredReasons, ", ")
		operation.Output = &ai.ModelResponse{
			Message: &ai.Message{
				Role:    ai.RoleModel,
				Content: []*ai.Part{ai.NewTextPart(fmt.Sprintf("Video generation blocked by safety filters. Reasons: %s", reasons))},
			},
			FinishReason:  ai.FinishReasonBlocked,
			FinishMessage: fmt.Sprintf("%d videos filtered due to RAI policies", veoOp.Response.RAIMediaFilteredCount),
			Raw:           veoOp.Response,
		}
		return operation
	}

	// Handle completed operations without valid response
	operation.Output = &ai.ModelResponse{
		Message: &ai.Message{
			Role:    ai.RoleModel,
			Content: []*ai.Part{ai.NewTextPart("Video generation completed but no videos were generated")},
		},
		FinishReason: ai.FinishReasonStop,
		Raw:          veoOp.Response,
	}

	return operation
}

// checkVeoOperation checks the status of a long-running Veo video generation operation.
func checkVeoOperation(ctx context.Context, client *genai.Client, ops *core.Operation[*ai.ModelResponse]) (*genai.GenerateVideosOperation, error) {
	genaiOps := &genai.GenerateVideosOperation{
		Name: ops.ID,
	}
	return client.Operations.GetVideosOperation(ctx, genaiOps, nil)
}
