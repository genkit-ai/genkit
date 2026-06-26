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

package openai

import (
	"context"
	"fmt"

	"github.com/firebase/genkit/go/ai"
	"github.com/openai/openai-go/v3"
	"github.com/openai/openai-go/v3/responses"
)

// generate is the entry point function to request content generation to the OpenAI client
func generate(ctx context.Context, client *openai.Client, model string, input *ai.ModelRequest, cb func(context.Context, *ai.ModelResponseChunk) error,
) (*ai.ModelResponse, error) {
	req, err := toOpenAIResponseParams(model, input)
	if err != nil {
		return nil, err
	}

	// stream mode
	if cb != nil {
		resp, err := generateStream(ctx, client, req, input, cb)
		if err != nil {
			return nil, err
		}
		return resp, nil

	}

	resp, err := generateComplete(ctx, client, req, input)
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// generateStream starts a new streaming response
func generateStream(ctx context.Context, client *openai.Client, req *responses.ResponseNewParams, input *ai.ModelRequest, cb func(context.Context, *ai.ModelResponseChunk) error) (*ai.ModelResponse, error) {
	stream := client.Responses.NewStreaming(ctx, *req)
	defer stream.Close()

	var (
		toolRefMap = make(map[string]string)
		finalResp  *responses.Response
	)

	for stream.Next() {
		evt := stream.Current()
		chunk := &ai.ModelResponseChunk{}

		switch v := evt.AsAny().(type) {
		case responses.ResponseTextDeltaEvent:
			chunk.Content = append(chunk.Content, ai.NewTextPart(v.Delta))

		case responses.ResponseReasoningTextDeltaEvent:
			chunk.Content = append(chunk.Content, ai.NewReasoningPart(v.Delta, nil))

		case responses.ResponseFunctionCallArgumentsDeltaEvent:
			name := toolRefMap[v.ItemID]
			chunk.Content = append(chunk.Content, ai.NewToolRequestPart(&ai.ToolRequest{
				Ref:   v.ItemID,
				Name:  name,
				Input: v.Delta,
			}))

		case responses.ResponseOutputItemAddedEvent:
			switch item := v.Item.AsAny().(type) {
			case responses.ResponseFunctionToolCall:
				toolRefMap[item.CallID] = item.Name
				chunk.Content = append(chunk.Content, ai.NewToolRequestPart(&ai.ToolRequest{
					Ref:  item.CallID,
					Name: item.Name,
				}))
			}

		case responses.ResponseCompletedEvent:
			finalResp = &v.Response
		}

		if len(chunk.Content) > 0 {
			if err := cb(ctx, chunk); err != nil {
				return nil, fmt.Errorf("callback error: %w", err)
			}
		}
	}

	if err := stream.Err(); err != nil {
		return nil, fmt.Errorf("stream error: %w", err)
	}

	if finalResp != nil {
		mResp, err := translateResponse(finalResp)
		if err != nil {
			return nil, err
		}
		mResp.Request = input
		return mResp, nil
	}

	return &ai.ModelResponse{
		Request:       input,
		Message:       &ai.Message{Role: ai.RoleModel},
		FinishReason:  ai.FinishReasonOther,
		FinishMessage: "stream finished without a completion event; usage statistics and final message might be incomplete",
	}, nil
}

// generateComplete starts a new completion
func generateComplete(ctx context.Context, client *openai.Client, req *responses.ResponseNewParams, input *ai.ModelRequest) (*ai.ModelResponse, error) {
	resp, err := client.Responses.New(ctx, *req)
	if err != nil {
		return nil, err
	}

	modelResp, err := translateResponse(resp)
	if err != nil {
		return nil, err
	}
	modelResp.Request = input
	return modelResp, nil
}
