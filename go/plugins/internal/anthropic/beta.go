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

package anthropic

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/status"
)

// defaultBetas matches the JS BetaRunner defaults that are generally useful
// across the beta Messages surface. Values without SDK constants are still
// valid anthropic-beta header strings.
var defaultBetas = []anthropic.AnthropicBeta{
	anthropic.AnthropicBetaFilesAPI2025_04_14,
	"effort-2025-11-24",
	"structured-outputs-2025-11-13",
	"task-budgets-2026-03-13",
}

// toBetaRequest converts a stable MessageNewParams body into BetaMessageNewParams
// and attaches beta feature headers. A nil betas slice means "use defaults"; a
// non-nil empty slice means the caller explicitly requested no beta headers.
func toBetaRequest(req *anthropic.MessageNewParams, betas []anthropic.AnthropicBeta) (*anthropic.BetaMessageNewParams, error) {
	data, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("unable to marshal anthropic request for beta: %w", err)
	}
	var betaReq anthropic.BetaMessageNewParams
	if err := json.Unmarshal(data, &betaReq); err != nil {
		return nil, fmt.Errorf("unable to convert anthropic request to beta: %w", err)
	}
	if betas == nil {
		betas = append([]anthropic.AnthropicBeta(nil), defaultBetas...)
	}
	betaReq.Betas = betas
	return &betaReq, nil
}

func generateBeta(
	ctx context.Context,
	client anthropic.Client,
	req *anthropic.MessageNewParams,
	betas []anthropic.AnthropicBeta,
	input *ai.ModelRequest,
	cb func(context.Context, *ai.ModelResponseChunk) error,
) (*ai.ModelResponse, error) {
	betaReq, err := toBetaRequest(req, betas)
	if err != nil {
		return nil, err
	}

	if cb == nil {
		msg, err := client.Beta.Messages.New(ctx, *betaReq)
		if err != nil {
			return nil, err
		}
		r, err := toBetaGenkitResponse(msg)
		if err != nil {
			return nil, err
		}
		r.Request = input
		return r, nil
	}

	stream := client.Beta.Messages.NewStreaming(ctx, *betaReq)
	message := anthropic.BetaMessage{}
	for stream.Next() {
		event := stream.Current()
		if err := message.Accumulate(event); err != nil {
			return nil, err
		}

		switch event := event.AsAny().(type) {
		case anthropic.BetaRawContentBlockDeltaEvent:
			content := []*ai.Part{}
			if event.Delta.Type == "thinking_delta" {
				content = append(content, ai.NewReasoningPart(event.Delta.Thinking, []byte(event.Delta.Signature)))
			} else if event.Delta.Type == "text_delta" || event.Delta.Text != "" {
				content = append(content, ai.NewTextPart(event.Delta.Text))
			}
			if len(content) > 0 {
				if err := cb(ctx, &ai.ModelResponseChunk{Content: content}); err != nil {
					return nil, err
				}
			}
		case anthropic.BetaRawContentBlockStopEvent:
			if event.Index >= 0 && int(event.Index) < len(message.Content) {
				p, err := betaContentBlockToPart(message.Content[event.Index])
				if err != nil {
					return nil, err
				}
				if shouldEmitOnContentBlockStop(p) {
					if err := cb(ctx, &ai.ModelResponseChunk{Content: []*ai.Part{p}}); err != nil {
						return nil, err
					}
				}
			}
		case anthropic.BetaRawMessageStopEvent:
			r, err := toBetaGenkitResponse(&message)
			if err != nil {
				return nil, err
			}
			r.Request = input
			return r, nil
		}
	}
	if stream.Err() != nil {
		return nil, stream.Err()
	}
	return nil, nil
}

func toBetaGenkitResponse(m *anthropic.BetaMessage) (*ai.ModelResponse, error) {
	r := ai.ModelResponse{}

	switch m.StopReason {
	case anthropic.BetaStopReasonMaxTokens, anthropic.BetaStopReasonModelContextWindowExceeded:
		r.FinishReason = ai.FinishReasonLength
	case anthropic.BetaStopReasonStopSequence,
		anthropic.BetaStopReasonEndTurn,
		anthropic.BetaStopReasonToolUse,
		anthropic.BetaStopReasonPauseTurn:
		r.FinishReason = ai.FinishReasonStop
	case anthropic.BetaStopReasonRefusal:
		r.FinishReason = ai.FinishReasonOther
	case "":
		r.FinishReason = ai.FinishReasonUnknown
	default:
		r.FinishReason = ai.FinishReasonOther
	}

	msg := &ai.Message{Role: ai.RoleModel}
	for _, part := range m.Content {
		p, err := betaContentBlockToPart(part)
		if err != nil {
			return nil, err
		}
		if p != nil {
			msg.Content = append(msg.Content, p)
		}
	}

	r.Message = msg
	r.Raw = m.JSON
	r.Usage = &ai.GenerationUsage{
		InputTokens:         int(m.Usage.InputTokens),
		OutputTokens:        int(m.Usage.OutputTokens),
		CachedContentTokens: int(m.Usage.CacheReadInputTokens),
	}
	return &r, nil
}

func betaContentBlockToPart(part anthropic.BetaContentBlockUnion) (*ai.Part, error) {
	switch b := part.AsAny().(type) {
	case anthropic.BetaThinkingBlock:
		return ai.NewReasoningPart(b.Thinking, []byte(b.Signature)), nil
	case anthropic.BetaTextBlock:
		return ai.NewTextPart(string(b.Text)), nil
	case anthropic.BetaToolUseBlock:
		// Prefer the union's raw input (json.RawMessage) so streaming
		// accumulation stays intact; fall back to the concrete block value.
		input := betaToolInput(part.Input, b.Input)
		name := b.Name
		if name == "" {
			name = "unknown_tool"
		}
		return ai.NewToolRequestPart(&ai.ToolRequest{
			Ref:   b.ID,
			Input: input,
			Name:  name,
		}), nil
	case anthropic.BetaServerToolUseBlock:
		name := string(b.Name)
		if name == "" {
			name = "unknown_tool"
		}
		if part.ServerName != "" {
			name = part.ServerName + "/" + name
		}
		return serverToolUseToPart(b.ID, name, b.Input), nil
	case anthropic.BetaWebSearchToolResultBlock:
		return webSearchToolResultToPart(b.ToolUseID, parseJSONAny(b.Content.RawJSON())), nil
	case anthropic.BetaRedactedThinkingBlock:
		p := ai.NewCustomPart(map[string]any{"redactedThinking": b.Data})
		return p, nil
	case anthropic.BetaWebFetchToolResultBlock,
		anthropic.BetaCodeExecutionToolResultBlock,
		anthropic.BetaBashCodeExecutionToolResultBlock,
		anthropic.BetaTextEditorCodeExecutionToolResultBlock,
		anthropic.BetaToolSearchToolResultBlock,
		anthropic.BetaMCPToolUseBlock,
		anthropic.BetaMCPToolResultBlock,
		anthropic.BetaContainerUploadBlock:
		return nil, unsupportedServerToolError(part.Type)
	default:
		return nil, status.Errorf(ai.ErrInvalidPart, "unknown beta part: %#v", part)
	}
}

// betaToolInput normalizes tool_use input from either the content-block union's
// raw JSON or the concrete BetaToolUseBlock.Input (typed as any).
func betaToolInput(raw json.RawMessage, concrete any) any {
	if len(raw) > 0 {
		var input any
		if err := json.Unmarshal(raw, &input); err != nil {
			return json.RawMessage(append([]byte(nil), raw...))
		}
		return input
	}
	return concrete
}
