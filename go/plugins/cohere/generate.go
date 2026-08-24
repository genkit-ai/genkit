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

package cohere

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"

	cohere "github.com/cohere-ai/cohere-go/v2"
	cohereclient "github.com/cohere-ai/cohere-go/v2/client"
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/internal/base"
)

// generate runs a Cohere ChatV2 request. When cb is nil the call is
// non-streaming; otherwise each delta is dispatched to the callback and the
// aggregated response is returned at the end.
func generate(
	ctx context.Context,
	client *cohereclient.Client,
	model string,
	input *ai.ModelRequest,
	cb func(context.Context, *ai.ModelResponseChunk) error,
) (*ai.ModelResponse, error) {
	req, err := toCohereRequest(input)
	if err != nil {
		return nil, fmt.Errorf("cohere: %w", err)
	}
	req.Model = model

	if cb == nil {
		resp, err := client.V2.Chat(ctx, req)
		if err != nil {
			return nil, fmt.Errorf("cohere: %w", err)
		}
		r, err := toGenkitResponse(resp)
		if err != nil {
			return nil, err
		}
		r.Request = input
		return r, nil
	}

	return generateStream(ctx, client, req, input, cb)
}

// generateStream consumes the ChatV2 SSE stream, forwarding text and tool-call
// deltas to cb and assembling the final [ai.ModelResponse].
func generateStream(
	ctx context.Context,
	client *cohereclient.Client,
	req *cohere.V2ChatRequest,
	input *ai.ModelRequest,
	cb func(context.Context, *ai.ModelResponseChunk) error,
) (*ai.ModelResponse, error) {
	streamReq, err := toStreamRequest(req)
	if err != nil {
		return nil, fmt.Errorf("cohere: %w", err)
	}
	stream, err := client.V2.ChatStream(ctx, streamReq)
	if err != nil {
		return nil, fmt.Errorf("cohere: %w", err)
	}
	defer stream.Close()

	var text, thinking strings.Builder
	r := &ai.ModelResponse{
		Request:      input,
		Message:      &ai.Message{Role: ai.RoleModel},
		FinishReason: ai.FinishReasonUnknown,
	}

	// Tool calls arrive incrementally, keyed by their stream index.
	type toolAccumulator struct {
		id   string
		name string
		args strings.Builder
	}
	tools := map[int]*toolAccumulator{}
	var toolOrder []int
	var citations []*cohere.Citation

	for {
		event, err := stream.Recv()
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("cohere: %w", err)
		}

		switch {
		case event.ContentDelta != nil:
			textDelta, thinkingDelta := contentDeltaParts(event.ContentDelta)
			if thinkingDelta != "" {
				thinking.WriteString(thinkingDelta)
				if err := cb(ctx, &ai.ModelResponseChunk{
					Content: []*ai.Part{ai.NewReasoningPart(thinkingDelta, nil)},
				}); err != nil {
					return nil, err
				}
			}
			if textDelta != "" {
				text.WriteString(textDelta)
				if err := cb(ctx, &ai.ModelResponseChunk{
					Content: []*ai.Part{ai.NewTextPart(textDelta)},
				}); err != nil {
					return nil, err
				}
			}

		case event.ToolCallStart != nil:
			idx := derefInt(event.ToolCallStart.Index)
			acc := &toolAccumulator{}
			if d := event.ToolCallStart.Delta; d != nil && d.Message != nil && d.Message.ToolCalls != nil {
				tc := d.Message.ToolCalls
				acc.id = tc.Id
				if tc.Function != nil {
					if tc.Function.Name != nil {
						acc.name = *tc.Function.Name
					}
					if tc.Function.Arguments != nil {
						acc.args.WriteString(*tc.Function.Arguments)
					}
				}
			}
			tools[idx] = acc
			toolOrder = append(toolOrder, idx)

		case event.ToolCallDelta != nil:
			idx := derefInt(event.ToolCallDelta.Index)
			if acc := tools[idx]; acc != nil {
				if d := event.ToolCallDelta.Delta; d != nil && d.Message != nil &&
					d.Message.ToolCalls != nil && d.Message.ToolCalls.Function != nil &&
					d.Message.ToolCalls.Function.Arguments != nil {
					acc.args.WriteString(*d.Message.ToolCalls.Function.Arguments)
				}
			}

		case event.ToolCallEnd != nil:
			idx := derefInt(event.ToolCallEnd.Index)
			acc := tools[idx]
			if acc == nil {
				continue
			}
			part, err := toolCallPart(acc.id, acc.name, acc.args.String())
			if err != nil {
				return nil, err
			}
			if err := cb(ctx, &ai.ModelResponseChunk{Content: []*ai.Part{part}}); err != nil {
				return nil, err
			}

		case event.CitationStart != nil:
			if d := event.CitationStart.Delta; d != nil && d.Message != nil && d.Message.Citations != nil {
				citations = append(citations, d.Message.Citations)
			}

		case event.MessageEnd != nil:
			if d := event.MessageEnd.Delta; d != nil {
				if d.FinishReason != nil {
					r.FinishReason = toGenkitFinishReason(*d.FinishReason)
				}
				if usage := toGenkitUsage(d.Usage); usage != nil {
					r.Usage = usage
				}
			}
		}
	}

	// Cohere streams thinking before the answer; preserve that order.
	if thinking.Len() > 0 {
		r.Message.Content = append(r.Message.Content, ai.NewReasoningPart(thinking.String(), nil))
	}
	if text.Len() > 0 {
		r.Message.Content = append(r.Message.Content, ai.NewTextPart(text.String()))
	}
	for _, idx := range toolOrder {
		acc := tools[idx]
		part, err := toolCallPart(acc.id, acc.name, acc.args.String())
		if err != nil {
			return nil, err
		}
		r.Message.Content = append(r.Message.Content, part)
	}
	if len(citations) > 0 {
		r.Custom = map[string]any{"citations": citations}
	}

	return r, nil
}

// toCohereRequest translates an [ai.ModelRequest] into a Cohere ChatV2 request.
// Any caller-supplied config is used as the base, so fields such as documents,
// safety_mode and citation_options pass through untouched.
func toCohereRequest(input *ai.ModelRequest) (*cohere.V2ChatRequest, error) {
	req, err := configFromRequest(input)
	if err != nil {
		return nil, err
	}

	messages, err := toCohereMessages(input.Messages)
	if err != nil {
		return nil, err
	}
	req.Messages = messages

	tools, err := toCohereTools(input.Tools)
	if err != nil {
		return nil, err
	}
	if len(tools) > 0 {
		req.Tools = tools
	}

	if input.Output != nil && input.Output.Format == "json" && input.Output.Schema != nil && input.Output.Constrained {
		req.ResponseFormat = &cohere.ResponseFormatV2{
			Type:       "json_object",
			JsonObject: &cohere.JsonResponseFormatV2{JsonSchema: input.Output.Schema},
		}
	}

	return req, nil
}

// configFromRequest converts any supported config value into a
// [cohere.V2ChatRequest]. An absent config is treated as valid.
func configFromRequest(input *ai.ModelRequest) (*cohere.V2ChatRequest, error) {
	var result cohere.V2ChatRequest

	switch config := input.Config.(type) {
	case cohere.V2ChatRequest:
		result = config
	case *cohere.V2ChatRequest:
		if config != nil {
			result = *config
		}
	case map[string]any:
		var err error
		result, err = base.MapToStruct[cohere.V2ChatRequest](config)
		if err != nil {
			return nil, fmt.Errorf("unable to parse config: %w", err)
		}
	case nil:
		// Empty configuration is valid.
	default:
		return nil, fmt.Errorf("unexpected config type: %T", input.Config)
	}

	return &result, nil
}

// toCohereMessages maps Genkit messages to Cohere ChatV2 messages, splitting
// tool-result messages into one Cohere tool message per response part.
func toCohereMessages(messages []*ai.Message) (cohere.ChatMessages, error) {
	out := make(cohere.ChatMessages, 0, len(messages))

	for _, m := range messages {
		switch m.Role {
		case ai.RoleSystem:
			// Skip empty content: the SDK's content union cannot marshal an
			// empty string, and an empty system prompt carries no instruction.
			text := m.Text()
			if text == "" {
				continue
			}
			out = append(out, &cohere.ChatMessageV2{
				Role:   "system",
				System: &cohere.SystemMessageV2{Content: &cohere.SystemMessageV2Content{String: text}},
			})

		case ai.RoleUser:
			text := m.Text()
			if text == "" {
				continue
			}
			out = append(out, &cohere.ChatMessageV2{
				Role: "user",
				User: &cohere.UserMessageV2{Content: &cohere.UserMessageV2Content{String: text}},
			})

		case ai.RoleModel:
			asst, err := toCohereAssistantMessage(m)
			if err != nil {
				return nil, err
			}
			out = append(out, &cohere.ChatMessageV2{Role: "assistant", Assistant: asst})

		case ai.RoleTool:
			for _, p := range m.Content {
				if !p.IsToolResponse() {
					continue
				}
				tm, err := toCohereToolMessage(p.ToolResponse)
				if err != nil {
					return nil, err
				}
				out = append(out, tm)
			}

		default:
			return nil, fmt.Errorf("unsupported message role: %q", m.Role)
		}
	}

	return out, nil
}

// toCohereAssistantMessage builds an assistant message, carrying both any text
// content and tool calls (with their tool_call_id round-tripped via Ref).
func toCohereAssistantMessage(m *ai.Message) (*cohere.AssistantMessage, error) {
	asst := &cohere.AssistantMessage{}
	var text strings.Builder

	for _, p := range m.Content {
		switch {
		case p.IsText():
			text.WriteString(p.Text)
		case p.IsToolRequest():
			tr := p.ToolRequest
			args, err := json.Marshal(tr.Input)
			if err != nil {
				return nil, fmt.Errorf("unable to marshal tool request input: %w", err)
			}
			name := tr.Name
			argStr := string(args)
			asst.ToolCalls = append(asst.ToolCalls, &cohere.ToolCallV2{
				Id:       tr.Ref,
				Function: &cohere.ToolCallV2Function{Name: &name, Arguments: &argStr},
			})
		}
	}

	if text.Len() > 0 {
		asst.Content = &cohere.AssistantMessageV2Content{String: text.String()}
	}

	return asst, nil
}

// toCohereToolMessage maps a single tool response to a Cohere tool message.
func toCohereToolMessage(tr *ai.ToolResponse) (*cohere.ChatMessageV2, error) {
	output, err := json.Marshal(tr.Output)
	if err != nil {
		return nil, fmt.Errorf("unable to marshal tool response output: %w", err)
	}
	return &cohere.ChatMessageV2{
		Role: "tool",
		Tool: &cohere.ToolMessageV2{
			ToolCallId: tr.Ref,
			Content:    &cohere.ToolMessageV2Content{String: string(output)},
		},
	}, nil
}

// toCohereTools maps Genkit tool definitions to Cohere ToolV2 values. The
// InputSchema is passed through as the function's JSON Schema parameters.
func toCohereTools(tools []*ai.ToolDefinition) ([]*cohere.ToolV2, error) {
	if len(tools) == 0 {
		return nil, nil
	}

	out := make([]*cohere.ToolV2, 0, len(tools))
	for _, t := range tools {
		if t.Name == "" {
			return nil, errors.New("tool name is required")
		}
		params := t.InputSchema
		if len(params) == 0 {
			params = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		desc := t.Description
		out = append(out, &cohere.ToolV2{
			Function: &cohere.ToolV2Function{
				Name:        t.Name,
				Description: &desc,
				Parameters:  params,
			},
		})
	}

	return out, nil
}

// toGenkitResponse maps a Cohere ChatV2 response back to an [ai.ModelResponse].
// Citations, when present, are preserved under Custom["citations"].
func toGenkitResponse(resp *cohere.V2ChatResponse) (*ai.ModelResponse, error) {
	r := &ai.ModelResponse{FinishReason: toGenkitFinishReason(resp.FinishReason)}

	msg := &ai.Message{Role: ai.RoleModel}
	if resp.Message != nil {
		for _, item := range resp.Message.Content {
			if item == nil {
				continue
			}
			if item.Thinking != nil {
				// Cohere thinking carries no signature (unlike Anthropic).
				msg.Content = append(msg.Content, ai.NewReasoningPart(item.Thinking.Thinking, nil))
			}
			if item.Text != nil {
				msg.Content = append(msg.Content, ai.NewTextPart(item.Text.Text))
			}
		}
		for _, tc := range resp.Message.ToolCalls {
			if tc == nil || tc.Function == nil {
				continue
			}
			name := ""
			if tc.Function.Name != nil {
				name = *tc.Function.Name
			}
			args := ""
			if tc.Function.Arguments != nil {
				args = *tc.Function.Arguments
			}
			part, err := toolCallPart(tc.Id, name, args)
			if err != nil {
				return nil, err
			}
			msg.Content = append(msg.Content, part)
		}
		if len(resp.Message.Citations) > 0 {
			r.Custom = map[string]any{"citations": resp.Message.Citations}
		}
	}

	r.Message = msg
	r.Raw = resp
	r.Usage = toGenkitUsage(resp.Usage)
	return r, nil
}

// toolCallPart builds a tool-request part, parsing the JSON-encoded arguments
// string into a structured input value.
func toolCallPart(ref, name, args string) (*ai.Part, error) {
	var input any
	if args != "" {
		if err := json.Unmarshal([]byte(args), &input); err != nil {
			return nil, fmt.Errorf("unable to parse tool call arguments: %w", err)
		}
	}
	return ai.NewToolRequestPart(&ai.ToolRequest{
		Ref:   ref,
		Name:  name,
		Input: input,
	}), nil
}

// toGenkitFinishReason maps Cohere finish reasons onto Genkit's enum.
func toGenkitFinishReason(reason cohere.ChatFinishReason) ai.FinishReason {
	switch reason {
	case cohere.ChatFinishReasonComplete, cohere.ChatFinishReasonStopSequence, cohere.ChatFinishReasonToolCall:
		return ai.FinishReasonStop
	case cohere.ChatFinishReasonMaxTokens:
		return ai.FinishReasonLength
	default:
		return ai.FinishReasonUnknown
	}
}

// toGenkitUsage maps Cohere token usage onto Genkit's GenerationUsage.
func toGenkitUsage(u *cohere.Usage) *ai.GenerationUsage {
	if u == nil || u.Tokens == nil {
		return nil
	}
	usage := &ai.GenerationUsage{}
	if u.Tokens.InputTokens != nil {
		usage.InputTokens = int(*u.Tokens.InputTokens)
	}
	if u.Tokens.OutputTokens != nil {
		usage.OutputTokens = int(*u.Tokens.OutputTokens)
	}
	return usage
}

// toStreamRequest converts a V2ChatRequest into the field-identical
// V2ChatStreamRequest expected by the streaming endpoint. The two types share
// the same JSON shape, so a marshal round-trip transfers every field —
// including the union-typed documents/safety_mode/tool_choice — without
// per-field conversion.
func toStreamRequest(req *cohere.V2ChatRequest) (*cohere.V2ChatStreamRequest, error) {
	data, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("unable to encode chat request: %w", err)
	}
	var streamReq cohere.V2ChatStreamRequest
	if err := json.Unmarshal(data, &streamReq); err != nil {
		return nil, fmt.Errorf("unable to decode stream request: %w", err)
	}
	return &streamReq, nil
}

// contentDeltaParts extracts the text and thinking payloads from a
// content-delta event. Either may be empty.
func contentDeltaParts(e *cohere.ChatContentDeltaEvent) (text, thinking string) {
	if e == nil || e.Delta == nil || e.Delta.Message == nil || e.Delta.Message.Content == nil {
		return "", ""
	}
	content := e.Delta.Message.Content
	if content.Text != nil {
		text = *content.Text
	}
	if content.Thinking != nil {
		thinking = *content.Thinking
	}
	return text, thinking
}

// derefInt returns the pointed-to int, or 0 when nil.
func derefInt(p *int) int {
	if p == nil {
		return 0
	}
	return *p
}
