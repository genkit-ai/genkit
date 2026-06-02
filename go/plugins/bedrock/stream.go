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
	"encoding/json"
	"fmt"
	"sort"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	"github.com/firebase/genkit/go/ai"
)

// streamBlock accumulates the state of a single content block across delta
// events keyed by ContentBlockIndex.
type streamBlock struct {
	// text holds plain-text deltas concatenated in order.
	text strings.Builder
	// reasoning holds extended-thinking deltas concatenated in order.
	reasoning strings.Builder
	// reasoningSignature carries the signature announced by a reasoning
	// delta, when the provider emits one.
	reasoningSignature string
	// toolID, toolName carry the metadata announced at ContentBlockStart.
	toolID   string
	toolName string
	// toolInput collects partial JSON fragments arriving as
	// ContentBlockDeltaMemberToolUse.Value.Input. The fragments concatenate
	// into a complete JSON value at ContentBlockStop.
	toolInput strings.Builder
	// isTool flags whether this block represents a tool-use call.
	isTool bool
}

// generateStream drives the ConverseStream RPC. It dispatches text deltas to
// the chunk callback as they arrive, accumulates tool-use input fragments
// for assembly at MessageStop, and returns the final ai.ModelResponse.
func generateStream(ctx context.Context, client *bedrockruntime.Client, in *bedrockruntime.ConverseInput, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
	streamIn := &bedrockruntime.ConverseStreamInput{
		ModelId:                      in.ModelId,
		Messages:                     in.Messages,
		System:                       in.System,
		InferenceConfig:              in.InferenceConfig,
		ToolConfig:                   in.ToolConfig,
		AdditionalModelRequestFields: in.AdditionalModelRequestFields,
	}
	out, err := client.ConverseStream(ctx, streamIn)
	if err != nil {
		return nil, fmt.Errorf("bedrock: ConverseStream: %w", err)
	}
	stream := out.GetStream()
	defer stream.Close()

	blocks := map[int32]*streamBlock{}
	var stopReason types.StopReason
	var usage *types.TokenUsage

	for event := range stream.Events() {
		switch e := event.(type) {
		case *types.ConverseStreamOutputMemberMessageStart:
			// No-op; role is implicit (always assistant) on outbound.

		case *types.ConverseStreamOutputMemberContentBlockStart:
			idx := indexOf(e.Value.ContentBlockIndex)
			b := getOrInit(blocks, idx)
			if startTool, ok := e.Value.Start.(*types.ContentBlockStartMemberToolUse); ok {
				b.isTool = true
				b.toolID = aws.ToString(startTool.Value.ToolUseId)
				b.toolName = aws.ToString(startTool.Value.Name)
			}

		case *types.ConverseStreamOutputMemberContentBlockDelta:
			idx := indexOf(e.Value.ContentBlockIndex)
			b := getOrInit(blocks, idx)
			switch d := e.Value.Delta.(type) {
			case *types.ContentBlockDeltaMemberText:
				b.text.WriteString(d.Value)
				if cb != nil {
					if cberr := cb(ctx, &ai.ModelResponseChunk{Content: []*ai.Part{ai.NewTextPart(d.Value)}}); cberr != nil {
						return nil, cberr
					}
				}
			case *types.ContentBlockDeltaMemberToolUse:
				// Tool input arrives as concatenable JSON fragments.
				b.isTool = true
				b.toolInput.WriteString(aws.ToString(d.Value.Input))
			case *types.ContentBlockDeltaMemberReasoningContent:
				part := appendReasoningDelta(b, d.Value)
				if part != nil && cb != nil {
					if cberr := cb(ctx, &ai.ModelResponseChunk{Content: []*ai.Part{part}}); cberr != nil {
						return nil, cberr
					}
				}
			}

		case *types.ConverseStreamOutputMemberContentBlockStop:
			// No-op; we read accumulated state at MessageStop instead so we
			// surface tool calls only once their JSON is fully assembled.

		case *types.ConverseStreamOutputMemberMessageStop:
			stopReason = e.Value.StopReason

		case *types.ConverseStreamOutputMemberMetadata:
			usage = e.Value.Usage
		}
	}
	if err := stream.Err(); err != nil {
		return nil, fmt.Errorf("bedrock: stream: %w", err)
	}

	parts, err := blocksToParts(blocks)
	if err != nil {
		return nil, err
	}
	resp := &ai.ModelResponse{
		Message:      &ai.Message{Role: ai.RoleModel, Content: parts},
		FinishReason: mapStopReason(stopReason),
		Usage:        usageFromTokens(usage),
		Request:      req,
	}
	return resp, nil
}

// blocksToParts assembles the accumulated stream state into ai.Parts in
// ContentBlockIndex order. Tool-use blocks have their JSON input parsed
// before being emitted.
func blocksToParts(blocks map[int32]*streamBlock) ([]*ai.Part, error) {
	idxs := make([]int32, 0, len(blocks))
	for k := range blocks {
		idxs = append(idxs, k)
	}
	sort.Slice(idxs, func(i, j int) bool { return idxs[i] < idxs[j] })

	parts := make([]*ai.Part, 0, len(idxs))
	for _, i := range idxs {
		b := blocks[i]
		if b.isTool {
			input, err := decodeToolInput(b.toolInput.String())
			if err != nil {
				return nil, fmt.Errorf("bedrock: stream tool block %d: %w", i, err)
			}
			parts = append(parts, ai.NewToolRequestPart(&ai.ToolRequest{
				Ref:   b.toolID,
				Name:  b.toolName,
				Input: input,
			}))
			continue
		}
		if b.text.Len() > 0 {
			parts = append(parts, ai.NewTextPart(b.text.String()))
		}
		if b.reasoning.Len() > 0 {
			parts = append(parts, ai.NewReasoningPart(b.reasoning.String(), []byte(b.reasoningSignature)))
		}
	}
	return parts, nil
}

func appendReasoningDelta(b *streamBlock, delta types.ReasoningContentBlockDelta) *ai.Part {
	switch d := delta.(type) {
	case *types.ReasoningContentBlockDeltaMemberText:
		b.reasoning.WriteString(d.Value)
		return ai.NewReasoningPart(d.Value, nil)
	case *types.ReasoningContentBlockDeltaMemberSignature:
		b.reasoningSignature = d.Value
	case *types.ReasoningContentBlockDeltaMemberRedactedContent:
		// Redacted reasoning is intentionally not surfaced as text. Bedrock
		// may still use it internally for safety and quality.
	}
	return nil
}

// decodeToolInput parses the concatenated JSON fragments of a tool-use block.
// An empty fragment string decodes to nil so the caller can still surface a
// well-formed ToolRequest without input.
func decodeToolInput(s string) (any, error) {
	if strings.TrimSpace(s) == "" {
		return nil, nil
	}
	var v any
	if err := json.Unmarshal([]byte(s), &v); err != nil {
		return nil, err
	}
	return v, nil
}

func getOrInit(m map[int32]*streamBlock, idx int32) *streamBlock {
	if b, ok := m[idx]; ok {
		return b
	}
	b := &streamBlock{}
	m[idx] = b
	return b
}

func indexOf(p *int32) int32 {
	if p == nil {
		return 0
	}
	return *p
}
