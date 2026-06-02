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
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/document"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	"github.com/firebase/genkit/go/ai"
)

const defaultMaxTokens = 4096

// generate dispatches to either the streaming or the synchronous Converse
// path based on whether cb is nil.
func generate(ctx context.Context, client *bedrockruntime.Client, modelID string, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
	in, err := buildConverseInput(modelID, req)
	if err != nil {
		return nil, err
	}
	if cb != nil {
		return generateStream(ctx, client, in, req, cb)
	}
	out, err := client.Converse(ctx, in)
	if err != nil {
		return nil, fmt.Errorf("bedrock: Converse: %w", err)
	}
	resp, err := convertResponse(out)
	if err != nil {
		return nil, err
	}
	resp.Request = req
	return resp, nil
}

// buildConverseInput translates an ai.ModelRequest into a Bedrock Converse
// input.
func buildConverseInput(modelID string, req *ai.ModelRequest) (*bedrockruntime.ConverseInput, error) {
	cfg, err := configFromRequest(req)
	if err != nil {
		return nil, err
	}

	system, messages, err := convertMessages(req.Messages)
	if err != nil {
		return nil, err
	}

	// Bedrock requires that a tool-bearing conversation not end with an
	// assistant message (the API treats it as a stale step). If the caller
	// passes a transcript that ends with an assistant turn and tools are
	// configured, drop the trailing assistant turn so the request validates.
	if len(req.Tools) > 0 && len(messages) > 0 {
		if last := messages[len(messages)-1]; last.Role == types.ConversationRoleAssistant {
			messages = messages[:len(messages)-1]
		}
	}

	in := &bedrockruntime.ConverseInput{
		ModelId:         aws.String(modelID),
		Messages:        messages,
		System:          system,
		InferenceConfig: buildInferenceConfig(cfg),
	}
	if cfg != nil && len(cfg.AdditionalModelRequestFields) > 0 {
		in.AdditionalModelRequestFields = document.NewLazyDocument(cfg.AdditionalModelRequestFields)
	}
	if len(req.Tools) > 0 {
		tools, err := convertTools(req.Tools)
		if err != nil {
			return nil, err
		}
		choice, err := convertToolChoice(cfgToolChoice(cfg), req.Tools)
		if err != nil {
			return nil, err
		}
		in.ToolConfig = &types.ToolConfiguration{Tools: tools, ToolChoice: choice}
	}
	return in, nil
}

// buildInferenceConfig converts our Config into the SDK's
// InferenceConfiguration. A non-nil pointer is always returned because most
// models require MaxTokens.
func buildInferenceConfig(cfg *Config) *types.InferenceConfiguration {
	out := &types.InferenceConfiguration{}
	maxTokens := defaultMaxTokens
	if cfg != nil && cfg.MaxTokens > 0 {
		maxTokens = cfg.MaxTokens
	}
	mt := int32(maxTokens)
	out.MaxTokens = &mt
	if cfg != nil {
		out.Temperature = cfg.Temperature
		out.TopP = cfg.TopP
		if len(cfg.StopSequences) > 0 {
			out.StopSequences = cfg.StopSequences
		}
	}
	return out
}

func cfgToolChoice(cfg *Config) string {
	if cfg == nil {
		return ""
	}
	return cfg.ToolChoice
}

// configFromRequest decodes req.Config (which may be *Config, Config, or a
// map[string]any from JSON-deserialised resumed flows) into a *Config.
// Returns (nil, nil) when no config is provided.
func configFromRequest(req *ai.ModelRequest) (*Config, error) {
	if req.Config == nil {
		return nil, nil
	}
	switch v := req.Config.(type) {
	case *Config:
		return v, nil
	case Config:
		return &v, nil
	case map[string]any:
		b, err := json.Marshal(v)
		if err != nil {
			return nil, fmt.Errorf("bedrock: marshal config: %w", err)
		}
		var c Config
		if err := json.Unmarshal(b, &c); err != nil {
			return nil, fmt.Errorf("bedrock: decode config: %w", err)
		}
		return &c, nil
	default:
		return nil, fmt.Errorf("bedrock: unexpected config type %T, want *bedrock.Config", req.Config)
	}
}

// convertMessages walks the ai.ModelRequest messages and produces a system
// block list plus the user/assistant/tool conversation. ai.RoleSystem messages
// go to System; everything else goes to Messages.
func convertMessages(msgs []*ai.Message) ([]types.SystemContentBlock, []types.Message, error) {
	var system []types.SystemContentBlock
	var conv []types.Message
	for _, m := range msgs {
		if m.Role == ai.RoleSystem {
			for _, p := range m.Content {
				if isCachePoint(p) {
					system = append(system, &types.SystemContentBlockMemberCachePoint{Value: types.CachePointBlock{Type: types.CachePointTypeDefault}})
					continue
				}
				if p.Text != "" {
					system = append(system, &types.SystemContentBlockMemberText{Value: p.Text})
				}
			}
			continue
		}
		role, err := convertRole(m.Role)
		if err != nil {
			return nil, nil, err
		}
		blocks, err := partsToContentBlocks(m.Content)
		if err != nil {
			return nil, nil, err
		}
		conv = append(conv, types.Message{Role: role, Content: blocks})
	}
	return system, conv, nil
}

func convertRole(r ai.Role) (types.ConversationRole, error) {
	switch r {
	case ai.RoleUser, ai.RoleTool:
		// Tool responses ride on a user-role turn in Bedrock.
		return types.ConversationRoleUser, nil
	case ai.RoleModel:
		return types.ConversationRoleAssistant, nil
	default:
		return "", fmt.Errorf("bedrock: unsupported role %q", r)
	}
}

// partsToContentBlocks translates a single ai.Message's content parts into
// Bedrock ContentBlocks. Returns an error for unsupported media MIME types so
// the model never silently sees mis-categorised data.
func partsToContentBlocks(parts []*ai.Part) ([]types.ContentBlock, error) {
	var blocks []types.ContentBlock
	for _, p := range parts {
		switch {
		case isCachePoint(p):
			blocks = append(blocks, &types.ContentBlockMemberCachePoint{Value: types.CachePointBlock{Type: types.CachePointTypeDefault}})
		case p.IsToolRequest():
			tr := p.ToolRequest
			inputDoc := document.NewLazyDocument(tr.Input)
			blocks = append(blocks, &types.ContentBlockMemberToolUse{Value: types.ToolUseBlock{
				ToolUseId: aws.String(tr.Ref),
				Name:      aws.String(tr.Name),
				Input:     inputDoc,
			}})
		case p.IsToolResponse():
			tr := p.ToolResponse
			text, err := toolResponseText(tr.Output)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, &types.ContentBlockMemberToolResult{Value: types.ToolResultBlock{
				ToolUseId: aws.String(tr.Ref),
				Content: []types.ToolResultContentBlock{
					&types.ToolResultContentBlockMemberText{Value: text},
				},
				Status: types.ToolResultStatusSuccess,
			}})
		case p.IsMedia():
			block, err := mediaToBlock(p)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, block)
		case p.Kind == ai.PartReasoning:
			blocks = append(blocks, reasoningPartToContentBlocks(p)...)
		case p.Text != "":
			blocks = append(blocks, &types.ContentBlockMemberText{Value: p.Text})
		}
	}
	return blocks, nil
}

func reasoningPartToContentBlocks(p *ai.Part) []types.ContentBlock {
	var blocks []types.ContentBlock
	redacted := metadataBytes(p.Metadata, redactedReasoningMetadataKey)
	if len(redacted) > 0 {
		blocks = append(blocks, &types.ContentBlockMemberReasoningContent{
			Value: &types.ReasoningContentBlockMemberRedactedContent{Value: redacted},
		})
	}

	signature := metadataBytes(p.Metadata, reasoningSignatureMetadataKey)
	if p.Text != "" && len(signature) > 0 {
		sig := string(signature)
		blocks = append(blocks, &types.ContentBlockMemberReasoningContent{
			Value: &types.ReasoningContentBlockMemberReasoningText{
				Value: types.ReasoningTextBlock{
					Text:      aws.String(p.Text),
					Signature: aws.String(sig),
				},
			},
		})
	}
	return blocks
}

// toolResponseText renders a tool's output value as a plain string, since the
// Bedrock ToolResultContentBlock variants the plugin uses are text-based.
// Structured outputs are JSON-encoded.
func toolResponseText(out any) (string, error) {
	if out == nil {
		return "", nil
	}
	if s, ok := out.(string); ok {
		return s, nil
	}
	b, err := json.Marshal(out)
	if err != nil {
		return "", fmt.Errorf("bedrock: marshal tool response: %w", err)
	}
	return string(b), nil
}

// mediaToBlock routes an ai.Part containing a media (image or document) into
// the appropriate Bedrock ContentBlock variant. The MIME type drives the
// routing. Unknown MIME types return an error.
func mediaToBlock(p *ai.Part) (types.ContentBlock, error) {
	mime := strings.ToLower(strings.TrimSpace(p.ContentType))
	data, err := decodeMediaPayload(p.Text)
	if err != nil {
		return nil, err
	}
	switch {
	case imageFormatFor(mime) != "":
		return &types.ContentBlockMemberImage{Value: types.ImageBlock{
			Format: imageFormatFor(mime),
			Source: &types.ImageSourceMemberBytes{Value: data},
		}}, nil
	case documentFormatFor(mime) != "":
		return &types.ContentBlockMemberDocument{Value: types.DocumentBlock{
			Format: documentFormatFor(mime),
			Name:   aws.String("document"),
			Source: &types.DocumentSourceMemberBytes{Value: data},
		}}, nil
	default:
		return nil, fmt.Errorf("bedrock: unsupported media MIME type %q (must be image/* or one of pdf/csv/doc/docx/xls/xlsx/html/txt/md)", mime)
	}
}

// decodeMediaPayload accepts either a raw "data:<mime>;base64,..." URL or a
// bare base64 string and returns decoded bytes. Bedrock expects raw bytes;
// the SDK base64-encodes them for the wire.
func decodeMediaPayload(s string) ([]byte, error) {
	if s == "" {
		return nil, errors.New("bedrock: media part has empty data")
	}
	if i := strings.Index(s, ";base64,"); i >= 0 {
		s = s[i+len(";base64,"):]
	} else if strings.HasPrefix(s, "data:") {
		return nil, errors.New("bedrock: data URL must be base64-encoded (use ';base64,' prefix)")
	}
	b, err := base64.StdEncoding.DecodeString(s)
	if err != nil {
		return nil, fmt.Errorf("bedrock: decode base64 media: %w", err)
	}
	return b, nil
}

func imageFormatFor(mime string) types.ImageFormat {
	switch mime {
	case "image/png":
		return types.ImageFormatPng
	case "image/jpeg", "image/jpg":
		return types.ImageFormatJpeg
	case "image/gif":
		return types.ImageFormatGif
	case "image/webp":
		return types.ImageFormatWebp
	}
	return ""
}

func documentFormatFor(mime string) types.DocumentFormat {
	switch mime {
	case "application/pdf":
		return types.DocumentFormatPdf
	case "text/csv":
		return types.DocumentFormatCsv
	case "application/msword":
		return types.DocumentFormatDoc
	case "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
		return types.DocumentFormatDocx
	case "application/vnd.ms-excel":
		return types.DocumentFormatXls
	case "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
		return types.DocumentFormatXlsx
	case "text/html":
		return types.DocumentFormatHtml
	case "text/plain":
		return types.DocumentFormatTxt
	case "text/markdown":
		return types.DocumentFormatMd
	}
	return ""
}

// convertTools maps Genkit tool definitions to Bedrock Tool union members.
func convertTools(tools []*ai.ToolDefinition) ([]types.Tool, error) {
	out := make([]types.Tool, 0, len(tools))
	for _, t := range tools {
		if t.Name == "" {
			return nil, errors.New("bedrock: tool name required")
		}
		schema := t.InputSchema
		if len(schema) == 0 {
			schema = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		out = append(out, &types.ToolMemberToolSpec{Value: types.ToolSpecification{
			Name:        aws.String(t.Name),
			Description: aws.String(t.Description),
			InputSchema: &types.ToolInputSchemaMemberJson{Value: document.NewLazyDocument(schema)},
		}})
	}
	return out, nil
}

// convertToolChoice maps the string-form ToolChoice to the SDK union.
// "" or "auto" → Auto, "any" → Any, anything else → Tool(name) when that
// name matches a declared tool.
func convertToolChoice(choice string, tools []*ai.ToolDefinition) (types.ToolChoice, error) {
	switch choice {
	case "", ToolChoiceAuto:
		return &types.ToolChoiceMemberAuto{Value: types.AutoToolChoice{}}, nil
	case ToolChoiceAny:
		return &types.ToolChoiceMemberAny{Value: types.AnyToolChoice{}}, nil
	default:
		for _, t := range tools {
			if t.Name == choice {
				return &types.ToolChoiceMemberTool{Value: types.SpecificToolChoice{Name: aws.String(choice)}}, nil
			}
		}
		return nil, fmt.Errorf("bedrock: ToolChoice %q does not match any declared tool", choice)
	}
}

// convertResponse turns a synchronous ConverseOutput into an ai.ModelResponse.
func convertResponse(out *bedrockruntime.ConverseOutput) (*ai.ModelResponse, error) {
	msgMember, ok := out.Output.(*types.ConverseOutputMemberMessage)
	if !ok {
		return nil, fmt.Errorf("bedrock: unexpected output variant %T", out.Output)
	}
	parts, err := contentBlocksToParts(msgMember.Value.Content)
	if err != nil {
		return nil, err
	}
	resp := &ai.ModelResponse{
		Message:      &ai.Message{Role: ai.RoleModel, Content: parts},
		FinishReason: mapStopReason(out.StopReason),
		Usage:        usageFromTokens(out.Usage),
	}
	return resp, nil
}

// contentBlocksToParts converts a Bedrock message's content blocks into the
// Genkit Part shape used by ai.ModelResponse.
func contentBlocksToParts(blocks []types.ContentBlock) ([]*ai.Part, error) {
	out := make([]*ai.Part, 0, len(blocks))
	for _, b := range blocks {
		switch v := b.(type) {
		case *types.ContentBlockMemberText:
			out = append(out, ai.NewTextPart(v.Value))
		case *types.ContentBlockMemberToolUse:
			input, err := unwrapToolInput(v.Value.Input)
			if err != nil {
				return nil, err
			}
			out = append(out, ai.NewToolRequestPart(&ai.ToolRequest{
				Ref:   aws.ToString(v.Value.ToolUseId),
				Name:  aws.ToString(v.Value.Name),
				Input: input,
			}))
		case *types.ContentBlockMemberReasoningContent:
			part, err := reasoningBlockToPart(v.Value)
			if err != nil {
				return nil, err
			}
			if part != nil {
				out = append(out, part)
			}
		default:
			// Other variants (image, document, audio, video, etc.) aren't
			// produced by current text models. Surface them as-is so we
			// don't silently drop content if a future model returns them.
			return nil, fmt.Errorf("bedrock: unhandled response content variant %T", b)
		}
	}
	return out, nil
}

func reasoningBlockToPart(block types.ReasoningContentBlock) (*ai.Part, error) {
	switch rc := block.(type) {
	case *types.ReasoningContentBlockMemberReasoningText:
		if rc.Value.Text == nil && rc.Value.Signature == nil {
			return nil, nil
		}
		return newBedrockReasoningPart(aws.ToString(rc.Value.Text), aws.ToString(rc.Value.Signature), nil), nil
	case *types.ReasoningContentBlockMemberRedactedContent:
		if len(rc.Value) == 0 {
			return nil, nil
		}
		return newBedrockReasoningPart("", "", rc.Value), nil
	default:
		return nil, fmt.Errorf("bedrock: unhandled reasoning content variant %T", block)
	}
}

func unwrapToolInput(d document.Interface) (any, error) {
	if d == nil {
		return nil, nil
	}
	var v any
	if err := d.UnmarshalSmithyDocument(&v); err != nil {
		return nil, fmt.Errorf("bedrock: decode tool input: %w", err)
	}
	return v, nil
}

// mapStopReason translates Bedrock's StopReason enum into ai.FinishReason.
// Covers all eight values the v1.52 SDK exposes, with a forward-compatible
// default for future additions.
func mapStopReason(r types.StopReason) ai.FinishReason {
	switch r {
	case types.StopReasonEndTurn, types.StopReasonStopSequence, types.StopReasonToolUse:
		return ai.FinishReasonStop
	case types.StopReasonMaxTokens, types.StopReasonModelContextWindowExceeded:
		return ai.FinishReasonLength
	case types.StopReasonGuardrailIntervened, types.StopReasonContentFiltered:
		return ai.FinishReasonBlocked
	case types.StopReasonMalformedModelOutput, types.StopReasonMalformedToolUse:
		// The model produced output the service could not interpret. Surface
		// as "other" rather than "stop" so callers don't treat the partial
		// response as a successful completion.
		return ai.FinishReasonOther
	default:
		return ai.FinishReasonOther
	}
}

func usageFromTokens(u *types.TokenUsage) *ai.GenerationUsage {
	if u == nil {
		return nil
	}
	out := &ai.GenerationUsage{}
	if u.InputTokens != nil {
		out.InputTokens = int(*u.InputTokens)
	}
	if u.OutputTokens != nil {
		out.OutputTokens = int(*u.OutputTokens)
	}
	if u.TotalTokens != nil {
		out.TotalTokens = int(*u.TotalTokens)
	}
	if u.CacheReadInputTokens != nil {
		out.CachedContentTokens = int(*u.CacheReadInputTokens)
	}
	return out
}
