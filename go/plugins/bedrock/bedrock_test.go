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
	"testing"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"

	"github.com/firebase/genkit/go/ai"
)

func TestStripInferenceProfilePrefix(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"anthropic.claude-3-haiku-20240307-v1:0", "anthropic.claude-3-haiku-20240307-v1:0"},
		{"us.anthropic.claude-3-haiku-20240307-v1:0", "anthropic.claude-3-haiku-20240307-v1:0"},
		{"eu.anthropic.claude-3-5-sonnet-20241022-v2:0", "anthropic.claude-3-5-sonnet-20241022-v2:0"},
		{"apac.amazon.nova-pro-v1:0", "amazon.nova-pro-v1:0"},
		{"jp.amazon.nova-lite-v1:0", "amazon.nova-lite-v1:0"},
		{"au.amazon.nova-micro-v1:0", "amazon.nova-micro-v1:0"},
		{"global.amazon.nova-premier-v1:0", "amazon.nova-premier-v1:0"},
		{"us-gov.anthropic.claude-3-haiku-20240307-v1:0", "anthropic.claude-3-haiku-20240307-v1:0"},
		// non-matching prefixes are returned untouched
		{"useless.prefix", "useless.prefix"},
	}
	for _, c := range cases {
		got := stripInferenceProfilePrefix(c.in)
		if got != c.want {
			t.Errorf("stripInferenceProfilePrefix(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

func TestMapStopReason(t *testing.T) {
	cases := map[types.StopReason]ai.FinishReason{
		types.StopReasonEndTurn:                    ai.FinishReasonStop,
		types.StopReasonStopSequence:               ai.FinishReasonStop,
		types.StopReasonToolUse:                    ai.FinishReasonStop,
		types.StopReasonMaxTokens:                  ai.FinishReasonLength,
		types.StopReasonModelContextWindowExceeded: ai.FinishReasonLength,
		types.StopReasonGuardrailIntervened:        ai.FinishReasonBlocked,
		types.StopReasonContentFiltered:            ai.FinishReasonBlocked,
		types.StopReasonMalformedModelOutput:       ai.FinishReasonOther,
		types.StopReasonMalformedToolUse:           ai.FinishReasonOther,
		types.StopReason("future_reason_xyz"):      ai.FinishReasonOther,
	}
	for in, want := range cases {
		if got := mapStopReason(in); got != want {
			t.Errorf("mapStopReason(%q) = %q, want %q", in, got, want)
		}
	}
}

func TestPartsToContentBlocks_SkipsReasoning(t *testing.T) {
	parts := []*ai.Part{
		ai.NewTextPart("question"),
		ai.NewReasoningPart("internal monologue from a prior model turn", nil),
		ai.NewTextPart("more question"),
	}
	blocks, err := partsToContentBlocks(parts)
	if err != nil {
		t.Fatal(err)
	}
	// Two text blocks, no leakage of the reasoning content.
	if len(blocks) != 2 {
		t.Fatalf("len(blocks) = %d, want 2", len(blocks))
	}
	for i, b := range blocks {
		text, ok := b.(*types.ContentBlockMemberText)
		if !ok {
			t.Fatalf("blocks[%d] = %T, want *ContentBlockMemberText", i, b)
		}
		if text.Value == "internal monologue from a prior model turn" {
			t.Errorf("reasoning text leaked into block %d", i)
		}
	}
}

func TestPartsToContentBlocks_RoundTripsBedrockReasoning(t *testing.T) {
	reasoning := newBedrockReasoningPart("signed thought", "sig", nil)
	redacted := ai.NewReasoningPart("", nil)
	redacted.Metadata[redactedReasoningMetadataKey] = base64.StdEncoding.EncodeToString([]byte("encrypted"))

	blocks, err := partsToContentBlocks([]*ai.Part{reasoning, redacted})
	if err != nil {
		t.Fatal(err)
	}
	if len(blocks) != 2 {
		t.Fatalf("len(blocks) = %d, want 2", len(blocks))
	}

	textBlock, ok := blocks[0].(*types.ContentBlockMemberReasoningContent)
	if !ok {
		t.Fatalf("blocks[0] = %T, want reasoning content", blocks[0])
	}
	reasoningText, ok := textBlock.Value.(*types.ReasoningContentBlockMemberReasoningText)
	if !ok {
		t.Fatalf("blocks[0].Value = %T, want reasoning text", textBlock.Value)
	}
	if aws.ToString(reasoningText.Value.Text) != "signed thought" {
		t.Errorf("reasoning text = %q", aws.ToString(reasoningText.Value.Text))
	}
	if aws.ToString(reasoningText.Value.Signature) != "sig" {
		t.Errorf("signature = %q, want sig", aws.ToString(reasoningText.Value.Signature))
	}

	redactedBlock, ok := blocks[1].(*types.ContentBlockMemberReasoningContent)
	if !ok {
		t.Fatalf("blocks[1] = %T, want reasoning content", blocks[1])
	}
	redactedContent, ok := redactedBlock.Value.(*types.ReasoningContentBlockMemberRedactedContent)
	if !ok {
		t.Fatalf("blocks[1].Value = %T, want redacted content", redactedBlock.Value)
	}
	if string(redactedContent.Value) != "encrypted" {
		t.Errorf("redacted value = %q, want encrypted", string(redactedContent.Value))
	}
}

func TestPartsToContentBlocks_DoesNotRoundTripGenericReasoningSignature(t *testing.T) {
	reasoning := ai.NewReasoningPart("signed elsewhere", []byte("foreign-sig"))
	blocks, err := partsToContentBlocks([]*ai.Part{reasoning})
	if err != nil {
		t.Fatal(err)
	}
	if len(blocks) != 0 {
		t.Fatalf("len(blocks) = %d, want 0", len(blocks))
	}
}

func TestContentBlocksToParts_ReasoningSignatureAndRedacted(t *testing.T) {
	redacted := []byte("encrypted")
	parts, err := contentBlocksToParts([]types.ContentBlock{
		&types.ContentBlockMemberReasoningContent{
			Value: &types.ReasoningContentBlockMemberReasoningText{
				Value: types.ReasoningTextBlock{
					Text:      aws.String("thinking"),
					Signature: aws.String("sig"),
				},
			},
		},
		&types.ContentBlockMemberReasoningContent{
			Value: &types.ReasoningContentBlockMemberRedactedContent{Value: redacted},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(parts) != 2 {
		t.Fatalf("len(parts) = %d, want 2", len(parts))
	}
	if !parts[0].IsReasoning() || parts[0].Text != "thinking" {
		t.Fatalf("parts[0] = %+v, want reasoning text", parts[0])
	}
	gotSig, ok := parts[0].Metadata["signature"].([]byte)
	if !ok {
		t.Fatalf("signature metadata = %T, want []byte", parts[0].Metadata["signature"])
	}
	if string(gotSig) != "sig" {
		t.Errorf("signature = %q, want sig", string(gotSig))
	}
	gotBedrockSig, ok := parts[0].Metadata[reasoningSignatureMetadataKey].([]byte)
	if !ok {
		t.Fatalf("bedrock signature metadata = %T, want []byte", parts[0].Metadata[reasoningSignatureMetadataKey])
	}
	if string(gotBedrockSig) != "sig" {
		t.Errorf("bedrock signature = %q, want sig", string(gotBedrockSig))
	}
	if !parts[1].IsReasoning() {
		t.Fatalf("parts[1] should be reasoning, got kind %v", parts[1].Kind)
	}
	gotRedacted, ok := parts[1].Metadata[redactedReasoningMetadataKey].([]byte)
	if !ok {
		t.Fatalf("redacted metadata = %T, want []byte", parts[1].Metadata[redactedReasoningMetadataKey])
	}
	if string(gotRedacted) != string(redacted) {
		t.Errorf("redacted metadata = %q, want %q", string(gotRedacted), string(redacted))
	}
}

func TestDefaultModelOptions_UnknownIsUnstable(t *testing.T) {
	opts := defaultModelOptions("vendor.future-model-v0")
	if opts.Stage != ai.ModelStageUnstable {
		t.Errorf("unknown model Stage = %q, want %q", opts.Stage, ai.ModelStageUnstable)
	}
	known := defaultModelOptions("anthropic.claude-3-5-sonnet-20241022-v2:0")
	if known.Stage != ai.ModelStageStable {
		t.Errorf("known model Stage = %q, want %q", known.Stage, ai.ModelStageStable)
	}
}

func TestImageFormatFor(t *testing.T) {
	for mime, want := range map[string]types.ImageFormat{
		"image/png":  types.ImageFormatPng,
		"image/jpeg": types.ImageFormatJpeg,
		"image/jpg":  types.ImageFormatJpeg,
		"image/gif":  types.ImageFormatGif,
		"image/webp": types.ImageFormatWebp,
		"image/heic": "", // unknown
		"":           "",
	} {
		if got := imageFormatFor(mime); got != want {
			t.Errorf("imageFormatFor(%q) = %q, want %q", mime, got, want)
		}
	}
}

func TestDocumentFormatFor(t *testing.T) {
	for mime, want := range map[string]types.DocumentFormat{
		"application/pdf": types.DocumentFormatPdf,
		"text/csv":        types.DocumentFormatCsv,
		"text/html":       types.DocumentFormatHtml,
		"text/plain":      types.DocumentFormatTxt,
		"text/markdown":   types.DocumentFormatMd,
		"image/png":       "", // not a document
		"application/zip": "", // unknown
	} {
		if got := documentFormatFor(mime); got != want {
			t.Errorf("documentFormatFor(%q) = %q, want %q", mime, got, want)
		}
	}
}

func TestMediaToBlock_ImageBytes(t *testing.T) {
	payload := []byte{0xff, 0xd8, 0xff, 0xe0}
	p := ai.NewMediaPart("image/jpeg", "data:image/jpeg;base64,"+base64.StdEncoding.EncodeToString(payload))
	block, err := mediaToBlock(p)
	if err != nil {
		t.Fatal(err)
	}
	imageBlock, ok := block.(*types.ContentBlockMemberImage)
	if !ok {
		t.Fatalf("got %T, want *ContentBlockMemberImage", block)
	}
	if imageBlock.Value.Format != types.ImageFormatJpeg {
		t.Errorf("format = %q, want %q", imageBlock.Value.Format, types.ImageFormatJpeg)
	}
	bytesSrc, ok := imageBlock.Value.Source.(*types.ImageSourceMemberBytes)
	if !ok {
		t.Fatalf("source = %T, want *ImageSourceMemberBytes", imageBlock.Value.Source)
	}
	if string(bytesSrc.Value) != string(payload) {
		t.Errorf("decoded bytes mismatch")
	}
}

func TestMediaToBlock_UnsupportedMIME(t *testing.T) {
	p := ai.NewMediaPart("application/zip", "data:application/zip;base64,UEsFBg==")
	_, err := mediaToBlock(p)
	if err == nil {
		t.Fatal("expected error for unsupported MIME, got nil")
	}
}

func TestConvertMessages_SystemAndUser(t *testing.T) {
	msgs := []*ai.Message{
		{Role: ai.RoleSystem, Content: []*ai.Part{ai.NewTextPart("you are helpful")}},
		{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("hi")}},
	}
	system, conv, err := convertMessages(msgs)
	if err != nil {
		t.Fatal(err)
	}
	if len(system) != 1 {
		t.Fatalf("len(system) = %d, want 1", len(system))
	}
	st, ok := system[0].(*types.SystemContentBlockMemberText)
	if !ok {
		t.Fatalf("system[0] = %T, want *SystemContentBlockMemberText", system[0])
	}
	if st.Value != "you are helpful" {
		t.Errorf("system text = %q", st.Value)
	}
	if len(conv) != 1 || conv[0].Role != types.ConversationRoleUser {
		t.Fatalf("conv[0] role = %v", conv[0].Role)
	}
}

func TestConvertMessages_CachePointInSystem(t *testing.T) {
	msgs := []*ai.Message{
		{Role: ai.RoleSystem, Content: []*ai.Part{
			ai.NewTextPart("long system prompt"),
			NewCachePointPart(),
		}},
	}
	system, _, err := convertMessages(msgs)
	if err != nil {
		t.Fatal(err)
	}
	if len(system) != 2 {
		t.Fatalf("len(system) = %d, want 2 (text + cache)", len(system))
	}
	if _, ok := system[1].(*types.SystemContentBlockMemberCachePoint); !ok {
		t.Errorf("system[1] = %T, want *SystemContentBlockMemberCachePoint", system[1])
	}
}

func TestConvertTools(t *testing.T) {
	tools := []*ai.ToolDefinition{
		{
			Name:        "get_weather",
			Description: "Get current weather",
			InputSchema: map[string]any{
				"type":       "object",
				"properties": map[string]any{"location": map[string]any{"type": "string"}},
			},
		},
	}
	got, err := convertTools(tools)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 1 {
		t.Fatalf("len = %d, want 1", len(got))
	}
	spec, ok := got[0].(*types.ToolMemberToolSpec)
	if !ok {
		t.Fatalf("got %T, want *ToolMemberToolSpec", got[0])
	}
	if aws.ToString(spec.Value.Name) != "get_weather" {
		t.Errorf("Name = %q", aws.ToString(spec.Value.Name))
	}
}

func TestConvertToolChoice_Auto(t *testing.T) {
	choice, err := convertToolChoice("", []*ai.ToolDefinition{{Name: "a"}})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := choice.(*types.ToolChoiceMemberAuto); !ok {
		t.Errorf("got %T, want *ToolChoiceMemberAuto", choice)
	}
}

func TestConvertToolChoice_Any(t *testing.T) {
	choice, err := convertToolChoice(ToolChoiceAny, []*ai.ToolDefinition{{Name: "a"}})
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := choice.(*types.ToolChoiceMemberAny); !ok {
		t.Errorf("got %T, want *ToolChoiceMemberAny", choice)
	}
}

func TestConvertToolChoice_SpecificTool(t *testing.T) {
	choice, err := convertToolChoice("foo", []*ai.ToolDefinition{{Name: "foo"}, {Name: "bar"}})
	if err != nil {
		t.Fatal(err)
	}
	spec, ok := choice.(*types.ToolChoiceMemberTool)
	if !ok {
		t.Fatalf("got %T, want *ToolChoiceMemberTool", choice)
	}
	if aws.ToString(spec.Value.Name) != "foo" {
		t.Errorf("tool name = %q", aws.ToString(spec.Value.Name))
	}
}

func TestConvertToolChoice_UnknownToolErrors(t *testing.T) {
	_, err := convertToolChoice("missing", []*ai.ToolDefinition{{Name: "a"}})
	if err == nil {
		t.Fatal("expected error for unknown tool name")
	}
}

func TestConfigFromRequest_Variants(t *testing.T) {
	cases := []struct {
		name string
		in   any
		want int
	}{
		{"nil", nil, 0},
		{"ptr", &Config{MaxTokens: 100}, 100},
		{"value", Config{MaxTokens: 200}, 200},
		{"map", map[string]any{"maxTokens": float64(300)}, 300},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			req := &ai.ModelRequest{Config: c.in}
			cfg, err := configFromRequest(req)
			if err != nil {
				t.Fatal(err)
			}
			if c.want == 0 {
				if cfg != nil {
					t.Errorf("want nil cfg, got %+v", cfg)
				}
				return
			}
			if cfg == nil || cfg.MaxTokens != c.want {
				t.Errorf("MaxTokens = %v, want %d", cfg, c.want)
			}
		})
	}
}

func TestConfigFromRequest_UnknownType(t *testing.T) {
	req := &ai.ModelRequest{Config: "string-is-not-a-config"}
	_, err := configFromRequest(req)
	if err == nil {
		t.Fatal("expected error for unknown config type")
	}
}

func TestDefaultModelOptions_KnownAndUnknown(t *testing.T) {
	known := defaultModelOptions("anthropic.claude-3-5-sonnet-20241022-v2:0")
	if !known.Supports.Media {
		t.Error("known multimodal model should support media")
	}
	if !known.Supports.Tools {
		t.Error("known tool-capable model should support tools")
	}
	// Inference profile prefix should still resolve.
	prefixed := defaultModelOptions("us.anthropic.claude-3-5-sonnet-20241022-v2:0")
	if !prefixed.Supports.Media {
		t.Error("prefixed model should still resolve capabilities")
	}
	// Unknown model gets the conservative default.
	unknown := defaultModelOptions("vendor.future-model-v0")
	if !unknown.Supports.Multiturn {
		t.Error("unknown model should default to multiturn=true")
	}
}

func TestDecodeMediaPayload(t *testing.T) {
	want := []byte("hello world")
	encoded := base64.StdEncoding.EncodeToString(want)

	// raw base64 string
	got, err := decodeMediaPayload(encoded)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(want) {
		t.Errorf("raw decode mismatch: got %q want %q", string(got), string(want))
	}
	// data URL form
	got, err = decodeMediaPayload("data:text/plain;base64," + encoded)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != string(want) {
		t.Errorf("data-url decode mismatch: got %q want %q", string(got), string(want))
	}
	// non-base64 data URL is rejected
	if _, err := decodeMediaPayload("data:text/plain,foo"); err == nil {
		t.Error("expected error for non-base64 data URL")
	}
	// empty payload
	if _, err := decodeMediaPayload(""); err == nil {
		t.Error("expected error for empty payload")
	}
}

func TestTitanEmbedPayload_TextModelRequiresText(t *testing.T) {
	body, err := titanEmbedPayload("amazon.titan-embed-text-v2:0", ai.DocumentFromText("hello", nil))
	if err != nil {
		t.Fatal(err)
	}
	if body.InputText != "hello" {
		t.Errorf("InputText = %q, want hello", body.InputText)
	}
	if body.InputImage != "" {
		t.Errorf("InputImage = %q, want empty", body.InputImage)
	}

	_, err = titanEmbedPayload("amazon.titan-embed-text-v2:0", &ai.Document{
		Content: []*ai.Part{ai.NewMediaPart("image/png", "data:image/png;base64,aGVsbG8=")},
	})
	if err == nil {
		t.Fatal("expected text-only Titan embedder to reject image-only input")
	}
}

func TestTitanEmbedPayload_ImageModelAcceptsImage(t *testing.T) {
	encoded := base64.StdEncoding.EncodeToString([]byte("fake png"))
	body, err := titanEmbedPayload("amazon.titan-embed-image-v1", &ai.Document{
		Content: []*ai.Part{
			ai.NewTextPart("caption"),
			ai.NewMediaPart("image/png", "data:image/png;base64,"+encoded),
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if body.InputText != "caption" {
		t.Errorf("InputText = %q, want caption", body.InputText)
	}
	if body.InputImage != encoded {
		t.Errorf("InputImage = %q, want %q", body.InputImage, encoded)
	}
}

func TestIsModernStabilityModel(t *testing.T) {
	for _, modelID := range []string{
		"stability.sd3-large-v1:0",
		"stability.sd3-5-large-v1:0",
		"stability.stable-image-core-v1:0",
		"stability.stable-image-ultra-v1:0",
	} {
		if !isModernStabilityModel(modelID) {
			t.Errorf("isModernStabilityModel(%q) = false, want true", modelID)
		}
	}
	if isModernStabilityModel("stability.stable-diffusion-xl-v1") {
		t.Error("Stable Diffusion XL should use the legacy text_prompts payload")
	}
}

func TestBlocksToParts_StreamReassembly(t *testing.T) {
	// Synthesise an out-of-order pair of blocks: text at index 0, tool at 1.
	blocks := map[int32]*streamBlock{}
	blocks[1] = &streamBlock{
		isTool:   true,
		toolID:   "call_1",
		toolName: "get_weather",
	}
	blocks[1].toolInput.WriteString(`{"location":"NYC"}`)
	blocks[0] = &streamBlock{}
	blocks[0].text.WriteString("Looking up the weather…")

	parts, err := blocksToParts(blocks)
	if err != nil {
		t.Fatal(err)
	}
	if len(parts) != 2 {
		t.Fatalf("len(parts) = %d, want 2", len(parts))
	}
	if parts[0].Text != "Looking up the weather…" {
		t.Errorf("parts[0] = %q", parts[0].Text)
	}
	if !parts[1].IsToolRequest() {
		t.Fatal("parts[1] should be a tool request")
	}
	if parts[1].ToolRequest.Name != "get_weather" {
		t.Errorf("tool name = %q", parts[1].ToolRequest.Name)
	}
	input, ok := parts[1].ToolRequest.Input.(map[string]any)
	if !ok {
		t.Fatalf("tool input = %T, want map[string]any", parts[1].ToolRequest.Input)
	}
	if input["location"] != "NYC" {
		t.Errorf("location = %v", input["location"])
	}
}

func TestAppendReasoningDelta(t *testing.T) {
	block := &streamBlock{}
	part, err := appendReasoningDelta(block, &types.ReasoningContentBlockDeltaMemberText{Value: "thinking"})
	if err != nil {
		t.Fatal(err)
	}
	if part == nil {
		t.Fatal("appendReasoningDelta returned nil for text delta")
	}
	if !part.IsReasoning() {
		t.Fatalf("part kind = %v, want reasoning", part.Kind)
	}
	if part.Text != "thinking" {
		t.Errorf("part.Text = %q, want thinking", part.Text)
	}
	if got := block.reasoning.String(); got != "thinking" {
		t.Errorf("block reasoning = %q, want thinking", got)
	}

	part, err = appendReasoningDelta(block, &types.ReasoningContentBlockDeltaMemberSignature{Value: "sig"})
	if err != nil {
		t.Fatal(err)
	}
	if part != nil {
		t.Fatalf("signature delta returned part %v, want nil", part)
	}
	if block.reasoningSignature != "sig" {
		t.Errorf("signature = %q, want sig", block.reasoningSignature)
	}

	part, err = appendReasoningDelta(block, &types.ReasoningContentBlockDeltaMemberRedactedContent{Value: []byte("encrypted")})
	if err != nil {
		t.Fatal(err)
	}
	if part != nil {
		t.Fatalf("redacted delta returned part %v, want nil", part)
	}
	if string(block.redactedReasoning) != "encrypted" {
		t.Errorf("redacted reasoning = %q, want encrypted", string(block.redactedReasoning))
	}
}

func TestAppendReasoningDelta_UnknownErrors(t *testing.T) {
	_, err := appendReasoningDelta(&streamBlock{}, &types.UnknownUnionMember{Tag: "future_reasoning_delta"})
	if err == nil {
		t.Fatal("expected error for unknown reasoning delta")
	}
}

func TestAppendContentBlockDelta_UnsupportedErrors(t *testing.T) {
	err := appendContentBlockDelta(t.Context(), &streamBlock{}, &types.ContentBlockDeltaMemberCitation{}, nil)
	if err == nil {
		t.Fatal("expected error for unsupported citation delta")
	}
}

func TestToolBlockToPart_ParsesCompleteInput(t *testing.T) {
	block := &streamBlock{
		toolID:   "call_1",
		toolName: "get_weather",
	}
	block.toolInput.WriteString(`{"location":"NYC"}`)

	part, err := toolBlockToPart(2, block)
	if err != nil {
		t.Fatal(err)
	}
	if !part.IsToolRequest() {
		t.Fatalf("part should be a tool request, got kind %v", part.Kind)
	}
	if part.ToolRequest.Ref != "call_1" {
		t.Errorf("ref = %q, want call_1", part.ToolRequest.Ref)
	}
	if part.ToolRequest.Name != "get_weather" {
		t.Errorf("name = %q, want get_weather", part.ToolRequest.Name)
	}
	input, ok := part.ToolRequest.Input.(map[string]any)
	if !ok {
		t.Fatalf("input = %T, want map[string]any", part.ToolRequest.Input)
	}
	if input["location"] != "NYC" {
		t.Errorf("location = %v, want NYC", input["location"])
	}
}

func TestToolBlockToPart_MalformedInputErrors(t *testing.T) {
	block := &streamBlock{toolID: "call_1", toolName: "get_weather"}
	block.toolInput.WriteString("{not valid")
	if _, err := toolBlockToPart(2, block); err == nil {
		t.Fatal("expected malformed tool JSON to error")
	}
}

func TestEmitToolBlockStop_CallbackReceivesToolRequest(t *testing.T) {
	block := &streamBlock{
		isTool:   true,
		toolID:   "call_1",
		toolName: "get_weather",
	}
	block.toolInput.WriteString(`{"location":"NYC"}`)

	var got *ai.ModelResponseChunk
	err := emitToolBlockStop(t.Context(), 2, block, func(ctx context.Context, c *ai.ModelResponseChunk) error {
		got = c
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if got == nil {
		t.Fatal("callback was not called")
	}
	if len(got.Content) != 1 || !got.Content[0].IsToolRequest() {
		t.Fatalf("callback content = %+v, want one tool request", got.Content)
	}
	if got.Content[0].ToolRequest.Name != "get_weather" {
		t.Errorf("tool name = %q, want get_weather", got.Content[0].ToolRequest.Name)
	}
}

func TestEmitToolBlockStop_MalformedInputErrors(t *testing.T) {
	block := &streamBlock{isTool: true, toolID: "call_1", toolName: "get_weather"}
	block.toolInput.WriteString("{not valid")
	if err := emitToolBlockStop(t.Context(), 2, block, func(ctx context.Context, c *ai.ModelResponseChunk) error {
		t.Fatal("callback should not be called")
		return nil
	}); err == nil {
		t.Fatal("expected malformed tool JSON to error")
	}
}

func TestBlocksToParts_StreamReasoningReassembly(t *testing.T) {
	blocks := map[int32]*streamBlock{}
	blocks[0] = &streamBlock{reasoningSignature: "sig", redactedReasoning: []byte("encrypted")}
	blocks[0].reasoning.WriteString("First thought. ")
	blocks[0].reasoning.WriteString("Second thought.")
	blocks[1] = &streamBlock{}
	blocks[1].text.WriteString("Final answer.")

	parts, err := blocksToParts(blocks)
	if err != nil {
		t.Fatal(err)
	}
	if len(parts) != 2 {
		t.Fatalf("len(parts) = %d, want 2", len(parts))
	}
	if !parts[0].IsReasoning() {
		t.Fatalf("parts[0] should be reasoning, got kind %v", parts[0].Kind)
	}
	if parts[0].Text != "First thought. Second thought." {
		t.Errorf("reasoning = %q", parts[0].Text)
	}
	gotSig, ok := parts[0].Metadata["signature"].([]byte)
	if !ok {
		t.Fatalf("signature metadata = %T, want []byte", parts[0].Metadata["signature"])
	}
	if string(gotSig) != "sig" {
		t.Errorf("signature = %q, want sig", string(gotSig))
	}
	gotBedrockSig, ok := parts[0].Metadata[reasoningSignatureMetadataKey].([]byte)
	if !ok {
		t.Fatalf("bedrock signature metadata = %T, want []byte", parts[0].Metadata[reasoningSignatureMetadataKey])
	}
	if string(gotBedrockSig) != "sig" {
		t.Errorf("bedrock signature = %q, want sig", string(gotBedrockSig))
	}
	gotRedacted, ok := parts[0].Metadata[redactedReasoningMetadataKey].([]byte)
	if !ok {
		t.Fatalf("redacted metadata = %T, want []byte", parts[0].Metadata[redactedReasoningMetadataKey])
	}
	if string(gotRedacted) != "encrypted" {
		t.Errorf("redacted metadata = %q, want encrypted", string(gotRedacted))
	}
	if parts[1].Text != "Final answer." {
		t.Errorf("text = %q, want Final answer.", parts[1].Text)
	}
}

func TestDecodeToolInput_EmptyAndMalformed(t *testing.T) {
	v, err := decodeToolInput("")
	if err != nil || v != nil {
		t.Errorf("decodeToolInput(\"\") = (%v, %v), want (nil, nil)", v, err)
	}
	if _, err := decodeToolInput("{not valid"); err == nil {
		t.Error("expected error on malformed JSON")
	}
}
