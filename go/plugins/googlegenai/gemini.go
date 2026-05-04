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
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"slices"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/internal"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/firebase/genkit/go/plugins/internal/uri"
	"github.com/invopop/jsonschema"
	"google.golang.org/genai"
)

var (
	// Attribution header
	xGoogApiClientHeader = http.CanonicalHeaderKey("x-goog-api-client")
	genkitClientHeader   = http.Header{
		xGoogApiClientHeader: {fmt.Sprintf("genkit-go/%s", internal.Version)},
	}
)

// configToMap converts a config struct to a map[string]any.
func configToMap(config any) map[string]any {
	r := jsonschema.Reflector{
		DoNotReference: true, // Prevent $ref usage
		ExpandedStruct: true, // Include all fields directly
		// NOTE: keep track of updated fields in [genai.GenerateContentConfig] since
		// they could create runtime panics when parsing fields with type recursion
		IgnoredTypes: []any{
			genai.Schema{},
		},
	}

	schema := r.Reflect(config)
	applyConfigOverrides(schema, overridesFor(config))
	result := base.SchemaAsMap(schema)
	return result
}

// configFromRequest converts any supported config type to [genai.GenerateContentConfig].
func configFromRequest(input *ai.ModelRequest) (*genai.GenerateContentConfig, error) {
	var result genai.GenerateContentConfig

	switch config := input.Config.(type) {
	case genai.GenerateContentConfig:
		result = config
	case *genai.GenerateContentConfig:
		result = *config
	case map[string]any:
		// TODO: Log warnings if unknown parameters are found.
		var err error
		result, err = base.MapToStruct[genai.GenerateContentConfig](config)
		if err != nil {
			return nil, core.NewPublicError(core.INVALID_ARGUMENT, fmt.Sprintf("The configuration settings are not in the correct format. Check that the names and values match what the model expects: %v", err), nil)
		}
	case nil:
		// Empty but valid config
	default:
		return nil, core.NewPublicError(core.INVALID_ARGUMENT, fmt.Sprintf("Invalid configuration type: %T. Expected *genai.GenerateContentConfig. Ensure you are using the correct ModelRef helper (e.g., ModelRef) or passing the correct configuration struct.", input.Config), nil)
	}

	return &result, nil
}

// newModel creates a model without registering it.
func newModel(client *genai.Client, name string, opts ai.ModelOptions) ai.Model {
	provider := googleAIProvider
	if client.ClientConfig().Backend == genai.BackendVertexAI {
		provider = vertexAIProvider
	}

	mt := ClassifyModel(name)

	if opts.ConfigSchema == nil {
		if config := mt.DefaultConfig(); config != nil {
			opts.ConfigSchema = configToMap(config)
		}
	}

	meta := &ai.ModelOptions{
		Label:        opts.Label,
		Supports:     opts.Supports,
		Versions:     opts.Versions,
		ConfigSchema: opts.ConfigSchema,
		Stage:        opts.Stage,
	}

	fn := func(
		ctx context.Context,
		input *ai.ModelRequest,
		cb func(context.Context, *ai.ModelResponseChunk) error,
	) (*ai.ModelResponse, error) {
		switch mt {
		case ModelTypeImagen:
			return generateImage(ctx, client, name, input, cb)
		default:
			return generate(ctx, client, name, input, cb)
		}
	}

	// the gemini api doesn't support downloading media from http(s)
	if opts.Supports.Media {
		fn = core.ChainMiddleware(ai.DownloadRequestMedia(&ai.DownloadMediaOptions{
			MaxBytes: 1024 * 1024 * 20, // 20MB
			Filter: func(part *ai.Part) bool {
				u, err := url.Parse(part.Text)
				if err != nil {
					return true
				}
				// Gemini can handle these URLs
				return !slices.Contains(
					[]string{
						"generativelanguage.googleapis.com",
						"www.youtube.com", "youtube.com", "youtu.be",
					},
					u.Hostname(),
				)
			},
		}))(fn)
	}
	return ai.NewModel(api.NewName(provider, name), meta, fn)
}

// generate requests generate call to the specified model with the provided
// configuration.
func generate(
	ctx context.Context,
	client *genai.Client,
	model string,
	input *ai.ModelRequest,
	cb func(context.Context, *ai.ModelResponseChunk) error,
) (*ai.ModelResponse, error) {
	if model == "" {
		return nil, errors.New("model not provided")
	}

	cache, err := handleCache(ctx, client, input, model)
	if err != nil {
		return nil, err
	}

	gcc, err := toGeminiRequest(input, cache)
	if err != nil {
		return nil, err
	}

	var contents []*genai.Content
	for _, m := range input.Messages {
		// system parts are handled separately
		if m.Role == ai.RoleSystem {
			continue
		}

		parts, err := toGeminiParts(m.Content)
		if err != nil {
			return nil, err
		}

		contents = append(contents, &genai.Content{
			Parts: parts,
			Role:  string(m.Role),
		})
	}
	if len(contents) == 0 {
		return nil, fmt.Errorf("at least one message is required in generate request")
	}

	// Send out the actual request.
	if cb == nil {
		resp, err := client.Models.GenerateContent(ctx, model, contents, gcc)
		if err != nil {
			return nil, wrapAPIError(err)
		}
		r, err := translateResponse(resp)
		if err != nil {
			return nil, err
		}

		r.Request = input
		if cache != nil {
			r.Message.Metadata = cacheMetadata(r.Message.Metadata, cache)
		}
		return r, nil
	}

	// Streaming version.
	iter := client.Models.GenerateContentStream(ctx, model, contents, gcc)

	var r *ai.ModelResponse
	var genaiResp *genai.GenerateContentResponse

	genaiParts := []*genai.Part{}
	chunks := []*ai.Part{}
	for chunk, err := range iter {
		// abort stream if error found in the iterator items
		if err != nil {
			return nil, wrapAPIError(err)
		}
		for _, c := range chunk.Candidates {
			tc, err := translateCandidate(c)
			if err != nil {
				return nil, err
			}
			err = cb(ctx, &ai.ModelResponseChunk{
				Content: tc.Message.Content,
				Role:    ai.RoleModel,
			})
			if err != nil {
				return nil, err
			}
			genaiParts = append(genaiParts, c.Content.Parts...)
			chunks = append(chunks, tc.Message.Content...)
		}
		genaiResp = chunk

	}

	if len(genaiResp.Candidates) == 0 {
		return nil, fmt.Errorf("no valid candidates found")
	}

	// preserve original parts since they will be included in the
	// "custom" response field
	merged := []*genai.Candidate{
		{
			FinishReason: genaiResp.Candidates[0].FinishReason,
			Content: &genai.Content{
				Role:  string(ai.RoleModel),
				Parts: genaiParts,
			},
		},
	}

	genaiResp.Candidates = merged
	r, err = translateResponse(genaiResp)
	r.Message.Content = chunks

	if err != nil {
		return nil, fmt.Errorf("failed to generate contents: %w", err)
	}
	r.Request = input
	if cache != nil {
		r.Message.Metadata = cacheMetadata(r.Message.Metadata, cache)
	}

	return r, nil
}

// toGeminiRequest translates an [*ai.ModelRequest] to
// *genai.GenerateContentConfig
func toGeminiRequest(input *ai.ModelRequest, cache *genai.CachedContent) (*genai.GenerateContentConfig, error) {
	gcc, err := configFromRequest(input)
	if err != nil {
		return nil, err
	}

	// candidate count might not be set to 1 and will keep its zero value if not set
	// e.g. default value from reflection server is 0
	if gcc.CandidateCount == 0 {
		gcc.CandidateCount = 1
	}

	// Genkit primitive fields must be used instead of go-genai fields
	// i.e.: system prompt, tools, cached content, response schema, etc
	if gcc.CandidateCount != 1 {
		return nil, errors.New("multiple candidates is not supported")
	}
	if gcc.SystemInstruction != nil {
		return nil, errors.New("system instruction must be set using Genkit feature: ai.WithSystemPrompt()")
	}
	if gcc.CachedContent != "" {
		return nil, errors.New("cached content must be set using Genkit feature: ai.WithCacheTTL()")
	}
	if gcc.ResponseSchema != nil {
		return nil, errors.New("response schema must be set using Genkit feature: ai.WithTools() or ai.WithOuputType()")
	}
	if gcc.ResponseMIMEType != "" {
		return nil, errors.New("response MIME type must be set using Genkit feature: ai.WithOuputType(), ai.WithOutputSchema(), ai.WithOutputSchemaByName()")
	}
	if gcc.ResponseJsonSchema != nil {
		return nil, errors.New("response JSON schema must be set using Genkit feature: ai.WithOutputSchema()")
	}
	for _, t := range gcc.Tools {
		if t != nil && len(t.FunctionDeclarations) > 0 {
			return nil, errors.New("custom function tools must be set using Genkit feature: ai.WithTools(); the config-level tools field is reserved for built-in API tools (GoogleSearch, Retrieval, CodeExecution, etc.)")
		}
	}

	// Set response MIME type and schema based on output format.
	// Gemini supports constrained output with application/json and text/x.enum.
	hasOutput := input.Output != nil
	// JSON mode is not compatible with tools
	if hasOutput && len(input.Tools) == 0 {
		switch {
		case input.Output.ContentType == "application/json" || input.Output.Format == "json":
			gcc.ResponseMIMEType = "application/json"
		case input.Output.ContentType == "text/enum" || input.Output.Format == "enum":
			gcc.ResponseMIMEType = "text/x.enum"
		}
	}

	if input.Output != nil && input.Output.Constrained && gcc.ResponseMIMEType != "" {
		schema, err := toGeminiSchema(input.Output.Schema, input.Output.Schema)
		if err != nil {
			return nil, err
		}
		gcc.ResponseSchema = schema
	}

	// Add tool configuration from input.Tools and input.ToolChoice directly
	// Merge with existing tools to preserve Gemini-specific tools (Retrieval, GoogleSearch, CodeExecution)
	if len(input.Tools) > 0 {
		// First convert the tools
		tools, err := toGeminiTools(input.Tools)
		if err != nil {
			return nil, err
		}
		gcc.Tools = mergeTools(append(gcc.Tools, tools...))

		// Then set up the tool configuration based on ToolChoice
		tc, err := toGeminiToolChoice(gcc.ToolConfig, input.ToolChoice, input.Tools)
		if err != nil {
			return nil, err
		}
		gcc.ToolConfig = tc
	}

	var systemParts []*genai.Part
	for _, m := range input.Messages {
		if m.Role == ai.RoleSystem {
			parts, err := toGeminiParts(m.Content)
			if err != nil {
				return nil, err
			}
			systemParts = append(systemParts, parts...)
		}
	}

	if len(systemParts) > 0 {
		gcc.SystemInstruction = &genai.Content{
			Parts: systemParts,
			Role:  string(ai.RoleSystem),
		}
	}

	if cache != nil {
		gcc.CachedContent = cache.Name
	}

	return gcc, nil
}

// translateCandidate translates from a genai.GenerateContentResponse to an ai.ModelResponse.
func translateCandidate(cand *genai.Candidate) (*ai.ModelResponse, error) {
	m := &ai.ModelResponse{}
	switch cand.FinishReason {
	case genai.FinishReasonStop:
		m.FinishReason = ai.FinishReasonStop
	case genai.FinishReasonMaxTokens:
		m.FinishReason = ai.FinishReasonLength
	case genai.FinishReasonSafety,
		genai.FinishReasonRecitation,
		genai.FinishReasonLanguage,
		genai.FinishReasonBlocklist,
		genai.FinishReasonProhibitedContent,
		genai.FinishReasonSPII,
		genai.FinishReasonImageSafety,
		genai.FinishReasonImageProhibitedContent,
		genai.FinishReasonImageRecitation:
		m.FinishReason = ai.FinishReasonBlocked
	case genai.FinishReasonMalformedFunctionCall,
		genai.FinishReasonUnexpectedToolCall,
		genai.FinishReasonNoImage,
		genai.FinishReasonImageOther,
		genai.FinishReasonOther:
		m.FinishReason = ai.FinishReasonOther
	case "MISSING_THOUGHT_SIGNATURE":
		// Gemini 3 returns this when thought signatures are missing from the request.
		// The SDK may not have this constant yet, so we match on the string value.
		m.FinishReason = ai.FinishReasonOther
	default:
		if cand.FinishReason != "" && cand.FinishReason != genai.FinishReasonUnspecified {
			m.FinishReason = ai.FinishReasonUnknown
		}
	}

	m.FinishMessage = cand.FinishMessage
	if cand.Content == nil {
		return nil, fmt.Errorf("no valid candidates were found in the generate response")
	}
	msg := &ai.Message{}
	msg.Role = ai.Role(cand.Content.Role)
	// A single genai.Part may have several fields populated at once (e.g.
	// image-generation models can return text alongside InlineData). Emit a
	// separate ai.Part for each populated field rather than failing.
	for _, part := range cand.Content.Parts {
		var emitted []*ai.Part

		if part.Thought {
			emitted = append(emitted, ai.NewReasoningPart(part.Text, part.ThoughtSignature))
		} else if part.Text != "" {
			emitted = append(emitted, ai.NewTextPart(part.Text))
		}
		if part.InlineData != nil {
			emitted = append(emitted, ai.NewMediaPart(part.InlineData.MIMEType, "data:"+part.InlineData.MIMEType+";base64,"+base64.StdEncoding.EncodeToString(part.InlineData.Data)))
		}
		if part.FileData != nil {
			emitted = append(emitted, ai.NewMediaPart(part.FileData.MIMEType, part.FileData.FileURI))
		}
		if part.FunctionCall != nil {
			emitted = append(emitted, ai.NewToolRequestPart(&ai.ToolRequest{
				Name:  part.FunctionCall.Name,
				Input: part.FunctionCall.Args,
			}))
		}
		if part.CodeExecutionResult != nil {
			emitted = append(emitted, newCodeExecutionResultPart(
				string(part.CodeExecutionResult.Outcome),
				part.CodeExecutionResult.Output,
			))
		}
		if part.ExecutableCode != nil {
			emitted = append(emitted, newExecutableCodePart(
				string(part.ExecutableCode.Language),
				part.ExecutableCode.Code,
			))
		}

		if len(emitted) == 0 {
			continue
		}

		// Attach the thought signature to the first emitted part so that a
		// subsequent request round-trips a single signature per genai.Part.
		if len(part.ThoughtSignature) > 0 {
			first := emitted[0]
			if first.Metadata == nil {
				first.Metadata = make(map[string]any)
			}
			first.Metadata["signature"] = part.ThoughtSignature
		}

		msg.Content = append(msg.Content, emitted...)
	}
	m.Message = msg
	return m, nil
}

// translateResponse translates from a genai.GenerateContentResponse to a ai.ModelResponse.
func translateResponse(resp *genai.GenerateContentResponse) (*ai.ModelResponse, error) {
	var r *ai.ModelResponse
	var err error

	if len(resp.Candidates) > 0 {
		r, err = translateCandidate(resp.Candidates[0])
		if err != nil {
			return nil, err
		}
	} else {
		r = &ai.ModelResponse{}
	}

	if r.Usage == nil {
		r.Usage = &ai.GenerationUsage{}
	}

	// populate "custom" with plugin custom information
	custom := make(map[string]any)
	custom["candidates"] = resp.Candidates

	if u := resp.UsageMetadata; u != nil {
		r.Usage.InputTokens = int(u.PromptTokenCount)
		r.Usage.OutputTokens = int(u.CandidatesTokenCount)
		r.Usage.TotalTokens = int(u.TotalTokenCount)
		r.Usage.CachedContentTokens = int(u.CachedContentTokenCount)
		r.Usage.ThoughtsTokens = int(u.ThoughtsTokenCount)
		custom["usageMetadata"] = resp.UsageMetadata
	}

	r.Custom = custom
	return r, nil
}

// toGeminiParts converts a slice of [ai.Part] to a slice of [genai.Part].
func toGeminiParts(parts []*ai.Part) ([]*genai.Part, error) {
	res := make([]*genai.Part, 0, len(parts))
	for _, p := range parts {
		part, err := toGeminiPart(p)
		if err != nil {
			return nil, err
		}
		res = append(res, part)
	}
	return res, nil
}

// toGeminiPart converts a [ai.Part] to a [genai.Part].
func toGeminiPart(p *ai.Part) (*genai.Part, error) {
	var gp *genai.Part
	switch {
	case p.IsReasoning():
		gp = genai.NewPartFromText(p.Text)
		gp.Thought = true
	case p.IsText():
		gp = genai.NewPartFromText(p.Text)
	case p.IsMedia():
		if strings.HasPrefix(p.Text, "data:") {
			contentType, data, err := uri.Data(p)
			if err != nil {
				return nil, err
			}
			gp = genai.NewPartFromBytes(data, contentType)
		} else {
			gp = genai.NewPartFromURI(p.Text, p.ContentType)
		}
	case p.IsData():
		contentType, data, err := uri.Data(p)
		if err != nil {
			return nil, err
		}
		gp = genai.NewPartFromBytes(data, contentType)
	case p.IsToolResponse():
		toolResp := p.ToolResponse
		var output map[string]any
		if m, ok := toolResp.Output.(map[string]any); ok {
			output = m
		} else {
			output = map[string]any{
				"name":    toolResp.Name,
				"content": toolResp.Output,
			}
		}
		var isMultipart bool
		if multiPart, ok := p.Metadata["multipart"].(bool); ok {
			isMultipart = multiPart
		}
		if len(toolResp.Content) > 0 {
			isMultipart = true
		}
		if isMultipart {
			toolRespParts, err := toGeminiFunctionResponsePart(toolResp.Content)
			if err != nil {
				return nil, err
			}
			gp = genai.NewPartFromFunctionResponseWithParts(toolResp.Name, output, toolRespParts)
		} else {
			gp = genai.NewPartFromFunctionResponse(toolResp.Name, output)
		}
	case p.IsToolRequest():
		toolReq := p.ToolRequest
		var input map[string]any
		if m, ok := toolReq.Input.(map[string]any); ok {
			input = m
		} else {
			input = map[string]any{
				"input": toolReq.Input,
			}
		}
		fc := genai.NewPartFromFunctionCall(toolReq.Name, input)
		// Restore ThoughtSignature if present in metadata.
		if p.Metadata != nil {
			fc.ThoughtSignature = metadataSignature(p.Metadata)
		}
		return fc, nil
	default:
		return nil, fmt.Errorf("unknown part in the request: %q", p.Kind)
	}

	// Restore ThoughtSignature if present in metadata.
	if p.Metadata != nil {
		gp.ThoughtSignature = metadataSignature(p.Metadata)
	}

	return gp, nil
}

// metadataSignature extracts the thought signature from part metadata.
// It handles both []byte (original value) and string (base64-encoded
// after a JSON clone roundtrip).
func metadataSignature(metadata map[string]any) []byte {
	switch sig := metadata["signature"].(type) {
	case []byte:
		return sig
	case string:
		decoded, err := base64.StdEncoding.DecodeString(sig)
		if err != nil {
			return nil
		}
		return decoded
	}
	return nil
}
