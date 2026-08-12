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

	"github.com/invopop/jsonschema"
	"google.golang.org/genai"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal"
	"github.com/firebase/genkit/go/internal/base"
	plugininternal "github.com/firebase/genkit/go/plugins/internal"
	"github.com/firebase/genkit/go/plugins/internal/uri"
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

	schema := base.SchemaAsMap(r.Reflect(config))
	plugininternal.ApplySchemaOverrides(schema, overridesFor(config))
	return schema
}

// newModel creates a model without registering it. The model's config type
// follows its modality: image models take a [genai.GenerateImagesConfig],
// everything else speaks generateContent and takes a
// [genai.GenerateContentConfig]. The framework validates and deserializes the
// request's config into that type before the model function runs, which
// yields the request's own copy; the plugin passes that copy by pointer from
// here down rather than copying the struct again at every hop.
func newModel(client *genai.Client, id string, opts ai.ModelOptions) *ai.ModelAction {
	provider := googleAIProvider
	if client.ClientConfig().Backend == genai.BackendVertexAI {
		provider = vertexAIProvider
	}

	mt := ClassifyModel(id)
	if opts.ConfigSchema == nil {
		opts.ConfigSchema = mt.configSchema()
	}

	// Image generation reads only the request's text prompt, so imagen models
	// skip the media-download middleware the generateContent path gets.
	if mt == ModelTypeImagen {
		return ai.NewModelAction(api.NewName(provider, id), &opts,
			func(ctx context.Context, input *ai.ModelRequest, config genai.GenerateImagesConfig, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
				return generateImage(ctx, client, id, input, &config, cb)
			})
	}

	// The gemini api doesn't support downloading media from http(s).
	var download ai.ModelMiddleware
	if opts.Supports != nil && opts.Supports.Media {
		download = ai.DownloadRequestMedia(&ai.DownloadMediaOptions{
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
		})
	}

	return ai.NewModelAction(api.NewName(provider, id), &opts,
		func(ctx context.Context, input *ai.ModelRequest, config genai.GenerateContentConfig, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			// The middleware wraps per call because it must run over the
			// request while generate needs the per-request typed config.
			if download != nil {
				return download(func(ctx context.Context, input *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
					return generate(ctx, client, id, input, &config, cb)
				})(ctx, input, cb)
			}
			return generate(ctx, client, id, input, &config, cb)
		})
}

// resolveVertexModelName prepares a model name for the google.golang.org/genai
// SDK. The SDK transforms most names into `publishers/google/models/NAME`,
// which is wrong for tuned endpoints. For a short-form tuned endpoint name
// (`endpoints/ID`), this expands it to the full resource path
// `projects/PROJECT/locations/LOCATION/endpoints/ID` using the client's
// configured project and location. Other names are returned unchanged.
func resolveVertexModelName(client *genai.Client, name string) string {
	if !isTunedGeminiName(name) {
		return name
	}
	if strings.HasPrefix(name, "projects/") {
		return name
	}
	cc := client.ClientConfig()
	if cc.Backend != genai.BackendVertexAI || cc.Project == "" || cc.Location == "" {
		return name
	}
	return fmt.Sprintf("projects/%s/locations/%s/%s", cc.Project, cc.Location, name)
}

// generate requests generate call to the specified model with the provided
// configuration.
func generate(
	ctx context.Context,
	client *genai.Client,
	model string,
	input *ai.ModelRequest,
	config *genai.GenerateContentConfig,
	cb func(context.Context, *ai.ModelResponseChunk) error,
) (*ai.ModelResponse, error) {
	if model == "" {
		return nil, errors.New("model not provided")
	}
	model = resolveVertexModelName(client, model)

	cache, err := handleCache(ctx, client, input, model)
	if err != nil {
		return nil, err
	}

	gcc, err := toGeminiRequest(input, config, cache, model)
	if err != nil {
		return nil, err
	}

	contents, err := toGeminiContents(input)
	if err != nil {
		return nil, err
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

	var (
		sawChunk   bool
		usage      *genai.GenerateContentResponseUsageMetadata
		feedback   *genai.GenerateContentResponsePromptFeedback
		merged     *genai.Candidate // candidate metadata merged across chunks
		genaiParts []*genai.Part
		chunks     []*ai.Part
	)
	for chunk, err := range iter {
		// abort stream if error found in the iterator items
		if err != nil {
			return nil, wrapAPIError(err)
		}
		sawChunk = true
		for _, c := range chunk.Candidates {
			if merged == nil {
				merged = &genai.Candidate{}
			}
			mergeCandidateMetadata(merged, c)
			// Metadata-only candidates (safety ratings, citations, or a
			// terminal finish reason without content) carry nothing to
			// stream; their metadata is merged above.
			if c.Content == nil {
				continue
			}
			tc, err := translateCandidate(c)
			if err != nil {
				return nil, err
			}
			if len(tc.Message.Content) > 0 {
				err = cb(ctx, &ai.ModelResponseChunk{
					Content: tc.Message.Content,
					Role:    ai.RoleModel,
				})
				if err != nil {
					return nil, err
				}
			}
			// preserve original parts since they will be included in the
			// "custom" response field
			genaiParts = append(genaiParts, c.Content.Parts...)
			chunks = append(chunks, tc.Message.Content...)
		}
		if chunk.UsageMetadata != nil {
			usage = chunk.UsageMetadata
		}
		if chunk.PromptFeedback != nil {
			feedback = chunk.PromptFeedback
		}
	}
	if !sawChunk {
		// A stream can end without yielding a chunk: the SDK only logs a
		// scanner failure rather than surfacing it, so an empty or truncated
		// 2xx body reaches here with no error to report.
		return nil, errors.New("model stream returned no responses")
	}

	// Fold the stream back into a single response: one candidate carrying the
	// accumulated parts plus the metadata merged across chunks, and the last
	// usage metadata and prompt feedback seen (neither arrives on every
	// chunk).
	resp := &genai.GenerateContentResponse{
		UsageMetadata:  usage,
		PromptFeedback: feedback,
	}
	if merged != nil {
		merged.Content = &genai.Content{
			Role:  string(ai.RoleModel),
			Parts: genaiParts,
		}
		resp.Candidates = []*genai.Candidate{merged}
	}

	r, err := translateResponse(resp)
	if err != nil {
		return nil, err
	}
	r.Message.Content = chunks
	r.Request = input
	if cache != nil {
		r.Message.Metadata = cacheMetadata(r.Message.Metadata, cache)
	}

	return r, nil
}

// mergeCandidateMetadata folds the metadata of a streamed candidate into dst.
// Content is accumulated separately by the caller. Citations accumulate
// across chunks; every other field describes the response as a whole, so the
// latest chunk that carries a value wins.
func mergeCandidateMetadata(dst, src *genai.Candidate) {
	if src.FinishReason != "" {
		dst.FinishReason = src.FinishReason
	}
	if src.FinishMessage != "" {
		dst.FinishMessage = src.FinishMessage
	}
	if src.TokenCount != 0 {
		dst.TokenCount = src.TokenCount
	}
	if src.AvgLogprobs != 0 {
		dst.AvgLogprobs = src.AvgLogprobs
	}
	if src.LogprobsResult != nil {
		dst.LogprobsResult = src.LogprobsResult
	}
	if src.GroundingMetadata != nil {
		dst.GroundingMetadata = src.GroundingMetadata
	}
	if src.URLContextMetadata != nil {
		dst.URLContextMetadata = src.URLContextMetadata
	}
	if len(src.SafetyRatings) > 0 {
		dst.SafetyRatings = src.SafetyRatings
	}
	if src.CitationMetadata != nil {
		if dst.CitationMetadata == nil {
			dst.CitationMetadata = &genai.CitationMetadata{}
		}
		dst.CitationMetadata.Citations = append(dst.CitationMetadata.Citations, src.CitationMetadata.Citations...)
	}
}

// toGeminiContents converts the non-system messages of an [*ai.ModelRequest]
// to a slice of [*genai.Content]. System messages are handled separately via
// the request's system instruction.
func toGeminiContents(input *ai.ModelRequest) ([]*genai.Content, error) {
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
		// The API rejects contents with zero parts. History can legitimately
		// carry a contentless message (e.g. a blocked model turn replayed
		// via resp.History()), so skip it rather than fail the request.
		if len(parts) == 0 {
			continue
		}

		contents = append(contents, &genai.Content{
			Parts: parts,
			Role:  toGeminiRole(m.Role),
		})
	}
	return contents, nil
}

// toGeminiRole maps a Genkit [ai.Role] to a Gemini content role. The Gemini
// Content API only accepts "user" or "model"; tool responses are sent under
// the "user" role.
func toGeminiRole(role ai.Role) string {
	switch role {
	case ai.RoleModel:
		return string(ai.RoleModel)
	case ai.RoleTool:
		return string(ai.RoleUser)
	default:
		return string(ai.RoleUser)
	}
}

// toGeminiRequest folds an [*ai.ModelRequest] into the config the framework
// deserialized for the request, and returns the result to send to the API.
// The Genkit primitives (system prompt, tools, cache, output schema) own the
// equivalent config fields, so setting those directly is rejected. config is
// the request's own copy and is amended in place; that copy is shallow, so
// nested pointers and slices are still shared with the caller and must be
// cloned before being amended.
func toGeminiRequest(input *ai.ModelRequest, config *genai.GenerateContentConfig, cache *genai.CachedContent, modelName ...string) (*genai.GenerateContentConfig, error) {
	gcc := config

	isTTS := len(modelName) > 0 && isTTSModelName(modelName[0])

	// candidate count might not be set to 1 and will keep its zero value if not set
	// e.g. default value from reflection server is 0. TTS models leave it unset.
	if gcc.CandidateCount == 0 && !isTTS {
		gcc.CandidateCount = 1
	}
	if isTTS && len(gcc.ResponseModalities) == 0 {
		gcc.ResponseModalities = []string{"AUDIO"}
	}
	// TTS generateContent requires a speechConfig with a voice; the API rejects
	// audio requests without one. Supply a default voice so the dedicated TTS
	// models are runnable from a bare prompt (e.g. the dev UI), while still
	// letting callers override it via ai.WithConfig.
	if isTTS && !hasSpeechVoiceConfig(gcc.SpeechConfig) {
		// Clone before amending: the caller's own SpeechConfig rides the
		// shallow config copy, and the default voice must not be written
		// into it.
		sc := genai.SpeechConfig{}
		if gcc.SpeechConfig != nil {
			sc = *gcc.SpeechConfig
		}
		sc.VoiceConfig = &genai.VoiceConfig{
			PrebuiltVoiceConfig: &genai.PrebuiltVoiceConfig{VoiceName: defaultTTSVoice},
		}
		gcc.SpeechConfig = &sc
	}

	// Genkit primitive fields must be used instead of go-genai fields
	// i.e.: system prompt, tools, cached content, response schema, etc
	//
	// Classified ErrInvalidArgument: the request is the caller's to fix, so
	// these reach the dev UI and any HTTP transport as a 400 rather than a
	// 500. Not ErrInvalidInput, which means a value failed the action's input
	// schema; these pass the schema and are refused on what they mean.
	if !isTTS && gcc.CandidateCount != 1 {
		return nil, status.Errorf(status.ErrInvalidArgument, "multiple candidates is not supported")
	}
	if isTTS && gcc.CandidateCount > 1 {
		return nil, status.Errorf(status.ErrInvalidArgument, "multiple candidates is not supported")
	}
	if gcc.SystemInstruction != nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "system instruction must be set using Genkit feature: ai.WithSystemPrompt()")
	}
	if gcc.CachedContent != "" {
		return nil, status.Errorf(status.ErrInvalidArgument, "cached content must be set using Genkit feature: ai.WithCacheTTL()")
	}
	if gcc.ResponseSchema != nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "response schema must be set using Genkit feature: ai.WithTools() or ai.WithOuputType()")
	}
	if gcc.ResponseMIMEType != "" {
		return nil, status.Errorf(status.ErrInvalidArgument, "response MIME type must be set using Genkit feature: ai.WithOuputType(), ai.WithOutputSchema(), ai.WithOutputSchemaByName()")
	}
	if gcc.ResponseJsonSchema != nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "response JSON schema must be set using Genkit feature: ai.WithOutputSchema()")
	}
	for _, t := range gcc.Tools {
		if t != nil && len(t.FunctionDeclarations) > 0 {
			return nil, status.Errorf(status.ErrInvalidArgument, "custom function tools must be set using Genkit feature: ai.WithTools(); the config-level tools field is reserved for built-in API tools (GoogleSearch, Retrieval, CodeExecution, etc.)")
		}
	}

	// Set response MIME type and schema based on output format.
	// Gemini supports constrained output with application/json and text/x.enum.
	hasOutput := input.Output != nil
	if hasOutput {
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
		// Clip so the append cannot write into the caller's backing array
		// through the shallow config copy.
		gcc.Tools = mergeTools(append(slices.Clip(gcc.Tools), tools...))

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

// defaultTTSVoice is the prebuilt voice used for TTS requests that don't
// specify one. It must be a valid Gemini TTS voice name.
const defaultTTSVoice = "Algenib"

func hasSpeechVoiceConfig(sc *genai.SpeechConfig) bool {
	if sc == nil {
		return false
	}
	if sc.MultiSpeakerVoiceConfig != nil {
		return true
	}
	if sc.VoiceConfig == nil {
		return false
	}
	return sc.VoiceConfig.PrebuiltVoiceConfig != nil || sc.VoiceConfig.ReplicatedVoiceConfig != nil
}

func isTTSModelName(name string) bool {
	return strings.Contains(strings.TrimPrefix(name, "googleai/"), "-tts")
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
	msg := &ai.Message{}
	if cand.Content == nil {
		// Candidates terminated before producing content (safety blocks and
		// other early terminations) arrive without Content. Return them as a
		// contentless response with the mapped finish reason rather than
		// failing the request; a candidate with neither content nor a finish
		// reason is malformed.
		if m.FinishReason == "" {
			return nil, fmt.Errorf("no valid candidates were found in the generate response")
		}
		msg.Role = ai.RoleModel
		m.Message = msg
		return m, nil
	}
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

// promptBlocked reports whether the service refused the prompt outright: no
// candidates come back and the reason is reported through PromptFeedback.
// Serving this as a blocked response is a deliberate divergence from the JS
// plugin, which throws FAILED_PRECONDITION on every zero-candidate response:
// the finish reason and message carry why the prompt was refused, where the
// error names only the absence of candidates.
func promptBlocked(resp *genai.GenerateContentResponse) bool {
	fb := resp.PromptFeedback
	return fb != nil && fb.BlockReason != "" && fb.BlockReason != genai.BlockedReasonUnspecified
}

// translateResponse translates from a genai.GenerateContentResponse to a ai.ModelResponse.
func translateResponse(resp *genai.GenerateContentResponse) (*ai.ModelResponse, error) {
	var r *ai.ModelResponse
	var err error

	switch {
	case len(resp.Candidates) > 0:
		r, err = translateCandidate(resp.Candidates[0])
		if err != nil {
			return nil, err
		}
	case promptBlocked(resp):
		msg := resp.PromptFeedback.BlockReasonMessage
		if msg == "" {
			msg = fmt.Sprintf("prompt blocked: %s", resp.PromptFeedback.BlockReason)
		}
		r = &ai.ModelResponse{
			FinishReason:  ai.FinishReasonBlocked,
			FinishMessage: msg,
			Message:       &ai.Message{Role: ai.RoleModel},
		}
	default:
		return nil, errors.New("model returned no candidates")
	}

	if r.Usage == nil {
		r.Usage = &ai.GenerationUsage{}
	}

	// populate "custom" with plugin custom information
	custom := make(map[string]any)
	custom["candidates"] = resp.Candidates
	if resp.PromptFeedback != nil {
		custom["promptFeedback"] = resp.PromptFeedback
	}

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
