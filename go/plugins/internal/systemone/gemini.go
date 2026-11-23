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

package systemone

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"

	"github.com/firebase/genkit/go/core/status"
	"google.golang.org/genai"
	"google.golang.org/genai/interactions/models/apierrors"
	"google.golang.org/genai/interactions/models/decisions"
	"google.golang.org/genai/interactions/models/interactions"
	"google.golang.org/genai/interactions/models/operations"
)

// geminiAPIVersion is the version the Decisions API is served on. The
// SDK's default is v1, which does not serve it.
const geminiAPIVersion = "v1beta"

// Gemini is the transport of the Gemini API's Decisions API, on the genai
// SDK's client. The questions are System One's, with three differences:
// the input is one string or a list of content blocks, with media as typed
// blocks beside the state's text; instructions and criteria are text only,
// so guidance and structured instructions go as their JSON text; and a
// noul question's criteria are a true and a false description. Its answers
// are System One's, and its usage counts tokens by modality.
type Gemini struct {
	// Name names the endpoint in errors.
	Name   string
	Client *genai.Client
}

// Decide implements [Transport]. The answers are read from the response
// body as the API sent it, so the fields the SDK's answer types do not
// declare, such as log likelihoods, stay on [Info.Answers].
func (g *Gemini) Decide(ctx context.Context, model string, r *Request) (*Response, error) {
	body, err := geminiRequest(model, r)
	if err != nil {
		return nil, err
	}
	version := geminiAPIVersion
	res, err := g.Client.Decisions.Create(ctx, operations.CreateDecisionRequest{APIVersion: &version, Body: body})
	if err != nil {
		var apiErr *apierrors.APIError
		if errors.As(err, &apiErr) {
			if block := geminiBlocked(apiErr.StatusCode, []byte(apiErr.Body)); block != nil {
				return &Response{Info: Info{Model: model}, Blocked: block}, nil
			}
		}
		return nil, g.failure(ctx, err)
	}
	if res.HTTPMeta.Response == nil {
		return nil, status.Errorf(status.ErrInternal, "%s: the SDK returned no response", g.Name)
	}
	raw, err := io.ReadAll(res.HTTPMeta.Response.Body)
	if err != nil {
		return nil, status.Errorf(status.ErrUnavailable, "%s: reading response: %w", g.Name, err)
	}
	resp, err := geminiDecode(model, raw)
	if err != nil {
		return nil, status.Errorf(status.ErrInternal, "%s: %w", g.Name, err)
	}
	if resp.Answers == nil && resp.Blocked == nil {
		return nil, status.Errorf(status.ErrInternal, "%s: response carries no answers", g.Name)
	}
	return resp, nil
}

// failure maps an SDK error onto a Genkit status. An API error is mapped
// as an error response is on the native wire, after the SDK has retried
// what was worth retrying; a context that ended is reported as such.
func (g *Gemini) failure(ctx context.Context, err error) error {
	if ctx.Err() != nil {
		return stopped(ctx, err)
	}
	var apiErr *apierrors.APIError
	if errors.As(err, &apiErr) {
		return httpError(g.Name, apiErr.StatusCode, []byte(apiErr.Body))
	}
	return status.Errorf(status.ErrUnavailable, "%s: %w", g.Name, err)
}

// geminiRequest builds a Decisions request.
func geminiRequest(model string, r *Request) (decisions.CreateDecisionRequest, error) {
	input, err := geminiInput(r)
	if err != nil {
		return decisions.CreateDecisionRequest{}, err
	}
	questions := make(map[string]decisions.Question, len(r.Questions))
	for id, q := range r.Questions {
		wire, err := geminiQuestion(q)
		if err != nil {
			return decisions.CreateDecisionRequest{}, status.Errorf(status.ErrInvalidArgument, "systemone: question %q: %w", id, err)
		}
		questions[id] = wire
	}
	return decisions.CreateDecisionRequest{Model: model, Input: input, Questions: questions}, nil
}

// geminiInput is the input: the state's text alone as a string, or the
// media as content blocks followed by the state's text, as the other
// servers place media before the state. A request of media alone sends no
// text block, since the API refuses an empty one.
func geminiInput(r *Request) (decisions.Input, error) {
	text, err := stateText(r.State)
	if err != nil {
		return decisions.Input{}, err
	}
	if len(r.Media.Parts) == 0 {
		return decisions.NewInput(text), nil
	}
	blocks := make([]decisions.DecisionInputContent, 0, len(r.Media.Parts)+1)
	for _, part := range r.Media.Parts {
		blocks = append(blocks, geminiMedia(part, r.MediaResolution))
	}
	if text != "" {
		blocks = append(blocks, decisions.NewDecisionInputContent(interactions.TextContent{Text: text}))
	}
	return decisions.NewInput(blocks), nil
}

// geminiMedia is one media part as a content block: its base64 content or
// its URL, its media type, and the resolution for an image or a video.
// ultra_high is for images only, so a video keeps the API's default.
func geminiMedia(part MediaPart, resolution string) decisions.DecisionInputContent {
	data, uri := optional(part.Data), optional(part.URI)
	var res *interactions.MediaResolution
	if resolution != "" && (part.Kind == MediaImage || part.Kind == MediaVideo && resolution != "ultra_high") {
		r := interactions.MediaResolution(resolution)
		res = &r
	}
	switch part.Kind {
	case MediaImage:
		mt := interactions.ImageContentMimeType(part.ContentType)
		return decisions.NewDecisionInputContent(interactions.ImageContent{Data: data, URI: uri, MimeType: &mt, Resolution: res})
	case MediaAudio:
		mt := interactions.AudioContentMimeType(part.ContentType)
		return decisions.NewDecisionInputContent(interactions.AudioContent{Data: data, URI: uri, MimeType: &mt})
	case MediaVideo:
		mt := interactions.VideoContentMimeType(part.ContentType)
		return decisions.NewDecisionInputContent(interactions.VideoContent{Data: data, URI: uri, MimeType: &mt, Resolution: res})
	default:
		mt := interactions.DocumentContentMimeType(part.ContentType)
		return decisions.NewDecisionInputContent(interactions.DocumentContent{Data: data, URI: uri, MimeType: &mt})
	}
}

// optional is a string field the SDK leaves out when nil.
func optional(s string) *string {
	if s == "" {
		return nil
	}
	return &s
}

// stateText is the state as the model reads it: a string state as it is,
// and any other as its JSON.
func stateText(state any) (string, error) {
	if s, ok := state.(string); ok {
		return s, nil
	}
	data, err := json.Marshal(state)
	if err != nil {
		return "", status.Errorf(status.ErrInvalidArgument, "systemone: the state is not JSON: %w", err)
	}
	return string(data), nil
}

// geminiQuestion is one question on the Gemini wire. A choice's options
// go as a map, which the SDK sends in key order, not the order declared.
func geminiQuestion(q Question) (decisions.Question, error) {
	text, blocks := geminiInstructions(q.Instructions)
	switch q.Type {
	case KindChoice:
		options, _ := q.Criteria.(Options)
		criteria := make(map[string]string, len(options))
		for _, option := range options {
			criteria[option.Name] = asText(option.Description)
		}
		instructions := decisions.NewChoiceQuestionInstructions(text)
		if blocks != nil {
			instructions = decisions.NewChoiceQuestionInstructions(blocks)
		}
		return decisions.NewQuestion(decisions.ChoiceQuestion{Criteria: criteria, Instructions: instructions}), nil
	case KindScore:
		levels, _ := q.Criteria.([]any)
		criteria := make([]string, len(levels))
		for i, level := range levels {
			criteria[i] = asText(level)
		}
		instructions := decisions.NewScoreQuestionInstructions(text)
		if blocks != nil {
			instructions = decisions.NewScoreQuestionInstructions(blocks)
		}
		return decisions.NewQuestion(decisions.ScoreQuestion{Criteria: criteria, Instructions: instructions}), nil
	case KindNoul:
		var criteria *decisions.NoulCriteria
		if sides, ok := q.Criteria.(map[string]any); ok {
			yes, no := asText(sides["true"]), asText(sides["false"])
			criteria = &decisions.NoulCriteria{TrueDescription: &yes, FalseDescription: &no}
		}
		instructions := decisions.NewNoulQuestionInstructions(text)
		if blocks != nil {
			instructions = decisions.NewNoulQuestionInstructions(blocks)
		}
		return decisions.NewQuestion(decisions.NoulQuestion{Criteria: criteria, Instructions: instructions}), nil
	}
	return decisions.Question{}, status.Errorf(status.ErrInvalidSchema, "unknown type %q", q.Type)
}

// geminiInstructions is a question's instructions as the API takes them:
// a string as it is, and the elements of structured instructions as text
// blocks, each a string as it is and any other value as its JSON. Blocks
// is nil for instructions that go as one string.
func geminiInstructions(instructions any) (text string, blocks []decisions.DecisionInstructionContent) {
	if s, ok := instructions.(string); ok {
		return s, nil
	}
	items, ok := jsonArray(instructions)
	if !ok {
		return asText(instructions), nil
	}
	blocks = make([]decisions.DecisionInstructionContent, len(items))
	for i, item := range items {
		blocks[i] = decisions.NewDecisionInstructionContent(interactions.TextContent{Text: asText(item)})
	}
	return "", blocks
}

// asText is a value as the text-only fields take it: a string as it is,
// a text block's text, and any other value as its JSON, so a guidance
// object reaches the model whole.
func asText(v any) string {
	switch t := v.(type) {
	case nil:
		return ""
	case string:
		return t
	case map[string]any:
		if text, ok := t["text"].(string); ok && t["type"] == "text" && len(t) == 2 {
			return text
		}
	}
	data, err := json.Marshal(v)
	if err != nil {
		return ""
	}
	return string(data)
}

// geminiResponse is a Decisions response, with the answers as the API
// sent them.
type geminiResponse struct {
	Answers        map[string]map[string]any `json:"answers"`
	PromptFeedback *decisions.PromptFeedback `json:"prompt_feedback"`
	Usage          interactions.Usage        `json:"usage"`
}

// geminiDecode reads a Decisions response. The response names no model
// version, so the model is the one the request named.
func geminiDecode(model string, body []byte) (*Response, error) {
	var wire geminiResponse
	if err := json.Unmarshal(body, &wire); err != nil {
		return nil, status.Errorf(status.ErrInternal, "response is not a Decisions answer: %w", err)
	}
	resp := &Response{Info: Info{Model: model, Answers: wire.Answers}}
	if n := wire.Usage.TotalInputTokens; n != nil {
		resp.Usage.InputTokens = *n
	}
	if n := wire.Usage.TotalOutputTokens; n != nil {
		resp.Usage.OutputTokens = *n
	}
	if by := wire.Usage.InputTokensByModality; len(by) > 0 {
		resp.Usage.InputByModality = make(map[string]int, len(by))
		for _, m := range by {
			if m.Modality != nil && m.Tokens != nil {
				resp.Usage.InputByModality[strings.ToLower(string(*m.Modality))] += *m.Tokens
			}
		}
	}
	if fb := wire.PromptFeedback; fb != nil && fb.BlockReason != nil && len(wire.Answers) == 0 {
		resp.Blocked = &Block{Reason: string(*fb.BlockReason)}
		if fb.BlockReasonMessage != nil {
			resp.Blocked.Message = *fb.BlockReasonMessage
		}
	}
	return resp, nil
}

// geminiBlockReasons are the codes a refused input is reported with.
var geminiBlockReasons = map[string]bool{
	"safety":             true,
	"prohibited_content": true,
	"blocklist":          true,
	"image_safety":       true,
	"content_blocked":    true,
}

// geminiBlocked reads a 400 that refuses the input as unsafe. The error
// names the reason in its code, or, in the shape the other Google APIs
// use, as the reason of an ErrorInfo detail; any other 400 is an error.
func geminiBlocked(code int, body []byte) *Block {
	if code != http.StatusBadRequest {
		return nil
	}
	var envelope struct {
		Error struct {
			Code    json.RawMessage `json:"code"`
			Message string          `json:"message"`
			Details []struct {
				Reason string `json:"reason"`
			} `json:"details"`
		} `json:"error"`
	}
	if json.Unmarshal(body, &envelope) != nil {
		return nil
	}
	reasons := make([]string, 0, len(envelope.Error.Details)+1)
	var reason string
	if json.Unmarshal(envelope.Error.Code, &reason) == nil {
		reasons = append(reasons, reason)
	}
	for _, d := range envelope.Error.Details {
		reasons = append(reasons, d.Reason)
	}
	for _, r := range reasons {
		if r := strings.ToLower(r); geminiBlockReasons[r] {
			return &Block{Reason: r, Message: envelope.Error.Message}
		}
	}
	return nil
}
