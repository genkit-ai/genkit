// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/internal"
	"google.golang.org/genai"
)

// antigravityDefaultEnvironment is the sandbox environment used when a request
// does not specify one. "remote" requests a server-managed sandbox.
//
// This is a deliberate deviation from the JS plugin, which leaves `environment`
// optional and relies on callers to set it. In practice the interactions
// endpoint rejects requests without one ("Missing required field 'environment'",
// HTTP 400), so we default it here to keep the model runnable from a bare prompt
// (e.g. the dev UI), mirroring the existing TTS-voice default in this plugin.
// Callers can still override it, including reusing an environment ID returned by
// a prior turn for multi-turn continuity.
const antigravityDefaultEnvironment = "remote"

// AntigravityConfig is the per-request configuration for Antigravity preview
// models. It mirrors the explicit fields of the JS AntigravityConfigSchema.
// Antigravity models are served by the Gemini "interactions" endpoint rather
// than generateContent, so this config drives that request.
type AntigravityConfig struct {
	// PreviousInteractionID continues a prior interaction (multi-turn). When
	// set, the server resumes from that interaction's state.
	PreviousInteractionID string `json:"previousInteractionId,omitempty"`
	// Store indicates whether the response and request should be stored for
	// later retrieval.
	Store *bool `json:"store,omitempty"`
	// Environment is the sandbox environment configuration. It may be an
	// environment ID (string) or an inline configuration object.
	Environment any `json:"environment,omitempty"`
	// ResponseModalities lists the modalities the response may use (e.g.
	// "TEXT", "IMAGE"). Values are lower-cased on the wire.
	ResponseModalities []string `json:"responseModalities,omitempty"`
}

// antigravityKnownConfigKeys are the AntigravityConfig fields (plus the client
// option overrides) that are handled explicitly. Any other key supplied via a
// map config is forwarded to the interactions request verbatim, mirroring the
// `.passthrough()` behaviour of the JS schema.
var antigravityKnownConfigKeys = map[string]bool{
	"previousInteractionId": true,
	"store":                 true,
	"environment":           true,
	"responseModalities":    true,
	"apiKey":                true,
	"baseUrl":               true,
	"apiVersion":            true,
}

// antigravityConfigFromRequest decodes the request config into an
// AntigravityConfig. When the config arrives as a map, unknown keys are
// returned separately as passthrough fields for the interactions request.
func antigravityConfigFromRequest(input *ai.ModelRequest) (*AntigravityConfig, map[string]any, error) {
	switch c := input.Config.(type) {
	case *AntigravityConfig:
		if c == nil {
			return &AntigravityConfig{}, nil, nil
		}
		return c, nil, nil
	case AntigravityConfig:
		return &c, nil, nil
	case map[string]any:
		cfg := &AntigravityConfig{}
		b, err := json.Marshal(c)
		if err != nil {
			return nil, nil, core.NewPublicError(core.INVALID_ARGUMENT, fmt.Sprintf("invalid Antigravity configuration: %v", err), nil)
		}
		if err := json.Unmarshal(b, cfg); err != nil {
			return nil, nil, core.NewPublicError(core.INVALID_ARGUMENT, fmt.Sprintf("invalid Antigravity configuration: %v", err), nil)
		}
		var passthrough map[string]any
		for k, v := range c {
			if antigravityKnownConfigKeys[k] {
				continue
			}
			if passthrough == nil {
				passthrough = map[string]any{}
			}
			passthrough[k] = v
		}
		return cfg, passthrough, nil
	case nil:
		return &AntigravityConfig{}, nil, nil
	default:
		return nil, nil, core.NewPublicError(core.INVALID_ARGUMENT, fmt.Sprintf("invalid Antigravity configuration type: %T. Expected *AntigravityConfig or map[string]any.", input.Config), nil)
	}
}

// generateAntigravity executes a request against the Gemini interactions
// endpoint. Antigravity is sync-only (no server-sent streaming); when a stream
// callback is supplied the full response is emitted as a single chunk.
func generateAntigravity(
	ctx context.Context,
	client *genai.Client,
	model string,
	input *ai.ModelRequest,
	cb func(context.Context, *ai.ModelResponseChunk) error,
) (*ai.ModelResponse, error) {
	if model == "" {
		return nil, core.NewError(core.INVALID_ARGUMENT, "model not provided")
	}

	cfg, passthrough, err := antigravityConfigFromRequest(input)
	if err != nil {
		return nil, err
	}

	// agent identifies the Antigravity model. Antigravity does not support
	// system instructions; system messages are emitted as user input by
	// toInteractionSteps (any non-model role becomes a user_input step).
	body := map[string]any{
		"agent": model,
		"input": toInteractionSteps(input.Messages),
	}
	if cfg.PreviousInteractionID != "" {
		body["previous_interaction_id"] = cfg.PreviousInteractionID
	}
	if cfg.Store != nil {
		body["store"] = *cfg.Store
	}
	// environment is required by the endpoint; see antigravityDefaultEnvironment.
	if cfg.Environment != nil {
		body["environment"] = cfg.Environment
	} else {
		body["environment"] = antigravityDefaultEnvironment
	}
	if len(cfg.ResponseModalities) > 0 {
		mods := make([]string, len(cfg.ResponseModalities))
		for i, m := range cfg.ResponseModalities {
			mods[i] = strings.ToLower(m)
		}
		body["response_modalities"] = mods
	}
	for k, v := range passthrough {
		body[k] = v
	}

	interaction, err := createInteraction(ctx, client, body)
	if err != nil {
		return nil, err
	}

	resp, err := fromInteractionSync(interaction)
	if err != nil {
		return nil, err
	}
	resp.Request = input

	if cb != nil && resp.Message != nil && len(resp.Message.Content) > 0 {
		if err := cb(ctx, &ai.ModelResponseChunk{
			Content: resp.Message.Content,
			Role:    ai.RoleModel,
		}); err != nil {
			return nil, err
		}
	}

	return resp, nil
}

// createInteraction issues a POST to the Gemini interactions endpoint. The
// genai SDK has no binding for this endpoint, so the call is made directly,
// reusing the client's API key, base URL, and HTTP client. Mirrors the JS
// createInteraction in client.ts.
func createInteraction(ctx context.Context, client *genai.Client, body map[string]any) (*geminiInteraction, error) {
	cc := client.ClientConfig()

	baseURL := cc.HTTPOptions.BaseURL
	if baseURL == "" {
		baseURL = defaultGoogleAIBaseURL
	}
	apiVersion := cc.HTTPOptions.APIVersion
	if apiVersion == "" {
		apiVersion = defaultGoogleAIAPIVersion
	}
	url := fmt.Sprintf("%s/%s/interactions", strings.TrimSuffix(baseURL, "/"), apiVersion)

	payload, err := json.Marshal(body)
	if err != nil {
		return nil, core.NewError(core.INTERNAL, "failed to encode interaction request: %v", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return nil, core.NewError(core.INTERNAL, "failed to build interaction request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-goog-api-client", fmt.Sprintf("genkit-go/%s", internal.Version))
	req.Header.Set("Api-Revision", antigravityAPIRevision)
	if cc.APIKey != "" {
		req.Header.Set("x-goog-api-key", cc.APIKey)
	}

	httpClient := cc.HTTPClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}

	httpResp, err := httpClient.Do(req)
	if err != nil {
		return nil, core.NewError(core.UNAVAILABLE, "interaction request to %s failed: %v", url, err)
	}
	defer httpResp.Body.Close()

	data, err := io.ReadAll(httpResp.Body)
	if err != nil {
		return nil, core.NewError(core.INTERNAL, "failed to read interaction response: %v", err)
	}

	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		return nil, interactionHTTPError(url, httpResp.StatusCode, data)
	}

	var interaction geminiInteraction
	if err := json.Unmarshal(data, &interaction); err != nil {
		return nil, core.NewError(core.INTERNAL, "failed to decode interaction response: %v", err)
	}
	return &interaction, nil
}

// interactionHTTPError maps an interactions HTTP failure to a Genkit error,
// extracting the API error message when the body is the standard
// {"error": {"message": ...}} shape.
func interactionHTTPError(url string, status int, body []byte) error {
	message := strings.TrimSpace(string(body))
	var parsed struct {
		Error struct {
			Message string `json:"message"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &parsed); err == nil && parsed.Error.Message != "" {
		message = parsed.Error.Message
	}

	code := core.UNKNOWN
	switch status {
	case http.StatusTooManyRequests:
		code = core.RESOURCE_EXHAUSTED
	case http.StatusBadRequest:
		code = core.INVALID_ARGUMENT
	case 499:
		code = core.CANCELLED
	case http.StatusInternalServerError:
		code = core.INTERNAL
	case http.StatusServiceUnavailable:
		code = core.UNAVAILABLE
	}
	return core.NewPublicError(code, fmt.Sprintf("error fetching from %s: [%d] %s", url, status, message), nil)
}

// toInteractionSteps converts Genkit messages into interaction steps, mirroring
// the JS toInteractionSteps. Tool requests/responses and reasoning become their
// own steps; remaining text/media parts are grouped into a single
// model_output (for model messages) or user_input step.
func toInteractionSteps(messages []*ai.Message) []any {
	steps := []any{}

	// Auto-IDs assigned to tool requests that lack a Ref, consumed in order by
	// subsequent tool responses (mirrors ensureToolIds).
	var generatedIDs []string
	idCounter := 0

	for _, message := range messages {
		var normal []map[string]any

		for _, part := range message.Content {
			switch {
			case part.IsToolRequest():
				ref := part.ToolRequest.Ref
				if ref == "" {
					ref = fmt.Sprintf("genkit-auto-id-%d", idCounter)
					idCounter++
					generatedIDs = append(generatedIDs, ref)
				}
				steps = append(steps, map[string]any{
					"type":      "function_call",
					"name":      part.ToolRequest.Name,
					"arguments": toolArguments(part.ToolRequest.Input),
					"id":        ref,
				})
			case part.IsToolResponse():
				ref := part.ToolResponse.Ref
				if ref == "" {
					if len(generatedIDs) > 0 {
						ref = generatedIDs[0]
						generatedIDs = generatedIDs[1:]
					} else {
						ref = fmt.Sprintf("genkit-orphan-id-%d", idCounter)
						idCounter++
					}
				}
				steps = append(steps, map[string]any{
					"type":    "function_result",
					"name":    part.ToolResponse.Name,
					"result":  toolResult(part.ToolResponse.Output),
					"call_id": ref,
				})
			case part.IsReasoning():
				thought := map[string]any{
					"type":    "thought",
					"summary": []map[string]any{{"type": "text", "text": part.Text}},
				}
				if sig := metadataSignature(part.Metadata); len(sig) > 0 {
					thought["signature"] = string(sig)
				}
				steps = append(steps, thought)
			default:
				if content, ok := toInteractionContent(part); ok {
					normal = append(normal, content)
				}
			}
		}

		if len(normal) > 0 {
			stepType := "user_input"
			if message.Role == ai.RoleModel {
				stepType = "model_output"
			}
			steps = append(steps, map[string]any{
				"type":    stepType,
				"content": normal,
			})
		}
	}

	return steps
}

// toInteractionContent converts a single text or media part into an interaction
// content block. Unsupported parts are skipped (ok=false).
func toInteractionContent(part *ai.Part) (map[string]any, bool) {
	switch {
	case part.IsText():
		return map[string]any{"type": "text", "text": part.Text}, true
	case part.IsMedia():
		return toInteractionMedia(part)
	default:
		return nil, false
	}
}

func toInteractionMedia(part *ai.Part) (map[string]any, bool) {
	contentType := part.ContentType
	if contentType == "" {
		return nil, false
	}
	out := map[string]any{"mime_type": contentType}
	if strings.HasPrefix(part.Text, "data:") {
		if i := strings.Index(part.Text, ","); i >= 0 {
			out["data"] = part.Text[i+1:]
		}
	} else {
		out["uri"] = part.Text
	}

	switch {
	case strings.HasPrefix(contentType, "image/"):
		out["type"] = "image"
	case strings.HasPrefix(contentType, "audio/"):
		out["type"] = "audio"
	case strings.HasPrefix(contentType, "video/"):
		out["type"] = "video"
	case contentType == "application/pdf":
		out["type"] = "document"
	default:
		return nil, false
	}
	return out, true
}

func toolArguments(input any) map[string]any {
	switch v := input.(type) {
	case map[string]any:
		return v
	case nil:
		return nil
	default:
		return map[string]any{"input": v}
	}
}

func toolResult(output any) any {
	switch output.(type) {
	case map[string]any, string, nil:
		return output
	default:
		return map[string]any{"result": output}
	}
}

// fromInteractionSync converts an interaction response into a ModelResponse,
// mirroring the JS fromInteractionSync.
func fromInteractionSync(interaction *geminiInteraction) (*ai.ModelResponse, error) {
	if interaction.Status == "failed" {
		return nil, core.NewError(core.INTERNAL, "interaction failed")
	}

	msg := &ai.Message{Role: ai.RoleModel}
	if interaction.ID != "" || interaction.EnvironmentID != "" {
		msg.Metadata = map[string]any{}
		if interaction.ID != "" {
			msg.Metadata["interactionId"] = interaction.ID
		}
		if interaction.EnvironmentID != "" {
			msg.Metadata["environmentId"] = interaction.EnvironmentID
		}
	}

	resp := &ai.ModelResponse{
		FinishReason: ai.FinishReasonStop,
		Message:      msg,
		Custom:       interaction,
		Raw:          interaction,
	}

	if interaction.Status == "cancelled" {
		resp.FinishReason = ai.FinishReasonInterrupted
		resp.FinishMessage = "Operation cancelled"
		msg.Content = []*ai.Part{ai.NewTextPart("Operation cancelled.")}
		return resp, nil
	}

	for _, step := range interaction.Steps {
		for _, p := range fromInteractionStep(step) {
			if p != nil {
				msg.Content = append(msg.Content, p)
			}
		}
	}

	if u := interaction.Usage; u != nil {
		resp.Usage = &ai.GenerationUsage{
			InputTokens:         u.TotalInputTokens,
			OutputTokens:        u.TotalOutputTokens,
			TotalTokens:         u.TotalTokens,
			CachedContentTokens: u.TotalCachedTokens,
			ThoughtsTokens:      u.TotalThoughtTokens,
		}
		applyModalityUsage(resp.Usage, u.InputTokensByModality, true)
		applyModalityUsage(resp.Usage, u.OutputTokensByModality, false)
	}

	return resp, nil
}

func applyModalityUsage(usage *ai.GenerationUsage, modalities []modalityTokens, input bool) {
	for _, mt := range modalities {
		switch mt.Modality {
		case "text":
			if input {
				usage.InputCharacters = mt.Tokens
			} else {
				usage.OutputCharacters = mt.Tokens
			}
		case "image":
			if input {
				usage.InputImages = mt.Tokens
			} else {
				usage.OutputImages = mt.Tokens
			}
		case "audio":
			if input {
				usage.InputAudioFiles = mt.Tokens
			} else {
				usage.OutputAudioFiles = mt.Tokens
			}
		}
	}
}

// fromInteractionStep converts one interaction step into Genkit parts. Steps
// that carry no model-visible content (e.g. user_input echoes) return nil, and
// unrecognised steps are preserved losslessly as a custom part.
func fromInteractionStep(step interactionStep) []*ai.Part {
	switch step.Type {
	case "model_output":
		parts := make([]*ai.Part, 0, len(step.Content))
		for _, c := range step.Content {
			parts = append(parts, fromInteractionContent(c))
		}
		return parts
	case "user_input":
		return nil
	case "thought":
		return []*ai.Part{fromThought(step.Summary, step.Signature)}
	case "function_call":
		return []*ai.Part{ai.NewToolRequestPart(&ai.ToolRequest{
			Name:  step.Name,
			Input: step.Arguments,
			Ref:   step.ID,
		})}
	case "function_result":
		return []*ai.Part{ai.NewToolResponsePart(&ai.ToolResponse{
			Name:   step.Name,
			Output: step.Result,
			Ref:    step.CallID,
		})}
	default:
		return []*ai.Part{ai.NewCustomPart(map[string]any{"unknownStep": step})}
	}
}

// fromInteractionContent converts an interaction content block into a Genkit
// part, mirroring the JS fromInteractionContent.
func fromInteractionContent(content interactionContent) *ai.Part {
	switch content.Type {
	case "text":
		part := ai.NewTextPart(content.Text)
		if len(content.Annotations) > 0 {
			part.Metadata = map[string]any{"annotations": content.Annotations}
		}
		return part
	case "image", "audio", "video", "document":
		return fromInteractionMedia(content)
	case "thought":
		return fromThought(content.Summary, content.Signature)
	case "function_call":
		return ai.NewToolRequestPart(&ai.ToolRequest{
			Name:  content.Name,
			Input: content.Arguments,
			Ref:   content.ID,
		})
	case "function_result":
		return ai.NewToolResponsePart(&ai.ToolResponse{
			Name:   content.Name,
			Output: content.Result,
			Ref:    content.CallID,
		})
	default:
		return ai.NewCustomPart(map[string]any{"unknownContent": content})
	}
}

func fromInteractionMedia(content interactionContent) *ai.Part {
	url := content.URI
	if content.Data != "" && content.MimeType != "" {
		url = "data:" + content.MimeType + ";base64," + content.Data
	}
	part := ai.NewMediaPart(content.MimeType, url)
	if content.Resolution != "" {
		part.Metadata = map[string]any{"resolution": content.Resolution}
	}
	return part
}

func fromThought(summary []interactionContent, signature string) *ai.Part {
	var b strings.Builder
	for i, c := range summary {
		if i > 0 {
			b.WriteString("\n")
		}
		if c.Type == "text" {
			b.WriteString(c.Text)
		} else {
			b.WriteString("[Image]")
		}
	}
	return ai.NewReasoningPart(b.String(), []byte(signature))
}
