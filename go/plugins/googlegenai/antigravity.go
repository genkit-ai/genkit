// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal"
	"google.golang.org/genai"
)

// antigravityDefaultEnvironment is the sandbox environment sent when neither
// the request nor a previous turn specifies one.
//
// The interactions endpoint rejects a request without an environment (HTTP 400,
// "Missing required field 'environment'", verified against the live API for
// antigravity-preview-09-2026), so a default is needed to run the model from a
// bare prompt. It is the object form {"type": "remote"}, as the Python plugin
// sends; the API also accepts the bare string "remote" (the JS plugin's
// "remote" shorthand), and both are recognised as "the default" when deciding
// whether a previous turn's environment ID should be reused.
func antigravityDefaultEnvironment() any {
	return map[string]any{"type": "remote"}
}

// maxInteractionResponseBytes caps how much of an interactions response is
// read. Responses can embed generated media as base64, so the cap is generous.
const maxInteractionResponseBytes = 64 << 20 // 64 MiB

// AntigravityConfig is the per-request configuration for Antigravity preview
// models. It mirrors the JS AntigravityConfigSchema, which is declared with
// `.passthrough()`: any key not declared here is forwarded to the interactions
// request verbatim (see Extra).
//
// Antigravity models are served by the Gemini "interactions" endpoint rather
// than generateContent, so this config drives that request.
type AntigravityConfig struct {
	// APIKey overrides the API key for this request. Note that request config
	// is recorded in traces; prefer configuring the key on the plugin.
	APIKey string `json:"apiKey,omitempty"`
	// BaseURL overrides the plugin-configured API endpoint for this request.
	BaseURL string `json:"baseUrl,omitempty"`
	// APIVersion overrides the plugin-configured API version for this request.
	APIVersion string `json:"apiVersion,omitempty"`
	// PreviousInteractionID continues a prior interaction (multi-turn). When
	// set, the server resumes from that interaction's state and only the
	// messages after the last model message are sent.
	PreviousInteractionID string `json:"previousInteractionId,omitempty"`
	// Store indicates whether the response and request should be stored for
	// later retrieval. It is sent only when set.
	Store *bool `json:"store,omitempty"`
	// Environment is the sandbox environment configuration. It may be an
	// environment ID (string) or an inline configuration object.
	Environment any `json:"environment,omitempty"`
	// ResponseModalities lists the modalities the response may use ("TEXT",
	// "IMAGE"). They are sent as response_format entries.
	ResponseModalities []string `json:"responseModalities,omitempty"`
	// Model is the Gemini model the agent runs on, sent as agent_config.model.
	Model string `json:"model,omitempty"`
	// Tools lists built-in interactions tools (for example
	// {"type": "google_search"} or {"type": "code_execution"}). They are sent
	// ahead of the request's own function tools. The shorthand keys
	// googleSearch, codeExecution and urlContext are accepted as well.
	Tools []map[string]any `json:"tools,omitempty"`

	// Extra holds every config key not declared above. It is forwarded as-is to
	// the top level of the interactions request. It is populated when the
	// config is decoded from JSON and is merged back in when encoded.
	Extra map[string]any `json:"-"`
}

// antigravityConfigAlias has AntigravityConfig's fields without its methods,
// so the custom (un)marshalers can use the default struct handling.
type antigravityConfigAlias AntigravityConfig

// antigravityDeclaredKeys are the JSON keys AntigravityConfig declares.
var antigravityDeclaredKeys = []string{
	"apiKey", "baseUrl", "apiVersion", "previousInteractionId", "store",
	"environment", "responseModalities", "model", "tools",
}

// UnmarshalJSON decodes the declared fields and collects every other key into
// Extra. The framework-owned "version" key is dropped.
func (c *AntigravityConfig) UnmarshalJSON(data []byte) error {
	var alias antigravityConfigAlias
	if err := json.Unmarshal(data, &alias); err != nil {
		return err
	}
	var all map[string]any
	if err := json.Unmarshal(data, &all); err != nil {
		return err
	}
	for _, k := range antigravityDeclaredKeys {
		delete(all, k)
	}
	delete(all, "version")
	if len(all) > 0 {
		alias.Extra = all
	}
	*c = AntigravityConfig(alias)
	return nil
}

// MarshalJSON encodes the declared fields and merges Extra at the top level.
func (c AntigravityConfig) MarshalJSON() ([]byte, error) {
	data, err := json.Marshal(antigravityConfigAlias(c))
	if err != nil {
		return nil, err
	}
	if len(c.Extra) == 0 {
		return data, nil
	}
	var m map[string]any
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	for k, v := range c.Extra {
		if _, declared := m[k]; !declared {
			m[k] = v
		}
	}
	return json.Marshal(m)
}

// antigravityConfigSchema is the config schema Antigravity models advertise and
// enforce. Declared keys are type-checked; additionalProperties is true because
// the JS schema is `.passthrough()` and undeclared keys are forwarded to the
// interactions request. It is hand-written because the reflected schema of a
// struct would forbid the undeclared keys.
var antigravityConfigSchema = map[string]any{
	"type": "object",
	"properties": map[string]any{
		"apiKey":                map[string]any{"type": "string", "description": "Overrides the API key provided at plugin initialization."},
		"baseUrl":               map[string]any{"type": "string", "description": "Overrides the plugin-configured or default baseUrl."},
		"apiVersion":            map[string]any{"type": "string", "description": "Overrides the plugin-configured or default apiVersion."},
		"previousInteractionId": map[string]any{"type": "string", "description": "The ID of the previous interaction, if any."},
		"store":                 map[string]any{"type": "boolean", "description": "Whether to store the response and request for later retrieval."},
		"environment": map[string]any{
			"oneOf": []any{
				map[string]any{"type": "string"},
				map[string]any{"type": "object", "additionalProperties": true},
			},
			"description": "The environment configuration for the sandbox: an environment ID or an inline configuration object.",
		},
		"responseModalities": map[string]any{
			"type":        "array",
			"items":       map[string]any{"type": "string", "enum": []any{"TEXT", "IMAGE"}},
			"description": "The modalities to be used in response.",
		},
		"model": map[string]any{"type": "string", "description": "The Gemini model the agent runs on. The server picks a default if not set."},
		"tools": map[string]any{
			"type":        "array",
			"items":       map[string]any{"type": "object", "additionalProperties": true},
			"description": "Built-in interactions tools, e.g. {\"type\": \"google_search\"} or {\"type\": \"code_execution\"}.",
		},
	},
	"additionalProperties": true,
}

// generateAntigravity executes a request against the Gemini interactions
// endpoint.
//
// Like the JS plugin, Antigravity is sync-only: the endpoint is called without
// server-sent streaming. When a stream callback is supplied the full response
// is emitted as a single chunk so streaming callers still receive the content.
func generateAntigravity(
	ctx context.Context,
	client *genai.Client,
	model string,
	input *ai.ModelRequest,
	cfg *AntigravityConfig,
	cb func(context.Context, *ai.ModelResponseChunk) error,
) (*ai.ModelResponse, error) {
	if model == "" {
		return nil, status.Errorf(status.ErrInvalidArgument, "model not provided")
	}
	if client.ClientConfig().Backend == genai.BackendVertexAI {
		return nil, errAntigravityVertex(model)
	}
	if cfg == nil {
		cfg = &AntigravityConfig{}
	}

	messages, previousID, environment := selectAntigravityContinuation(input.Messages, cfg)
	steps, err := toInteractionSteps(messages)
	if err != nil {
		return nil, err
	}
	if len(steps) == 0 {
		return nil, status.Errorf(status.ErrInvalidArgument, "antigravity request has no input to send")
	}

	tools, err := antigravityTools(cfg, input.Tools)
	if err != nil {
		return nil, err
	}

	body := map[string]any{}
	for k, v := range cfg.Extra {
		body[k] = v
	}
	if _, ok := body["agent"]; ok {
		return nil, status.Errorf(status.ErrInvalidArgument, "config key %q is set from the model name and cannot be overridden", "agent")
	}
	if _, ok := body["input"]; ok {
		return nil, status.Errorf(status.ErrInvalidArgument, "config key %q is built from the request messages and cannot be overridden", "input")
	}
	body["agent"] = model
	body["input"] = steps
	body["environment"] = environment
	if previousID != "" {
		body["previous_interaction_id"] = previousID
	}
	// Antigravity does not work with `store: false` by default on the server,
	// so store is only sent when the caller sets it.
	if cfg.Store != nil {
		body["store"] = *cfg.Store
	}
	if rf := antigravityResponseFormat(cfg.ResponseModalities); rf != nil {
		body["response_format"] = rf
	}
	if cfg.Model != "" {
		// A top-level `model` would conflict with `agent`.
		body["agent_config"] = map[string]any{"type": "antigravity", "model": cfg.Model}
	}
	if len(tools) > 0 {
		body["tools"] = tools
	}

	interaction, err := createInteraction(ctx, client, cfg, body)
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

// errAntigravityVertex is the error for Antigravity models requested through a
// Vertex AI client. The interactions endpoint is a Google AI (Gemini API)
// feature; the JS plugin registers Antigravity under googleai only.
func errAntigravityVertex(model string) error {
	return status.Errorf(status.ErrInvalidArgument,
		"model %q is only available through the Google AI plugin (Gemini API), not Vertex AI", model)
}

// selectAntigravityContinuation decides which messages to send and which
// interaction and environment to continue, mirroring the JS plugin and the
// Python select_antigravity_continuation.
//
// Antigravity keeps conversation state on the server and rejects function_call
// and code-execution steps replayed in `input`, and a fresh sandbox would lose
// files from earlier turns. So it continues from the last model message that
// carries metadata["interactionId"], sends only the messages after it and
// reuses that message's environmentId, unless the caller pinned a specific
// environment. An explicit previousInteractionId in the config wins; history up
// to the last model message is then assumed to be on the server already.
func selectAntigravityContinuation(messages []*ai.Message, cfg *AntigravityConfig) (newMessages []*ai.Message, previousID string, environment any) {
	newMessages = messages
	previousID = cfg.PreviousInteractionID
	environment = cfg.Environment
	if environment == nil {
		environment = antigravityDefaultEnvironment()
	}

	lastModel := -1
	for i := len(messages) - 1; i >= 0; i-- {
		if messages[i].Role != ai.RoleModel {
			continue
		}
		if previousID != "" {
			lastModel = i
			break
		}
		id, _ := messages[i].Metadata["interactionId"].(string)
		if id == "" {
			continue
		}
		lastModel = i
		previousID = id
		if envID, _ := messages[i].Metadata["environmentId"].(string); envID != "" && isDefaultAntigravityEnvironment(cfg.Environment) {
			environment = envID
		}
		break
	}
	if lastModel >= 0 {
		newMessages = messages[lastModel+1:]
	}
	return newMessages, previousID, environment
}

// isDefaultAntigravityEnvironment reports whether env is unset or one of the
// "new remote sandbox" spellings, which a previous turn's environment ID may
// replace.
func isDefaultAntigravityEnvironment(env any) bool {
	switch e := env.(type) {
	case nil:
		return true
	case string:
		return e == "remote"
	case map[string]any:
		t, _ := e["type"].(string)
		return len(e) == 1 && t == "remote"
	}
	return false
}

// antigravityResponseFormat builds the interactions response_format from the
// requested modalities: a single object for one modality, a list for several,
// nil for none (the server default). The deprecated top-level
// response_modalities field is not used.
func antigravityResponseFormat(modalities []string) any {
	var formats []any
	seen := map[string]bool{}
	for _, m := range modalities {
		m = strings.ToLower(m)
		if seen[m] {
			continue
		}
		seen[m] = true
		formats = append(formats, map[string]any{"type": m})
	}
	switch len(formats) {
	case 0:
		return nil
	case 1:
		return formats[0]
	default:
		return formats
	}
}

// antigravityTools builds the interactions tool list: built-in tools from the
// config first, then the request's function tools.
func antigravityTools(cfg *AntigravityConfig, defs []*ai.ToolDefinition) ([]any, error) {
	var tools []any
	for _, t := range cfg.Tools {
		tool, err := antigravityConfigTool(t)
		if err != nil {
			return nil, err
		}
		tools = append(tools, tool)
	}
	for _, def := range defs {
		fn := map[string]any{"type": "function", "name": def.Name}
		if def.Description != "" {
			fn["description"] = def.Description
		}
		if def.InputSchema != nil {
			fn["parameters"] = cleanInteractionSchema(def.InputSchema)
		}
		tools = append(tools, fn)
	}
	return tools, nil
}

// antigravityToolShorthands maps the camelCase/snake_case shorthand keys of a
// config tool to the interactions tool type.
var antigravityToolShorthands = map[string]string{
	"googleSearch":   "google_search",
	"google_search":  "google_search",
	"codeExecution":  "code_execution",
	"code_execution": "code_execution",
	"urlContext":     "url_context",
	"url_context":    "url_context",
}

// antigravityConfigTool converts one entry of config.tools. An entry with a
// "type" is already in wire form and is sent as-is; otherwise a shorthand key
// such as {"googleSearch": {}} is expanded.
func antigravityConfigTool(t map[string]any) (map[string]any, error) {
	if _, ok := t["type"]; ok {
		return t, nil
	}
	for key, typ := range antigravityToolShorthands {
		v, ok := t[key]
		if !ok {
			continue
		}
		out := map[string]any{"type": typ}
		if opts, ok := v.(map[string]any); ok {
			for k, val := range opts {
				out[k] = val
			}
		}
		return out, nil
	}
	return nil, status.Errorf(status.ErrInvalidArgument, "config.tools entry must have a \"type\" (e.g. \"google_search\") or a googleSearch/codeExecution/urlContext key: %v", t)
}

// cleanInteractionSchema removes the JSON Schema keywords the interactions API
// does not accept ($schema, additionalProperties) and collapses nullable type
// arrays to their first non-null type, as the JS cleanSchema does.
func cleanInteractionSchema(v any) any {
	switch s := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(s))
		for k, val := range s {
			if k == "$schema" || k == "additionalProperties" {
				continue
			}
			if k == "type" {
				if types, ok := val.([]any); ok {
					for _, t := range types {
						if t != "null" {
							out[k] = t
							break
						}
					}
					continue
				}
			}
			out[k] = cleanInteractionSchema(val)
		}
		return out
	case []any:
		out := make([]any, len(s))
		for i, val := range s {
			out[i] = cleanInteractionSchema(val)
		}
		return out
	default:
		return v
	}
}

// createInteraction issues a POST to the Gemini interactions endpoint. The
// genai SDK has no binding for this endpoint, so the call is made directly,
// reusing the client's API key, base URL, API version, HTTP client, timeout and
// headers. Per-request apiKey/baseUrl/apiVersion in cfg take precedence.
// Mirrors the JS createInteraction in client.ts.
func createInteraction(ctx context.Context, client *genai.Client, cfg *AntigravityConfig, body map[string]any) (*geminiInteraction, error) {
	cc := client.ClientConfig()

	baseURL := firstNonEmpty(cfg.BaseURL, cc.HTTPOptions.BaseURL, defaultGoogleAIBaseURL)
	apiVersion := firstNonEmpty(cfg.APIVersion, cc.HTTPOptions.APIVersion, defaultGoogleAIAPIVersion)
	apiKey := firstNonEmpty(cfg.APIKey, cc.APIKey)
	url := fmt.Sprintf("%s/%s/interactions", strings.TrimSuffix(baseURL, "/"), apiVersion)

	payload, err := json.Marshal(body)
	if err != nil {
		return nil, status.Errorf(status.ErrInternal, "failed to encode interaction request: %w", err)
	}

	if t := cc.HTTPOptions.Timeout; t != nil && *t > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, *t)
		defer cancel()
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(payload))
	if err != nil {
		return nil, status.Errorf(status.ErrInternal, "failed to build interaction request: %w", err)
	}
	for k, vs := range cc.HTTPOptions.Headers {
		for _, v := range vs {
			req.Header.Add(k, v)
		}
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-goog-api-client", fmt.Sprintf("genkit-go/%s", internal.Version))
	req.Header.Set("Api-Revision", antigravityAPIRevision)
	if apiKey != "" {
		req.Header.Set("x-goog-api-key", apiKey)
	}

	httpClient := cc.HTTPClient
	if httpClient == nil {
		httpClient = &http.Client{}
	}

	httpResp, err := httpClient.Do(req)
	if err != nil {
		return nil, interactionTransportError(ctx, url, err)
	}
	defer httpResp.Body.Close()

	// Read one byte past the cap so an oversized body is detected, not
	// silently truncated into invalid JSON.
	data, err := io.ReadAll(io.LimitReader(httpResp.Body, maxInteractionResponseBytes+1))
	if err != nil {
		return nil, interactionTransportError(ctx, url, err)
	}
	if len(data) > maxInteractionResponseBytes {
		return nil, status.Errorf(status.ErrInternal, "interaction response from %s exceeds the %d byte limit", url, maxInteractionResponseBytes)
	}

	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		return nil, interactionHTTPError(url, httpResp.StatusCode, data)
	}

	var interaction geminiInteraction
	if err := json.Unmarshal(data, &interaction); err != nil {
		return nil, status.Errorf(status.ErrInternal, "failed to decode interaction response from %s (%w): %s", url, err, excerpt(data))
	}
	return &interaction, nil
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}

// excerpt returns a short, single-line excerpt of a response body for errors.
func excerpt(data []byte) string {
	const max = 200
	s := strings.Join(strings.Fields(string(data)), " ")
	if len(s) > max {
		s = s[:max] + "..."
	}
	return s
}

// interactionTransportError maps a failure to send the request or read the
// response. Context cancellation and deadlines keep their cause (so
// errors.Is(err, context.Canceled) works) and map to the matching status; other
// timeouts are DEADLINE_EXCEEDED and anything else is UNAVAILABLE.
func interactionTransportError(ctx context.Context, url string, err error) error {
	switch {
	case errors.Is(err, context.Canceled), errors.Is(ctx.Err(), context.Canceled):
		return status.Errorf(status.ErrCancelled, "interaction request to %s cancelled: %w", url, context.Canceled)
	case errors.Is(err, context.DeadlineExceeded), errors.Is(ctx.Err(), context.DeadlineExceeded):
		return status.Errorf(status.ErrDeadlineExceeded, "interaction request to %s timed out: %w", url, context.DeadlineExceeded)
	}
	var ne net.Error
	if errors.As(err, &ne) && ne.Timeout() {
		return status.Errorf(status.ErrDeadlineExceeded, "interaction request to %s timed out: %w", url, err)
	}
	return status.Errorf(status.ErrUnavailable, "interaction request to %s failed: %w", url, err)
}

// interactionHTTPError maps a non-2xx interactions response to a Genkit error
// through the plugin's [wrapAPIError], so the status and any
// google.rpc.RetryInfo delay are reported as for the other models. The body is
// either Google-style ({"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
// "details": [...]}}) or the Interactions style ({"error": {"code":
// "invalid_request", "message": ...}}); a body that is neither is quoted.
func interactionHTTPError(url string, httpStatus int, body []byte) error {
	apiErr := genai.APIError{Code: httpStatus}

	var parsed struct {
		Error struct {
			Code    any              `json:"code"`
			Message string           `json:"message"`
			Status  string           `json:"status"`
			Details []map[string]any `json:"details"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &parsed); err == nil && (parsed.Error.Message != "" || parsed.Error.Code != nil || parsed.Error.Status != "") {
		apiErr.Message = parsed.Error.Message
		apiErr.Details = parsed.Error.Details
		if status.Name(parsed.Error.Status).IsValid() {
			apiErr.Status = parsed.Error.Status
		} else if n, ok := interactionErrorCodeToStatus(parsed.Error.Code); ok {
			apiErr.Status = string(n)
		}
	}
	if apiErr.Message == "" {
		apiErr.Message = excerpt(body)
	}
	return wrapAPIError(fmt.Errorf("interaction request to %s: %w", url, apiErr))
}

// interactionContentBlockCodes are the Interactions error codes that report a
// content-policy block. They map to FAILED_PRECONDITION when thrown (the same
// input would be blocked again) and to a blocked response when reported on a
// failed interaction. Mirrors the JS INTERACTION_CONTENT_BLOCK_CODES.
var interactionContentBlockCodes = map[string]bool{
	"safety": true, "recitation": true, "language": true, "prohibited_content": true,
	"spii": true, "blocklist": true, "image_safety": true, "image_prohibited_content": true,
	"image_recitation": true, "image_other": true, "content_blocked": true,
	"jailbreak": true, "model_armor": true,
}

// interactionErrorCodeStatus maps Interactions error codes to Genkit statuses,
// mirroring the JS INTERACTION_ERROR_CODE_TO_STATUS.
var interactionErrorCodeStatus = map[string]status.Name{
	// Generation failures after the server's own retries: the server advises
	// retrying, so ABORTED (retried by the retry middleware by default).
	"malformed_function_call": status.Aborted,
	"unexpected_tool_call":    status.Aborted,
	"no_image":                status.Aborted,
	"service_unavailable":     status.Unavailable,
	"rate_limit_exceeded":     status.ResourceExhausted,
	"quota_exceeded":          status.ResourceExhausted,
	"invalid_request":         status.InvalidArgument,
	"parameter_unknown":       status.InvalidArgument,
	"failed_precondition":     status.FailedPrecondition,
	"out_of_range":            status.OutOfRange,
	// Deterministic for a given request, so deliberately not RESOURCE_EXHAUSTED
	// (which is retryable).
	"agent_max_token_limit": status.InvalidArgument,
	"model_not_found":       status.NotFound,
	"not_found":             status.NotFound,
	"authentication":        status.Unauthenticated,
	"permission_denied":     status.PermissionDenied,
	"already_exists":        status.AlreadyExists,
	"aborted":               status.Aborted,
	"cancelled":             status.Cancelled,
	"deadline_exceeded":     status.DeadlineExceeded,
	"api_error":             status.Internal,
	"unimplemented":         status.Unimplemented,
}

func isInteractionContentBlockCode(code string) bool {
	return interactionContentBlockCodes[strings.ToLower(strings.TrimSpace(code))]
}

// interactionErrorCodeToStatus maps an Interactions error code (a string, or a
// numeric HTTP code) to a Genkit status. The second result is false when the
// code is unknown.
func interactionErrorCodeToStatus(code any) (status.Name, bool) {
	switch c := code.(type) {
	case float64:
		if n := status.FromHTTPCode(int(c)); n != status.Unknown {
			return n, true
		}
	case string:
		norm := strings.ToLower(strings.TrimSpace(c))
		if norm == "" {
			return "", false
		}
		if n, ok := interactionErrorCodeStatus[norm]; ok {
			return n, true
		}
		if isInteractionContentBlockCode(norm) {
			return status.FailedPrecondition, true
		}
		var num int
		if _, err := fmt.Sscanf(norm, "%d", &num); err == nil && fmt.Sprint(num) == norm {
			return interactionErrorCodeToStatus(float64(num))
		}
	}
	return "", false
}

// toInteractionSteps converts Genkit messages into interaction steps, mirroring
// the JS toInteractionSteps. Tool requests/responses and reasoning become their
// own steps; consecutive text/media parts are grouped into a model_output (for
// model messages) or user_input step. Steps are emitted in the parts' original
// order: a tool or thought step ends the current group, so text before and
// after it stays on either side.
//
// Server-side steps the model produced (search, code execution, server
// function calls and their results) are not replayed: Antigravity keeps them
// in the stored interaction and rejects them in `input`. Unsupported parts and
// media return an error rather than being dropped.
func toInteractionSteps(messages []*ai.Message) ([]any, error) {
	steps := []any{}

	// Auto-IDs assigned to tool requests that lack a Ref, consumed in order by
	// subsequent tool responses (mirrors ensureToolIds).
	var generatedIDs []string
	idCounter := 0

	for mi, message := range messages {
		var normal []map[string]any
		flush := func() {
			if len(normal) == 0 {
				return
			}
			stepType := "user_input"
			if message.Role == ai.RoleModel {
				stepType = "model_output"
			}
			steps = append(steps, map[string]any{"type": stepType, "content": normal})
			normal = nil
		}

		for pi, part := range message.Content {
			switch {
			case part.IsToolRequest():
				flush()
				ref := part.ToolRequest.Ref
				if ref == "" {
					ref = fmt.Sprintf("genkit-auto-id-%d", idCounter)
					idCounter++
					generatedIDs = append(generatedIDs, ref)
				}
				call := map[string]any{
					"type":      "function_call",
					"name":      part.ToolRequest.Name,
					"arguments": toolArguments(part.ToolRequest.Input),
					"id":        ref,
				}
				if sig := metadataSignature(part.Metadata); len(sig) > 0 {
					call["signature"] = string(sig)
				}
				steps = append(steps, call)
			case part.IsToolResponse():
				flush()
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
				flush()
				thought := map[string]any{
					"type":    "thought",
					"summary": []map[string]any{{"type": "text", "text": part.Text}},
				}
				if sig := metadataSignature(part.Metadata); len(sig) > 0 {
					thought["signature"] = string(sig)
				}
				steps = append(steps, thought)
			case part.IsText():
				normal = append(normal, map[string]any{"type": "text", "text": part.Text})
			case part.IsMedia():
				content, err := toInteractionMedia(part)
				if err != nil {
					return nil, status.Errorf(status.ErrInvalidArgument, "message %d part %d: %w", mi, pi, err)
				}
				normal = append(normal, content)
			case part.IsCustom() && isServerSidePart(part):
				// Owned by the stored interaction; see the function comment.
			default:
				return nil, status.Errorf(status.ErrInvalidArgument, "message %d part %d: unsupported part kind %v for the interactions endpoint", mi, pi, part.Kind)
			}
		}
		flush()
	}

	return steps, nil
}

// isServerSidePart reports whether a custom part records a step the server ran
// (as produced by fromInteractionStep) rather than anything the caller added.
func isServerSidePart(part *ai.Part) bool {
	m := part.Custom
	for _, k := range []string{
		"googleSearchCall", "googleSearchResult", "executableCode",
		"codeExecutionResult", "serverFunctionCall", "serverFunctionResult",
		"unknownStep", "unknownContent",
	} {
		if _, ok := m[k]; ok {
			return true
		}
	}
	return false
}

// toInteractionMedia converts a media part into an image/audio/video/document
// content block. The MIME type comes from the part's ContentType, else from a
// data: URL. Like the JS plugin, any type that is not image, audio or video is
// sent as a document and the server decides whether it accepts it; a part with
// no usable data or type is an error.
func toInteractionMedia(part *ai.Part) (map[string]any, error) {
	url := part.Text
	if url == "" {
		return nil, errors.New("media part has no url")
	}
	contentType := part.ContentType
	isData := strings.HasPrefix(url, "data:")

	out := map[string]any{}
	if isData {
		header, data, ok := strings.Cut(strings.TrimPrefix(url, "data:"), ",")
		if !ok {
			return nil, errors.New("malformed data URL: missing comma")
		}
		mime, params, _ := strings.Cut(header, ";")
		if !strings.Contains(";"+params+";", ";base64;") {
			return nil, errors.New("data URL must be base64-encoded")
		}
		if contentType == "" {
			contentType = mime
		}
		if contentType == "" {
			return nil, errors.New("media part has no content type")
		}
		out["data"] = data
	} else {
		out["uri"] = url
		if contentType == "" {
			// A URI whose type can't be determined (a YouTube link or a Files
			// API URI): the server works the type out itself, as on the
			// generateContent path.
			if isYouTubeURL(url) {
				out["type"] = "video"
			} else {
				out["type"] = "document"
			}
			return out, nil
		}
	}
	out["mime_type"] = contentType

	switch {
	case strings.HasPrefix(contentType, "image/"):
		out["type"] = "image"
	case strings.HasPrefix(contentType, "audio/"):
		out["type"] = "audio"
	case strings.HasPrefix(contentType, "video/"):
		out["type"] = "video"
	default:
		out["type"] = "document"
	}
	return out, nil
}

func isYouTubeURL(url string) bool {
	for _, p := range []string{"https://www.youtube.com/", "https://youtube.com/", "https://youtu.be/", "http://www.youtube.com/", "http://youtube.com/", "http://youtu.be/"} {
		if strings.HasPrefix(url, p) {
			return true
		}
	}
	return false
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
	case map[string]any, string:
		return output
	case nil:
		return map[string]any{}
	default:
		return map[string]any{"result": output}
	}
}

// interactionTerminalStatuses lists the statuses of a finished interaction.
// requires_action is also handled (pending client function calls), and an
// empty status is treated as completed.

// fromInteractionSync converts an interaction response into a ModelResponse,
// mirroring the JS fromInteractionSync.
//
//   - completed (or unset): the steps become parts, finish reason "stop".
//   - requires_action: client function calls the server is waiting on become
//     tool requests; the generate loop runs them and calls again.
//   - incomplete: token/step limit reached; finish reason "length" with the
//     partial content.
//   - failed: a content-policy block returns a response with finish reason
//     "blocked"; any other failure is an error carrying the server's code and
//     message, with the status mapped from the code.
//   - cancelled: finish reason "aborted", no content.
//   - in_progress, queued or anything else: the call is synchronous, so the
//     interaction should be finished. Partial content is returned with finish
//     reason "other"; with no content at all it is an error rather than an
//     empty "stop" message.
func fromInteractionSync(interaction *geminiInteraction) (*ai.ModelResponse, error) {
	if interaction.Status == "failed" {
		return fromFailedInteraction(interaction)
	}

	msg := &ai.Message{Role: ai.RoleModel, Metadata: interactionMessageMetadata(interaction)}
	resp := &ai.ModelResponse{
		FinishReason: ai.FinishReasonStop,
		Message:      msg,
		Custom:       interaction,
		Raw:          interaction,
	}
	if interaction.Usage != nil {
		resp.Usage = fromInteractionUsage(interaction.Usage)
	}

	if interaction.Status == "cancelled" {
		resp.FinishReason = ai.FinishReasonAborted
		resp.FinishMessage = "Operation cancelled"
		return resp, nil
	}

	pending := pendingFunctionCallIDs(interaction.Steps, interaction.Status)
	for _, step := range interaction.Steps {
		isPending := step.Type == "function_call" && step.ID != "" && pending[step.ID]
		for _, p := range fromInteractionStep(step, isPending) {
			if p != nil {
				msg.Content = append(msg.Content, p)
			}
		}
	}

	switch interaction.Status {
	case "", "completed", "requires_action":
	case "incomplete":
		resp.FinishReason = ai.FinishReasonLength
		resp.FinishMessage = "Interaction incomplete (truncated output)"
	default:
		if len(msg.Content) == 0 {
			return nil, status.Errorf(status.ErrFailedPrecondition,
				"interaction %q returned status %q with no content; background interactions are not supported", interaction.ID, interaction.Status)
		}
		resp.FinishReason = ai.FinishReasonOther
		resp.FinishMessage = fmt.Sprintf("Interaction not finished (status: %s)", interaction.Status)
	}

	return resp, nil
}

// interactionMessageMetadata carries the interaction and environment IDs on the
// model message so a later turn can continue them (see
// selectAntigravityContinuation).
func interactionMessageMetadata(interaction *geminiInteraction) map[string]any {
	if interaction.ID == "" && interaction.EnvironmentID == "" {
		return nil
	}
	md := map[string]any{}
	if interaction.ID != "" {
		md["interactionId"] = interaction.ID
	}
	if interaction.EnvironmentID != "" {
		md["environmentId"] = interaction.EnvironmentID
	}
	return md
}

// interactionFailureMessage builds a message from a failed interaction's errors.
func interactionFailureMessage(interaction *geminiInteraction) string {
	var details []string
	for _, e := range interaction.Errors {
		d := strings.TrimSpace(strings.Join([]string{bracketed(e.Code), e.Message}, " "))
		if d != "" {
			details = append(details, d)
		}
	}
	if len(details) == 0 {
		return "Interaction failed"
	}
	return "Interaction failed: " + strings.Join(details, "; ")
}

func bracketed(code string) string {
	if code == "" {
		return ""
	}
	return "[" + code + "]"
}

// fromFailedInteraction converts a status "failed" interaction; see
// fromInteractionSync.
func fromFailedInteraction(interaction *geminiInteraction) (*ai.ModelResponse, error) {
	var first interactionError
	if len(interaction.Errors) > 0 {
		first = interaction.Errors[0]
	}
	if isInteractionContentBlockCode(first.Code) {
		msg := first.Message
		if msg == "" {
			msg = interactionFailureMessage(interaction)
		}
		resp := &ai.ModelResponse{
			FinishReason:  ai.FinishReasonBlocked,
			FinishMessage: msg,
			Message:       &ai.Message{Role: ai.RoleModel, Metadata: interactionMessageMetadata(interaction)},
			Custom:        interaction,
			Raw:           interaction,
		}
		if interaction.Usage != nil {
			resp.Usage = fromInteractionUsage(interaction.Usage)
		}
		return resp, nil
	}
	name, ok := interactionErrorCodeToStatus(first.Code)
	if !ok {
		name = status.Unknown
	}
	return nil, status.Errorf(status.Base(name), "%s", interactionFailureMessage(interaction)).
		WithDetails(map[string]any{"interactionId": interaction.ID, "code": first.Code})
}

// pendingFunctionCallIDs returns the IDs of function_call steps that the server
// is waiting on the caller to run: those of a requires_action interaction that
// have no function_result. Every other function_call was run by the server (for
// example by a built-in tool), so Genkit's tool loop must not try to run it.
func pendingFunctionCallIDs(steps []interactionStep, interactionStatus string) map[string]bool {
	pending := map[string]bool{}
	if interactionStatus != "requires_action" {
		return pending
	}
	answered := map[string]bool{}
	for _, s := range steps {
		if s.Type == "function_result" && s.CallID != "" {
			answered[s.CallID] = true
		}
	}
	for _, s := range steps {
		if s.Type == "function_call" && s.ID != "" && !answered[s.ID] {
			pending[s.ID] = true
		}
	}
	return pending
}

// fromInteractionUsage converts interaction usage following the token
// convention of ai.GenerationUsage: InputTokens is the whole prompt (cached
// tokens included; tool-use tokens are input too, as for generateContent),
// OutputTokens and ThoughtsTokens do not overlap, and TotalTokens is the
// server's. Per-modality token counts go in Custom (inputTextTokens,
// outputImageTokens, ...), not in the character and media-count fields, which
// the framework fills from the request and response.
func fromInteractionUsage(u *interactionUsage) *ai.GenerationUsage {
	usage := &ai.GenerationUsage{
		InputTokens:         u.TotalInputTokens + u.TotalToolUseTokens,
		OutputTokens:        u.TotalOutputTokens,
		ThoughtsTokens:      u.TotalThoughtTokens,
		CachedContentTokens: u.TotalCachedTokens,
		TotalTokens:         u.TotalTokens,
	}
	if usage.TotalTokens == 0 {
		usage.TotalTokens = usage.InputTokens + usage.OutputTokens + usage.ThoughtsTokens
	}

	custom := map[string]float64{}
	addModality := func(direction string, breakdown []modalityTokens) {
		for _, mt := range breakdown {
			if mt.Modality == "" {
				continue
			}
			key := direction + strings.ToUpper(mt.Modality[:1]) + mt.Modality[1:] + "Tokens"
			custom[key] += float64(mt.Tokens)
		}
	}
	addModality("input", u.InputTokensByModality)
	addModality("output", u.OutputTokensByModality)
	if u.TotalToolUseTokens > 0 {
		custom["toolUseTokens"] = float64(u.TotalToolUseTokens)
	}
	for _, g := range u.GroundingToolCount {
		if g.Type != "" {
			custom[snakeToCamel(g.Type)+"Count"] += float64(g.Count)
		}
	}
	if len(custom) > 0 {
		usage.Custom = custom
	}
	return usage
}

// snakeToCamel converts google_search to googleSearch.
func snakeToCamel(s string) string {
	parts := strings.Split(s, "_")
	for i := 1; i < len(parts); i++ {
		if parts[i] != "" {
			parts[i] = strings.ToUpper(parts[i][:1]) + parts[i][1:]
		}
	}
	return strings.Join(parts, "")
}

// fromInteractionStep converts one interaction step into Genkit parts. Steps
// that carry no model-visible content (e.g. user_input echoes) return nil, and
// unrecognised steps are preserved losslessly as a custom part.
func fromInteractionStep(step interactionStep, isPending bool) []*ai.Part {
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
		if !isPending {
			// Run by the server: record it, but not as a tool request, so the
			// tool loop does not try to run a tool it may not have.
			p := ai.NewCustomPart(map[string]any{
				"serverFunctionCall": map[string]any{"id": step.ID, "name": step.Name, "arguments": step.Arguments},
			})
			setPartMeta(p, "signature", signatureBytes(step.Signature))
			return []*ai.Part{p}
		}
		p := ai.NewToolRequestPart(&ai.ToolRequest{
			Name:  step.Name,
			Input: step.Arguments,
			Ref:   step.ID,
		})
		// Keep the signature so it is sent back with the call.
		setPartMeta(p, "signature", signatureBytes(step.Signature))
		return []*ai.Part{p}
	case "function_result":
		return []*ai.Part{ai.NewCustomPart(map[string]any{
			"serverFunctionResult": map[string]any{"callId": step.CallID, "name": step.Name, "result": step.Result, "isError": step.IsError},
		})}
	case "google_search_call":
		p := ai.NewCustomPart(map[string]any{
			"googleSearchCall": map[string]any{"id": step.ID, "arguments": step.Arguments},
		})
		setPartMeta(p, "signature", signatureBytes(step.Signature))
		return []*ai.Part{p}
	case "google_search_result":
		p := ai.NewCustomPart(map[string]any{
			"googleSearchResult": map[string]any{"callId": step.CallID, "result": step.Result},
		})
		setPartMeta(p, "signature", signatureBytes(step.Signature))
		return []*ai.Part{p}
	case "code_execution_call":
		code, _ := step.Arguments["code"].(string)
		language, _ := step.Arguments["language"].(string)
		if language == "" {
			language = "PYTHON"
		}
		p := newExecutableCodePart(language, code)
		setPartMeta(p, "callId", step.ID)
		setPartMeta(p, "signature", signatureBytes(step.Signature))
		return []*ai.Part{p}
	case "code_execution_result":
		p := newCodeExecutionResultPart("OUTCOME_OK", resultToString(step.Result))
		setPartMeta(p, "callId", step.CallID)
		setPartMeta(p, "signature", signatureBytes(step.Signature))
		return []*ai.Part{p}
	default:
		return []*ai.Part{ai.NewCustomPart(map[string]any{"unknownStep": step})}
	}
}

// setPartMeta sets a metadata key on a part, allocating the map on first use and
// skipping empty values.
func setPartMeta(p *ai.Part, key string, val any) {
	switch v := val.(type) {
	case nil:
		return
	case string:
		if v == "" {
			return
		}
	case []byte:
		if len(v) == 0 {
			return
		}
	}
	if p.Metadata == nil {
		p.Metadata = map[string]any{}
	}
	p.Metadata[key] = val
}

// resultToString renders an interaction step result (string or object) as a
// string for the code-execution-result custom part.
func resultToString(v any) string {
	if v == nil {
		return ""
	}
	if s, ok := v.(string); ok {
		return s
	}
	b, err := json.Marshal(v)
	if err != nil {
		return fmt.Sprintf("%v", v)
	}
	return string(b)
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

// signatureBytes converts a step's opaque signature to the []byte form Genkit
// parts carry in Metadata["signature"] (nil when empty).
func signatureBytes(sig string) []byte {
	if sig == "" {
		return nil
	}
	return []byte(sig)
}
