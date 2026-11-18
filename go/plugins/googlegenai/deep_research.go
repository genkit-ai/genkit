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
	"net/url"
	"regexp"
	"strings"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal/base"
	"google.golang.org/genai"
)

const (
	deepResearchAgentType   = "deep-research"
	deepResearchAPIRevision = "2026-05-20"

	// deepResearchMaxResponseBytes caps how much of an interactions response is read.
	deepResearchMaxResponseBytes = 64 << 20 // 64 MiB
)

var (
	// deepResearchOperationIDPattern matches the interaction IDs the service
	// issues. Operations are caller-supplied, so the ID is validated before it
	// is placed in a request path.
	deepResearchOperationIDPattern = regexp.MustCompile(`^[A-Za-z0-9._~-]{1,512}$`)
	// deepResearchAPIVersionPattern matches a plausible API version ("v1beta").
	deepResearchAPIVersionPattern = regexp.MustCompile(`^v[0-9a-z]+$`)
)

// DeepResearchConfig configures Google AI Deep Research interactions.
type DeepResearchConfig struct {
	// APIKey overrides the client's API key for the start request only. Check
	// and cancel calls use the plugin client's key, because the key is never
	// stored in the operation; an operation started with a per-request key
	// must therefore be polled by a client that can also access it.
	APIKey                string                  `json:"apiKey,omitempty"`
	BaseURL               string                  `json:"baseUrl,omitempty"`
	APIVersion            string                  `json:"apiVersion,omitempty"`
	ThinkingSummaries     string                  `json:"thinkingSummaries,omitempty"`
	PreviousInteractionID string                  `json:"previousInteractionId,omitempty"`
	Store                 *bool                   `json:"store,omitempty"`
	ResponseModalities    []string                `json:"responseModalities,omitempty"`
	Visualization         string                  `json:"visualization,omitempty"`
	CollaborativePlanning *bool                   `json:"collaborativePlanning,omitempty"`
	GoogleSearch          any                     `json:"googleSearch,omitempty"`
	URLContext            any                     `json:"urlContext,omitempty"`
	CodeExecution         any                     `json:"codeExecution,omitempty"`
	FileSearch            *DeepResearchFileSearch `json:"fileSearch,omitempty"`
	MCPServers            []DeepResearchMCPServer `json:"mcpServers,omitempty"`
}

// DeepResearchFileSearch configures file-search stores for Deep Research.
type DeepResearchFileSearch struct {
	FileSearchStoreNames []string       `json:"fileSearchStoreNames,omitempty"`
	Extra                map[string]any `json:"-"`
}

// DeepResearchMCPServer configures a remote MCP server for Deep Research.
type DeepResearchMCPServer struct {
	Name         string            `json:"name,omitempty"`
	URL          string            `json:"url,omitempty"`
	Headers      map[string]string `json:"headers,omitempty"`
	AllowedTools []string          `json:"allowedTools,omitempty"`
}

type deepResearchClientOptions struct {
	APIKey     string `json:"apiKey,omitempty"`
	BaseURL    string `json:"baseUrl,omitempty"`
	APIVersion string `json:"apiVersion,omitempty"`
}

type interactionRequest struct {
	PreviousInteractionID string           `json:"previous_interaction_id,omitempty"`
	Agent                 string           `json:"agent,omitempty"`
	Input                 any              `json:"input"`
	Tools                 []map[string]any `json:"tools,omitempty"`
	ResponseFormat        any              `json:"response_format,omitempty"`
	Store                 *bool            `json:"store,omitempty"`
	Background            bool             `json:"background,omitempty"`
	AgentConfig           map[string]any   `json:"agent_config,omitempty"`
}

type geminiInteraction struct {
	Model                 string             `json:"model,omitempty"`
	Agent                 string             `json:"agent,omitempty"`
	EnvironmentID         string             `json:"environment_id,omitempty"`
	ID                    string             `json:"id,omitempty"`
	PreviousInteractionID string             `json:"previous_interaction_id,omitempty"`
	Status                string             `json:"status,omitempty"`
	Created               string             `json:"created,omitempty"`
	Updated               string             `json:"updated,omitempty"`
	Role                  string             `json:"role,omitempty"`
	Steps                 []interactionStep  `json:"steps,omitempty"`
	Usage                 *interactionUsage  `json:"usage,omitempty"`
	Error                 *interactionError  `json:"error,omitempty"`
	Errors                []interactionError `json:"errors,omitempty"`
}

type interactionError struct {
	Code    string `json:"code,omitempty"`
	Status  string `json:"status,omitempty"`
	Message string `json:"message,omitempty"`
}

type interactionStep struct {
	Type      string               `json:"type,omitempty"`
	Content   []interactionContent `json:"content,omitempty"`
	Text      string               `json:"text,omitempty"`
	Data      string               `json:"data,omitempty"`
	URI       string               `json:"uri,omitempty"`
	MIMEType  string               `json:"mime_type,omitempty"`
	Name      string               `json:"name,omitempty"`
	Arguments map[string]any       `json:"arguments,omitempty"`
	ID        string               `json:"id,omitempty"`
	Result    any                  `json:"result,omitempty"`
	CallID    string               `json:"call_id,omitempty"`
	Summary   []interactionContent `json:"summary,omitempty"`
	Signature string               `json:"signature,omitempty"`
}

type interactionContent struct {
	Type        string               `json:"type,omitempty"`
	Text        string               `json:"text,omitempty"`
	Data        string               `json:"data,omitempty"`
	URI         string               `json:"uri,omitempty"`
	MIMEType    string               `json:"mime_type,omitempty"`
	Name        string               `json:"name,omitempty"`
	Arguments   map[string]any       `json:"arguments,omitempty"`
	ID          string               `json:"id,omitempty"`
	Result      any                  `json:"result,omitempty"`
	CallID      string               `json:"call_id,omitempty"`
	Summary     []interactionContent `json:"summary,omitempty"`
	Signature   string               `json:"signature,omitempty"`
	Annotations []map[string]any     `json:"annotations,omitempty"`
}

type interactionUsage struct {
	TotalInputTokens       int              `json:"total_input_tokens,omitempty"`
	TotalOutputTokens      int              `json:"total_output_tokens,omitempty"`
	TotalTokens            int              `json:"total_tokens,omitempty"`
	TotalCachedTokens      int              `json:"total_cached_tokens,omitempty"`
	TotalThoughtTokens     int              `json:"total_thought_tokens,omitempty"`
	InputTokensByModality  []modalityTokens `json:"input_tokens_by_modality,omitempty"`
	OutputTokensByModality []modalityTokens `json:"output_tokens_by_modality,omitempty"`
}

type modalityTokens struct {
	Modality string `json:"modality,omitempty"`
	Tokens   int    `json:"tokens,omitempty"`
}

// newDeepResearchModel defines a Deep Research background model using the Google AI interactions API.
//
// Deep Research is a Gemini Developer API feature: it is served only from the
// generativelanguage endpoint and is not exposed on Vertex AI, so the model is
// always registered under the Google AI provider (matching the JS plugin).
func newDeepResearchModel(client *genai.Client, name string, info ai.ModelOptions) *ai.BackgroundModelAction {
	provider := googleAIProvider

	startFunc := func(ctx context.Context, req *ai.ModelRequest, cfg DeepResearchConfig) (*ai.ModelOperation, error) {
		config := &cfg
		clientOptions := deepResearchOptionsFromConfig(client, config)

		interactionReq, err := toDeepResearchInteractionRequest(name, req, config)
		if err != nil {
			return nil, err
		}

		interaction, err := doDeepResearchRequest(ctx, client, clientOptions, http.MethodPost, "interactions", interactionReq)
		if err != nil {
			return nil, err
		}

		op := fromDeepResearchInteraction(interaction)
		if op.Metadata == nil {
			op.Metadata = make(map[string]any)
		}
		// Persist only the API version for later check/cancel calls. The API key
		// and base URL are deliberately omitted: operation metadata is serialized
		// back to the caller, who may hand it back altered, so both are
		// re-derived from the live client rather than trusted from the operation.
		op.Metadata["clientOptions"] = deepResearchClientOptions{APIVersion: clientOptions.APIVersion}
		// RFC3339 string so it survives a JSON round trip.
		op.Metadata["startTime"] = time.Now().UTC().Format(time.RFC3339Nano)
		return op, nil
	}

	checkFunc := func(ctx context.Context, op *ai.ModelOperation) (*ai.ModelOperation, error) {
		if op == nil {
			return nil, core.NewPublicError(core.INVALID_ARGUMENT, "Deep Research operation is nil", nil)
		}
		if err := validateDeepResearchOperationID(op.ID); err != nil {
			return nil, err
		}
		clientOptions := deepResearchOptionsFromOperation(client, op)
		interaction, err := doDeepResearchRequest(ctx, client, clientOptions, http.MethodGet, "interactions/"+url.PathEscape(op.ID), nil)
		if err != nil {
			return nil, err
		}

		updatedOp := fromDeepResearchInteraction(interaction)
		restoreDeepResearchMetadata(updatedOp, op)
		if updatedOp.Done && updatedOp.Output != nil {
			// Token usage comes from the interaction's own usage block; this only
			// backfills any output char/media counts the API left unset.
			ai.CalculateInputOutputUsage(nil, updatedOp.Output)
			if startTime, ok := deepResearchStartTime(updatedOp.Metadata["startTime"]); ok && updatedOp.Output.LatencyMs == 0 {
				updatedOp.Output.LatencyMs = float64(time.Since(startTime).Nanoseconds()) / 1e6
			}
		}
		return updatedOp, nil
	}

	cancelFunc := func(ctx context.Context, op *ai.ModelOperation) (*ai.ModelOperation, error) {
		if op == nil {
			return nil, core.NewPublicError(core.INVALID_ARGUMENT, "Deep Research operation is nil", nil)
		}
		if err := validateDeepResearchOperationID(op.ID); err != nil {
			return nil, err
		}
		clientOptions := deepResearchOptionsFromOperation(client, op)
		escapedID := url.PathEscape(op.ID)
		interaction, err := doDeepResearchRequest(ctx, client, clientOptions, http.MethodPost, "interactions/"+escapedID+"/cancel", nil)
		if err != nil {
			// The service reports an interaction that is already cancelled (or
			// is cancelling) as CANCELLED (HTTP 499). That is the outcome the
			// caller asked for, unless it is our own context that was cancelled.
			if ctx.Err() == nil && status.Of(err) == status.Cancelled {
				interaction = &geminiInteraction{ID: op.ID, Status: "cancelled"}
			} else {
				return nil, err
			}
		} else if interaction.ID == "" && interaction.Status == "" {
			// An empty 2xx body does not say what state the interaction is in.
			interaction, err = doDeepResearchRequest(ctx, client, clientOptions, http.MethodGet, "interactions/"+escapedID, nil)
			if err != nil {
				return nil, err
			}
		}

		updatedOp := fromDeepResearchInteraction(interaction)
		if updatedOp.ID == "" {
			updatedOp.ID = op.ID
		}
		restoreDeepResearchMetadata(updatedOp, op)
		return updatedOp, nil
	}

	return ai.NewBackgroundModelAction(
		api.NewName(provider, name),
		&ai.BackgroundModelOptions{ModelOptions: info, Cancel: cancelFunc},
		startFunc,
		checkFunc,
	)
}

func deepResearchOptionsFromConfig(client *genai.Client, config *DeepResearchConfig) deepResearchClientOptions {
	opts := deepResearchClientOptions{}
	if client != nil {
		clientConfig := client.ClientConfig()
		opts.APIKey = clientConfig.APIKey
		opts.BaseURL = clientConfig.HTTPOptions.BaseURL
		opts.APIVersion = clientConfig.HTTPOptions.APIVersion
	}
	if config != nil {
		if config.APIKey != "" {
			opts.APIKey = config.APIKey
		}
		if config.BaseURL != "" {
			opts.BaseURL = config.BaseURL
		}
		if config.APIVersion != "" {
			opts.APIVersion = config.APIVersion
		}
	}
	if opts.BaseURL == "" {
		opts.BaseURL = "https://generativelanguage.googleapis.com"
	}
	if opts.APIVersion == "" {
		opts.APIVersion = "v1beta"
	}
	return opts
}

func deepResearchOptionsFromOperation(client *genai.Client, op *ai.ModelOperation) deepResearchClientOptions {
	// The API key always comes from the live client, never from operation
	// metadata (which is serialized and persisted). Only non-secret routing
	// pinned at start time is restored from metadata.
	opts := deepResearchOptionsFromConfig(client, nil)
	if op == nil || op.Metadata == nil {
		return opts
	}

	var stored deepResearchClientOptions
	switch raw := op.Metadata["clientOptions"].(type) {
	case deepResearchClientOptions:
		stored = raw
	case map[string]any:
		if s, err := base.MapToStruct[deepResearchClientOptions](raw); err == nil {
			stored = s
		}
	}
	// The base URL is never read from the operation: it is caller-supplied and
	// requests carry the API key. The version is only honored if well-formed.
	if deepResearchAPIVersionPattern.MatchString(stored.APIVersion) {
		opts.APIVersion = stored.APIVersion
	}
	return opts
}

// validateDeepResearchOperationID rejects operation IDs that could alter the
// request path or query.
func validateDeepResearchOperationID(id string) error {
	if id == "" {
		return core.NewPublicError(core.INVALID_ARGUMENT, "Deep Research operation is missing an ID", nil)
	}
	if !deepResearchOperationIDPattern.MatchString(id) || strings.Contains(id, "..") {
		return core.NewPublicError(core.INVALID_ARGUMENT, "Deep Research operation has an invalid ID", nil)
	}
	return nil
}

// deepResearchStartTime reads the start time stored in operation metadata,
// tolerating the forms it takes after a JSON round trip.
func deepResearchStartTime(v any) (time.Time, bool) {
	switch t := v.(type) {
	case time.Time:
		return t, !t.IsZero()
	case string:
		if parsed, err := time.Parse(time.RFC3339Nano, t); err == nil {
			return parsed, true
		}
	case float64:
		return time.UnixMilli(int64(t)), true
	case int64:
		return time.UnixMilli(t), true
	}
	return time.Time{}, false
}

func toDeepResearchInteractionRequest(model string, req *ai.ModelRequest, config *DeepResearchConfig) (*interactionRequest, error) {
	var messages []*ai.Message
	if req != nil {
		messages = make([]*ai.Message, 0, len(req.Messages))
		for _, msg := range req.Messages {
			if msg == nil {
				continue
			}
			cp := msg.Clone()
			if cp.Role == ai.RoleSystem {
				cp.Role = ai.RoleUser
			}
			messages = append(messages, cp)
		}
	}

	steps, err := toInteractionSteps(messages)
	if err != nil {
		return nil, err
	}

	agentConfig := map[string]any{
		"type": deepResearchAgentType,
	}
	if config != nil {
		if config.ThinkingSummaries != "" {
			agentConfig["thinking_summaries"] = strings.ToLower(config.ThinkingSummaries)
		}
		if config.Visualization != "" {
			agentConfig["visualization"] = strings.ToLower(config.Visualization)
		}
		if config.CollaborativePlanning != nil {
			agentConfig["collaborative_planning"] = *config.CollaborativePlanning
		}
	}

	interactionReq := &interactionRequest{
		Agent:          model,
		Input:          steps,
		Background:     true,
		AgentConfig:    agentConfig,
		Tools:          toInteractionTools(req, config),
		ResponseFormat: toDeepResearchResponseFormat(req, config),
	}
	if config != nil {
		interactionReq.PreviousInteractionID = config.PreviousInteractionID
		interactionReq.Store = config.Store
	}
	return interactionReq, nil
}

// toDeepResearchResponseFormat builds the interactions response_format: one
// entry per requested modality, with JSON output folded into the text entry.
// It returns a single entry when there is one, a list when there are several
// and nil when nothing is requested. The deprecated top-level
// response_modalities field is not used.
func toDeepResearchResponseFormat(req *ai.ModelRequest, config *DeepResearchConfig) any {
	var entries []map[string]any
	index := map[string]map[string]any{}
	entry := func(typ string) map[string]any {
		if e, ok := index[typ]; ok {
			return e
		}
		e := map[string]any{"type": typ}
		index[typ] = e
		entries = append(entries, e)
		return e
	}
	if config != nil {
		for _, modality := range config.ResponseModalities {
			entry(strings.ToLower(modality))
		}
	}
	if req != nil && req.Output != nil &&
		(req.Output.Format == ai.OutputFormatJSON || req.Output.ContentType == "application/json") {
		text := entry("text")
		text["mime_type"] = "application/json"
		if req.Output.Schema != nil {
			text["schema"] = deepResearchCleanSchema(req.Output.Schema)
		}
	}
	switch len(entries) {
	case 0:
		return nil
	case 1:
		return entries[0]
	default:
		return entries
	}
}

// deepResearchCleanSchema removes the JSON Schema keywords the interactions
// API does not accept ($schema, additionalProperties) and collapses nullable
// type arrays to their first non-null type, like the JS cleanSchema.
func deepResearchCleanSchema(v any) any {
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
				if types, ok := val.([]string); ok {
					for _, t := range types {
						if t != "null" {
							out[k] = t
							break
						}
					}
					continue
				}
			}
			out[k] = deepResearchCleanSchema(val)
		}
		return out
	case []any:
		out := make([]any, len(s))
		for i, val := range s {
			out[i] = deepResearchCleanSchema(val)
		}
		return out
	default:
		return v
	}
}

func toInteractionTools(req *ai.ModelRequest, config *DeepResearchConfig) []map[string]any {
	var tools []map[string]any
	if req != nil {
		for _, tool := range req.Tools {
			if tool == nil {
				continue
			}
			interactionTool := map[string]any{
				"type":        "function",
				"name":        tool.Name,
				"description": tool.Description,
			}
			if tool.InputSchema != nil {
				interactionTool["parameters"] = deepResearchCleanSchema(tool.InputSchema)
			}
			tools = append(tools, interactionTool)
		}
	}
	if config == nil {
		return tools
	}
	tools = appendOptionalInteractionTool(tools, "google_search", config.GoogleSearch)
	tools = appendOptionalInteractionTool(tools, "url_context", config.URLContext)
	tools = appendOptionalInteractionTool(tools, "code_execution", config.CodeExecution)
	if config.FileSearch != nil {
		tool := map[string]any{"type": "file_search"}
		if len(config.FileSearch.FileSearchStoreNames) > 0 {
			tool["file_search_store_names"] = config.FileSearch.FileSearchStoreNames
		}
		for k, v := range config.FileSearch.Extra {
			tool[k] = v
		}
		tools = append(tools, tool)
	}
	for _, server := range config.MCPServers {
		tool := map[string]any{"type": "mcp_server"}
		if server.Name != "" {
			tool["name"] = server.Name
		}
		if server.URL != "" {
			tool["url"] = server.URL
		}
		if len(server.Headers) > 0 {
			tool["headers"] = server.Headers
		}
		if len(server.AllowedTools) > 0 {
			tool["allowed_tools"] = server.AllowedTools
		}
		tools = append(tools, tool)
	}
	return tools
}

func appendOptionalInteractionTool(tools []map[string]any, toolType string, value any) []map[string]any {
	switch typed := value.(type) {
	case nil:
		return tools
	case bool:
		if typed {
			return append(tools, map[string]any{"type": toolType})
		}
	case map[string]any:
		tool := map[string]any{"type": toolType}
		for k, v := range typed {
			tool[k] = v
		}
		return append(tools, tool)
	}
	return tools
}

func toInteractionSteps(messages []*ai.Message) ([]interactionStep, error) {
	steps := []interactionStep{}
	for _, message := range messages {
		normalContent := []interactionContent{}
		for _, part := range message.Content {
			if part == nil {
				continue
			}
			switch {
			case part.IsToolRequest():
				if part.ToolRequest == nil {
					continue
				}
				args, _ := part.ToolRequest.Input.(map[string]any)
				steps = append(steps, interactionStep{
					Type:      "function_call",
					Name:      part.ToolRequest.Name,
					Arguments: args,
					ID:        part.ToolRequest.Ref,
				})
			case part.IsToolResponse():
				if part.ToolResponse == nil {
					continue
				}
				steps = append(steps, interactionStep{
					Type:   "function_result",
					Name:   part.ToolResponse.Name,
					Result: part.ToolResponse.Output,
					CallID: part.ToolResponse.Ref,
				})
			case part.IsCustom():
				if step, ok := customPartToInteractionStep(part); ok {
					steps = append(steps, step)
				}
			default:
				content, err := toInteractionContent(part)
				if err != nil {
					return nil, err
				}
				if content.Type != "" {
					normalContent = append(normalContent, content)
				}
			}
		}
		if len(normalContent) > 0 {
			stepType := "user_input"
			if message.Role == ai.RoleModel {
				stepType = "model_output"
			}
			steps = append(steps, interactionStep{
				Type:    stepType,
				Content: normalContent,
			})
		}
	}
	return steps, nil
}

func toInteractionContent(part *ai.Part) (interactionContent, error) {
	if part.IsText() {
		return interactionContent{Type: "text", Text: part.Text}, nil
	}
	if part.IsData() {
		// The interactions API has no unstructured-data content (JS sends only
		// text and media), so a data part is sent as its string form.
		return interactionContent{Type: "text", Text: part.DataString()}, nil
	}
	if part.IsReasoning() {
		content := interactionContent{
			Type:    "thought",
			Summary: []interactionContent{{Type: "text", Text: part.Text}},
		}
		if part.Metadata != nil {
			if signature, ok := part.Metadata["thoughtSignature"].(string); ok {
				content.Signature = signature
			}
		}
		return content, nil
	}
	if !part.IsMedia() {
		return interactionContent{}, nil
	}
	if part.ContentType == "" {
		return interactionContent{}, fmt.Errorf("media part missing content type")
	}

	content := interactionContent{
		MIMEType: part.ContentType,
	}
	switch {
	case part.IsImage():
		content.Type = "image"
	case part.IsAudio():
		content.Type = "audio"
	case part.IsVideo():
		content.Type = "video"
	case part.ContentType == "application/pdf":
		content.Type = "document"
	default:
		return interactionContent{}, fmt.Errorf("unsupported media type: %s", part.ContentType)
	}
	if strings.HasPrefix(part.Text, "data:") {
		if _, data, ok := strings.Cut(part.Text, ","); ok {
			content.Data = data
		}
	} else {
		content.URI = part.Text
	}
	return content, nil
}

func doDeepResearchRequest(ctx context.Context, client *genai.Client, opts deepResearchClientOptions, method, resourcePath string, body any) (*geminiInteraction, error) {
	if client == nil {
		return nil, core.NewError(core.FAILED_PRECONDITION, "Deep Research client is nil")
	}
	reqURL, err := deepResearchURL(opts, resourcePath)
	if err != nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "invalid Deep Research endpoint: %w", err)
	}

	clientConfig := client.ClientConfig()
	if t := clientConfig.HTTPOptions.Timeout; t != nil && *t > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, *t)
		defer cancel()
	}

	var bodyReader io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return nil, status.Errorf(status.ErrInternal, "encode Deep Research request: %w", err)
		}
		bodyReader = bytes.NewReader(data)
	}

	req, err := http.NewRequestWithContext(ctx, method, reqURL, bodyReader)
	if err != nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "build Deep Research request: %w", err)
	}
	for k, values := range clientConfig.HTTPOptions.Headers {
		for _, value := range values {
			req.Header.Add(k, value)
		}
	}
	if req.Header.Get(xGoogApiClientHeader) == "" {
		req.Header.Set(xGoogApiClientHeader, genkitClientHeader.Get(xGoogApiClientHeader))
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Api-Revision", deepResearchAPIRevision)
	if opts.APIKey != "" {
		req.Header.Set("x-goog-api-key", opts.APIKey)
	}

	httpClient := clientConfig.HTTPClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, deepResearchTransportError(ctx, method, resourcePath, err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(io.LimitReader(resp.Body, deepResearchMaxResponseBytes+1))
	if err != nil {
		return nil, deepResearchTransportError(ctx, method, resourcePath, err)
	}
	if len(respBody) > deepResearchMaxResponseBytes {
		return nil, status.Errorf(status.ErrInternal, "Deep Research response for %s %s exceeds the %d byte limit", method, resourcePath, deepResearchMaxResponseBytes)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, deepResearchHTTPError(method, resourcePath, resp.StatusCode, respBody)
	}

	var interaction geminiInteraction
	if len(respBody) > 0 {
		if err := json.Unmarshal(respBody, &interaction); err != nil {
			return nil, status.Errorf(status.ErrInternal, "decode Deep Research response for %s %s: %w", method, resourcePath, err)
		}
	}
	return &interaction, nil
}

// deepResearchTransportError maps a failure to send the request or read the
// response. Context cancellation and deadlines keep their cause and map to the
// matching status; other timeouts are DEADLINE_EXCEEDED, anything else is
// UNAVAILABLE.
func deepResearchTransportError(ctx context.Context, method, resourcePath string, err error) error {
	switch {
	case errors.Is(err, context.Canceled), errors.Is(ctx.Err(), context.Canceled):
		return status.Errorf(status.ErrCancelled, "Deep Research request %s %s cancelled: %w", method, resourcePath, context.Canceled)
	case errors.Is(err, context.DeadlineExceeded), errors.Is(ctx.Err(), context.DeadlineExceeded):
		return status.Errorf(status.ErrDeadlineExceeded, "Deep Research request %s %s timed out: %w", method, resourcePath, context.DeadlineExceeded)
	}
	var ne net.Error
	if errors.As(err, &ne) && ne.Timeout() {
		return status.Errorf(status.ErrDeadlineExceeded, "Deep Research request %s %s timed out: %w", method, resourcePath, err)
	}
	return status.Errorf(status.ErrUnavailable, "Deep Research request %s %s failed: %w", method, resourcePath, err)
}

// deepResearchHTTPError maps a non-2xx response through [wrapAPIError], so the
// status and any google.rpc.RetryInfo delay are reported as for other models.
// The body is parsed as a Google or Interactions style error; the raw body is
// never echoed, only the parsed message (or the HTTP status text).
func deepResearchHTTPError(method, resourcePath string, httpStatus int, body []byte) error {
	apiErr := genai.APIError{Code: httpStatus}
	var parsed struct {
		Error struct {
			Code    any              `json:"code"`
			Message string           `json:"message"`
			Status  string           `json:"status"`
			Details []map[string]any `json:"details"`
		} `json:"error"`
	}
	if err := json.Unmarshal(body, &parsed); err == nil {
		apiErr.Message = parsed.Error.Message
		apiErr.Details = parsed.Error.Details
		if status.Name(parsed.Error.Status).IsValid() {
			apiErr.Status = parsed.Error.Status
		} else if n, ok := deepResearchErrorCodeToStatus(parsed.Error.Code); ok {
			apiErr.Status = string(n)
		}
	}
	if apiErr.Message == "" {
		apiErr.Message = http.StatusText(httpStatus)
	}
	return wrapAPIError(fmt.Errorf("Deep Research request %s %s: %w", method, resourcePath, apiErr))
}

// deepResearchContentBlockCodes are the Interactions error codes that report a
// content-policy block.
var deepResearchContentBlockCodes = map[string]bool{
	"safety": true, "recitation": true, "language": true, "prohibited_content": true,
	"spii": true, "blocklist": true, "image_safety": true, "image_prohibited_content": true,
	"image_recitation": true, "image_other": true, "content_blocked": true,
	"jailbreak": true, "model_armor": true,
}

// deepResearchErrorCodeStatus maps Interactions error codes to Genkit statuses.
var deepResearchErrorCodeStatus = map[string]status.Name{
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
	"agent_max_token_limit":   status.InvalidArgument,
	"model_not_found":         status.NotFound,
	"not_found":               status.NotFound,
	"authentication":          status.Unauthenticated,
	"permission_denied":       status.PermissionDenied,
	"already_exists":          status.AlreadyExists,
	"aborted":                 status.Aborted,
	"cancelled":               status.Cancelled,
	"deadline_exceeded":       status.DeadlineExceeded,
	"api_error":               status.Internal,
	"unimplemented":           status.Unimplemented,
}

func isDeepResearchContentBlockCode(code string) bool {
	return deepResearchContentBlockCodes[strings.ToLower(strings.TrimSpace(code))]
}

// deepResearchErrorCodeToStatus maps an Interactions error code (a string, or
// a numeric HTTP code) to a Genkit status. The second result is false when the
// code is unknown.
func deepResearchErrorCodeToStatus(code any) (status.Name, bool) {
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
		if n, ok := deepResearchErrorCodeStatus[norm]; ok {
			return n, true
		}
		if isDeepResearchContentBlockCode(norm) {
			return status.FailedPrecondition, true
		}
		if upper := status.Name(strings.ToUpper(norm)); upper.IsValid() {
			return upper, true
		}
		var num int
		if _, err := fmt.Sscanf(norm, "%d", &num); err == nil && fmt.Sprint(num) == norm {
			return deepResearchErrorCodeToStatus(float64(num))
		}
	}
	return "", false
}

func deepResearchURL(opts deepResearchClientOptions, resourcePath string) (string, error) {
	baseURL := opts.BaseURL
	if baseURL == "" {
		baseURL = "https://generativelanguage.googleapis.com"
	}
	apiVersion := opts.APIVersion
	if apiVersion == "" {
		apiVersion = "v1beta"
	}
	u, err := url.Parse(baseURL)
	if err != nil {
		return "", err
	}
	return u.JoinPath(apiVersion, resourcePath).String(), nil
}

func fromDeepResearchInteraction(interaction *geminiInteraction) *ai.ModelOperation {
	if interaction == nil {
		return &ai.ModelOperation{Metadata: make(map[string]any)}
	}
	op := &ai.ModelOperation{
		ID:       interaction.ID,
		Metadata: make(map[string]any),
	}
	if interaction.ID != "" {
		op.Metadata["interactionId"] = interaction.ID
	}
	if interaction.EnvironmentID != "" {
		op.Metadata["environmentId"] = interaction.EnvironmentID
	}

	switch interaction.Status {
	case "in_progress", "queued", "":
		op.Done = false
	case "cancelled":
		op.Done = true
		op.Output = deepResearchCancelledOutput(interaction)
	case "failed":
		op.Done = true
		first := deepResearchFirstError(interaction)
		if isDeepResearchContentBlockCode(first.Code) {
			op.Output = deepResearchBlockedOutput(interaction, first)
			break
		}
		name := status.Unknown
		for _, code := range []any{first.Code, first.Status} {
			if n, ok := deepResearchErrorCodeToStatus(code); ok {
				name = n
				break
			}
		}
		op.Error = status.Errorf(status.Base(name), "deep research interaction failed: %s", deepResearchFailureDetail(interaction)).
			WithDetails(map[string]any{"interactionId": interaction.ID, "code": first.Code})
	case "completed":
		op.Done = true
		op.Output = fromCompletedDeepResearchInteraction(interaction)
	case "requires_action":
		// A deliberate pause (e.g. a plan awaiting approval or a client-side
		// function call), not a failure: finish the operation and surface what
		// was produced. The caller continues with previousInteractionId.
		op.Done = true
		op.Output = fromCompletedDeepResearchInteraction(interaction)
		op.Output.Message.Metadata["interactionStatus"] = interaction.Status
		if len(interaction.Steps) == 0 {
			// Do not claim completion when nothing was produced.
			op.Output.Message.Content = nil
		}
	case "incomplete":
		op.Done = true
		op.Output = fromCompletedDeepResearchInteraction(interaction)
		op.Output.FinishReason = ai.FinishReasonLength
		op.Output.FinishMessage = "Interaction incomplete (truncated output)"
	default:
		// Finish rather than leave Done unset, which would make callers poll forever.
		op.Done = true
		op.Error = status.Errorf(status.ErrUnknown, "Unknown interaction status: %s", interaction.Status)
	}
	return op
}

func deepResearchFirstError(interaction *geminiInteraction) interactionError {
	if len(interaction.Errors) > 0 {
		return interaction.Errors[0]
	}
	if interaction.Error != nil {
		return *interaction.Error
	}
	return interactionError{}
}

func deepResearchCancelledOutput(interaction *geminiInteraction) *ai.ModelResponse {
	metadata := map[string]any{"interactionId": interaction.ID}
	if interaction.EnvironmentID != "" {
		metadata["environmentId"] = interaction.EnvironmentID
	}
	resp := &ai.ModelResponse{
		Message: &ai.Message{
			Role:     ai.RoleModel,
			Content:  []*ai.Part{ai.NewTextPart("Operation cancelled.")},
			Metadata: metadata,
		},
		FinishReason:  ai.FinishReasonInterrupted,
		FinishMessage: "Operation cancelled",
		Raw:           interaction,
	}
	if interaction.Usage != nil {
		resp.Usage = fromInteractionUsage(interaction.Usage)
	}
	return resp
}

func deepResearchBlockedOutput(interaction *geminiInteraction, first interactionError) *ai.ModelResponse {
	msg := first.Message
	if msg == "" {
		msg = deepResearchFailureDetail(interaction)
	}
	metadata := map[string]any{"interactionId": interaction.ID}
	if interaction.EnvironmentID != "" {
		metadata["environmentId"] = interaction.EnvironmentID
	}
	resp := &ai.ModelResponse{
		Message:       &ai.Message{Role: ai.RoleModel, Metadata: metadata},
		FinishReason:  ai.FinishReasonBlocked,
		FinishMessage: msg,
		Raw:           interaction,
	}
	if interaction.Usage != nil {
		resp.Usage = fromInteractionUsage(interaction.Usage)
	}
	return resp
}

// deepResearchFailureDetail extracts a human-readable reason from a failed interaction,
// falling back to a generic message when the API provides no detail.
func deepResearchFailureDetail(interaction *geminiInteraction) string {
	if interaction == nil {
		return "no error detail provided"
	}
	err := deepResearchFirstError(interaction)
	switch {
	case err.Message != "":
		return err.Message
	case err.Status != "":
		return err.Status
	case err.Code != "":
		return err.Code
	default:
		return "no error detail provided"
	}
}

func fromCompletedDeepResearchInteraction(interaction *geminiInteraction) *ai.ModelResponse {
	if interaction == nil {
		return &ai.ModelResponse{
			Message:      &ai.Message{Role: ai.RoleModel, Content: []*ai.Part{ai.NewTextPart("Deep Research completed.")}},
			FinishReason: ai.FinishReasonStop,
		}
	}
	content := []*ai.Part{}
	for _, step := range interaction.Steps {
		content = append(content, fromInteractionStep(step)...)
	}
	if len(content) == 0 {
		content = append(content, ai.NewTextPart("Deep Research completed."))
	}

	resp := &ai.ModelResponse{
		Message: &ai.Message{
			Role:    ai.RoleModel,
			Content: content,
			Metadata: map[string]any{
				"interactionId": interaction.ID,
			},
		},
		FinishReason: ai.FinishReasonStop,
		Raw:          interaction,
	}
	if interaction.EnvironmentID != "" {
		resp.Message.Metadata["environmentId"] = interaction.EnvironmentID
	}
	if interaction.Usage != nil {
		resp.Usage = fromInteractionUsage(interaction.Usage)
	}
	return resp
}

func fromInteractionStep(step interactionStep) []*ai.Part {
	switch step.Type {
	case "model_output":
		return fromInteractionContents(step.Content)
	case "user_input":
		return nil
	case "google_search_call":
		part := ai.NewCustomPart(map[string]any{
			"googleSearchCall": map[string]any{
				"id":        step.ID,
				"arguments": step.Arguments,
			},
		})
		addInteractionSignature(part, step.Signature)
		return []*ai.Part{part}
	case "google_search_result":
		part := ai.NewCustomPart(map[string]any{
			"googleSearchResult": map[string]any{
				"callId": step.CallID,
				"result": step.Result,
			},
		})
		addInteractionSignature(part, step.Signature)
		return []*ai.Part{part}
	case "code_execution_call":
		part := ai.NewCustomPart(map[string]any{
			"executableCode": step.Arguments,
		})
		if part.Metadata == nil {
			part.Metadata = make(map[string]any)
		}
		part.Metadata["callId"] = step.ID
		addInteractionSignature(part, step.Signature)
		return []*ai.Part{part}
	case "code_execution_result":
		part := ai.NewCustomPart(map[string]any{
			"codeExecutionResult": map[string]any{
				"output": step.Result,
			},
		})
		if part.Metadata == nil {
			part.Metadata = make(map[string]any)
		}
		part.Metadata["callId"] = step.CallID
		addInteractionSignature(part, step.Signature)
		return []*ai.Part{part}
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
	case "text", "image", "audio", "video", "document", "thought":
		return fromInteractionContents([]interactionContent{{
			Type:      step.Type,
			Text:      step.Text,
			Data:      step.Data,
			URI:       step.URI,
			MIMEType:  step.MIMEType,
			Summary:   step.Summary,
			Signature: step.Signature,
		}})
	default:
		return []*ai.Part{ai.NewCustomPart(map[string]any{"unknownStep": step})}
	}
}

func customPartToInteractionStep(part *ai.Part) (interactionStep, bool) {
	if part == nil || part.Custom == nil {
		return interactionStep{}, false
	}
	signature, _ := metadataString(part.Metadata, "thoughtSignature")
	if raw, ok := part.Custom["googleSearchCall"].(map[string]any); ok {
		id, _ := raw["id"].(string)
		args, _ := raw["arguments"].(map[string]any)
		return interactionStep{Type: "google_search_call", ID: id, Arguments: args, Signature: signature}, true
	}
	if raw, ok := part.Custom["googleSearchResult"].(map[string]any); ok {
		callID, _ := raw["callId"].(string)
		result, _ := raw["result"].(map[string]any)
		return interactionStep{Type: "google_search_result", CallID: callID, Result: result, Signature: signature}, true
	}
	if raw, ok := part.Custom["executableCode"].(map[string]any); ok {
		callID, _ := metadataString(part.Metadata, "callId")
		return interactionStep{Type: "code_execution_call", ID: callID, Arguments: raw, Signature: signature}, true
	}
	if raw, ok := part.Custom["codeExecutionResult"].(map[string]any); ok {
		callID, _ := metadataString(part.Metadata, "callId")
		return interactionStep{Type: "code_execution_result", CallID: callID, Result: raw["output"], Signature: signature}, true
	}
	return interactionStep{}, false
}

func metadataString(metadata map[string]any, key string) (string, bool) {
	if metadata == nil {
		return "", false
	}
	value, ok := metadata[key].(string)
	return value, ok
}

func addInteractionSignature(part *ai.Part, signature string) {
	if signature == "" {
		return
	}
	if part.Metadata == nil {
		part.Metadata = make(map[string]any)
	}
	part.Metadata["thoughtSignature"] = signature
}

func fromInteractionContents(contents []interactionContent) []*ai.Part {
	parts := []*ai.Part{}
	for _, content := range contents {
		switch content.Type {
		case "text":
			part := ai.NewTextPart(content.Text)
			if len(content.Annotations) > 0 {
				part.Metadata = map[string]any{"annotations": content.Annotations}
			}
			parts = append(parts, part)
		case "image", "audio", "video", "document":
			mimeType := content.MIMEType
			if mimeType == "" {
				mimeType = content.Type + "/*"
			}
			source := content.URI
			if source == "" && content.Data != "" {
				source = "data:" + mimeType + ";base64," + content.Data
			}
			if source != "" {
				parts = append(parts, ai.NewMediaPart(mimeType, source))
			}
		case "thought":
			summaryText := ""
			for _, summary := range content.Summary {
				if summary.Type == "text" {
					summaryText += summary.Text
				}
			}
			if summaryText == "" {
				summaryText = content.Text
			}
			part := ai.NewReasoningPart(summaryText, nil)
			if content.Signature != "" {
				part.Metadata = map[string]any{"thoughtSignature": content.Signature}
			}
			parts = append(parts, part)
		case "function_call":
			parts = append(parts, ai.NewToolRequestPart(&ai.ToolRequest{
				Name:  content.Name,
				Input: content.Arguments,
				Ref:   content.ID,
			}))
		case "function_result":
			parts = append(parts, ai.NewToolResponsePart(&ai.ToolResponse{
				Name:   content.Name,
				Output: content.Result,
				Ref:    content.CallID,
			}))
		}
	}
	return parts
}

func fromInteractionUsage(usage *interactionUsage) *ai.GenerationUsage {
	if usage == nil {
		return nil
	}
	result := &ai.GenerationUsage{
		InputTokens:         usage.TotalInputTokens,
		OutputTokens:        usage.TotalOutputTokens,
		TotalTokens:         usage.TotalTokens,
		CachedContentTokens: usage.TotalCachedTokens,
		ThoughtsTokens:      usage.TotalThoughtTokens,
	}
	for _, modality := range usage.InputTokensByModality {
		switch modality.Modality {
		case "text":
			result.InputCharacters = modality.Tokens
		case "image":
			result.InputImages = modality.Tokens
		case "audio":
			result.InputAudioFiles = modality.Tokens
		case "video":
			result.InputVideos = modality.Tokens
		}
	}
	for _, modality := range usage.OutputTokensByModality {
		switch modality.Modality {
		case "text":
			result.OutputCharacters = modality.Tokens
		case "image":
			result.OutputImages = modality.Tokens
		case "audio":
			result.OutputAudioFiles = modality.Tokens
		case "video":
			result.OutputVideos = modality.Tokens
		}
	}
	return result
}

func restoreDeepResearchMetadata(updatedOp, originalOp *ai.ModelOperation) {
	if updatedOp.Metadata == nil {
		updatedOp.Metadata = make(map[string]any)
	}
	if originalOp == nil || originalOp.Metadata == nil {
		return
	}
	for k, v := range originalOp.Metadata {
		updatedOp.Metadata[k] = v
	}
}
