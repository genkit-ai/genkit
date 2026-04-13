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

package ollama

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"regexp"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/internal/uri"
	"github.com/invopop/jsonschema"
)

const provider = "ollama"

var (
	mediaSupportedModels = []string{"llava", "bakllava", "llava-llama3", "llava:13b", "llava:7b", "llava:latest", "gemma3:4b", "gemma3:12b", "gemma3:27b"}
	toolSupportedModels  = []string{
		"qwq", "mistral-small3.1", "llama3.3", "llama3.2", "llama3.1", "mistral",
		"qwen2.5", "qwen2.5-coder", "qwen2", "mistral-nemo", "mixtral", "smollm2",
		"mistral-small", "command-r", "hermes3", "mistral-large", "command-r-plus",
		"phi4-mini", "granite3.1-dense", "granite3-dense", "granite3.2", "athene-v2",
		"nemotron-mini", "nemotron", "llama3-groq-tool-use", "aya-expanse", "granite3-moe",
		"granite3.2-vision", "granite3.1-moe", "cogito", "command-r7b", "firefunction-v2",
		"granite3.3", "command-a", "command-r7b-arabic", "gpt-oss",
	}
	roleMapping = map[ai.Role]string{
		ai.RoleUser:   "user",
		ai.RoleModel:  "assistant",
		ai.RoleSystem: "system",
		ai.RoleTool:   "tool",
	}
	// defaultOllamaSupports defines the default capabilities for dynamically
	// discovered Ollama models. All capabilities are enabled since local models
	// vary widely and we can't query their capabilities individually.
	defaultOllamaSupports = ai.ModelSupports{
		Multiturn:  true,
		Media:      true,
		Tools:      true,
		SystemRole: true,
	}

	// thinkingRegex matches <think> or <thinking> tags case-insensitively across multiple lines.
	// It uses non-greedy matching (.*?) to correctly extract individual blocks when
	// multiple blocks are present in a single response.
	thinkingRegex = regexp.MustCompile("(?si)<(think|thinking)>(.*?)</(?:think|thinking)>")
)

// ollamaTagsResponse represents the response from GET /api/tags.
type ollamaTagsResponse struct {
	Models []ollamaLocalModel `json:"models"`
}

// ollamaLocalModel represents a locally available Ollama model from /api/tags.
type ollamaLocalModel struct {
	Name  string `json:"name"`
	Model string `json:"model"`
}

// ollamaShowResponse represents the response from POST /api/show.
type ollamaShowResponse struct {
	Capabilities []string `json:"capabilities"`
}

// getModelCapabilities calls POST /api/show to retrieve the model's capabilities.
// Returns nil if the endpoint is unavailable or the model doesn't report capabilities.
func (o *Ollama) getModelCapabilities(ctx context.Context, modelName string) []string {
	body, err := json.Marshal(map[string]string{"model": modelName})
	if err != nil {
		return nil
	}
	req, err := http.NewRequestWithContext(ctx, "POST", o.ServerAddress+"/api/show", bytes.NewReader(body))
	if err != nil {
		return nil
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := o.client.Do(req)
	if err != nil {
		return nil
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil
	}
	var showResp ollamaShowResponse
	if err := json.NewDecoder(resp.Body).Decode(&showResp); err != nil {
		return nil
	}
	return showResp.Capabilities
}

// modelSupportsFromCapabilities derives ModelSupports from capabilities reported
// by the Ollama /api/show endpoint. Falls back to the static allowlists when
// the server doesn't report capabilities (older Ollama versions).
func modelSupportsFromCapabilities(caps []string, modelName string) *ai.ModelSupports {
	if len(caps) > 0 {
		return &ai.ModelSupports{
			Multiturn:  true,
			SystemRole: true,
			Tools:      slices.Contains(caps, "tools"),
			Media:      slices.Contains(caps, "vision") || slices.Contains(caps, "audio"),
		}
	}
	// Fallback: use static allowlists for older Ollama servers that don't
	// report capabilities via /api/show.
	return &ai.ModelSupports{
		Multiturn:  true,
		SystemRole: true,
		Tools:      slices.Contains(toolSupportedModels, modelName),
		Media:      slices.Contains(mediaSupportedModels, modelName),
	}
}

// listLocalModels calls GET /api/tags to list locally installed Ollama models.
func (o *Ollama) listLocalModels(ctx context.Context) ([]ollamaLocalModel, error) {
	req, err := http.NewRequestWithContext(ctx, "GET", o.ServerAddress+"/api/tags", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	resp, err := o.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch local models from Ollama: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ollama /api/tags returned status %d", resp.StatusCode)
	}

	var tagsResp ollamaTagsResponse
	if err := json.NewDecoder(resp.Body).Decode(&tagsResp); err != nil {
		return nil, fmt.Errorf("failed to decode /api/tags response: %w", err)
	}
	return tagsResp.Models, nil
}

func (o *Ollama) DefineModel(g *genkit.Genkit, model ModelDefinition, opts *ai.ModelOptions) ai.Model {
	// Check the init guard first under a brief lock — before any I/O — so
	// that a forgotten Init() panics immediately rather than after a timeout.
	o.mu.Lock()
	if !o.initted {
		o.mu.Unlock()
		panic("ollama.Init not called")
	}
	o.mu.Unlock()

	// Detect capabilities outside the lock to avoid holding it during HTTP I/O.
	var modelOpts ai.ModelOptions
	if opts != nil {
		modelOpts = *opts
	} else {
		// Query the Ollama server for the model's actual capabilities via
		// /api/show. This replaces the hardcoded allowlist approach so that
		// newly released models (e.g. gemma4) work automatically without
		// code changes. Falls back to the static list for older servers.
		ctx, cancel := context.WithTimeout(context.Background(), time.Duration(o.Timeout)*time.Second)
		defer cancel()
		caps := o.getModelCapabilities(ctx, model.Name)
		modelOpts = ai.ModelOptions{
			Label:    model.Name,
			Supports: modelSupportsFromCapabilities(caps, model.Name),
			Versions: []string{},
		}
	}

	// Re-acquire lock for the registration step.
	o.mu.Lock()
	defer o.mu.Unlock()
	meta := &ai.ModelOptions{
		Label:        "Ollama - " + model.Name,
		Supports:     modelOpts.Supports,
		Versions:     []string{},
		ConfigSchema: core.InferSchemaMap(GenerateContentConfig{}),
	}
	gen := &generator{model: model, serverAddress: o.ServerAddress, timeout: o.Timeout}
	return genkit.DefineModel(g, api.NewName(provider, model.Name), meta, gen.generate)
}

// IsDefinedModel reports whether a model is defined.
func IsDefinedModel(g *genkit.Genkit, name string) bool {
	return genkit.LookupModel(g, api.NewName(provider, name)) != nil
}

// Model returns the [ai.Model] with the given name.
// It returns nil if the model was not configured.
func Model(g *genkit.Genkit, name string) ai.Model {
	return genkit.LookupModel(g, api.NewName(provider, name))
}

// ModelDefinition represents a model with its name and api.
type ModelDefinition struct {
	Name string
	Type string
}

type generator struct {
	model         ModelDefinition
	serverAddress string
	timeout       int
}

type ollamaMessage struct {
	Role      string           `json:"role"`
	Content   string           `json:"content,omitempty"`
	Images    []string         `json:"images,omitempty"`
	ToolCalls []ollamaToolCall `json:"tool_calls,omitempty"`
	Thinking  string           `json:"thinking,omitempty"`
}

// ThinkOption controls thinking/reasoning behavior for models that support it.
// Use [ThinkEnabled] for Ollama models (e.g. deepseek-r1) or [ThinkEffort]
// for GPT-OSS models.
type ThinkOption struct {
	value any // bool or string; unexported to enforce use of constructors
}

// ThinkEnabled creates a ThinkOption that enables or disables thinking mode.
// This is used with Ollama models like deepseek-r1.
func ThinkEnabled(enabled bool) *ThinkOption {
	return &ThinkOption{value: enabled}
}

// ThinkEffort creates a ThinkOption with an effort level for GPT-OSS models.
// Valid values: "low", "medium", "high".
func ThinkEffort(level string) *ThinkOption {
	return &ThinkOption{value: level}
}

// IsEnabled reports whether thinking is active.
func (t *ThinkOption) IsEnabled() bool {
	if t == nil {
		return false
	}
	switch v := t.value.(type) {
	case bool:
		return v
	case string:
		return v != ""
	default:
		return false
	}
}

func (t ThinkOption) MarshalJSON() ([]byte, error) {
	return json.Marshal(t.value)
}

func (t *ThinkOption) UnmarshalJSON(data []byte) error {
	var b bool
	if err := json.Unmarshal(data, &b); err == nil {
		t.value = b
		return nil
	}
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		t.value = s
		return nil
	}
	return fmt.Errorf("think must be a boolean or string, got: %s", data)
}

// JSONSchema returns a schema allowing either a boolean or a string.
func (ThinkOption) JSONSchema() *jsonschema.Schema {
	return &jsonschema.Schema{
		OneOf: []*jsonschema.Schema{
			{Type: "boolean"},
			{Type: "string"},
		},
	}
}

type GenerateContentConfig struct {
	// Think controls thinking/reasoning mode.
	// Use ThinkEnabled(true/false) for Ollama models, or
	// ThinkEffort("low"/"medium"/"high") for GPT-OSS models.
	Think *ThinkOption `json:"think,omitempty"`

	// Runtime options
	Seed        *int     `json:"seed,omitempty"`
	Temperature *float64 `json:"temperature,omitempty"`
	TopK        *int     `json:"top_k,omitempty"`
	TopP        *float64 `json:"top_p,omitempty"`
	MinP        *float64 `json:"min_p,omitempty"`
	Stop        []string `json:"stop,omitempty"`
	NumCtx      *int     `json:"num_ctx,omitempty"`
	NumPredict  *int     `json:"num_predict,omitempty"`

	// Ollama-specific
	KeepAlive string `json:"keep_alive,omitempty"`
}

type ollamaModelRequest struct {
	System string   `json:"system,omitempty"`
	Images []string `json:"images,omitempty"`
	Model  string   `json:"model"`
	Prompt string   `json:"prompt"`
	Stream bool     `json:"stream"`
	Format string   `json:"format,omitempty"`
}

// Tool definition from Ollama API
type ollamaTool struct {
	Type     string         `json:"type"`
	Function ollamaFunction `json:"function"`
}

// Function definition for Ollama API
type ollamaFunction struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
}

// Tool Call from Ollama API
type ollamaToolCall struct {
	Function ollamaFunctionCall `json:"function"`
}

// Function Call for Ollama API
type ollamaFunctionCall struct {
	Name      string `json:"name"`
	Arguments any    `json:"arguments"`
}

// TODO: Add optional parameters (images, format, options, etc.) based on your use case
type ollamaChatResponse struct {
	Model     string `json:"model"`
	CreatedAt string `json:"created_at"`
	Message   struct {
		Role      string           `json:"role"`
		Content   string           `json:"content"`
		Thinking  string           `json:"thinking"`
		ToolCalls []ollamaToolCall `json:"tool_calls,omitempty"`
	} `json:"message"`
}

type ollamaModelResponse struct {
	Model     string `json:"model"`
	CreatedAt string `json:"created_at"`
	Response  string `json:"response"`
}

// Ollama provides configuration options for the Init function.
type Ollama struct {
	ServerAddress string // Server address of oLLama.
	Timeout       int    // Response timeout in seconds (defaulted to 30 seconds)

	mu      sync.Mutex   // Mutex to control access.
	initted bool         // Whether the plugin has been initialized.
	client  *http.Client // Shared HTTP client for API calls (e.g., /api/tags).
}

func (o *Ollama) Name() string {
	return provider
}

// Init initializes the plugin.
// Since Ollama models are locally hosted, the plugin doesn't initialize any default models.
// After downloading a model, call [DefineModel] to use it.
func (o *Ollama) Init(ctx context.Context) []api.Action {
	o.mu.Lock()
	defer o.mu.Unlock()
	if o.initted {
		panic("ollama.Init already called")
	}
	if o == nil || o.ServerAddress == "" {
		panic("ollama: need ServerAddress")
	}
	o.initted = true
	if o.Timeout == 0 {
		o.Timeout = 30
	}
	o.client = &http.Client{}
	return []api.Action{}
}

// newModel creates an Ollama model without registering it in the Genkit registry.
// It is used by ListActions (to generate ActionDesc) and ResolveAction (to return an Action).
func (o *Ollama) newModel(name string, opts ai.ModelOptions) ai.Model {
	meta := &ai.ModelOptions{
		Label:        "Ollama - " + name,
		Supports:     opts.Supports,
		Versions:     []string{},
		ConfigSchema: core.InferSchemaMap(GenerateContentConfig{}),
	}
	gen := &generator{
		model:         ModelDefinition{Name: name, Type: "chat"},
		serverAddress: o.ServerAddress,
		timeout:       o.Timeout,
	}
	return ai.NewModel(api.NewName(provider, name), meta, gen.generate)
}

// ListActions calls /api/tags to discover locally installed Ollama models.
func (o *Ollama) ListActions(ctx context.Context) []api.ActionDesc {
	models, err := o.listLocalModels(ctx)
	if err != nil {
		slog.Error("unable to list ollama models", "error", err)
		return nil
	}

	var actions []api.ActionDesc
	for _, m := range models {
		name := m.Name
		// Filter out embedding models (following JS: !m.model.includes('embed'))
		if strings.Contains(name, "embed") {
			continue
		}
		// Check for context cancellation before each potentially slow HTTP call.
		if ctx.Err() != nil {
			break
		}
		// Query each model's actual capabilities from the Ollama server.
		caps := o.getModelCapabilities(ctx, name)
		supports := modelSupportsFromCapabilities(caps, name)
		model := o.newModel(name, ai.ModelOptions{Supports: supports})
		if action, ok := model.(api.Action); ok {
			actions = append(actions, action.Desc())
		}
	}
	return actions
}

// ResolveAction dynamically creates a model action on demand.
func (o *Ollama) ResolveAction(atype api.ActionType, name string) api.Action {
	if atype != api.ActionTypeModel {
		return nil
	}
	// Query the model's actual capabilities from the Ollama server.
	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(o.Timeout)*time.Second)
	defer cancel()
	caps := o.getModelCapabilities(ctx, name)
	supports := modelSupportsFromCapabilities(caps, name)
	model := o.newModel(name, ai.ModelOptions{Supports: supports})
	if action, ok := model.(api.Action); ok {
		return action
	}
	return nil
}

// Ptr returns a pointer to the given value.
func Ptr[T any](v T) *T {
	return &v
}

// Generate makes a request to the Ollama API and processes the response.
func (g *generator) generate(ctx context.Context, input *ai.ModelRequest, cb func(context.Context, *ai.ModelResponseChunk) error) (*ai.ModelResponse, error) {
	stream := cb != nil
	var payload any
	var thinkingEnabled bool
	isChatModel := g.model.Type == "chat"

	// Extract images from the request. Ollama will handle unsupported media
	// gracefully, matching the JS plugin behavior of unconditionally forwarding images.
	images, err := concatImages(input, []ai.Role{ai.RoleUser, ai.RoleModel})
	if err != nil {
		return nil, fmt.Errorf("failed to grab image parts: %v", err)
	}

	if !isChatModel {
		payload = ollamaModelRequest{
			Model:  g.model.Name,
			Prompt: concatMessages(input, []ai.Role{ai.RoleUser, ai.RoleModel, ai.RoleTool}),
			System: concatMessages(input, []ai.Role{ai.RoleSystem}),
			Images: images,
			Stream: stream,
		}
	} else {
		var messages []*ollamaMessage
		// Translate all messages to ollama message format.
		for _, m := range input.Messages {
			message, err := convertParts(m.Role, m.Content)
			if err != nil {
				return nil, fmt.Errorf("failed to convert message parts: %v", err)
			}
			messages = append(messages, message)
		}

		chatReq := ollamaChatRequest{
			Messages: messages,
			Model:    g.model.Name,
			Stream:   stream,
			Images:   images,
		}
		if err := chatReq.ApplyOptions(input.Config); err != nil {
			return nil, fmt.Errorf("failed to apply options: %v", err)
		}
		thinkingEnabled = chatReq.Think.IsEnabled()

		if len(input.Tools) > 0 {
			tools, err := convertTools(input.Tools)
			if err != nil {
				return nil, fmt.Errorf("failed to convert tools: %v", err)
			}
			chatReq.Tools = tools
		}
		payload = chatReq
	}

	client := &http.Client{Timeout: time.Duration(g.timeout) * time.Second}
	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}

	// Determine the correct endpoint
	endpoint := g.serverAddress + "/api/chat"
	if !isChatModel {
		endpoint = g.serverAddress + "/api/generate"
	}

	req, err := http.NewRequest("POST", endpoint, bytes.NewReader(payloadBytes))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req = req.WithContext(ctx)

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to send request: %v", err)
	}
	defer resp.Body.Close()

	if cb == nil {
		// Existing behavior for non-streaming responses
		var err error
		body, err := io.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("failed to read response body: %v", err)
		}
		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("server returned non-200 status: %d, body: %s", resp.StatusCode, body)
		}

		var response *ai.ModelResponse
		if isChatModel {
			response, err = translateChatResponse(body, thinkingEnabled)
		} else {
			response, err = translateModelResponse(body)
		}
		response.Request = input
		if err != nil {
			return nil, fmt.Errorf("failed to parse response: %v", err)
		}
		return response, nil
	} else {
		var chunks []*ai.ModelResponseChunk
		decoder := json.NewDecoder(resp.Body)
		chunkCount := 0

		for {
			var raw json.RawMessage
			if err := decoder.Decode(&raw); err == io.EOF {
				break
			} else if err != nil {
				return nil, fmt.Errorf("reading response stream: %v", err)
			}
			chunkCount++

			var chunk *ai.ModelResponseChunk
			if isChatModel {
				chunk, err = translateChatChunk(string(raw))
			} else {
				chunk, err = translateGenerateChunk(string(raw))
			}
			if err != nil {
				return nil, fmt.Errorf("failed to translate chunk: %v", err)
			}
			chunks = append(chunks, chunk)
			cb(ctx, chunk)
		}

		// Create a final response with the merged chunks
		finalResponse := &ai.ModelResponse{
			Request:      input,
			FinishReason: ai.FinishReason("stop"),
			Message: &ai.Message{
				Role: ai.RoleModel,
			},
		}
		// Add all the merged content to the final response's candidate
		for _, chunk := range chunks {
			finalResponse.Message.Content = append(finalResponse.Message.Content, chunk.Content...)
		}
		return finalResponse, nil // Return the final merged response

	}
}

// convertTools converts Genkit tool definitions to Ollama tool format
func convertTools(tools []*ai.ToolDefinition) ([]ollamaTool, error) {
	ollamaTools := make([]ollamaTool, 0, len(tools))
	for _, tool := range tools {
		ollamaTools = append(ollamaTools, ollamaTool{
			Type: "function",
			Function: ollamaFunction{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  tool.InputSchema,
			},
		})
	}
	return ollamaTools, nil
}

func convertParts(role ai.Role, parts []*ai.Part) (*ollamaMessage, error) {
	message := &ollamaMessage{
		Role: roleMapping[role],
	}
	var contentBuilder strings.Builder
	var toolCalls []ollamaToolCall
	var images []string
	for _, part := range parts {
		if part.IsText() {
			contentBuilder.WriteString(part.Text)
		} else if part.IsMedia() {
			_, data, err := uri.Data(part)
			if err != nil {
				return nil, fmt.Errorf("failed to extract media data: %v", err)
			}
			base64Encoded := base64.StdEncoding.EncodeToString(data)
			images = append(images, base64Encoded)
		} else if part.IsToolRequest() {
			toolReq := part.ToolRequest
			toolCalls = append(toolCalls, ollamaToolCall{
				Function: ollamaFunctionCall{
					Name:      toolReq.Name,
					Arguments: toolReq.Input,
				},
			})
		} else if part.IsToolResponse() {
			toolResp := part.ToolResponse
			outputJSON, err := json.Marshal(toolResp.Output)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal tool response: %v", err)
			}
			contentBuilder.WriteString(string(outputJSON))
		} else if part.IsReasoning() {
			contentBuilder.WriteString(part.Text)
		} else {
			return nil, errors.New("unsupported content type")
		}
	}

	message.Content = contentBuilder.String()
	if len(toolCalls) > 0 {
		message.ToolCalls = toolCalls
	}
	if len(images) > 0 {
		message.Images = images
	}
	return message, nil
}

// translateChatResponse translates Ollama chat response into a genkit response.
// When thinkingEnabled is true, the function will also parse <think>/<thinking>
// tags from content text as a fallback for models that don't return a dedicated
// "thinking" JSON field.
func translateChatResponse(responseData []byte, thinkingEnabled bool) (*ai.ModelResponse, error) {
	var response ollamaChatResponse

	if err := json.Unmarshal(responseData, &response); err != nil {
		return nil, fmt.Errorf("failed to parse response JSON: %v", err)
	}

	modelResponse := &ai.ModelResponse{
		FinishReason: ai.FinishReason("stop"),
		Message: &ai.Message{
			Role: ai.RoleModel,
		},
	}

	// Check for thinking/reasoning in the dedicated JSON field first.
	if response.Message.Thinking != "" {
		aiPart := ai.NewReasoningPart(response.Message.Thinking, nil)
		modelResponse.Message.Content = append(modelResponse.Message.Content, aiPart)
	} else if thinkingEnabled {
		// Only parse <think>/<thinking> tags from content when thinking was
		// explicitly requested. Without this guard, a model could legitimately
		// return these tags as part of normal text output and they would be
		// incorrectly hijacked.
		thinking, content := parseThinking(response.Message.Content)
		if thinking != "" {
			aiPart := ai.NewReasoningPart(thinking, nil)
			modelResponse.Message.Content = append(modelResponse.Message.Content, aiPart)
			response.Message.Content = content
		}
	}

	if len(response.Message.ToolCalls) > 0 {
		for _, toolCall := range response.Message.ToolCalls {
			toolRequest := &ai.ToolRequest{
				Name:  toolCall.Function.Name,
				Input: toolCall.Function.Arguments,
			}
			toolPart := ai.NewToolRequestPart(toolRequest)
			modelResponse.Message.Content = append(modelResponse.Message.Content, toolPart)
		}
	}

	// Add remaining content as text if present
	if response.Message.Content != "" {
		aiPart := ai.NewTextPart(response.Message.Content)
		modelResponse.Message.Content = append(modelResponse.Message.Content, aiPart)
	}

	return modelResponse, nil
}

// translateModelResponse translates Ollama generate response into a genkit response.
func translateModelResponse(responseData []byte) (*ai.ModelResponse, error) {
	var response ollamaModelResponse

	if err := json.Unmarshal(responseData, &response); err != nil {
		return nil, fmt.Errorf("failed to parse response JSON: %v", err)
	}

	modelResponse := &ai.ModelResponse{
		FinishReason: ai.FinishReason("stop"),
		Message: &ai.Message{
			Role: ai.RoleModel,
		},
	}

	aiPart := ai.NewTextPart(response.Response)
	modelResponse.Message.Content = append(modelResponse.Message.Content, aiPart)
	modelResponse.Usage = &ai.GenerationUsage{} // TODO: can we get any of this info?
	return modelResponse, nil
}

func translateChatChunk(input string) (*ai.ModelResponseChunk, error) {
	var response ollamaChatResponse

	if err := json.Unmarshal([]byte(input), &response); err != nil {
		return nil, fmt.Errorf("failed to parse response JSON: %v", err)
	}
	chunk := &ai.ModelResponseChunk{}

	// Check for thinking/reasoning first
	if response.Message.Thinking != "" {
		aiPart := ai.NewReasoningPart(response.Message.Thinking, nil)
		chunk.Content = append(chunk.Content, aiPart)
	}

	if response.Message.Content != "" {
		aiPart := ai.NewTextPart(response.Message.Content)
		chunk.Content = append(chunk.Content, aiPart)
	}
	if len(response.Message.ToolCalls) > 0 {
		for _, toolCall := range response.Message.ToolCalls {
			toolRequest := &ai.ToolRequest{
				Name:  toolCall.Function.Name,
				Input: toolCall.Function.Arguments,
			}
			toolPart := ai.NewToolRequestPart(toolRequest)
			chunk.Content = append(chunk.Content, toolPart)
		}
	}

	return chunk, nil
}

func translateGenerateChunk(input string) (*ai.ModelResponseChunk, error) {
	var response ollamaModelResponse

	if err := json.Unmarshal([]byte(input), &response); err != nil {
		return nil, fmt.Errorf("failed to parse response JSON: %v", err)
	}
	chunk := &ai.ModelResponseChunk{}
	aiPart := ai.NewTextPart(response.Response)
	chunk.Content = append(chunk.Content, aiPart)
	return chunk, nil
}

// concatMessages translates a list of messages into a prompt-style format
func concatMessages(input *ai.ModelRequest, roles []ai.Role) string {
	roleSet := make(map[ai.Role]bool)
	for _, role := range roles {
		roleSet[role] = true // Create a set for faster lookup
	}
	var sb strings.Builder
	for _, message := range input.Messages {
		// Check if the message role is in the allowed set
		if !roleSet[message.Role] {
			continue
		}
		for _, part := range message.Content {
			if !part.IsText() {
				continue
			}
			sb.WriteString(part.Text)
		}
	}
	return sb.String()
}

// concatImages grabs the images from genkit message parts
func concatImages(input *ai.ModelRequest, roleFilter []ai.Role) ([]string, error) {
	roleSet := make(map[ai.Role]bool)
	for _, role := range roleFilter {
		roleSet[role] = true
	}

	var images []string

	for _, message := range input.Messages {
		// Check if the message role is in the allowed set
		if roleSet[message.Role] {
			for _, part := range message.Content {
				if !part.IsMedia() {
					continue
				}

				// Get the media type and data
				mediaType, data, err := uri.Data(part)
				if err != nil {
					return nil, fmt.Errorf("failed to extract image data: %v", err)
				}

				// Only include image media types
				if !strings.HasPrefix(mediaType, "image/") {
					continue
				}

				base64Encoded := base64.StdEncoding.EncodeToString(data)
				images = append(images, base64Encoded)
			}
		}
	}
	return images, nil
}

// parseThinking extracts the thinking content from the response string.
func parseThinking(content string) (string, string) {
	matches := thinkingRegex.FindAllStringSubmatch(content, -1)
	if len(matches) == 0 {
		return "", content
	}

	var thinkingParts []string
	for _, match := range matches {
		thinkingParts = append(thinkingParts, strings.TrimSpace(match[2]))
	}

	rest := thinkingRegex.ReplaceAllString(content, "")
	return strings.Join(thinkingParts, "\n\n"), strings.TrimSpace(rest)
}
