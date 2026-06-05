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
	"net/url"
	"strings"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/internal/base"
	"google.golang.org/genai"
)

const (
	deepResearchAgentType   = "deep-research"
	deepResearchAPIRevision = "2026-05-20"
)

// DeepResearchConfig configures Google AI Deep Research interactions.
type DeepResearchConfig struct {
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
	ResponseFormat        map[string]any   `json:"response_format,omitempty"`
	ResponseModalities    []string         `json:"response_modalities,omitempty"`
	Store                 *bool            `json:"store,omitempty"`
	Background            bool             `json:"background,omitempty"`
	AgentConfig           map[string]any   `json:"agent_config,omitempty"`
}

type geminiInteraction struct {
	Model                 string            `json:"model,omitempty"`
	Agent                 string            `json:"agent,omitempty"`
	EnvironmentID         string            `json:"environment_id,omitempty"`
	ID                    string            `json:"id,omitempty"`
	PreviousInteractionID string            `json:"previous_interaction_id,omitempty"`
	Status                string            `json:"status,omitempty"`
	Created               string            `json:"created,omitempty"`
	Updated               string            `json:"updated,omitempty"`
	Role                  string            `json:"role,omitempty"`
	Steps                 []interactionStep `json:"steps,omitempty"`
	Usage                 *interactionUsage `json:"usage,omitempty"`
	Error                 *interactionError `json:"error,omitempty"`
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
func newDeepResearchModel(client *genai.Client, name string, info ai.ModelOptions) ai.BackgroundModel {
	provider := googleAIProvider
	if client != nil && client.ClientConfig().Backend == genai.BackendVertexAI {
		provider = vertexAIProvider
	}

	startFunc := func(ctx context.Context, req *ai.ModelRequest) (*ai.ModelOperation, error) {
		config, err := deepResearchConfigFromRequest(req)
		if err != nil {
			return nil, err
		}
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
		op.Metadata["clientOptions"] = clientOptions
		op.Metadata["inputRequest"] = req
		op.Metadata["startTime"] = time.Now()
		return op, nil
	}

	checkFunc := func(ctx context.Context, op *ai.ModelOperation) (*ai.ModelOperation, error) {
		if op == nil {
			return nil, core.NewPublicError(core.INVALID_ARGUMENT, "Deep Research operation is nil", nil)
		}
		if op.ID == "" {
			return nil, core.NewPublicError(core.INVALID_ARGUMENT, "Deep Research operation is missing an ID", nil)
		}
		clientOptions := deepResearchOptionsFromOperation(client, op)
		interaction, err := doDeepResearchRequest(ctx, client, clientOptions, http.MethodGet, "interactions/"+op.ID, nil)
		if err != nil {
			return nil, err
		}

		updatedOp := fromDeepResearchInteraction(interaction)
		restoreDeepResearchMetadata(updatedOp, op)
		if updatedOp.Done && updatedOp.Output != nil {
			if req, ok := updatedOp.Metadata["inputRequest"].(*ai.ModelRequest); ok {
				ai.CalculateInputOutputUsage(req, updatedOp.Output)
			} else {
				ai.CalculateInputOutputUsage(nil, updatedOp.Output)
			}
			if startTime, ok := updatedOp.Metadata["startTime"].(time.Time); ok && updatedOp.Output.LatencyMs == 0 {
				updatedOp.Output.LatencyMs = float64(time.Since(startTime).Nanoseconds()) / 1e6
			}
		}
		return updatedOp, nil
	}

	cancelFunc := func(ctx context.Context, op *ai.ModelOperation) (*ai.ModelOperation, error) {
		if op == nil {
			return nil, core.NewPublicError(core.INVALID_ARGUMENT, "Deep Research operation is nil", nil)
		}
		if op.ID == "" {
			return nil, core.NewPublicError(core.INVALID_ARGUMENT, "Deep Research operation is missing an ID", nil)
		}
		clientOptions := deepResearchOptionsFromOperation(client, op)
		interaction, err := doDeepResearchRequest(ctx, client, clientOptions, http.MethodPost, "interactions/"+op.ID+"/cancel", nil)
		if err != nil {
			return nil, err
		}

		updatedOp := fromDeepResearchInteraction(interaction)
		if updatedOp.ID == "" {
			updatedOp.ID = op.ID
		}
		restoreDeepResearchMetadata(updatedOp, op)
		return updatedOp, nil
	}

	return ai.NewBackgroundModel(
		api.NewName(provider, name),
		&ai.BackgroundModelOptions{ModelOptions: info, Cancel: cancelFunc},
		startFunc,
		checkFunc,
	)
}

func deepResearchConfigFromRequest(input *ai.ModelRequest) (*DeepResearchConfig, error) {
	var result DeepResearchConfig
	if input == nil {
		return &result, nil
	}

	switch config := input.Config.(type) {
	case nil:
	case DeepResearchConfig:
		result = config
	case *DeepResearchConfig:
		if config != nil {
			result = *config
		}
	case map[string]any:
		var err error
		result, err = base.MapToStruct[DeepResearchConfig](config)
		if err != nil {
			return nil, core.NewPublicError(core.INVALID_ARGUMENT, fmt.Sprintf("The Deep Research configuration settings are not in the correct format. Check that the names and values match what the model expects: %v", err), nil)
		}
	default:
		return nil, core.NewPublicError(core.INVALID_ARGUMENT, fmt.Sprintf("Invalid configuration type: %T. Expected *googlegenai.DeepResearchConfig. Ensure you are using DeepResearchModelRef or passing the correct configuration struct.", input.Config), nil)
	}

	return &result, nil
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
	if op != nil && op.Metadata != nil {
		if stored, ok := op.Metadata["clientOptions"].(deepResearchClientOptions); ok {
			return stored
		}
		if storedMap, ok := op.Metadata["clientOptions"].(map[string]any); ok {
			if stored, err := base.MapToStruct[deepResearchClientOptions](storedMap); err == nil {
				return stored
			}
		}
	}
	return deepResearchOptionsFromConfig(client, nil)
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
		Agent:              model,
		Input:              steps,
		Background:         true,
		AgentConfig:        agentConfig,
		Tools:              toInteractionTools(req, config),
		ResponseFormat:     toDeepResearchResponseFormat(req),
		ResponseModalities: toDeepResearchResponseModalities(config),
	}
	if config != nil {
		interactionReq.PreviousInteractionID = config.PreviousInteractionID
		interactionReq.Store = config.Store
	}
	return interactionReq, nil
}

func toDeepResearchResponseFormat(req *ai.ModelRequest) map[string]any {
	if req == nil || req.Output == nil {
		return nil
	}
	if req.Output.Format != ai.OutputFormatJSON && req.Output.ContentType != "application/json" {
		return nil
	}
	responseFormat := map[string]any{
		"type":      "text",
		"mime_type": "application/json",
	}
	if req.Output.Schema != nil {
		responseFormat["schema"] = req.Output.Schema
	}
	return responseFormat
}

func toDeepResearchResponseModalities(config *DeepResearchConfig) []string {
	if config == nil || len(config.ResponseModalities) == 0 {
		return nil
	}
	modalities := make([]string, 0, len(config.ResponseModalities))
	for _, modality := range config.ResponseModalities {
		modalities = append(modalities, strings.ToLower(modality))
	}
	return modalities
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
				interactionTool["parameters"] = tool.InputSchema
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
	if part.IsText() || part.IsData() {
		return interactionContent{Type: "text", Text: part.Text}, nil
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
	url, err := deepResearchURL(opts, resourcePath)
	if err != nil {
		return nil, err
	}

	var bodyReader io.Reader
	if body != nil {
		data, err := json.Marshal(body)
		if err != nil {
			return nil, err
		}
		bodyReader = bytes.NewReader(data)
	}

	req, err := http.NewRequestWithContext(ctx, method, url, bodyReader)
	if err != nil {
		return nil, err
	}
	for k, values := range client.ClientConfig().HTTPOptions.Headers {
		for _, value := range values {
			req.Header.Add(k, value)
		}
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Api-Revision", deepResearchAPIRevision)
	if opts.APIKey != "" {
		req.Header.Set("x-goog-api-key", opts.APIKey)
	}

	httpClient := client.ClientConfig().HTTPClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, core.NewError(core.StatusFromHTTPCode(resp.StatusCode), "deep research request failed: %s", strings.TrimSpace(string(respBody)))
	}

	var interaction geminiInteraction
	if len(respBody) > 0 {
		if err := json.Unmarshal(respBody, &interaction); err != nil {
			return nil, err
		}
	}
	return &interaction, nil
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
	case "in_progress", "requires_action", "":
		op.Done = false
	case "cancelled":
		op.Done = true
		op.Output = &ai.ModelResponse{
			Message: &ai.Message{
				Role:    ai.RoleModel,
				Content: []*ai.Part{ai.NewTextPart("Operation cancelled.")},
				Metadata: map[string]any{
					"interactionId": interaction.ID,
				},
			},
			FinishReason:  ai.FinishReasonInterrupted,
			FinishMessage: "Operation cancelled",
			Raw:           interaction,
		}
	case "failed":
		op.Done = true
		op.Error = fmt.Errorf("deep research interaction failed: %s", deepResearchFailureDetail(interaction))
	case "completed":
		op.Done = true
		op.Output = fromCompletedDeepResearchInteraction(interaction)
	default:
		op.Done = false
	}
	return op
}

// deepResearchFailureDetail extracts a human-readable reason from a failed interaction,
// falling back to a generic message when the API provides no detail.
func deepResearchFailureDetail(interaction *geminiInteraction) string {
	if interaction == nil || interaction.Error == nil {
		return "no error detail provided"
	}
	err := interaction.Error
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
