// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"google.golang.org/genai"
)

func TestDeepResearchModelType(t *testing.T) {
	t.Parallel()

	mt := ClassifyModel(deepResearchPreview042026)
	if mt != ModelTypeDeepResearch {
		t.Fatalf("ClassifyModel(%q) = %v, want %v", deepResearchPreview042026, mt, ModelTypeDeepResearch)
	}
	if mt.ActionType() != api.ActionTypeBackgroundModel {
		t.Errorf("Deep Research ActionType = %v, want %v", mt.ActionType(), api.ActionTypeBackgroundModel)
	}
	if mt.DefaultSupports() != &DeepResearchSupports {
		t.Errorf("Deep Research DefaultSupports = %v, want DeepResearchSupports", mt.DefaultSupports())
	}
	if _, ok := mt.DefaultConfig().(*DeepResearchConfig); !ok {
		t.Errorf("Deep Research DefaultConfig = %T, want *DeepResearchConfig", mt.DefaultConfig())
	}
}

func TestDeepResearchModelOptions(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		wantTools   bool
		wantMedia   bool
		wantOutputs int
	}{
		{name: deepResearchProPreview122025, wantTools: false, wantMedia: false, wantOutputs: 1},
		{name: deepResearchPreview042026, wantTools: true, wantMedia: true, wantOutputs: 2},
		{name: deepResearchMaxPreview042026, wantTools: true, wantMedia: true, wantOutputs: 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			opts := GetModelOptions(tt.name, googleAIProvider)
			if opts.Supports == nil {
				t.Fatal("GetModelOptions().Supports is nil")
			}
			if !opts.Supports.LongRunning {
				t.Error("Deep Research model should be long-running")
			}
			if opts.Supports.Tools != tt.wantTools {
				t.Errorf("Tools = %t, want %t", opts.Supports.Tools, tt.wantTools)
			}
			if opts.Supports.Media != tt.wantMedia {
				t.Errorf("Media = %t, want %t", opts.Supports.Media, tt.wantMedia)
			}
			if len(opts.Supports.Output) != tt.wantOutputs {
				t.Errorf("Output count = %d, want %d", len(opts.Supports.Output), tt.wantOutputs)
			}
			if opts.ConfigSchema == nil {
				t.Error("ConfigSchema is nil")
			}
		})
	}
}

func TestResolveDeepResearchActions(t *testing.T) {
	t.Parallel()

	client := newDeepResearchTestClient(t, "http://127.0.0.1")
	if got := resolveAction(client, googleAIProvider, api.ActionTypeModel, deepResearchPreview042026); got != nil {
		t.Fatalf("ActionTypeModel resolved Deep Research action %v, want nil", got)
	}
	if got := resolveAction(client, googleAIProvider, api.ActionTypeBackgroundModel, deepResearchPreview042026); got == nil {
		t.Fatal("ActionTypeBackgroundModel returned nil for Deep Research")
	}
	if got := resolveAction(client, googleAIProvider, api.ActionTypeCheckOperation, deepResearchPreview042026); got == nil {
		t.Fatal("ActionTypeCheckOperation returned nil for Deep Research")
	}
	if got := resolveAction(client, googleAIProvider, api.ActionTypeCancelOperation, deepResearchPreview042026); got == nil {
		t.Fatal("ActionTypeCancelOperation returned nil for Deep Research")
	}
}

func TestListGenaiModelsIncludesDeepResearch(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			t.Errorf("method = %s, want GET", r.Method)
		}
		_ = json.NewEncoder(w).Encode(map[string]any{
			"models": []map[string]any{
				{"name": "models/" + deepResearchPreview042026},
				{"name": "models/gemini-2.5-flash"},
				{"name": "models/veo-3.1-generate-preview"},
			},
		})
	}))
	defer server.Close()

	client := newDeepResearchTestClient(t, server.URL)
	models, err := listGenaiModels(context.Background(), client)
	if err != nil {
		t.Fatalf("listGenaiModels() error = %v", err)
	}
	if len(models.deep) != 1 || models.deep[0] != deepResearchPreview042026 {
		t.Fatalf("models.deep = %v, want [%s]", models.deep, deepResearchPreview042026)
	}
}

func TestDeepResearchBackgroundModelStartCheckCancel(t *testing.T) {
	ctx := context.Background()

	var startRequest map[string]any
	var sawCheck bool
	var sawCancel bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Api-Revision"); got != deepResearchAPIRevision {
			t.Errorf("Api-Revision = %q, want %q", got, deepResearchAPIRevision)
		}
		// The start request carries the per-request key override; check and cancel
		// re-derive the key from the live client (the override is not persisted in
		// operation metadata), so they send the client's key instead.
		wantKey := "test-key"
		if r.Method == http.MethodPost && r.URL.Path == "/v1beta/interactions" {
			wantKey = "override-key"
		}
		if got := r.Header.Get("x-goog-api-key"); got != wantKey {
			t.Errorf("x-goog-api-key = %q, want %q", got, wantKey)
		}

		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1beta/interactions":
			if err := json.NewDecoder(r.Body).Decode(&startRequest); err != nil {
				t.Errorf("decode start request: %v", err)
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id":     "interaction-1",
				"status": "in_progress",
			})
		case r.Method == http.MethodGet && r.URL.Path == "/v1beta/interactions/interaction-1":
			sawCheck = true
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id":     "interaction-1",
				"status": "completed",
				"steps": []map[string]any{
					{
						"type": "model_output",
						"content": []map[string]any{
							{"type": "text", "text": "research result"},
							{"type": "image", "mime_type": "image/png", "data": "aW1hZ2U="},
						},
					},
					{
						"type":      "google_search_call",
						"id":        "search-1",
						"arguments": map[string]any{"queries": []string{"genkit"}},
						"signature": "sig-1",
					},
				},
				"usage": map[string]any{
					"total_input_tokens":  11,
					"total_output_tokens": 7,
					"total_tokens":        18,
				},
			})
		case r.Method == http.MethodPost && r.URL.Path == "/v1beta/interactions/interaction-1/cancel":
			sawCancel = true
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id":     "interaction-1",
				"status": "cancelled",
			})
		default:
			t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	client := newDeepResearchTestClient(t, server.URL)
	model := newDeepResearchModel(client, deepResearchPreview042026, GetModelOptions(deepResearchPreview042026, googleAIProvider))

	store := true
	collaborativePlanning := true
	op, err := model.Start(ctx, &ai.ModelRequest{
		Config: &DeepResearchConfig{
			APIKey:                "override-key",
			BaseURL:               server.URL,
			APIVersion:            "v1beta",
			ThinkingSummaries:     "AUTO",
			Visualization:         "OFF",
			CollaborativePlanning: &collaborativePlanning,
			PreviousInteractionID: "previous-1",
			Store:                 &store,
			ResponseModalities:    []string{"TEXT", "IMAGE"},
			GoogleSearch:          true,
		},
		Messages: []*ai.Message{
			{Role: ai.RoleSystem, Content: []*ai.Part{ai.NewTextPart("system guidance")}},
			{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("research this")}},
		},
		Tools: []*ai.ToolDefinition{
			{
				Name:        "lookup",
				Description: "Look up a fact",
				InputSchema: map[string]any{
					"type": "object",
				},
			},
		},
		Output: &ai.ModelOutputConfig{
			Format:      ai.OutputFormatJSON,
			ContentType: "application/json",
			Schema:      map[string]any{"type": "object"},
		},
	})
	if err != nil {
		t.Fatalf("Start() error = %v", err)
	}
	if op.ID != "interaction-1" {
		t.Fatalf("op.ID = %q, want interaction-1", op.ID)
	}
	if op.Done {
		t.Fatal("Start() returned done operation, want in-progress")
	}

	assertDeepResearchStartRequest(t, startRequest)

	checked, err := model.Check(ctx, op)
	if err != nil {
		t.Fatalf("Check() error = %v", err)
	}
	if !sawCheck {
		t.Fatal("server did not receive check request")
	}
	if !checked.Done {
		t.Fatal("Check() returned not done, want done")
	}
	if checked.Output == nil || checked.Output.Message == nil {
		t.Fatal("Check() output message is nil")
	}
	if got := checked.Output.Message.Content[0].Text; got != "research result" {
		t.Errorf("output text = %q, want research result", got)
	}
	if got := checked.Output.Message.Content[1].Text; got != "data:image/png;base64,aW1hZ2U=" {
		t.Errorf("output media = %q, want data URI", got)
	}
	if got := checked.Output.Message.Content[2].Custom["googleSearchCall"].(map[string]any)["id"]; got != "search-1" {
		t.Errorf("googleSearchCall.id = %v, want search-1", got)
	}
	if checked.Output.Usage == nil || checked.Output.Usage.TotalTokens != 18 {
		t.Errorf("usage = %#v, want total tokens 18", checked.Output.Usage)
	}

	cancelled, err := model.Cancel(ctx, op)
	if err != nil {
		t.Fatalf("Cancel() error = %v", err)
	}
	if !sawCancel {
		t.Fatal("server did not receive cancel request")
	}
	if !cancelled.Done || cancelled.Output == nil || cancelled.Output.FinishReason != ai.FinishReasonInterrupted {
		t.Fatalf("Cancel() = %#v, want done interrupted output", cancelled)
	}
}

func TestDeepResearchCheckFailedStatus(t *testing.T) {
	ctx := context.Background()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case r.Method == http.MethodPost && r.URL.Path == "/v1beta/interactions":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id":     "interaction-1",
				"status": "in_progress",
			})
		case r.Method == http.MethodGet && r.URL.Path == "/v1beta/interactions/interaction-1":
			_ = json.NewEncoder(w).Encode(map[string]any{
				"id":     "interaction-1",
				"status": "failed",
				"error": map[string]any{
					"code":    "INTERNAL",
					"message": "research backend exploded",
				},
			})
		default:
			t.Errorf("unexpected request %s %s", r.Method, r.URL.Path)
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	client := newDeepResearchTestClient(t, server.URL)
	model := newDeepResearchModel(client, deepResearchPreview042026, GetModelOptions(deepResearchPreview042026, googleAIProvider))

	op, err := model.Start(ctx, &ai.ModelRequest{
		Messages: []*ai.Message{
			{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("research this")}},
		},
	})
	if err != nil {
		t.Fatalf("Start() error = %v", err)
	}

	checked, err := model.Check(ctx, op)
	if err != nil {
		t.Fatalf("Check() error = %v", err)
	}
	if !checked.Done {
		t.Fatal("Check() returned not done, want done")
	}
	if checked.Error == nil {
		t.Fatal("Check() error field is nil, want failure detail")
	}
	if got := checked.Error.Error(); !strings.Contains(got, "research backend exploded") {
		t.Errorf("error = %q, want it to contain the API failure message", got)
	}
}

func TestDeepResearchDefensiveHandling(t *testing.T) {
	t.Parallel()

	if _, err := deepResearchConfigFromRequest(nil); err != nil {
		t.Fatalf("deepResearchConfigFromRequest(nil) error = %v", err)
	}
	req, err := toDeepResearchInteractionRequest(deepResearchPreview042026, nil, nil)
	if err != nil {
		t.Fatalf("toDeepResearchInteractionRequest(nil) error = %v", err)
	}
	if req == nil || req.Input == nil {
		t.Fatalf("toDeepResearchInteractionRequest(nil) = %#v, want non-nil request and input", req)
	}
	if got := toDeepResearchResponseFormat(nil); got != nil {
		t.Fatalf("toDeepResearchResponseFormat(nil) = %#v, want nil", got)
	}
	if got := toInteractionTools(nil, &DeepResearchConfig{GoogleSearch: true}); len(got) != 1 || got[0]["type"] != "google_search" {
		t.Fatalf("toInteractionTools(nil, googleSearch) = %#v, want google_search tool", got)
	}
	if _, err := toInteractionSteps([]*ai.Message{
		{
			Role: ai.RoleUser,
			Content: []*ai.Part{
				{Kind: ai.PartToolRequest},
				{Kind: ai.PartToolResponse},
				nil,
				ai.NewCustomPart(map[string]any{
					"googleSearchCall": map[string]any{
						"id":        "search-1",
						"arguments": map[string]any{"queries": []string{"genkit"}},
					},
				}),
				ai.NewTextPart("hello"),
			},
		},
	}); err != nil {
		t.Fatalf("toInteractionSteps() with malformed tool parts error = %v", err)
	}
	if _, err := doDeepResearchRequest(context.Background(), nil, deepResearchClientOptions{}, http.MethodGet, "interactions/1", nil); err == nil {
		t.Fatal("doDeepResearchRequest(nil client) error = nil, want error")
	}

	model := newDeepResearchModel(nil, deepResearchPreview042026, GetModelOptions(deepResearchPreview042026, googleAIProvider))
	if _, err := model.Check(context.Background(), nil); err == nil {
		t.Fatal("Check(nil) error = nil, want error")
	}
	if _, err := model.Cancel(context.Background(), &ai.ModelOperation{}); err == nil {
		t.Fatal("Cancel(empty ID) error = nil, want error")
	}
}

func assertDeepResearchStartRequest(t *testing.T, req map[string]any) {
	t.Helper()

	if req["agent"] != deepResearchPreview042026 {
		t.Errorf("agent = %v, want %s", req["agent"], deepResearchPreview042026)
	}
	if req["background"] != true {
		t.Errorf("background = %v, want true", req["background"])
	}
	if req["previous_interaction_id"] != "previous-1" {
		t.Errorf("previous_interaction_id = %v, want previous-1", req["previous_interaction_id"])
	}
	if req["store"] != true {
		t.Errorf("store = %v, want true", req["store"])
	}

	agentConfig, ok := req["agent_config"].(map[string]any)
	if !ok {
		t.Fatalf("agent_config = %T, want object", req["agent_config"])
	}
	if agentConfig["type"] != "deep-research" {
		t.Errorf("agent_config.type = %v, want deep-research", agentConfig["type"])
	}
	if agentConfig["thinking_summaries"] != "auto" {
		t.Errorf("thinking_summaries = %v, want auto", agentConfig["thinking_summaries"])
	}
	if agentConfig["visualization"] != "off" {
		t.Errorf("visualization = %v, want off", agentConfig["visualization"])
	}
	if agentConfig["collaborative_planning"] != true {
		t.Errorf("collaborative_planning = %v, want true", agentConfig["collaborative_planning"])
	}

	input, ok := req["input"].([]any)
	if !ok || len(input) != 3 {
		t.Fatalf("input = %#v, want simulated system prompt plus user input", req["input"])
	}
	firstStep := input[0].(map[string]any)
	if firstStep["type"] != "user_input" {
		t.Errorf("input[0].type = %v, want user_input", firstStep["type"])
	}
	secondStep := input[1].(map[string]any)
	if secondStep["type"] != "model_output" {
		t.Errorf("input[1].type = %v, want model_output", secondStep["type"])
	}
	thirdStep := input[2].(map[string]any)
	if thirdStep["type"] != "user_input" {
		t.Errorf("input[2].type = %v, want user_input", thirdStep["type"])
	}

	tools, ok := req["tools"].([]any)
	if !ok || len(tools) != 2 {
		t.Fatalf("tools = %#v, want function tool and google_search", req["tools"])
	}
	tool0 := tools[0].(map[string]any)
	if tool0["type"] != "function" || tool0["name"] != "lookup" {
		t.Errorf("function tool = %#v, want lookup function", tool0)
	}
	tool1 := tools[1].(map[string]any)
	if tool1["type"] != "google_search" {
		t.Errorf("second tool = %#v, want google_search", tool1)
	}

	responseFormat := req["response_format"].(map[string]any)
	if responseFormat["mime_type"] != "application/json" {
		t.Errorf("response_format.mime_type = %v, want application/json", responseFormat["mime_type"])
	}
	modalities := req["response_modalities"].([]any)
	if modalities[0] != "text" || modalities[1] != "image" {
		t.Errorf("response_modalities = %v, want [text image]", modalities)
	}
}

func newDeepResearchTestClient(t *testing.T, baseURL string) *genai.Client {
	t.Helper()

	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		Backend: genai.BackendGeminiAPI,
		APIKey:  "test-key",
		HTTPOptions: genai.HTTPOptions{
			BaseURL:    baseURL,
			APIVersion: "v1beta",
			Headers:    genkitClientHeader,
		},
	})
	if err != nil {
		t.Fatalf("genai.NewClient() error = %v", err)
	}
	return client
}
