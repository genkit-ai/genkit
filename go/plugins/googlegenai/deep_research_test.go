// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
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
	if got := resolveAction(client, catalog{provider: googleAIProvider}, api.ActionTypeModel, deepResearchPreview042026); got != nil {
		t.Fatalf("ActionTypeModel resolved Deep Research action %v, want nil", got)
	}
	if got := resolveAction(client, catalog{provider: googleAIProvider}, api.ActionTypeBackgroundModel, deepResearchPreview042026); got == nil {
		t.Fatal("ActionTypeBackgroundModel returned nil for Deep Research")
	}
	if got := resolveAction(client, catalog{provider: googleAIProvider}, api.ActionTypeCheckOperation, deepResearchPreview042026+"/check"); got == nil {
		t.Fatal("ActionTypeCheckOperation returned nil for Deep Research")
	}
	if got := resolveAction(client, catalog{provider: googleAIProvider}, api.ActionTypeCancelOperation, deepResearchPreview042026+"/cancel"); got == nil {
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

	req, err := toDeepResearchInteractionRequest(deepResearchPreview042026, nil, nil)
	if err != nil {
		t.Fatalf("toDeepResearchInteractionRequest(nil) error = %v", err)
	}
	if req == nil || req.Input == nil {
		t.Fatalf("toDeepResearchInteractionRequest(nil) = %#v, want non-nil request and input", req)
	}
	if got := toDeepResearchResponseFormat(nil, nil); got != nil {
		t.Fatalf("toDeepResearchResponseFormat(nil, nil) = %#v, want nil", got)
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

	if _, ok := req["response_modalities"]; ok {
		t.Error("deprecated response_modalities was sent")
	}
	formats := req["response_format"].([]any)
	if len(formats) != 2 {
		t.Fatalf("response_format = %v, want 2 entries", formats)
	}
	text := formats[0].(map[string]any)
	if text["type"] != "text" || text["mime_type"] != "application/json" {
		t.Errorf("response_format[0] = %v, want JSON text entry", text)
	}
	if image := formats[1].(map[string]any); image["type"] != "image" {
		t.Errorf("response_format[1] = %v, want image entry", image)
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

// deepResearchTestServer serves handler and returns a model on it.
func deepResearchTestModel(t *testing.T, handler http.HandlerFunc) (*ai.BackgroundModelAction, *genai.Client, *httptest.Server) {
	t.Helper()
	server := httptest.NewServer(handler)
	t.Cleanup(server.Close)
	client := newDeepResearchTestClient(t, server.URL)
	return newDeepResearchModel(client, deepResearchPreview042026, GetModelOptions(deepResearchPreview042026, googleAIProvider)), client, server
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func TestDeepResearchStatuses(t *testing.T) {
	t.Parallel()
	tests := []struct {
		status     string
		wantDone   bool
		wantFinish ai.FinishReason
		wantMeta   bool
		wantErr    string
	}{
		{status: "requires_action", wantDone: true, wantFinish: ai.FinishReasonStop, wantMeta: true},
		{status: "incomplete", wantDone: true, wantFinish: ai.FinishReasonLength},
		{status: "queued", wantDone: false},
		{status: "bogus", wantDone: true, wantErr: "Unknown interaction status"},
	}
	for _, tt := range tests {
		t.Run(tt.status, func(t *testing.T) {
			t.Parallel()
			model, _, _ := deepResearchTestModel(t, func(w http.ResponseWriter, r *http.Request) {
				writeJSON(w, 200, map[string]any{
					"id": "i-1", "status": tt.status,
					"steps": []map[string]any{{"type": "model_output", "content": []map[string]any{{"type": "text", "text": "partial"}}}},
				})
			})
			op, err := model.Check(context.Background(), &ai.ModelOperation{ID: "i-1"})
			if err != nil {
				t.Fatalf("Check() error = %v", err)
			}
			if op.Done != tt.wantDone {
				t.Fatalf("Done = %t, want %t", op.Done, tt.wantDone)
			}
			if tt.wantErr != "" {
				if op.Error == nil || !strings.Contains(op.Error.Error(), tt.wantErr) {
					t.Fatalf("Error = %v, want %q", op.Error, tt.wantErr)
				}
				return
			}
			if !tt.wantDone {
				return
			}
			if op.Output.FinishReason != tt.wantFinish {
				t.Errorf("FinishReason = %q, want %q", op.Output.FinishReason, tt.wantFinish)
			}
			if got := op.Output.Message.Metadata["interactionStatus"]; (got != nil) != tt.wantMeta {
				t.Errorf("interactionStatus = %v, wantMeta %t", got, tt.wantMeta)
			}
			if tt.status == "incomplete" && !strings.Contains(op.Output.FinishMessage, "incomplete") {
				t.Errorf("FinishMessage = %q", op.Output.FinishMessage)
			}
			if op.Output.Message.Content[0].Text != "partial" {
				t.Errorf("content = %v", op.Output.Message.Content)
			}
		})
	}
}

func TestDeepResearchFailedErrorMapping(t *testing.T) {
	t.Parallel()
	model, _, _ := deepResearchTestModel(t, func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, 200, map[string]any{
			"id": "i-1", "status": "failed",
			"errors": []map[string]any{{"code": "rate_limit_exceeded", "message": "slow down"}},
		})
	})
	op, err := model.Check(context.Background(), &ai.ModelOperation{ID: "i-1"})
	if err != nil {
		t.Fatal(err)
	}
	if got := status.Of(op.Error); got != status.ResourceExhausted {
		t.Errorf("status = %v, want RESOURCE_EXHAUSTED", got)
	}
	// The error must survive JSON marshaling.
	data, err := json.Marshal(op)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "RESOURCE_EXHAUSTED") || !strings.Contains(string(data), "slow down") {
		t.Errorf("marshaled op lost the error: %s", data)
	}
}

func TestDeepResearchFailedSafetyIsBlocked(t *testing.T) {
	t.Parallel()
	model, _, _ := deepResearchTestModel(t, func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, 200, map[string]any{
			"id": "i-1", "status": "failed",
			"errors": []map[string]any{{"code": "safety", "message": "blocked by policy"}},
		})
	})
	op, err := model.Check(context.Background(), &ai.ModelOperation{ID: "i-1"})
	if err != nil {
		t.Fatal(err)
	}
	if !op.Done || op.Error != nil || op.Output == nil || op.Output.FinishReason != ai.FinishReasonBlocked {
		t.Fatalf("op = %#v, want done blocked output", op)
	}
}

func TestDeepResearchResponseOverCap(t *testing.T) {
	t.Parallel()
	model, _, _ := deepResearchTestModel(t, func(w http.ResponseWriter, r *http.Request) {
		chunk := bytes.Repeat([]byte("a"), 1<<20)
		w.Header().Set("Content-Type", "application/json")
		for i := 0; i <= deepResearchMaxResponseBytes>>20; i++ {
			if _, err := w.Write(chunk); err != nil {
				return
			}
		}
	})
	_, err := model.Check(context.Background(), &ai.ModelOperation{ID: "i-1"})
	if err == nil || !strings.Contains(err.Error(), "byte limit") {
		t.Fatalf("Check() error = %v, want byte limit error", err)
	}
}

func TestDeepResearchHTTPErrorMapping(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name   string
		code   int
		body   string
		want   status.Name
		secret string
	}{
		{"404", 404, `{"error":{"code":404,"message":"no such interaction","status":"NOT_FOUND"}}`, status.NotFound, ""},
		{"429", 429, `{"error":{"code":429,"message":"quota","status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.RetryInfo","retryDelay":"7s"}]}}`, status.ResourceExhausted, ""},
		{"opaque", 502, `<html>SECRET-BODY</html>`, status.Internal, "SECRET-BODY"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			model, _, _ := deepResearchTestModel(t, func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.code)
				_, _ = w.Write([]byte(tt.body))
			})
			_, err := model.Check(context.Background(), &ai.ModelOperation{ID: "i-1"})
			if err == nil {
				t.Fatal("Check() error = nil")
			}
			if got := status.Of(err); got != tt.want {
				t.Errorf("status = %v, want %v (err %v)", got, tt.want, err)
			}
			if tt.secret != "" && strings.Contains(err.Error(), tt.secret) {
				t.Errorf("error echoes raw body: %v", err)
			}
			if tt.code == 429 {
				if d, ok := RetryDelay(err); !ok || d != 7*time.Second {
					t.Errorf("RetryDelay = %v, %t, want 7s", d, ok)
				}
			}
		})
	}
}

func TestDeepResearchTamperedOperation(t *testing.T) {
	t.Parallel()
	var evil atomic.Int32
	evilServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		evil.Add(1)
		if r.Header.Get("x-goog-api-key") != "" {
			t.Errorf("API key reached attacker server")
		}
	}))
	defer evilServer.Close()

	var paths []string
	var mu sync.Mutex
	model, _, _ := deepResearchTestModel(t, func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		paths = append(paths, r.URL.RequestURI())
		mu.Unlock()
		writeJSON(w, 200, map[string]any{"id": "i-1", "status": "in_progress"})
	})

	op := &ai.ModelOperation{ID: "i-1", Metadata: map[string]any{
		"clientOptions": map[string]any{"baseUrl": evilServer.URL, "apiVersion": "v1beta/../../x"},
	}}
	if _, err := model.Check(context.Background(), op); err != nil {
		t.Fatalf("Check() error = %v", err)
	}
	if evil.Load() != 0 {
		t.Fatal("request was sent to the tampered baseUrl")
	}
	if len(paths) != 1 || paths[0] != "/v1beta/interactions/i-1" {
		t.Errorf("paths = %v, want the default version path", paths)
	}

	for _, id := range []string{"a/b", "../x", "a?b=c", "a#b", "a b", "..", "a%2fb"} {
		for name, fn := range map[string]func(context.Context, *ai.ModelOperation) (*ai.ModelOperation, error){"check": model.Check, "cancel": model.Cancel} {
			if _, err := fn(context.Background(), &ai.ModelOperation{ID: id}); err == nil || status.Of(err) != status.InvalidArgument {
				t.Errorf("%s(%q) error = %v, want INVALID_ARGUMENT", name, id, err)
			}
		}
	}
	if len(paths) != 1 {
		t.Errorf("invalid IDs reached the server: %v", paths)
	}
}

func TestDeepResearchJSONRoundTrippedOperation(t *testing.T) {
	t.Parallel()
	model, _, _ := deepResearchTestModel(t, func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodPost {
			writeJSON(w, 200, map[string]any{"id": "i-1", "status": "in_progress"})
			return
		}
		writeJSON(w, 200, map[string]any{
			"id": "i-1", "status": "completed",
			"steps": []map[string]any{{"type": "model_output", "content": []map[string]any{{"type": "text", "text": "done"}}}},
		})
	})
	op, err := model.Start(context.Background(), &ai.ModelRequest{
		Messages: []*ai.Message{{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("go")}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(op)
	if err != nil {
		t.Fatal(err)
	}
	var rt ai.ModelOperation
	if err := json.Unmarshal(data, &rt); err != nil {
		t.Fatal(err)
	}
	if _, ok := deepResearchStartTime(rt.Metadata["startTime"]); !ok {
		t.Fatalf("startTime = %#v did not survive the round trip", rt.Metadata["startTime"])
	}
	checked, err := model.Check(context.Background(), &rt)
	if err != nil {
		t.Fatalf("Check() error = %v", err)
	}
	if !checked.Done || checked.Output.LatencyMs <= 0 {
		t.Errorf("checked = done %t latency %v, want done with latency", checked.Done, checked.Output)
	}
}

func TestDeepResearchCancel(t *testing.T) {
	t.Parallel()
	t.Run("CANCELLED is success", func(t *testing.T) {
		t.Parallel()
		model, _, _ := deepResearchTestModel(t, func(w http.ResponseWriter, r *http.Request) {
			writeJSON(w, 499, map[string]any{"error": map[string]any{"code": 499, "status": "CANCELLED", "message": "already cancelled"}})
		})
		op, err := model.Cancel(context.Background(), &ai.ModelOperation{ID: "i-1"})
		if err != nil {
			t.Fatalf("Cancel() error = %v", err)
		}
		if op.ID != "i-1" || !op.Done || op.Output == nil || op.Output.FinishReason != ai.FinishReasonInterrupted {
			t.Errorf("op = %#v, want cancelled op with ID", op)
		}
	})
	t.Run("empty body follows up with GET", func(t *testing.T) {
		t.Parallel()
		var gets atomic.Int32
		model, _, _ := deepResearchTestModel(t, func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodGet {
				gets.Add(1)
				writeJSON(w, 200, map[string]any{"id": "i-1", "status": "cancelled", "usage": map[string]any{"total_tokens": 9}})
				return
			}
			w.WriteHeader(200)
		})
		op, err := model.Cancel(context.Background(), &ai.ModelOperation{ID: "i-1"})
		if err != nil {
			t.Fatal(err)
		}
		if gets.Load() != 1 || !op.Done {
			t.Errorf("gets = %d, done = %t", gets.Load(), op.Done)
		}
		if op.Output.Usage == nil || op.Output.Usage.TotalTokens != 9 {
			t.Errorf("cancelled output usage = %#v, want 9 tokens", op.Output.Usage)
		}
	})
	t.Run("terminal operation", func(t *testing.T) {
		t.Parallel()
		model, _, _ := deepResearchTestModel(t, func(w http.ResponseWriter, r *http.Request) {
			writeJSON(w, 200, map[string]any{"id": "i-1", "status": "completed"})
		})
		op, err := model.Cancel(context.Background(), &ai.ModelOperation{ID: "i-1"})
		if err != nil {
			t.Fatal(err)
		}
		if !op.Done || op.ID != "i-1" {
			t.Errorf("op = %#v", op)
		}
	})
	t.Run("own context cancellation is an error", func(t *testing.T) {
		t.Parallel()
		model, _, _ := deepResearchTestModel(t, func(w http.ResponseWriter, r *http.Request) {
			writeJSON(w, 200, map[string]any{"id": "i-1", "status": "cancelled"})
		})
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, err := model.Cancel(ctx, &ai.ModelOperation{ID: "i-1"}); err == nil || !errors.Is(err, context.Canceled) {
			t.Errorf("Cancel() error = %v, want context.Canceled", err)
		}
	})
}

func TestDeepResearchVertexBackendNotListedOrResolved(t *testing.T) {
	t.Parallel()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, 200, map[string]any{"models": []map[string]any{
			{"name": "models/" + deepResearchPreview042026},
			{"name": "models/gemini-2.5-flash"},
		}})
	}))
	defer server.Close()
	client := newDeepResearchTestClient(t, server.URL)
	vertex := catalog{provider: vertexAIProvider}

	for _, desc := range listActions(context.Background(), client, vertex) {
		if strings.Contains(desc.Key, "deep-research") {
			t.Errorf("Vertex listActions returned Deep Research action %q", desc.Key)
		}
	}
	found := false
	for _, desc := range listActions(context.Background(), client, catalog{provider: googleAIProvider}) {
		found = found || strings.Contains(desc.Key, "deep-research")
	}
	if !found {
		t.Error("Google AI listActions did not return Deep Research")
	}
	for atype, id := range map[api.ActionType]string{
		api.ActionTypeBackgroundModel: deepResearchPreview042026,
		api.ActionTypeCheckOperation:  deepResearchPreview042026 + "/check",
		api.ActionTypeCancelOperation: deepResearchPreview042026 + "/cancel",
	} {
		if got := resolveAction(client, vertex, atype, id); got != nil {
			t.Errorf("Vertex resolveAction(%s, %s) = %v, want nil", atype, id, got)
		}
	}
}

func TestResolveCancelKeyRefusedForVeo(t *testing.T) {
	t.Parallel()
	client := newDeepResearchTestClient(t, "http://127.0.0.1")
	cat := catalog{provider: googleAIProvider}
	if got := resolveAction(client, cat, api.ActionTypeCancelOperation, "veo-3.1-generate-preview/cancel"); got != nil {
		t.Errorf("cancel key resolved for Veo (no cancel support): %v", got)
	}
	if got := resolveAction(client, cat, api.ActionTypeCheckOperation, "veo-3.1-generate-preview/check"); got == nil {
		t.Error("check key not resolved for Veo")
	}
}

func TestDeepResearchRequestShape(t *testing.T) {
	t.Parallel()
	req, err := toDeepResearchInteractionRequest(deepResearchPreview042026, &ai.ModelRequest{
		Output: &ai.ModelOutputConfig{
			Format: ai.OutputFormatJSON,
			Schema: map[string]any{
				"$schema":              "http://json-schema.org/draft-07/schema#",
				"type":                 "object",
				"additionalProperties": false,
				"properties":           map[string]any{"a": map[string]any{"type": []any{"string", "null"}}},
			},
		},
	}, &DeepResearchConfig{ResponseModalities: []string{"TEXT"}})
	if err != nil {
		t.Fatal(err)
	}
	data, _ := json.Marshal(req)
	got := string(data)
	for _, bad := range []string{"response_modalities", "$schema", "additionalProperties", "null"} {
		if strings.Contains(got, bad) {
			t.Errorf("request contains %q: %s", bad, got)
		}
	}
	rf, ok := req.ResponseFormat.(map[string]any)
	if !ok || rf["type"] != "text" || rf["mime_type"] != "application/json" {
		t.Errorf("response_format = %#v, want single folded text entry", req.ResponseFormat)
	}
}

func TestDeepResearchModelRef(t *testing.T) {
	t.Parallel()
	cfg := &DeepResearchConfig{ThinkingSummaries: "auto"}
	ref := DeepResearchModelRef("googleai/"+deepResearchPreview042026, cfg)
	if ref.Name() != "googleai/"+deepResearchPreview042026 {
		t.Errorf("Name() = %q", ref.Name())
	}
	if ref.Config() != cfg {
		t.Errorf("Config() = %v, want the supplied config", ref.Config())
	}
}

func TestDeepResearchProIsDeprecated(t *testing.T) {
	t.Parallel()
	if got := GetModelOptions(deepResearchProPreview122025, googleAIProvider).Stage; got != ai.ModelStageDeprecated {
		t.Errorf("Stage = %q, want deprecated", got)
	}
}
