// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/status"
	"google.golang.org/genai"
)

// agRecorder is a fake interactions endpoint. It records every request body
// and replies with the next queued response (the last one repeats).
type agRecorder struct {
	mu        sync.Mutex
	requests  []map[string]any
	headers   []http.Header
	paths     []string
	responses []any
	status    int
}

func (r *agRecorder) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	data, _ := io.ReadAll(req.Body)
	var body map[string]any
	_ = json.Unmarshal(data, &body)

	r.mu.Lock()
	idx := len(r.requests)
	r.requests = append(r.requests, body)
	r.headers = append(r.headers, req.Header.Clone())
	r.paths = append(r.paths, req.URL.Path)
	resp := r.responses[min(idx, len(r.responses)-1)]
	st := r.status
	r.mu.Unlock()

	if st == 0 {
		st = http.StatusOK
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(st)
	out, _ := json.Marshal(resp)
	_, _ = w.Write(out)
}

func (r *agRecorder) last(t *testing.T) map[string]any {
	t.Helper()
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.requests) == 0 {
		t.Fatal("no request reached the interactions endpoint")
	}
	return r.requests[len(r.requests)-1]
}

func agClient(t *testing.T, baseURL string, mutate func(*genai.ClientConfig)) *genai.Client {
	t.Helper()
	cfg := &genai.ClientConfig{
		Backend:     genai.BackendGeminiAPI,
		APIKey:      "test-api-key-plugin",
		HTTPOptions: genai.HTTPOptions{BaseURL: baseURL},
	}
	if mutate != nil {
		mutate(cfg)
	}
	client, err := genai.NewClient(context.Background(), cfg)
	if err != nil {
		t.Fatalf("genai.NewClient: %v", err)
	}
	return client
}

// agSetup starts a fake endpoint replying with responses and returns the
// recorder and a model action registered the way the plugin builds it.
func agSetup(t *testing.T, responses ...any) (*agRecorder, *ai.ModelAction) {
	t.Helper()
	rec := &agRecorder{responses: responses}
	srv := httptest.NewServer(rec)
	t.Cleanup(srv.Close)
	return rec, agModel(t, agClient(t, srv.URL, nil))
}

func agModel(t *testing.T, client *genai.Client) *ai.ModelAction {
	t.Helper()
	return newModel(client, antigravityPreview092026, GetModelOptions(antigravityPreview092026, googleAIProvider))
}

func okInteraction(text string) *geminiInteraction {
	return &geminiInteraction{
		ID:            "interaction-123",
		Status:        "completed",
		EnvironmentID: "env-123",
		Steps: []interactionStep{
			{Type: "model_output", Content: []interactionContent{{Type: "text", Text: text}}},
		},
	}
}

func userReq(text string, cfg any) *ai.ModelRequest {
	return &ai.ModelRequest{Messages: []*ai.Message{ai.NewUserTextMessage(text)}, Config: cfg}
}

func inputSteps(t *testing.T, body map[string]any) []map[string]any {
	t.Helper()
	raw, _ := body["input"].([]any)
	steps := make([]map[string]any, len(raw))
	for i, s := range raw {
		steps[i], _ = s.(map[string]any)
	}
	return steps
}

func stepText(step map[string]any) string {
	content, _ := step["content"].([]any)
	var b strings.Builder
	for _, c := range content {
		m, _ := c.(map[string]any)
		if s, ok := m["text"].(string); ok {
			b.WriteString(s)
		}
	}
	return b.String()
}

func TestAntigravityFirstTurnRequest(t *testing.T) {
	rec, m := agSetup(t, okInteraction("Hello human"))

	resp, err := m.Generate(context.Background(), userReq("Hello antigravity", nil), nil)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}

	if rec.paths[0] != "/v1beta/interactions" {
		t.Errorf("path = %q", rec.paths[0])
	}
	if got := rec.headers[0].Get("Api-Revision"); got != antigravityAPIRevision {
		t.Errorf("Api-Revision = %q", got)
	}
	if got := rec.headers[0].Get("x-goog-api-key"); got != "test-api-key-plugin" {
		t.Errorf("x-goog-api-key = %q", got)
	}
	body := rec.last(t)
	if body["agent"] != antigravityPreview092026 {
		t.Errorf("agent = %v", body["agent"])
	}
	// The endpoint requires an environment; the default is a remote sandbox.
	env, _ := body["environment"].(map[string]any)
	if env["type"] != "remote" {
		t.Errorf("environment = %v, want {type: remote}", body["environment"])
	}
	for _, k := range []string{"previous_interaction_id", "store", "tools", "response_format", "response_modalities", "agent_config"} {
		if _, ok := body[k]; ok {
			t.Errorf("%s should be absent on a bare first turn, body = %v", k, body)
		}
	}
	steps := inputSteps(t, body)
	if len(steps) != 1 || steps[0]["type"] != "user_input" || stepText(steps[0]) != "Hello antigravity" {
		t.Errorf("input = %v", body["input"])
	}

	if resp.Text() != "Hello human" || resp.FinishReason != ai.FinishReasonStop {
		t.Errorf("response = %q / %v", resp.Text(), resp.FinishReason)
	}
	if resp.Message.Metadata["interactionId"] != "interaction-123" || resp.Message.Metadata["environmentId"] != "env-123" {
		t.Errorf("metadata = %v", resp.Message.Metadata)
	}
}

func TestAntigravityContinuesFromMessageMetadata(t *testing.T) {
	rec, m := agSetup(t, okInteraction("second"))

	req := &ai.ModelRequest{Messages: []*ai.Message{
		ai.NewUserTextMessage("first question"),
		{
			Role:     ai.RoleModel,
			Content:  []*ai.Part{ai.NewTextPart("first answer"), ai.NewToolRequestPart(&ai.ToolRequest{Name: "t", Ref: "c1", Input: map[string]any{}})},
			Metadata: map[string]any{"interactionId": "prev-1", "environmentId": "env-prev"},
		},
		ai.NewUserTextMessage("second question"),
	}}
	if _, err := m.Generate(context.Background(), req, nil); err != nil {
		t.Fatalf("Generate: %v", err)
	}

	body := rec.last(t)
	if body["previous_interaction_id"] != "prev-1" {
		t.Errorf("previous_interaction_id = %v", body["previous_interaction_id"])
	}
	// Only the messages after the last model message are sent: replaying the
	// earlier turn (and its function_call) would be rejected by the server.
	steps := inputSteps(t, body)
	if len(steps) != 1 || stepText(steps[0]) != "second question" {
		t.Errorf("input = %v, want only the new user message", body["input"])
	}
	// The sandbox from the previous turn is reused, not a fresh remote one.
	if body["environment"] != "env-prev" {
		t.Errorf("environment = %v, want env-prev", body["environment"])
	}
}

func TestAntigravityContinuationUsesLastModelMessageWithID(t *testing.T) {
	rec, m := agSetup(t, okInteraction("ok"))
	req := &ai.ModelRequest{Messages: []*ai.Message{
		ai.NewUserTextMessage("a"),
		{Role: ai.RoleModel, Content: []*ai.Part{ai.NewTextPart("A")}, Metadata: map[string]any{"interactionId": "old"}},
		ai.NewUserTextMessage("b"),
		{Role: ai.RoleModel, Content: []*ai.Part{ai.NewTextPart("B")}, Metadata: map[string]any{"interactionId": "new"}},
		ai.NewUserTextMessage("c"),
	}}
	if _, err := m.Generate(context.Background(), req, nil); err != nil {
		t.Fatal(err)
	}
	body := rec.last(t)
	if body["previous_interaction_id"] != "new" {
		t.Errorf("previous_interaction_id = %v, want new", body["previous_interaction_id"])
	}
	if steps := inputSteps(t, body); len(steps) != 1 || stepText(steps[0]) != "c" {
		t.Errorf("input = %v", body["input"])
	}
}

func TestAntigravityExplicitPreviousInteractionID(t *testing.T) {
	rec, m := agSetup(t, okInteraction("ok"))
	req := &ai.ModelRequest{
		Config: map[string]any{"previousInteractionId": "explicit-9"},
		Messages: []*ai.Message{
			ai.NewUserTextMessage("old question"),
			{Role: ai.RoleModel, Content: []*ai.Part{ai.NewTextPart("old answer")}, Metadata: map[string]any{"interactionId": "ignored", "environmentId": "env-ignored"}},
			ai.NewUserTextMessage("new question"),
		},
	}
	if _, err := m.Generate(context.Background(), req, nil); err != nil {
		t.Fatal(err)
	}
	body := rec.last(t)
	if body["previous_interaction_id"] != "explicit-9" {
		t.Errorf("previous_interaction_id = %v, want the explicit one", body["previous_interaction_id"])
	}
	if steps := inputSteps(t, body); len(steps) != 1 || stepText(steps[0]) != "new question" {
		t.Errorf("history was replayed: input = %v", body["input"])
	}
}

func TestAntigravityEnvironmentSelection(t *testing.T) {
	history := func() []*ai.Message {
		return []*ai.Message{
			ai.NewUserTextMessage("a"),
			{Role: ai.RoleModel, Content: []*ai.Part{ai.NewTextPart("A")}, Metadata: map[string]any{"interactionId": "p", "environmentId": "env-prev"}},
			ai.NewUserTextMessage("b"),
		}
	}
	tests := []struct {
		name string
		cfg  any
		want any
	}{
		{"unset reuses previous", nil, "env-prev"},
		{"remote string reuses previous", map[string]any{"environment": "remote"}, "env-prev"},
		{"remote object reuses previous", map[string]any{"environment": map[string]any{"type": "remote"}}, "env-prev"},
		{"pinned id wins", map[string]any{"environment": "env-pinned"}, "env-pinned"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			rec, m := agSetup(t, okInteraction("ok"))
			if _, err := m.Generate(context.Background(), &ai.ModelRequest{Messages: history(), Config: tc.cfg}, nil); err != nil {
				t.Fatal(err)
			}
			if got := rec.last(t)["environment"]; got != tc.want {
				t.Errorf("environment = %v, want %v", got, tc.want)
			}
		})
	}

	t.Run("inline object is kept", func(t *testing.T) {
		rec, m := agSetup(t, okInteraction("ok"))
		inline := map[string]any{"type": "remote", "network": "off"}
		if _, err := m.Generate(context.Background(), &ai.ModelRequest{Messages: history(), Config: map[string]any{"environment": inline}}, nil); err != nil {
			t.Fatal(err)
		}
		got, _ := rec.last(t)["environment"].(map[string]any)
		if got["network"] != "off" {
			t.Errorf("environment = %v", rec.last(t)["environment"])
		}
	})
}

// TestAntigravityMultiTurnRoundTrip feeds a real response's message back as
// history, the way a chat does.
func TestAntigravityMultiTurnRoundTrip(t *testing.T) {
	rec, m := agSetup(t, okInteraction("first"), okInteraction("second"))
	ctx := context.Background()

	req1 := userReq("one", nil)
	resp1, err := m.Generate(ctx, req1, nil)
	if err != nil {
		t.Fatal(err)
	}
	req2 := &ai.ModelRequest{Messages: []*ai.Message{req1.Messages[0], resp1.Message, ai.NewUserTextMessage("two")}}
	if _, err := m.Generate(ctx, req2, nil); err != nil {
		t.Fatal(err)
	}
	body := rec.last(t)
	if body["previous_interaction_id"] != "interaction-123" || body["environment"] != "env-123" {
		t.Errorf("second request = %v", body)
	}
	if steps := inputSteps(t, body); len(steps) != 1 || stepText(steps[0]) != "two" {
		t.Errorf("input = %v", body["input"])
	}
}

// TestAntigravityConfigPassthrough goes through the real model action: keys the
// schema declares are mapped, and undeclared keys reach the request body, as the
// JS schema's .passthrough() does.
func TestAntigravityConfigPassthrough(t *testing.T) {
	rec, m := agSetup(t, okInteraction("ok"))
	cfg := map[string]any{
		"store":                 true,
		"responseModalities":    []any{"TEXT"},
		"model":                 "gemini-3.5-flash-lite",
		"tools":                 []any{map[string]any{"type": "google_search"}, map[string]any{"codeExecution": map[string]any{}}},
		"temperature":           0.2,
		"agent_config_extra":    map[string]any{"k": "v"},
		"version":               "ignored-framework-key",
		"previousInteractionId": "prev-x",
	}
	if _, err := m.Generate(context.Background(), userReq("hi", cfg), nil); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	body := rec.last(t)

	if body["store"] != true {
		t.Errorf("store = %v", body["store"])
	}
	if body["temperature"] != 0.2 {
		t.Errorf("temperature (undeclared key) = %v, want it passed through", body["temperature"])
	}
	if extra, _ := body["agent_config_extra"].(map[string]any); extra["k"] != "v" {
		t.Errorf("agent_config_extra = %v", body["agent_config_extra"])
	}
	if _, ok := body["version"]; ok {
		t.Errorf("framework-owned version key leaked into the body")
	}
	if ac, _ := body["agent_config"].(map[string]any); ac["model"] != "gemini-3.5-flash-lite" || ac["type"] != "antigravity" {
		t.Errorf("agent_config = %v", body["agent_config"])
	}
	if _, ok := body["model"]; ok {
		t.Errorf("a top-level model would conflict with agent: %v", body["model"])
	}
	if rf, _ := body["response_format"].(map[string]any); rf["type"] != "text" {
		t.Errorf("response_format = %v, want {type: text}", body["response_format"])
	}
	if _, ok := body["response_modalities"]; ok {
		t.Errorf("deprecated response_modalities must not be sent")
	}
	tools, _ := body["tools"].([]any)
	if len(tools) != 2 || tools[0].(map[string]any)["type"] != "google_search" || tools[1].(map[string]any)["type"] != "code_execution" {
		t.Errorf("tools = %v", body["tools"])
	}
	if body["previous_interaction_id"] != "prev-x" {
		t.Errorf("previous_interaction_id = %v", body["previous_interaction_id"])
	}
}

func TestAntigravityPerRequestClientOverrides(t *testing.T) {
	rec, m := agSetup(t, okInteraction("ok"))
	other := &agRecorder{responses: []any{okInteraction("other")}}
	srv := httptest.NewServer(other)
	defer srv.Close()

	cfg := map[string]any{"apiKey": "per-request-key", "baseUrl": srv.URL, "apiVersion": "v9"}
	if _, err := m.Generate(context.Background(), userReq("hi", cfg), nil); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(rec.requests) != 0 {
		t.Errorf("request went to the plugin's base URL")
	}
	if other.paths[0] != "/v9/interactions" {
		t.Errorf("path = %q", other.paths[0])
	}
	if got := other.headers[0].Get("x-goog-api-key"); got != "per-request-key" {
		t.Errorf("api key = %q", got)
	}
	for _, k := range []string{"apiKey", "baseUrl", "apiVersion"} {
		if _, ok := other.last(t)[k]; ok {
			t.Errorf("client option %q leaked into the body", k)
		}
	}
}

func TestAntigravityConfigRejectsBadTypes(t *testing.T) {
	rec, m := agSetup(t, okInteraction("ok"))
	for _, cfg := range []map[string]any{
		{"store": "yes"},
		{"responseModalities": []any{"VIDEO"}},
		{"previousInteractionId": 5},
		{"tools": "google_search"},
	} {
		_, err := m.Generate(context.Background(), userReq("hi", cfg), nil)
		if err == nil {
			t.Errorf("config %v: expected a validation error", cfg)
			continue
		}
		if got, _ := status.Classified(err); got != status.InvalidArgument {
			t.Errorf("config %v: status = %v (%v), want INVALID_ARGUMENT", cfg, got, err)
		}
	}
	if len(rec.requests) != 0 {
		t.Errorf("an invalid config reached the endpoint")
	}
}

func TestAntigravityConfigRejectsManagedKeys(t *testing.T) {
	rec, m := agSetup(t, okInteraction("ok"))
	for _, key := range []string{"agent", "input"} {
		_, err := m.Generate(context.Background(), userReq("hi", map[string]any{key: "x"}), nil)
		if got, _ := status.Classified(err); got != status.InvalidArgument {
			t.Errorf("%s: err = %v, want INVALID_ARGUMENT", key, err)
		}
	}
	if len(rec.requests) != 0 {
		t.Errorf("a managed key reached the endpoint")
	}
}

func TestAntigravityTypedConfig(t *testing.T) {
	rec, m := agSetup(t, okInteraction("ok"))
	store := false
	cfg := &AntigravityConfig{Store: &store, Extra: map[string]any{"temperature": 0.5}, Environment: "env-1"}
	if _, err := m.Generate(context.Background(), userReq("hi", cfg), nil); err != nil {
		t.Fatalf("Generate: %v", err)
	}
	body := rec.last(t)
	if body["store"] != false || body["temperature"] != 0.5 || body["environment"] != "env-1" {
		t.Errorf("body = %v", body)
	}
}

func TestAntigravityConfigJSONRoundTrip(t *testing.T) {
	in := []byte(`{"store":true,"temperature":0.3,"nested":{"a":1}}`)
	var cfg AntigravityConfig
	if err := json.Unmarshal(in, &cfg); err != nil {
		t.Fatal(err)
	}
	if cfg.Store == nil || !*cfg.Store || cfg.Extra["temperature"] != 0.3 {
		t.Fatalf("cfg = %+v", cfg)
	}
	out, err := json.Marshal(cfg)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	_ = json.Unmarshal(out, &m)
	if m["store"] != true || m["temperature"] != 0.3 || m["nested"] == nil {
		t.Errorf("marshalled = %s", out)
	}
	if err := json.Unmarshal([]byte(`{"store":"x"}`), &cfg); err == nil {
		t.Error("expected a type error for a bad store value")
	}
}

func TestAntigravityStreamingEmitsSingleChunk(t *testing.T) {
	_, m := agSetup(t, okInteraction("Hello"))
	var chunks []string
	_, err := m.Generate(context.Background(), userReq("hi", nil), func(_ context.Context, c *ai.ModelResponseChunk) error {
		chunks = append(chunks, c.Text())
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(chunks) != 1 || chunks[0] != "Hello" {
		t.Errorf("chunks = %v, want one chunk with the full response (sync-only endpoint)", chunks)
	}
}

// ---- capabilities, system prompt, tools ----

func TestAntigravityCapabilitiesMatchJS(t *testing.T) {
	s := AntigravitySupports
	if !s.Multiturn || !s.Media || !s.Tools || s.ToolChoice || s.SystemRole {
		t.Errorf("supports = %+v, want multiturn/media/tools true, toolChoice/systemRole false (as JS)", s)
	}
}

func TestAntigravitySystemMessageUsesSimulatedSystemPrompt(t *testing.T) {
	rec, m := agSetup(t, okInteraction("ok"))
	req := &ai.ModelRequest{Messages: []*ai.Message{
		ai.NewSystemTextMessage("Always answer in French"),
		ai.NewUserTextMessage("Hello"),
	}}
	if _, err := m.Generate(context.Background(), req, nil); err != nil {
		t.Fatal(err)
	}
	steps := inputSteps(t, rec.last(t))
	// SystemRole is false, so the framework's simulateSystemPrompt turns the
	// system message into a user/model pair before the handler runs; no step
	// carries a bare "system" role and the instruction text is sent once.
	var all strings.Builder
	for _, s := range steps {
		all.WriteString(stepText(s))
		all.WriteString("|")
	}
	if strings.Count(all.String(), "Always answer in French") != 1 {
		t.Errorf("system text should be sent exactly once, input = %v", rec.last(t)["input"])
	}
	if len(steps) != 3 || steps[0]["type"] != "user_input" || steps[1]["type"] != "model_output" || steps[2]["type"] != "user_input" {
		t.Errorf("input = %v, want simulated system user/model pair then the user turn", rec.last(t)["input"])
	}
	if !strings.Contains(stepText(steps[2]), "Hello") {
		t.Errorf("last step = %v", steps[2])
	}
}

func TestAntigravityFunctionToolsAreSent(t *testing.T) {
	rec, m := agSetup(t, okInteraction("ok"))
	req := &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserTextMessage("weather?")},
		Tools: []*ai.ToolDefinition{{
			Name:        "get_weather",
			Description: "Get weather",
			InputSchema: map[string]any{
				"$schema":              "http://json-schema.org/draft-07/schema#",
				"type":                 "object",
				"additionalProperties": false,
				"properties":           map[string]any{"city": map[string]any{"type": []any{"string", "null"}}},
			},
		}},
		Config: map[string]any{"tools": []any{map[string]any{"type": "google_search"}}},
	}
	if _, err := m.Generate(context.Background(), req, nil); err != nil {
		t.Fatal(err)
	}
	tools, _ := rec.last(t)["tools"].([]any)
	if len(tools) != 2 {
		t.Fatalf("tools = %v", rec.last(t)["tools"])
	}
	if tools[0].(map[string]any)["type"] != "google_search" {
		t.Errorf("built-in tool should come first: %v", tools)
	}
	fn := tools[1].(map[string]any)
	if fn["type"] != "function" || fn["name"] != "get_weather" || fn["description"] != "Get weather" {
		t.Errorf("function tool = %v", fn)
	}
	params, _ := fn["parameters"].(map[string]any)
	if _, ok := params["$schema"]; ok {
		t.Errorf("$schema must be stripped: %v", params)
	}
	if _, ok := params["additionalProperties"]; ok {
		t.Errorf("additionalProperties must be stripped: %v", params)
	}
	city := params["properties"].(map[string]any)["city"].(map[string]any)
	if city["type"] != "string" {
		t.Errorf("nullable type should collapse to string: %v", city)
	}
}

func TestAntigravityPendingVsServerFunctionCalls(t *testing.T) {
	resp, err := fromInteractionSync(&geminiInteraction{
		ID:     "i1",
		Status: "requires_action",
		Steps: []interactionStep{
			{Type: "function_call", ID: "srv-1", Name: "internal_search", Arguments: map[string]any{"q": "x"}},
			{Type: "function_result", CallID: "srv-1", Name: "internal_search", Result: "r"},
			{Type: "function_call", ID: "client-1", Name: "get_weather", Arguments: map[string]any{"city": "Paris"}, Signature: "sig-abc"},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	var reqs []*ai.Part
	for _, p := range resp.Message.Content {
		if p.IsToolRequest() {
			reqs = append(reqs, p)
		}
	}
	if len(reqs) != 1 || reqs[0].ToolRequest.Ref != "client-1" {
		t.Fatalf("only the unanswered call should be a tool request, content = %+v", resp.Message.Content)
	}
	if string(metadataSignature(reqs[0].Metadata)) != "sig-abc" {
		t.Errorf("signature not preserved on the function call: %v", reqs[0].Metadata)
	}
	if len(resp.ToolRequests()) != 1 {
		t.Errorf("ToolRequests() = %d", len(resp.ToolRequests()))
	}
	var sawServerCall, sawServerResult bool
	for _, p := range resp.Message.Content {
		sawServerCall = sawServerCall || (p.IsCustom() && p.Custom["serverFunctionCall"] != nil)
		sawServerResult = sawServerResult || (p.IsCustom() && p.Custom["serverFunctionResult"] != nil)
	}
	if !sawServerCall || !sawServerResult {
		t.Errorf("server-side call/result should be kept as custom parts: %+v", resp.Message.Content)
	}

	// Outside requires_action nothing is pending, whatever the steps say.
	resp, err = fromInteractionSync(&geminiInteraction{Status: "completed", Steps: []interactionStep{
		{Type: "function_call", ID: "c", Name: "n"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if len(resp.ToolRequests()) != 0 {
		t.Errorf("a completed interaction must not yield tool requests")
	}
}

// TestAntigravityToolResultContinuation: the tool loop sends the function
// result against the interaction that asked for it, without replaying the call.
func TestAntigravityToolResultContinuation(t *testing.T) {
	rec, m := agSetup(t, okInteraction("It is sunny"))
	req := &ai.ModelRequest{Messages: []*ai.Message{
		ai.NewUserTextMessage("weather in Paris?"),
		{
			Role:     ai.RoleModel,
			Content:  []*ai.Part{ai.NewToolRequestPart(&ai.ToolRequest{Name: "get_weather", Ref: "call-1", Input: map[string]any{"city": "Paris"}})},
			Metadata: map[string]any{"interactionId": "int-1", "environmentId": "env-1"},
		},
		{
			Role:    ai.RoleTool,
			Content: []*ai.Part{ai.NewToolResponsePart(&ai.ToolResponse{Name: "get_weather", Ref: "call-1", Output: map[string]any{"temp": 20}})},
		},
	}}
	if _, err := m.Generate(context.Background(), req, nil); err != nil {
		t.Fatal(err)
	}
	body := rec.last(t)
	steps := inputSteps(t, body)
	if len(steps) != 1 || steps[0]["type"] != "function_result" || steps[0]["call_id"] != "call-1" {
		t.Fatalf("input = %v", body["input"])
	}
	if body["previous_interaction_id"] != "int-1" || body["environment"] != "env-1" {
		t.Errorf("body = %v", body)
	}
}

func TestAntigravityFunctionCallSignatureRoundTrip(t *testing.T) {
	// A call without an interaction ID is replayed, with its signature.
	steps, err := toInteractionSteps([]*ai.Message{{
		Role: ai.RoleModel,
		Content: []*ai.Part{func() *ai.Part {
			p := ai.NewToolRequestPart(&ai.ToolRequest{Name: "f", Ref: "c1", Input: map[string]any{"a": 1}})
			p.Metadata = map[string]any{"signature": []byte("sig-xyz")}
			return p
		}()},
	}})
	if err != nil {
		t.Fatal(err)
	}
	if len(steps) != 1 || steps[0].(map[string]any)["signature"] != "sig-xyz" {
		t.Errorf("steps = %v", steps)
	}
}

// ---- toInteractionSteps ----

func TestToInteractionStepsPreservesPartOrder(t *testing.T) {
	steps, err := toInteractionSteps([]*ai.Message{{
		Role: ai.RoleModel,
		Content: []*ai.Part{
			ai.NewTextPart("before"),
			ai.NewReasoningPart("thinking", nil),
			ai.NewTextPart("middle"),
			ai.NewToolRequestPart(&ai.ToolRequest{Name: "f", Ref: "c1"}),
			ai.NewTextPart("after"),
		},
	}})
	if err != nil {
		t.Fatal(err)
	}
	var kinds []string
	for _, s := range steps {
		kinds = append(kinds, s.(map[string]any)["type"].(string))
	}
	want := []string{"model_output", "thought", "model_output", "function_call", "model_output"}
	if strings.Join(kinds, ",") != strings.Join(want, ",") {
		t.Errorf("step order = %v, want %v", kinds, want)
	}
}

func TestToInteractionStepsMedia(t *testing.T) {
	media := func(ct, url string) *ai.Part { return &ai.Part{Kind: ai.PartMedia, ContentType: ct, Text: url} }

	t.Run("data URL without ContentType", func(t *testing.T) {
		steps, err := toInteractionSteps([]*ai.Message{{Role: ai.RoleUser, Content: []*ai.Part{
			media("", "data:image/png;base64,AAAA"),
		}}})
		if err != nil {
			t.Fatal(err)
		}
		c := steps[0].(map[string]any)["content"].([]map[string]any)[0]
		if c["type"] != "image" || c["mime_type"] != "image/png" || c["data"] != "AAAA" {
			t.Errorf("content = %v", c)
		}
	})
	t.Run("kinds", func(t *testing.T) {
		for ct, want := range map[string]string{"audio/mp3": "audio", "video/mp4": "video", "application/pdf": "document", "text/csv": "document", "image/jpeg": "image"} {
			steps, err := toInteractionSteps([]*ai.Message{{Role: ai.RoleUser, Content: []*ai.Part{media(ct, "https://generativelanguage.googleapis.com/v1beta/files/x")}}})
			if err != nil {
				t.Fatalf("%s: %v", ct, err)
			}
			c := steps[0].(map[string]any)["content"].([]map[string]any)[0]
			if c["type"] != want || c["uri"] == nil {
				t.Errorf("%s: content = %v, want type %s with uri", ct, c, want)
			}
		}
	})
	t.Run("untyped youtube uri is video", func(t *testing.T) {
		steps, err := toInteractionSteps([]*ai.Message{{Role: ai.RoleUser, Content: []*ai.Part{media("", "https://www.youtube.com/watch?v=abc")}}})
		if err != nil {
			t.Fatal(err)
		}
		if c := steps[0].(map[string]any)["content"].([]map[string]any)[0]; c["type"] != "video" || c["mime_type"] != nil {
			t.Errorf("content = %v", c)
		}
	})
	for name, p := range map[string]*ai.Part{
		"data URL without a type":  media("", "data:;base64,AAAA"),
		"non-base64 data URL":      media("text/plain", "data:text/plain,hello"),
		"data URL without a comma": media("image/png", "data:image/png"),
		"empty url":                media("image/png", ""),
	} {
		t.Run("error: "+name, func(t *testing.T) {
			_, err := toInteractionSteps([]*ai.Message{{Role: ai.RoleUser, Content: []*ai.Part{p}}})
			if got, _ := status.Classified(err); err == nil || got != status.InvalidArgument {
				t.Errorf("err = %v, want INVALID_ARGUMENT", err)
			}
		})
	}
	t.Run("unsupported part kind", func(t *testing.T) {
		_, err := toInteractionSteps([]*ai.Message{{Role: ai.RoleUser, Content: []*ai.Part{ai.NewDataPart("{}")}}})
		if err == nil {
			t.Error("a data part must not be silently dropped")
		}
	})
	t.Run("server-side parts are not replayed", func(t *testing.T) {
		steps, err := toInteractionSteps([]*ai.Message{{Role: ai.RoleModel, Content: []*ai.Part{
			ai.NewCustomPart(map[string]any{"googleSearchCall": map[string]any{"id": "1"}}),
			ai.NewTextPart("answer"),
		}}})
		if err != nil || len(steps) != 1 || steps[0].(map[string]any)["type"] != "model_output" {
			t.Errorf("steps = %v, err = %v", steps, err)
		}
	})
}

func TestAntigravityEmptyInputIsRejected(t *testing.T) {
	rec, m := agSetup(t, okInteraction("ok"))
	// A history that ends on the model turn leaves nothing to send.
	req := &ai.ModelRequest{Messages: []*ai.Message{
		ai.NewUserTextMessage("a"),
		{Role: ai.RoleModel, Content: []*ai.Part{ai.NewTextPart("A")}, Metadata: map[string]any{"interactionId": "p"}},
	}}
	_, err := m.Generate(context.Background(), req, nil)
	if got, _ := status.Classified(err); got != status.InvalidArgument {
		t.Errorf("err = %v, want INVALID_ARGUMENT", err)
	}
	if len(rec.requests) != 0 {
		t.Errorf("an empty request reached the endpoint")
	}
}

func TestAntigravityDownloadsHTTPMedia(t *testing.T) {
	img := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "image/png")
		_, _ = w.Write([]byte("PNGDATA"))
	}))
	defer img.Close()

	rec, m := agSetup(t, okInteraction("ok"))
	req := &ai.ModelRequest{Messages: []*ai.Message{{Role: ai.RoleUser, Content: []*ai.Part{
		ai.NewTextPart("describe"),
		ai.NewMediaPart("image/png", img.URL+"/a.png"),
	}}}}
	if _, err := m.Generate(context.Background(), req, nil); err != nil {
		t.Fatal(err)
	}
	content := inputSteps(t, rec.last(t))[0]["content"].([]any)
	media := content[1].(map[string]any)
	// The interactions endpoint rejects plain http(s) URIs ("Unsupported file
	// uri"), so the media middleware inlines the bytes as it does for
	// generateContent.
	if media["uri"] != nil || media["data"] == nil || media["type"] != "image" {
		t.Errorf("media = %v, want inlined data", media)
	}
}

// ---- responses ----

func TestFromInteractionSyncStatuses(t *testing.T) {
	text := []interactionStep{{Type: "model_output", Content: []interactionContent{{Type: "text", Text: "partial"}}}}

	t.Run("incomplete", func(t *testing.T) {
		resp, err := fromInteractionSync(&geminiInteraction{ID: "i", Status: "incomplete", Steps: text})
		if err != nil {
			t.Fatal(err)
		}
		if resp.FinishReason != ai.FinishReasonLength || resp.Text() != "partial" || resp.FinishMessage == "" {
			t.Errorf("resp = %v %q %q", resp.FinishReason, resp.Text(), resp.FinishMessage)
		}
	})
	t.Run("cancelled", func(t *testing.T) {
		resp, err := fromInteractionSync(&geminiInteraction{ID: "i", Status: "cancelled"})
		if err != nil {
			t.Fatal(err)
		}
		if resp.FinishReason != ai.FinishReasonAborted || len(resp.Message.Content) != 0 {
			t.Errorf("finish = %v, content = %+v (want aborted and no synthetic text)", resp.FinishReason, resp.Message.Content)
		}
		if resp.Message.Metadata["interactionId"] != "i" {
			t.Errorf("metadata = %v", resp.Message.Metadata)
		}
	})
	t.Run("in_progress with no content is an error, not an empty stop", func(t *testing.T) {
		_, err := fromInteractionSync(&geminiInteraction{ID: "i", Status: "in_progress"})
		if got, _ := status.Classified(err); got != status.FailedPrecondition {
			t.Errorf("err = %v", err)
		}
	})
	t.Run("in_progress with content is not stop", func(t *testing.T) {
		resp, err := fromInteractionSync(&geminiInteraction{ID: "i", Status: "in_progress", Steps: text})
		if err != nil {
			t.Fatal(err)
		}
		if resp.FinishReason != ai.FinishReasonOther || resp.Text() != "partial" {
			t.Errorf("resp = %v %q", resp.FinishReason, resp.Text())
		}
	})
	t.Run("requires_action with a pending call", func(t *testing.T) {
		resp, err := fromInteractionSync(&geminiInteraction{Status: "requires_action", Steps: []interactionStep{{Type: "function_call", ID: "c", Name: "n"}}})
		if err != nil || len(resp.ToolRequests()) != 1 {
			t.Errorf("resp = %+v, err = %v", resp, err)
		}
	})
}

func TestFromInteractionSyncFailed(t *testing.T) {
	t.Run("content block is blocked", func(t *testing.T) {
		resp, err := fromInteractionSync(&geminiInteraction{
			ID: "i", Status: "failed",
			Errors: []interactionError{{Code: "SAFETY", Message: "blocked for safety"}},
			Usage:  &interactionUsage{TotalInputTokens: 3, TotalTokens: 3},
		})
		if err != nil {
			t.Fatal(err)
		}
		if resp.FinishReason != ai.FinishReasonBlocked || resp.FinishMessage != "blocked for safety" {
			t.Errorf("resp = %v %q", resp.FinishReason, resp.FinishMessage)
		}
		if resp.Usage == nil || resp.Usage.InputTokens != 3 {
			t.Errorf("usage = %+v", resp.Usage)
		}
	})
	for code, want := range map[string]status.Name{
		"malformed_function_call": status.Aborted,
		"service_unavailable":     status.Unavailable,
		"rate_limit_exceeded":     status.ResourceExhausted,
		"invalid_request":         status.InvalidArgument,
		"model_not_found":         status.NotFound,
		"authentication":          status.Unauthenticated,
		"503":                     status.Unavailable,
		"totally_new_code":        status.Unknown,
		"":                        status.Unknown,
	} {
		t.Run("code "+code, func(t *testing.T) {
			_, err := fromInteractionSync(&geminiInteraction{ID: "i", Status: "failed", Errors: []interactionError{{Code: code, Message: "server said no"}}})
			if err == nil {
				t.Fatal("expected an error")
			}
			if got, _ := status.Classified(err); got != want {
				t.Errorf("status = %v, want %v", got, want)
			}
			if !strings.Contains(err.Error(), "server said no") || (code != "" && !strings.Contains(err.Error(), code)) {
				t.Errorf("server code/message lost: %v", err)
			}
		})
	}
	t.Run("no errors", func(t *testing.T) {
		_, err := fromInteractionSync(&geminiInteraction{Status: "failed"})
		if err == nil || !strings.Contains(err.Error(), "Interaction failed") {
			t.Errorf("err = %v", err)
		}
	})
}

func TestFailedInteractionThroughModelAction(t *testing.T) {
	_, m := agSetup(t, map[string]any{"id": "i", "status": "failed", "errors": []any{map[string]any{"code": "rate_limit_exceeded", "message": "slow down"}}})
	_, err := m.Generate(context.Background(), userReq("hi", nil), nil)
	if got, _ := status.Classified(err); got != status.ResourceExhausted || !strings.Contains(err.Error(), "slow down") {
		t.Errorf("err = %v", err)
	}
}

func TestFromInteractionSyncUsage(t *testing.T) {
	resp, err := fromInteractionSync(&geminiInteraction{
		Status: "completed",
		Steps:  []interactionStep{{Type: "model_output", Content: []interactionContent{{Type: "text", Text: "hi"}}}},
		Usage: &interactionUsage{
			TotalInputTokens:       100,
			TotalCachedTokens:      40,
			TotalOutputTokens:      5,
			TotalThoughtTokens:     7,
			TotalToolUseTokens:     3,
			TotalTokens:            115,
			InputTokensByModality:  []modalityTokens{{Modality: "text", Tokens: 90}, {Modality: "image", Tokens: 10}},
			OutputTokensByModality: []modalityTokens{{Modality: "text", Tokens: 5}},
			GroundingToolCount:     []groundingCount{{Type: "google_search", Count: 2}},
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	u := resp.Usage
	if u.InputTokens != 103 || u.OutputTokens != 5 || u.ThoughtsTokens != 7 || u.CachedContentTokens != 40 || u.TotalTokens != 115 {
		t.Errorf("token fields = %+v", u)
	}
	// Character and media-count fields belong to the framework; per-modality
	// token counts must not be written there.
	if u.InputCharacters != 0 || u.InputImages != 0 || u.InputAudioFiles != 0 ||
		u.OutputCharacters != 0 || u.OutputImages != 0 || u.OutputAudioFiles != 0 {
		t.Errorf("per-modality tokens leaked into character/media fields: %+v", u)
	}
	for k, want := range map[string]float64{
		"inputTextTokens": 90, "inputImageTokens": 10, "outputTextTokens": 5,
		"toolUseTokens": 3, "googleSearchCount": 2,
	} {
		if u.Custom[k] != want {
			t.Errorf("Custom[%s] = %v, want %v (custom = %v)", k, u.Custom[k], want, u.Custom)
		}
	}
}

func TestFromInteractionSyncUsageTotalFallback(t *testing.T) {
	resp, _ := fromInteractionSync(&geminiInteraction{Status: "completed", Usage: &interactionUsage{TotalInputTokens: 10, TotalOutputTokens: 5, TotalThoughtTokens: 2}})
	if resp.Usage.TotalTokens != 17 {
		t.Errorf("TotalTokens = %d, want input+output+thoughts", resp.Usage.TotalTokens)
	}
}

func TestAntigravityUsageThroughModelAction(t *testing.T) {
	r := okInteraction("hello there")
	r.Usage = &interactionUsage{TotalInputTokens: 10, TotalOutputTokens: 5, TotalTokens: 15, InputTokensByModality: []modalityTokens{{Modality: "text", Tokens: 10}}}
	_, m := agSetup(t, r)
	resp, err := m.Generate(context.Background(), userReq("hi", nil), nil)
	if err != nil {
		t.Fatal(err)
	}
	// The framework fills the character counts from the actual text, not the
	// token breakdown.
	if resp.Usage.InputTokens != 10 || resp.Usage.InputCharacters != len("hi") || resp.Usage.OutputCharacters != len("hello there") {
		t.Errorf("usage = %+v", resp.Usage)
	}
	if resp.Usage.Custom["inputTextTokens"] != 10 {
		t.Errorf("custom = %v", resp.Usage.Custom)
	}
}

func TestFromInteractionStepExoticTypes(t *testing.T) {
	t.Run("google_search_call", func(t *testing.T) {
		parts := fromInteractionStep(interactionStep{Type: "google_search_call", ID: "gsc-1", Arguments: map[string]any{"queries": []any{"weather"}}, Signature: "sig-1"}, false)
		if len(parts) != 1 || parts[0].Custom["googleSearchCall"] == nil {
			t.Fatalf("parts = %+v", parts)
		}
		if string(metadataSignature(parts[0].Metadata)) != "sig-1" {
			t.Errorf("metadata = %v", parts[0].Metadata)
		}
	})
	t.Run("code_execution_call", func(t *testing.T) {
		parts := fromInteractionStep(interactionStep{Type: "code_execution_call", ID: "cec-1", Arguments: map[string]any{"code": "print(1)", "language": "PYTHON"}}, false)
		ec, ok := parts[0].Custom["executableCode"].(map[string]any)
		if !ok || ec["code"] != "print(1)" || parts[0].Metadata["callId"] != "cec-1" {
			t.Fatalf("parts = %+v", parts)
		}
	})
	t.Run("code_execution_result", func(t *testing.T) {
		parts := fromInteractionStep(interactionStep{Type: "code_execution_result", CallID: "cec-1", Result: "1\n"}, false)
		cr, ok := parts[0].Custom["codeExecutionResult"].(map[string]any)
		if !ok || cr["output"] != "1\n" {
			t.Fatalf("parts = %+v", parts)
		}
	})
	t.Run("unknown still preserved", func(t *testing.T) {
		parts := fromInteractionStep(interactionStep{Type: "some_future_type"}, false)
		if len(parts) != 1 || parts[0].Custom["unknownStep"] == nil {
			t.Fatalf("parts = %+v", parts)
		}
	})
}

// ---- HTTP layer ----

func agErrServer(t *testing.T, st int, body string, mutate func(*genai.ClientConfig)) *ai.ModelAction {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(st)
		_, _ = io.WriteString(w, body)
	}))
	t.Cleanup(srv.Close)
	return agModel(t, agClient(t, srv.URL, mutate))
}

func TestAntigravityHTTPErrorMapping(t *testing.T) {
	googleStyle := func(code int, st string) string {
		b, _ := json.Marshal(map[string]any{"error": map[string]any{"code": code, "message": "boom", "status": st}})
		return string(b)
	}
	interactionStyle := func(code string) string {
		b, _ := json.Marshal(map[string]any{"error": map[string]any{"code": code, "message": "boom"}})
		return string(b)
	}
	tests := []struct {
		name string
		http int
		body string
		want status.Name
	}{
		{"401 google", 401, googleStyle(401, "UNAUTHENTICATED"), status.Unauthenticated},
		{"401 interactions", 401, interactionStyle("authentication"), status.Unauthenticated},
		{"403", 403, googleStyle(403, "PERMISSION_DENIED"), status.PermissionDenied},
		{"404 non-json", 404, "not found", status.NotFound},
		{"409", 409, googleStyle(409, "ABORTED"), status.Aborted},
		{"429", 429, googleStyle(429, "RESOURCE_EXHAUSTED"), status.ResourceExhausted},
		{"400 invalid_request", 400, interactionStyle("invalid_request"), status.InvalidArgument},
		{"400 content block", 400, interactionStyle("safety"), status.FailedPrecondition},
		{"500", 500, "internal oops", status.Internal},
		{"503", 503, interactionStyle("service_unavailable"), status.Unavailable},
		{"504", 504, "", status.DeadlineExceeded},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			m := agErrServer(t, tc.http, tc.body, nil)
			_, err := m.Generate(context.Background(), userReq("hi", nil), nil)
			if err == nil {
				t.Fatal("expected an error")
			}
			if got, _ := status.Classified(err); got != tc.want {
				t.Errorf("status = %v, want %v (%v)", got, tc.want, err)
			}
		})
	}
}

func TestAntigravity429HonoursRetryInfo(t *testing.T) {
	body, _ := json.Marshal(map[string]any{"error": map[string]any{
		"code": 429, "message": "quota", "status": "RESOURCE_EXHAUSTED",
		"details": []any{map[string]any{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "17s"}},
	}})
	m := agErrServer(t, 429, string(body), nil)
	_, err := m.Generate(context.Background(), userReq("hi", nil), nil)
	if err == nil {
		t.Fatal("expected an error")
	}
	d, ok := RetryDelay(err)
	if !ok || d != 17*time.Second {
		t.Errorf("RetryDelay = %v, %v, want 17s", d, ok)
	}
	var se *status.Error
	if errors.As(err, &se) {
		if ms, _ := se.Details["retryAfterMs"].(int64); ms != 17000 {
			t.Errorf("details = %v", se.Details)
		}
	} else {
		t.Errorf("not a status error: %T", err)
	}
}

func TestAntigravityMalformedJSON(t *testing.T) {
	m := agErrServer(t, 200, "<html>gateway splash page that is not json</html>", nil)
	_, err := m.Generate(context.Background(), userReq("hi", nil), nil)
	if err == nil {
		t.Fatal("expected a decode error")
	}
	if !strings.Contains(err.Error(), "gateway splash page") {
		t.Errorf("error should include a body excerpt: %v", err)
	}
	if got, _ := status.Classified(err); got != status.Internal {
		t.Errorf("status = %v", got)
	}
}

func TestAntigravityResponseOverCap(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(200)
		chunk := []byte(strings.Repeat("x", 1<<20))
		for i := 0; i <= maxInteractionResponseBytes>>20; i++ {
			if _, err := w.Write(chunk); err != nil {
				return
			}
		}
	}))
	defer srv.Close()
	m := agModel(t, agClient(t, srv.URL, nil))
	_, err := m.Generate(context.Background(), userReq("hi", nil), nil)
	if err == nil || !strings.Contains(err.Error(), "limit") {
		t.Errorf("err = %v, want a size-limit error", err)
	}
}

func TestAntigravityContextCancellation(t *testing.T) {
	release := make(chan struct{})
	started := make(chan struct{}, 1)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.Copy(io.Discard, r.Body)
		started <- struct{}{}
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	defer srv.Close()
	defer close(release)
	m := agModel(t, agClient(t, srv.URL, nil))

	ctx, cancel := context.WithCancel(context.Background())
	go func() { <-started; cancel() }()
	_, err := m.Generate(ctx, userReq("hi", nil), nil)
	if err == nil {
		t.Fatal("expected an error")
	}
	if !errors.Is(err, context.Canceled) {
		t.Errorf("errors.Is(err, context.Canceled) = false: %v", err)
	}
	if got, _ := status.Classified(err); got != status.Cancelled {
		t.Errorf("status = %v, want CANCELLED (not UNAVAILABLE)", got)
	}
}

func TestAntigravityHonoursClientTimeoutAndHeaders(t *testing.T) {
	var gotHeader string
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotHeader = r.Header.Get("X-Custom")
		_, _ = io.Copy(io.Discard, r.Body)
		select {
		case <-release:
		case <-r.Context().Done():
		}
	}))
	defer srv.Close()
	defer close(release)

	timeout := 50 * time.Millisecond
	m := agModel(t, agClient(t, srv.URL, func(c *genai.ClientConfig) {
		c.HTTPOptions.Timeout = &timeout
		c.HTTPOptions.Headers = http.Header{"X-Custom": []string{"yes"}}
	}))
	start := time.Now()
	_, err := m.Generate(context.Background(), userReq("hi", nil), nil)
	if err == nil {
		t.Fatal("expected a timeout")
	}
	if got, _ := status.Classified(err); got != status.DeadlineExceeded || !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("err = %v (%v), want DEADLINE_EXCEEDED", err, got)
	}
	if time.Since(start) > 5*time.Second {
		t.Errorf("timeout not honoured")
	}
	if gotHeader != "yes" {
		t.Errorf("X-Custom header = %q, want the client config's header", gotHeader)
	}
}

func TestAntigravityUsesConfiguredHTTPClient(t *testing.T) {
	var used bool
	rec := &agRecorder{responses: []any{okInteraction("ok")}}
	srv := httptest.NewServer(rec)
	defer srv.Close()
	hc := &http.Client{Transport: roundTripFunc(func(r *http.Request) (*http.Response, error) {
		used = true
		return http.DefaultTransport.RoundTrip(r)
	})}
	m := agModel(t, agClient(t, srv.URL, func(c *genai.ClientConfig) { c.HTTPClient = hc }))
	if _, err := m.Generate(context.Background(), userReq("hi", nil), nil); err != nil {
		t.Fatal(err)
	}
	if !used {
		t.Error("the configured HTTPClient was not used")
	}
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }

// ---- Vertex scoping ----

func vertexClient(t *testing.T) *genai.Client {
	t.Helper()
	c, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		Backend: genai.BackendVertexAI, Project: "p", Location: "us-central1", HTTPClient: &http.Client{},
	})
	if err != nil {
		t.Fatalf("genai.NewClient (vertex): %v", err)
	}
	return c
}

func TestAntigravityNotResolvedOnVertex(t *testing.T) {
	v := vertexClient(t)
	cat := catalog{provider: vertexAIProvider}
	if a := resolveAction(v, cat, "model", antigravityPreview092026); a != nil {
		t.Errorf("resolveAction on Vertex returned %v, want nil", a)
	}
	// ... while Google AI still resolves it.
	g := agClient(t, "http://127.0.0.1:1", nil)
	if a := resolveAction(g, catalog{provider: googleAIProvider}, "model", antigravityPreview092026); a == nil {
		t.Error("resolveAction on Google AI returned nil")
	}
}

func TestAntigravityVertexDefineModelRejected(t *testing.T) {
	vp := &VertexAI{ProjectID: "p", Location: "us-central1"}
	vp.initted = true
	vp.gclient = vertexClient(t)
	for _, opts := range []*ai.ModelOptions{nil, {Supports: &AntigravitySupports}} {
		_, err := vp.DefineModel(nil, "vertexai/"+antigravityPreview092026, opts)
		if got, _ := status.Classified(err); got != status.InvalidArgument {
			t.Errorf("opts=%v: err = %v, want INVALID_ARGUMENT", opts, err)
		}
	}
}

func TestAntigravityVertexHandlerNeverCallsEndpoint(t *testing.T) {
	// Even a model action built directly on a Vertex client refuses the request
	// instead of POSTing to the Google AI host without a key.
	m := newModel(vertexClient(t), antigravityPreview092026, GetModelOptions(antigravityPreview092026, vertexAIProvider))
	_, err := m.Generate(context.Background(), userReq("hi", nil), nil)
	if got, _ := status.Classified(err); got != status.InvalidArgument || !strings.Contains(err.Error(), "Vertex") {
		t.Errorf("err = %v", err)
	}
}

func TestListGenaiModelsAntigravityByBackend(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"models":[{"name":"models/gemini-2.5-flash"},{"name":"models/antigravity-preview-09-2026"}]}`)
	}))
	defer srv.Close()

	got, err := listGenaiModels(context.Background(), agClient(t, srv.URL, nil))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(got.gemini, ",") != "gemini-2.5-flash,antigravity-preview-09-2026" {
		t.Errorf("Google AI listing = %v", got.gemini)
	}

	vc, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		Backend: genai.BackendVertexAI, Project: "p", Location: "l", HTTPClient: &http.Client{},
		HTTPOptions: genai.HTTPOptions{BaseURL: srv.URL},
	})
	if err != nil {
		t.Fatal(err)
	}
	got, err = listGenaiModels(context.Background(), vc)
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range got.gemini {
		if strings.HasPrefix(n, "antigravity") {
			t.Errorf("Vertex listing includes %q", n)
		}
	}
}
