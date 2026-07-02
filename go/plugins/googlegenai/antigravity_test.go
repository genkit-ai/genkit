// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

package googlegenai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"google.golang.org/genai"
)

// capturedRequest records what the fake interactions endpoint received.
type capturedRequest struct {
	method  string
	path    string
	headers http.Header
	body    map[string]any
}

// newInteractionServer returns a test server that records the incoming request
// and replies with respBody, plus a genai client pointed at it.
func newInteractionServer(t *testing.T, respBody any, status int) (*genai.Client, *capturedRequest) {
	t.Helper()
	captured := &capturedRequest{}

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		captured.method = r.Method
		captured.path = r.URL.Path
		captured.headers = r.Header.Clone()
		data, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(data, &captured.body)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		out, _ := json.Marshal(respBody)
		_, _ = w.Write(out)
	}))
	t.Cleanup(server.Close)

	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		Backend:     genai.BackendGeminiAPI,
		APIKey:      "test-api-key-plugin",
		HTTPOptions: genai.HTTPOptions{BaseURL: server.URL},
	})
	if err != nil {
		t.Fatalf("genai.NewClient: %v", err)
	}
	return client, captured
}

func mockInteractionResponse() *geminiInteraction {
	return &geminiInteraction{
		ID:            "interaction-123",
		Status:        "completed",
		EnvironmentID: "env-123",
		Steps: []interactionStep{
			{Type: "model_output", Content: []interactionContent{{Type: "text", Text: "Hello human"}}},
		},
	}
}

func TestAntigravitySetsAgentAndHitsInteractionsEndpoint(t *testing.T) {
	client, captured := newInteractionServer(t, mockInteractionResponse(), http.StatusOK)

	input := &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserTextMessage("Hello antigravity")},
	}

	_, err := generateAntigravity(context.Background(), client, antigravityPreview052026, input, nil)
	if err != nil {
		t.Fatalf("generateAntigravity: %v", err)
	}

	if captured.method != http.MethodPost {
		t.Errorf("method = %q, want POST", captured.method)
	}
	if captured.path != "/v1beta/interactions" {
		t.Errorf("path = %q, want /v1beta/interactions", captured.path)
	}
	if got := captured.headers.Get("Api-Revision"); got != antigravityAPIRevision {
		t.Errorf("Api-Revision = %q, want %q", got, antigravityAPIRevision)
	}
	if got := captured.headers.Get("x-goog-api-key"); got != "test-api-key-plugin" {
		t.Errorf("x-goog-api-key = %q, want test-api-key-plugin", got)
	}
	if got := captured.headers.Get("x-goog-api-client"); got == "" {
		t.Errorf("x-goog-api-client header missing")
	}
	if got := captured.body["agent"]; got != antigravityPreview052026 {
		t.Errorf("agent = %v, want %q", got, antigravityPreview052026)
	}
	// The interactions endpoint requires an environment; with no config we
	// default it to a remote sandbox.
	if got := captured.body["environment"]; got != "remote" {
		t.Errorf("environment = %v, want remote (default)", got)
	}
	// Empty config: store and tools must be absent.
	if _, ok := captured.body["store"]; ok {
		t.Errorf("store should be absent for empty config")
	}
	if _, ok := captured.body["tools"]; ok {
		t.Errorf("tools should be absent for empty config")
	}
}

func TestAntigravityMapsSystemMessagesToUserInput(t *testing.T) {
	client, captured := newInteractionServer(t, mockInteractionResponse(), http.StatusOK)

	input := &ai.ModelRequest{
		Messages: []*ai.Message{
			ai.NewSystemTextMessage("You are an agent"),
			ai.NewUserTextMessage("Hello"),
		},
	}

	if _, err := generateAntigravity(context.Background(), client, antigravityPreview052026, input, nil); err != nil {
		t.Fatalf("generateAntigravity: %v", err)
	}

	steps, ok := captured.body["input"].([]any)
	if !ok || len(steps) != 2 {
		t.Fatalf("input = %v, want 2 steps", captured.body["input"])
	}
	first := steps[0].(map[string]any)
	if first["type"] != "user_input" {
		t.Errorf("step[0].type = %v, want user_input (system should map to user)", first["type"])
	}
	firstContent := first["content"].([]any)[0].(map[string]any)
	if firstContent["text"] != "You are an agent" {
		t.Errorf("step[0] text = %v, want %q", firstContent["text"], "You are an agent")
	}
	second := steps[1].(map[string]any)
	if second["type"] != "user_input" {
		t.Errorf("step[1].type = %v, want user_input", second["type"])
	}
}

func TestAntigravityMapsExplicitConfigOptions(t *testing.T) {
	client, captured := newInteractionServer(t, mockInteractionResponse(), http.StatusOK)

	input := &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserTextMessage("Hello antigravity")},
		Config: map[string]any{
			"environment":           "remote",
			"previousInteractionId": "interaction-abc",
			"store":                 true,
			"tools":                 []any{map[string]any{"type": "google_search"}},
		},
	}

	if _, err := generateAntigravity(context.Background(), client, antigravityPreview052026, input, nil); err != nil {
		t.Fatalf("generateAntigravity: %v", err)
	}

	if captured.body["environment"] != "remote" {
		t.Errorf("environment = %v, want remote", captured.body["environment"])
	}
	if captured.body["store"] != true {
		t.Errorf("store = %v, want true", captured.body["store"])
	}
	if captured.body["previous_interaction_id"] != "interaction-abc" {
		t.Errorf("previous_interaction_id = %v, want interaction-abc", captured.body["previous_interaction_id"])
	}
	// tools is an unmapped key forwarded verbatim.
	if _, ok := captured.body["tools"]; !ok {
		t.Errorf("tools passthrough missing from request body")
	}
}

func TestAntigravityPassesUnmappedConfigThrough(t *testing.T) {
	client, captured := newInteractionServer(t, mockInteractionResponse(), http.StatusOK)

	input := &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserTextMessage("Hello antigravity")},
		Config: map[string]any{
			"temperature": 0.8,
			"topP":        0.9,
		},
	}

	if _, err := generateAntigravity(context.Background(), client, antigravityPreview052026, input, nil); err != nil {
		t.Fatalf("generateAntigravity: %v", err)
	}

	if captured.body["temperature"] != 0.8 {
		t.Errorf("temperature = %v, want 0.8", captured.body["temperature"])
	}
	if captured.body["topP"] != 0.9 {
		t.Errorf("topP = %v, want 0.9", captured.body["topP"])
	}
}

func TestAntigravitySurfacesInteractionMetadataAndContent(t *testing.T) {
	client, _ := newInteractionServer(t, mockInteractionResponse(), http.StatusOK)

	input := &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserTextMessage("Hello antigravity")},
	}

	resp, err := generateAntigravity(context.Background(), client, antigravityPreview052026, input, nil)
	if err != nil {
		t.Fatalf("generateAntigravity: %v", err)
	}

	if resp.Message == nil {
		t.Fatal("response message is nil")
	}
	if got := resp.Message.Metadata["interactionId"]; got != "interaction-123" {
		t.Errorf("interactionId = %v, want interaction-123", got)
	}
	if got := resp.Message.Metadata["environmentId"]; got != "env-123" {
		t.Errorf("environmentId = %v, want env-123", got)
	}
	if resp.Text() != "Hello human" {
		t.Errorf("response text = %q, want %q", resp.Text(), "Hello human")
	}
	if resp.FinishReason != ai.FinishReasonStop {
		t.Errorf("finishReason = %q, want stop", resp.FinishReason)
	}
}

func TestAntigravityStreamingEmitsSingleChunk(t *testing.T) {
	client, _ := newInteractionServer(t, mockInteractionResponse(), http.StatusOK)

	input := &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserTextMessage("Hello antigravity")},
	}

	var chunks int
	cb := func(_ context.Context, chunk *ai.ModelResponseChunk) error {
		chunks++
		return nil
	}

	if _, err := generateAntigravity(context.Background(), client, antigravityPreview052026, input, cb); err != nil {
		t.Fatalf("generateAntigravity: %v", err)
	}
	if chunks != 1 {
		t.Errorf("emitted %d chunks, want 1 (antigravity is sync-only)", chunks)
	}
}

func TestFromInteractionSyncCancelled(t *testing.T) {
	resp, err := fromInteractionSync(&geminiInteraction{ID: "x", Status: "cancelled"})
	if err != nil {
		t.Fatalf("fromInteractionSync: %v", err)
	}
	if resp.FinishReason != ai.FinishReasonInterrupted {
		t.Errorf("finishReason = %q, want interrupted", resp.FinishReason)
	}
	if resp.Text() != "Operation cancelled." {
		t.Errorf("text = %q, want %q", resp.Text(), "Operation cancelled.")
	}
}

func TestFromInteractionSyncFailed(t *testing.T) {
	if _, err := fromInteractionSync(&geminiInteraction{Status: "failed"}); err == nil {
		t.Fatal("expected error for failed interaction, got nil")
	}
}

func TestFromInteractionSyncUsage(t *testing.T) {
	resp, err := fromInteractionSync(&geminiInteraction{
		Status: "completed",
		Steps:  []interactionStep{{Type: "model_output", Content: []interactionContent{{Type: "text", Text: "hi"}}}},
		Usage: &interactionUsage{
			TotalInputTokens:      10,
			TotalOutputTokens:     5,
			TotalTokens:           15,
			InputTokensByModality: []modalityTokens{{Modality: "text", Tokens: 10}},
		},
	})
	if err != nil {
		t.Fatalf("fromInteractionSync: %v", err)
	}
	if resp.Usage.InputTokens != 10 || resp.Usage.OutputTokens != 5 || resp.Usage.TotalTokens != 15 {
		t.Errorf("usage tokens = %+v", resp.Usage)
	}
	if resp.Usage.InputCharacters != 10 {
		t.Errorf("inputCharacters = %d, want 10", resp.Usage.InputCharacters)
	}
}

func TestFromInteractionStepExoticTypes(t *testing.T) {
	t.Run("google_search_call", func(t *testing.T) {
		parts := fromInteractionStep(interactionStep{
			Type:      "google_search_call",
			ID:        "gsc-1",
			Arguments: map[string]any{"queries": []any{"weather"}},
			Signature: "sig-1",
		})
		if len(parts) != 1 || parts[0].Custom["googleSearchCall"] == nil {
			t.Fatalf("expected googleSearchCall custom part, got %+v", parts)
		}
		if parts[0].Metadata["signature"] != "sig-1" {
			t.Errorf("signature metadata = %v, want sig-1", parts[0].Metadata["signature"])
		}
	})

	t.Run("code_execution_call", func(t *testing.T) {
		parts := fromInteractionStep(interactionStep{
			Type:      "code_execution_call",
			ID:        "cec-1",
			Arguments: map[string]any{"code": "print(1)", "language": "PYTHON"},
		})
		if len(parts) != 1 {
			t.Fatalf("expected 1 part, got %d", len(parts))
		}
		ec, ok := parts[0].Custom["executableCode"].(map[string]any)
		if !ok {
			t.Fatalf("expected executableCode custom part, got %+v", parts[0].Custom)
		}
		if ec["code"] != "print(1)" {
			t.Errorf("code = %v, want print(1)", ec["code"])
		}
		if parts[0].Metadata["callId"] != "cec-1" {
			t.Errorf("callId = %v, want cec-1", parts[0].Metadata["callId"])
		}
	})

	t.Run("code_execution_result", func(t *testing.T) {
		parts := fromInteractionStep(interactionStep{
			Type:   "code_execution_result",
			CallID: "cec-1",
			Result: "1\n",
		})
		cr, ok := parts[0].Custom["codeExecutionResult"].(map[string]any)
		if !ok {
			t.Fatalf("expected codeExecutionResult custom part, got %+v", parts[0].Custom)
		}
		if cr["output"] != "1\n" {
			t.Errorf("output = %v, want %q", cr["output"], "1\n")
		}
	})

	t.Run("unknown still preserved", func(t *testing.T) {
		parts := fromInteractionStep(interactionStep{Type: "some_future_type"})
		if len(parts) != 1 || parts[0].Custom["unknownStep"] == nil {
			t.Fatalf("expected unknownStep fallback, got %+v", parts)
		}
	})
}

func TestAntigravityErrorMapping(t *testing.T) {
	client, _ := newInteractionServer(t, map[string]any{
		"error": map[string]any{"message": "bad request"},
	}, http.StatusBadRequest)

	input := &ai.ModelRequest{
		Messages: []*ai.Message{ai.NewUserTextMessage("Hello")},
	}

	_, err := generateAntigravity(context.Background(), client, antigravityPreview052026, input, nil)
	if err == nil {
		t.Fatal("expected error for 400 response, got nil")
	}
}
