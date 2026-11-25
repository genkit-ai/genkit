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
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/firebase/genkit/go/core/status"
	"google.golang.org/genai"
)

// geminiReply answers testQuestions as the Decisions API does.
const geminiReply = `{"answers":` + testAnswers + `,"usage":{"total_input_tokens":300,"input_tokens_by_modality":[{"modality":"TEXT","tokens":44},{"modality":"IMAGE","tokens":256}],"total_output_tokens":0,"total_tokens":300}}`

// newGemini is a client whose endpoint is the Gemini transport, on a genai
// SDK client pointed at the recorder.
func newGemini(t *testing.T, rec *recorder) *Client {
	t.Helper()
	srv := httptest.NewServer(rec)
	t.Cleanup(srv.Close)
	sdk, err := genai.NewClient(t.Context(), &genai.ClientConfig{
		APIKey:      "test-key",
		Backend:     genai.BackendGeminiAPI,
		HTTPClient:  srv.Client(),
		HTTPOptions: genai.HTTPOptions{BaseURL: srv.URL},
	})
	if err != nil {
		t.Fatal(err)
	}
	return &Client{Endpoint: &Endpoint{Name: "gemini", Transport: &Gemini{Name: "gemini", Client: sdk}}}
}

func TestGeminiRequest(t *testing.T) {
	rec := &recorder{reply: geminiReply}
	c := newGemini(t, rec)
	questions := map[string]Question{
		"department": {
			Type:         KindChoice,
			Instructions: []any{"The state is a ticket.", map[string]any{"question": "Which team?"}},
			Criteria:     Options{{"billing", map[string]any{"what": "Payments", "examples": []any{"Charged twice"}}}, {"technical", "Bugs"}},
		},
		"is_urgent": {Type: KindNoul, Instructions: "Urgent?", Criteria: map[string]any{"true": "Names a deadline", "false": "No time pressure"}},
		"plain":     {Type: KindNoul, Instructions: "Polite?"},
		"anger":     {Type: KindScore, Instructions: "How angry?", Criteria: []any{"Calm", map[string]any{"what": "Furious"}}},
	}
	_, err := c.Decide(t.Context(), "gemini-decision-1", &Request{
		State: map[string]any{"ticket": "hi"},
		Media: Media{
			{Kind: MediaImage, ContentType: "image/png", Data: "iVBORw0KGgo="},
			{Kind: MediaVideo, ContentType: "video/mp4", URI: "https://example.com/v.mp4"},
		},
		Questions:       questions,
		MediaResolution: "ultra_high",
	})
	if err != nil {
		t.Fatal(err)
	}
	req, body := rec.last(t)
	// The SDK's default version is v1, which does not serve decisions.
	if req.URL.Path != "/v1beta/decisions" || req.Header.Get("x-goog-api-key") != "test-key" {
		t.Errorf("path = %q, key header = %q", req.URL.Path, req.Header.Get("x-goog-api-key"))
	}
	// Media goes before the state's text, and ultra_high reaches images
	// only.
	wantInput := []any{
		map[string]any{"type": "image", "mime_type": "image/png", "data": "iVBORw0KGgo=", "resolution": "ultra_high"},
		map[string]any{"type": "video", "mime_type": "video/mp4", "uri": "https://example.com/v.mp4"},
		map[string]any{"type": "text", "text": `{"ticket":"hi"}`},
	}
	if !reflect.DeepEqual(body["input"], wantInput) {
		t.Errorf("input = %v, want %v", body["input"], wantInput)
	}
	// Criteria and instructions are text: guidance and structured
	// instructions go as their JSON, and a noul's sides are renamed.
	wantQuestions := map[string]any{
		"department": map[string]any{
			"type": "choice",
			"instructions": []any{
				map[string]any{"type": "text", "text": "The state is a ticket."},
				map[string]any{"type": "text", "text": `{"question":"Which team?"}`},
			},
			"criteria": map[string]any{"billing": `{"examples":["Charged twice"],"what":"Payments"}`, "technical": "Bugs"},
		},
		"is_urgent": map[string]any{"type": "noul", "instructions": "Urgent?", "criteria": map[string]any{"true_description": "Names a deadline", "false_description": "No time pressure"}},
		"plain":     map[string]any{"type": "noul", "instructions": "Polite?"},
		"anger":     map[string]any{"type": "score", "instructions": "How angry?", "criteria": []any{"Calm", `{"what":"Furious"}`}},
	}
	if !reflect.DeepEqual(body["questions"], wantQuestions) {
		t.Errorf("questions = %v, want %v", body["questions"], wantQuestions)
	}
}

func TestGeminiTextOnlyInputIsAString(t *testing.T) {
	rec := &recorder{reply: geminiReply}
	c := newGemini(t, rec)
	if _, err := c.Decide(t.Context(), "gemini-decision-1", &Request{State: "hi", Questions: testQuestions}); err != nil {
		t.Fatal(err)
	}
	if _, body := rec.last(t); body["input"] != "hi" {
		t.Errorf("input = %v, want the state as a string", body["input"])
	}
	// Media alone sends no empty text block, which the API refuses.
	_, err := c.Decide(t.Context(), "gemini-decision-1", &Request{
		State:     "",
		Media:     Media{{Kind: MediaImage, ContentType: "image/png", Data: "iVBORw0KGgo="}},
		Questions: testQuestions,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, body := rec.last(t); len(body["input"].([]any)) != 1 {
		t.Errorf("input = %v, want the image alone", body["input"])
	}
}

func TestGeminiResponse(t *testing.T) {
	c := newGemini(t, &recorder{reply: geminiReply})
	resp, err := c.Decide(t.Context(), "gemini-decision-1", &Request{State: "hi", Questions: testQuestions})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Model != "gemini-decision-1" || resp.Usage.InputTokens != 300 || resp.Usage.OutputTokens != 0 {
		t.Errorf("model = %q, usage = %+v", resp.Model, resp.Usage)
	}
	if want := map[string]int{"text": 44, "image": 256}; !reflect.DeepEqual(resp.Usage.InputByModality, want) {
		t.Errorf("by modality = %v, want %v", resp.Usage.InputByModality, want)
	}
}

func TestGeminiBlocked(t *testing.T) {
	for _, tt := range []struct {
		name       string
		code       int
		body       string
		wantReason string
	}{
		{"error code", 400, `{"error":{"code":"safety","message":"Input blocked for safety."}}`, "safety"},
		{"error detail", 400, `{"error":{"code":400,"message":"Blocked.","status":"INVALID_ARGUMENT","details":[{"@type":"type.googleapis.com/google.rpc.ErrorInfo","reason":"PROHIBITED_CONTENT"}]}}`, "prohibited_content"},
		{"prompt feedback", 200, `{"prompt_feedback":{"block_reason":"other","block_reason_message":"Blocked."},"usage":{"total_input_tokens":12}}`, "other"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			rec := &recorder{respond: func(w http.ResponseWriter, _ int) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(tt.code)
				_, _ = io.WriteString(w, tt.body)
			}}
			resp, err := newGemini(t, rec).Decide(t.Context(), "gemini-decision-1", &Request{State: "hi", Questions: testQuestions})
			if err != nil {
				t.Fatal(err)
			}
			if resp.Blocked == nil || resp.Blocked.Reason != tt.wantReason || resp.Blocked.Message == "" {
				t.Errorf("blocked = %+v, want reason %q with the message", resp.Blocked, tt.wantReason)
			}
		})
	}
	// Any other 400 is the request's fault.
	rec := &recorder{respond: func(w http.ResponseWriter, _ int) {
		w.WriteHeader(400)
		_, _ = io.WriteString(w, `{"error":{"code":400,"message":"Too many questions.","status":"INVALID_ARGUMENT"}}`)
	}}
	_, err := newGemini(t, rec).Decide(t.Context(), "gemini-decision-1", &Request{State: "hi", Questions: testQuestions})
	if !errors.Is(err, status.ErrInvalidArgument) {
		t.Errorf("error = %v, want INVALID_ARGUMENT", err)
	}
	// A success the SDK cannot read, such as a proxy's HTML page, is not
	// worth a retry and not the request's fault.
	rec = &recorder{respond: func(w http.ResponseWriter, _ int) {
		w.Header().Set("Content-Type", "text/html")
		_, _ = io.WriteString(w, "<html>Sign in to the proxy</html>")
	}}
	_, err = newGemini(t, rec).Decide(t.Context(), "gemini-decision-1", &Request{State: "hi", Questions: testQuestions})
	if !errors.Is(err, status.ErrInternal) {
		t.Errorf("error = %v, want INTERNAL for an unreadable success", err)
	}
}

func TestGeminiErrors(t *testing.T) {
	for _, tt := range []struct {
		name string
		code int
		body string
		want error
	}{
		// The Gemini API sends a rejected key as a 400.
		{"rejected key", 400, `{"error":{"code":400,"message":"API key not valid.","status":"INVALID_ARGUMENT","details":[{"@type":"type.googleapis.com/google.rpc.ErrorInfo","reason":"API_KEY_INVALID"}]}}`, status.ErrUnauthenticated},
		{"status wins over the HTTP code", 400, `{"error":{"code":400,"message":"Billing not enabled.","status":"FAILED_PRECONDITION"}}`, status.ErrFailedPrecondition},
		{"no status", 404, `{"error":{"message":"Not found."}}`, status.ErrNotFound},
		// The SDK reads the body after the response arrives, so a dropped
		// connection surfaces as a short read rather than a url.Error.
		{"connection dropped mid-body", 200, "", status.ErrUnavailable},
	} {
		t.Run(tt.name, func(t *testing.T) {
			rec := &recorder{respond: func(w http.ResponseWriter, _ int) {
				w.Header().Set("Content-Type", "application/json")
				if tt.body == "" {
					w.Header().Set("Content-Length", "100")
					w.WriteHeader(tt.code)
					_, _ = io.WriteString(w, `{"answers":`)
					return
				}
				w.WriteHeader(tt.code)
				_, _ = io.WriteString(w, tt.body)
			}}
			_, err := newGemini(t, rec).Decide(t.Context(), "gemini-decision-1", &Request{State: "hi", Questions: testQuestions})
			if !errors.Is(err, tt.want) {
				t.Errorf("error = %v, want %v", err, tt.want)
			}
		})
	}
}
