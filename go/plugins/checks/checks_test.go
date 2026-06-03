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

package checks

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
)

type fakeTokenSource struct{}

func (fakeTokenSource) Token() (*oauth2.Token, error) {
	return &oauth2.Token{AccessToken: "test-token", TokenType: "Bearer", Expiry: time.Now().Add(time.Hour)}, nil
}

func ptr(f float64) *float64 { return &f }

// writeResults responds with one policyResult per requested policy, marking a
// policy VIOLATIVE when the content contains "bad". The first policy carries a
// score; the rest omit it (to exercise the optional-score path).
func writeResults(w http.ResponseWriter, req classifyRequest) {
	violative := strings.Contains(strings.ToLower(req.Input.TextInput.Content), "bad")
	var results []policyResult
	for i, p := range req.Policies {
		r := policyResult{PolicyType: p.PolicyType, ViolationResult: "NOT_VIOLATIVE"}
		if violative {
			r.ViolationResult = "VIOLATIVE"
		}
		if i == 0 {
			r.Score = ptr(0.9)
		}
		results = append(results, r)
	}
	_ = json.NewEncoder(w).Encode(classifyResponse{PolicyResults: results})
}

func TestClassifyContent_RequestAndResponse(t *testing.T) {
	var gotAuth, gotProject, gotPath, gotMethod string
	var gotBody classifyRequest
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotProject = r.Header.Get("x-goog-user-project")
		gotPath = r.URL.Path
		gotMethod = r.Method
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(classifyResponse{PolicyResults: []policyResult{
			{PolicyType: "HATE_SPEECH", Score: ptr(0.82), ViolationResult: "VIOLATIVE"},
			{PolicyType: "HARASSMENT", ViolationResult: "NOT_VIOLATIVE"}, // no score
		}})
	}))
	defer srv.Close()

	g := newGuardrails(fakeTokenSource{}, "my-project", srv.URL)
	resp, err := g.ClassifyContent(context.Background(), "some text", []ChecksMetricConfig{
		{Type: HateSpeech, Threshold: ptr(0.5)},
		{Type: Harassment},
	})
	if err != nil {
		t.Fatalf("ClassifyContent: %v", err)
	}

	if gotMethod != http.MethodPost {
		t.Errorf("method = %s, want POST", gotMethod)
	}
	if gotAuth != "Bearer test-token" {
		t.Errorf("Authorization = %q, want Bearer test-token", gotAuth)
	}
	if gotProject != "my-project" {
		t.Errorf("x-goog-user-project = %q, want my-project", gotProject)
	}
	if gotPath == "" {
		t.Errorf("empty request path")
	}
	if gotBody.Input.TextInput.Content != "some text" {
		t.Errorf("content = %q, want 'some text'", gotBody.Input.TextInput.Content)
	}
	if len(gotBody.Policies) != 2 || gotBody.Policies[0].PolicyType != "HATE_SPEECH" || gotBody.Policies[0].Threshold == nil || *gotBody.Policies[0].Threshold != 0.5 {
		t.Errorf("policies mapped incorrectly: %+v", gotBody.Policies)
	}

	if len(resp.PolicyResults) != 2 {
		t.Fatalf("expected 2 policy results, got %d", len(resp.PolicyResults))
	}
	if resp.PolicyResults[0].Score == nil || *resp.PolicyResults[0].Score != 0.82 {
		t.Errorf("result[0] score = %v, want 0.82", resp.PolicyResults[0].Score)
	}
	if resp.PolicyResults[1].Score != nil {
		t.Errorf("result[1] score = %v, want nil (omitted)", resp.PolicyResults[1].Score)
	}
}

func TestEvaluator_FanOutAndMapping(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req classifyRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		writeResults(w, req)
	}))
	defer srv.Close()

	c := &Checks{
		Metrics:   []ChecksMetricConfig{{Type: HateSpeech}, {Type: Harassment}},
		ts:        fakeTokenSource{},
		projectID: "proj",
		endpoint:  srv.URL,
		initted:   true,
	}
	ev := newEvaluator(c)

	resp, err := ev.Evaluate(context.Background(), &ai.EvaluatorRequest{
		Dataset: []*ai.Example{
			{TestCaseId: "a", Output: "totally fine"},
			{TestCaseId: "b", Output: "this is bad content"},
		},
	})
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	results := *resp
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	// Order preserved.
	if results[0].TestCaseId != "a" || results[1].TestCaseId != "b" {
		t.Fatalf("order not preserved: %s, %s", results[0].TestCaseId, results[1].TestCaseId)
	}
	// Two policies => two scores per datapoint.
	if len(results[0].Evaluation) != 2 {
		t.Fatalf("expected 2 scores, got %d", len(results[0].Evaluation))
	}
	s0 := results[0].Evaluation[0]
	if s0.Id != "HATE_SPEECH" {
		t.Errorf("score id = %q, want HATE_SPEECH", s0.Id)
	}
	if s0.Score == nil || s0.Score.(float64) != 0.9 {
		t.Errorf("score = %v, want 0.9", s0.Score)
	}
	if got := s0.Details["reasoning"]; got != "Status NOT_VIOLATIVE" {
		t.Errorf("reasoning = %v, want 'Status NOT_VIOLATIVE'", got)
	}
	// Second policy of first datapoint has no score (omitted).
	if results[0].Evaluation[1].Score != nil {
		t.Errorf("expected nil score for second policy, got %v", results[0].Evaluation[1].Score)
	}
	// Datapoint b is violative.
	if got := results[1].Evaluation[0].Details["reasoning"]; got != "Status VIOLATIVE" {
		t.Errorf("datapoint b reasoning = %v, want 'Status VIOLATIVE'", got)
	}
}

func TestGuardrailMiddleware(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req classifyRequest
		_ = json.NewDecoder(r.Body).Decode(&req)
		writeResults(w, req)
	}))
	defer srv.Close()

	g := newGuardrails(fakeTokenSource{}, "proj", srv.URL)
	mw := GuardrailMiddleware(g, []ChecksMetricConfig{{Type: HateSpeech}})
	hooks, err := mw.New(context.Background())
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	userMsg := func(text string) *ai.ModelParams {
		return &ai.ModelParams{Request: &ai.ModelRequest{
			Messages: []*ai.Message{{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart(text)}}},
		}}
	}
	modelResp := func(text string) *ai.ModelResponse {
		return &ai.ModelResponse{Message: &ai.Message{Role: ai.RoleModel, Content: []*ai.Part{ai.NewTextPart(text)}}}
	}

	t.Run("blocks violative input", func(t *testing.T) {
		var nextCalled bool
		next := func(ctx context.Context, p *ai.ModelParams) (*ai.ModelResponse, error) {
			nextCalled = true
			return modelResp("ok"), nil
		}
		resp, err := hooks.WrapModel(context.Background(), userMsg("this is bad"), next)
		if err != nil {
			t.Fatal(err)
		}
		if nextCalled {
			t.Error("next should not be called when input is blocked")
		}
		if resp.FinishReason != ai.FinishReasonBlocked || !strings.Contains(resp.FinishMessage, "input") {
			t.Errorf("expected blocked input response, got %+v", resp)
		}
	})

	t.Run("blocks violative output", func(t *testing.T) {
		next := func(ctx context.Context, p *ai.ModelParams) (*ai.ModelResponse, error) {
			return modelResp("a bad answer"), nil
		}
		resp, err := hooks.WrapModel(context.Background(), userMsg("fine prompt"), next)
		if err != nil {
			t.Fatal(err)
		}
		if resp.FinishReason != ai.FinishReasonBlocked || !strings.Contains(resp.FinishMessage, "output") {
			t.Errorf("expected blocked output response, got %+v", resp)
		}
	})

	t.Run("passes through when clean", func(t *testing.T) {
		want := modelResp("a fine answer")
		next := func(ctx context.Context, p *ai.ModelParams) (*ai.ModelResponse, error) {
			return want, nil
		}
		resp, err := hooks.WrapModel(context.Background(), userMsg("fine prompt"), next)
		if err != nil {
			t.Fatal(err)
		}
		if resp != want {
			t.Errorf("expected pass-through response, got %+v", resp)
		}
	})
}

func TestClassifyContent_RetriesOn429(t *testing.T) {
	var calls int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if atomic.AddInt32(&calls, 1) == 1 {
			w.Header().Set("Retry-After", "0")
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
		_ = json.NewEncoder(w).Encode(classifyResponse{PolicyResults: []policyResult{
			{PolicyType: "HATE_SPEECH", Score: ptr(0.1), ViolationResult: "NOT_VIOLATIVE"},
		}})
	}))
	defer srv.Close()

	g := newGuardrails(fakeTokenSource{}, "proj", srv.URL)
	resp, err := g.ClassifyContent(context.Background(), "text", []ChecksMetricConfig{{Type: HateSpeech}})
	if err != nil {
		t.Fatalf("ClassifyContent after retry: %v", err)
	}
	if got := atomic.LoadInt32(&calls); got != 2 {
		t.Errorf("expected 2 calls (1 retry), got %d", got)
	}
	if len(resp.PolicyResults) != 1 {
		t.Errorf("expected 1 result, got %d", len(resp.PolicyResults))
	}
}

// TestEvaluatorRegistration verifies the name and the api.Action cast that
// Checks.Init relies on (Init itself needs ADC, so it isn't exercised offline).
func TestEvaluatorRegistration(t *testing.T) {
	c := &Checks{ts: fakeTokenSource{}, projectID: "p", Metrics: []ChecksMetricConfig{{Type: HateSpeech}}}
	ev := newEvaluator(c)
	if ev.Name() != "checks/guardrails" {
		t.Errorf("name = %q, want checks/guardrails", ev.Name())
	}
	if _, ok := ev.(api.Action); !ok {
		t.Fatal("evaluator does not implement api.Action — Init cast would panic")
	}
}

func TestOutputText(t *testing.T) {
	if got := outputText(nil); got != "" {
		t.Errorf("outputText(nil) = %q, want empty", got)
	}
	if got := outputText("hello"); got != "hello" {
		t.Errorf("outputText(string) = %q, want hello", got)
	}
	if got := outputText(42); got != "42" {
		t.Errorf("outputText(42) = %q, want 42", got)
	}
}

func TestResolveProjectID(t *testing.T) {
	t.Run("explicit wins", func(t *testing.T) {
		t.Setenv("GOOGLE_CLOUD_PROJECT", "env-proj")
		got := resolveProjectID("explicit-proj", &google.Credentials{ProjectID: "adc-proj"})
		if got != "explicit-proj" {
			t.Errorf("got %q, want explicit-proj", got)
		}
	})
	t.Run("env over adc", func(t *testing.T) {
		t.Setenv("GOOGLE_CLOUD_PROJECT", "env-proj")
		got := resolveProjectID("", &google.Credentials{ProjectID: "adc-proj"})
		if got != "env-proj" {
			t.Errorf("got %q, want env-proj", got)
		}
	})
	t.Run("adc fallback", func(t *testing.T) {
		t.Setenv("GOOGLE_CLOUD_PROJECT", "")
		got := resolveProjectID("", &google.Credentials{ProjectID: "adc-proj"})
		if got != "adc-proj" {
			t.Errorf("got %q, want adc-proj", got)
		}
	})
}
