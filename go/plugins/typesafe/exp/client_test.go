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

package exp

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

	"github.com/firebase/genkit/go/core/status"
)

// recorder is a fake endpoint that records what it was sent and answers
// with a canned body, or with a scripted sequence of statuses.
type recorder struct {
	mu       sync.Mutex
	requests []*http.Request
	bodies   []map[string]any
	// respond, when set, takes precedence over reply.
	respond func(w http.ResponseWriter, call int)
	reply   string
}

// record notes a request and its JSON body, and returns the call's index
// and the body.
func (r *recorder) record(req *http.Request) (call int, body map[string]any) {
	raw, _ := io.ReadAll(req.Body)
	if len(raw) > 0 {
		_ = json.Unmarshal(raw, &body)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.requests = append(r.requests, req)
	r.bodies = append(r.bodies, body)
	return len(r.requests) - 1, body
}

func (r *recorder) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	call, _ := r.record(req)
	if r.respond != nil {
		r.respond(w, call)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = io.WriteString(w, r.reply)
}

func (r *recorder) last(t *testing.T) (*http.Request, map[string]any) {
	t.Helper()
	r.mu.Lock()
	defer r.mu.Unlock()
	if len(r.requests) == 0 {
		t.Fatal("the endpoint was never called")
	}
	return r.requests[len(r.requests)-1], r.bodies[len(r.bodies)-1]
}

func (r *recorder) calls() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.requests)
}

const cannedReply = `{"model":"jev-1.13.0","answers":` + triageAnswers + `,"usage":{"input_tokens":312,"output_tokens":48}}`

func newClient(t *testing.T, rec *recorder, ep *Endpoint) *client {
	t.Helper()
	srv := httptest.NewServer(rec)
	t.Cleanup(srv.Close)
	return &client{http: srv.Client(), baseURL: srv.URL, apiKey: "test-key", ep: ep}
}

func TestDirectEndpoint(t *testing.T) {
	rec := &recorder{reply: cannedReply}
	c := newClient(t, rec, Direct())
	c.headers = http.Header{"X-Trace": {"abc"}}

	resp, err := c.decide(t.Context(), "jev-1.13.0", &request{State: "hi", Questions: triageQuestions})
	if err != nil {
		t.Fatal(err)
	}
	req, body := rec.last(t)
	if req.URL.Path != "/v1/systemone" || req.Method != http.MethodPost {
		t.Errorf("request = %s %s, want POST /v1/systemone", req.Method, req.URL.Path)
	}
	if got := req.Header.Get("Authorization"); got != "Bearer test-key" {
		t.Errorf("Authorization = %q", got)
	}
	if got := req.Header.Get("X-Trace"); got != "abc" {
		t.Errorf("extra header X-Trace = %q, want abc", got)
	}
	if body["model"] != "jev-1.13.0" || body["state"] != "hi" {
		t.Errorf("body = %v", body)
	}
	questions := body["questions"].(map[string]any)
	if q := questions["department"].(map[string]any); q["type"] != "choice" || q["instructions"] != "Which team should handle this?" {
		t.Errorf("department question on the wire = %v", q)
	}
	if resp.Model != "jev-1.13.0" || resp.Usage.InputTokens != 312 || resp.Answers["department"]["choice"] != "billing" {
		t.Errorf("response = %+v", resp)
	}
}

func TestOpenRouterEndpoint(t *testing.T) {
	rec := &recorder{reply: `{"id":"gen-1","provider":"TypeSafe","model":"typesafe/jev-1.13","answers":` + triageAnswers + `,"usage":{"input_tokens":1,"output_tokens":2,"cost":0.00004}}`}
	c := newClient(t, rec, OpenRouter())

	for id, want := range map[string]string{
		"jev-latest":            "~typesafe/jev-latest",
		"jev-preview":           "~typesafe/jev-preview",
		"jev-1.13":              "typesafe/jev-1.13",
		"typesafe/jev-1.13":     "typesafe/jev-1.13",
		"~typesafe/jev-preview": "~typesafe/jev-preview",
	} {
		resp, err := c.decide(t.Context(), id, &request{State: "hi", Questions: triageQuestions})
		if err != nil {
			t.Fatal(err)
		}
		req, body := rec.last(t)
		if req.URL.Path != "/api/alpha/decisions" {
			t.Errorf("path = %s, want /api/alpha/decisions", req.URL.Path)
		}
		if body["model"] != want {
			t.Errorf("model %q sent as %v, want %q", id, body["model"], want)
		}
		if resp.Provider != "TypeSafe" || resp.ID != "gen-1" || resp.Usage.Cost == nil || *resp.Usage.Cost != 0.00004 {
			t.Errorf("gateway fields not decoded: %+v", resp)
		}
	}

	calls := rec.calls()
	if _, err := c.decide(t.Context(), "jev-1.13.0", &request{State: "hi", Questions: triageQuestions}); err == nil || !errors.Is(err, status.ErrInvalidArgument) || !strings.Contains(err.Error(), "jev-1.13") {
		t.Errorf("patch version on OpenRouter: error = %v, want invalid argument naming jev-1.13", err)
	}
	if rec.calls() != calls {
		t.Error("a patch version was sent to OpenRouter instead of being refused")
	}
}

func TestCloudflareEndpoint(t *testing.T) {
	envelope := `{"result":` + cannedReply + `,"success":true,"errors":[],"messages":[]}`
	rec := &recorder{reply: envelope}
	c := newClient(t, rec, Cloudflare("acct-1"))

	resp, err := c.decide(t.Context(), "jev-latest", &request{State: "hi", Questions: triageQuestions})
	if err != nil {
		t.Fatal(err)
	}
	req, body := rec.last(t)
	if req.URL.Path != "/client/v4/accounts/acct-1/ai/run" {
		t.Errorf("path = %s", req.URL.Path)
	}
	if body["model"] != "typesafe/jev" {
		t.Errorf("model = %v, want typesafe/jev", body["model"])
	}
	input, _ := body["input"].(map[string]any)
	if input["state"] != "hi" || input["questions"] == nil || input["model"] != nil {
		t.Errorf("input = %v: want state and questions only", input)
	}
	if _, top := body["state"]; top {
		t.Error("state was sent at the top level; Cloudflare takes it under input")
	}
	if resp.Answers["department"]["choice"] != "billing" {
		t.Errorf("the envelope was not unwrapped: %+v", resp)
	}

	// The bare shape the model page documents works too.
	rec.reply = cannedReply
	if resp, err := c.decide(t.Context(), "jev-latest", &request{State: "hi", Questions: triageQuestions}); err != nil || resp.Model != "jev-1.13.0" {
		t.Errorf("bare response: %+v, %v", resp, err)
	}

	rec.reply = `{"result":null,"success":false,"errors":[{"code":7000,"message":"No route for that URI"}]}`
	if _, err := c.decide(t.Context(), "jev-latest", &request{State: "hi", Questions: triageQuestions}); err == nil || !strings.Contains(err.Error(), "No route") {
		t.Errorf("failed envelope error = %v", err)
	}

	calls := rec.calls()
	if _, err := c.decide(t.Context(), "jev-1.13.0", &request{State: "hi", Questions: triageQuestions}); err == nil || !errors.Is(err, status.ErrInvalidArgument) {
		t.Errorf("pinned version on Cloudflare: error = %v, want invalid argument", err)
	}
	if rec.calls() != calls {
		t.Error("a pinned version was sent to Cloudflare instead of being refused")
	}

	// The account ID comes from the environment when none is given, and is
	// escaped into the path.
	t.Setenv("CLOUDFLARE_ACCOUNT_ID", "acct/2")
	ep, err := Cloudflare("").withAccount()
	if err != nil {
		t.Fatal(err)
	}
	c = newClient(t, rec, ep)
	rec.reply = cannedReply
	if _, err := c.decide(t.Context(), "jev-latest", &request{State: "hi", Questions: triageQuestions}); err != nil {
		t.Fatal(err)
	}
	if req, _ := rec.last(t); req.URL.EscapedPath() != "/client/v4/accounts/acct%2F2/ai/run" {
		t.Errorf("path = %s, want the account from the environment, escaped", req.URL.EscapedPath())
	}
}

func TestExtraMergesTopLevelFields(t *testing.T) {
	rec := &recorder{reply: cannedReply}
	c := newClient(t, rec, OpenRouter())
	_, err := c.decide(t.Context(), "jev-latest", &request{
		State:     "hi",
		Questions: triageQuestions,
		Extra:     map[string]any{"session_id": "s-1", "model": "typesafe/jev-9"},
	})
	if err != nil {
		t.Fatal(err)
	}
	_, body := rec.last(t)
	if body["session_id"] != "s-1" {
		t.Errorf("session_id = %v, want s-1", body["session_id"])
	}
	if body["model"] != "typesafe/jev-9" {
		t.Errorf("model = %v: an extra wins over the field it collides with", body["model"])
	}
}

func TestRetriesOverloadAndHonorsRetryAfter(t *testing.T) {
	rec := &recorder{respond: func(w http.ResponseWriter, call int) {
		if call == 0 {
			w.Header().Set("Retry-After", "0")
			w.WriteHeader(529)
			_, _ = io.WriteString(w, `{"error":{"message":"overloaded"}}`)
			return
		}
		_, _ = io.WriteString(w, cannedReply)
	}}
	c := newClient(t, rec, Direct())
	start := time.Now()
	if _, err := c.decide(t.Context(), "jev-latest", &request{State: "hi", Questions: triageQuestions}); err != nil {
		t.Fatal(err)
	}
	if rec.calls() != 2 {
		t.Errorf("calls = %d, want 2", rec.calls())
	}
	if elapsed := time.Since(start); elapsed > 300*time.Millisecond {
		t.Errorf("Retry-After: 0 was not honored; the retry waited %v", elapsed)
	}
}

func TestCancelDuringBackoffReportsTheCancellation(t *testing.T) {
	// The server asks for a long wait; the caller gives up during it. The
	// error is the cancellation, not the 503 it interrupted.
	rec := &recorder{respond: func(w http.ResponseWriter, call int) {
		w.Header().Set("Retry-After", "5")
		w.WriteHeader(http.StatusServiceUnavailable)
	}}
	c := newClient(t, rec, Direct())
	ctx, cancel := context.WithCancel(t.Context())
	time.AfterFunc(50*time.Millisecond, cancel)
	start := time.Now()
	_, err := c.decide(ctx, "jev-latest", &request{State: "hi", Questions: triageQuestions})
	if !errors.Is(err, context.Canceled) {
		t.Errorf("error = %v, want context.Canceled", err)
	}
	if errors.Is(err, status.ErrUnavailable) {
		t.Errorf("error = %v still reads as unavailable, which a caller would retry", err)
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Errorf("the wait ran on for %v after the cancellation", elapsed)
	}
	if rec.calls() != 1 {
		t.Errorf("calls = %d, want 1", rec.calls())
	}
}

func TestErrorsMapToStatus(t *testing.T) {
	tests := []struct {
		code int
		body string
		want *status.Sentinel
		msg  string
	}{
		{401, `{"error":{"message":"Missing or invalid API key"}}`, status.ErrUnauthenticated, "Missing or invalid API key"},
		{422, `{"detail":[{"loc":["body","questions"],"msg":"field required"}]}`, status.ErrInvalidArgument, "body.questions: field required"},
		{400, `[{"code":"invalid_union","path":["questions","u","criteria","false"],"message":"Invalid input"},{"path":[],"message":"Unrecognized key"}]`, status.ErrInvalidArgument, "questions.u.criteria.false: Invalid input; Unrecognized key"},
		{400, `{"error":{"message":"HTTP 400: {\"detail\":\"Too many score levels\"}"}}`, status.ErrInvalidArgument, "Too many score levels"},
		{429, `{"error":"rate limited"}`, status.ErrResourceExhausted, "rate limited"},
		{503, `service unavailable`, status.ErrUnavailable, "service unavailable"},
		{500, ``, status.ErrInternal, "HTTP 500"},
	}
	for _, tt := range tests {
		t.Run(http.StatusText(tt.code), func(t *testing.T) {
			rec := &recorder{respond: func(w http.ResponseWriter, call int) {
				// An immediate Retry-After keeps the retried codes from
				// sleeping through the backoff.
				w.Header().Set("Retry-After", "0")
				w.WriteHeader(tt.code)
				_, _ = io.WriteString(w, tt.body)
			}}
			c := newClient(t, rec, Direct())
			_, err := c.decide(t.Context(), "jev-latest", &request{State: "hi", Questions: triageQuestions})
			if !errors.Is(err, tt.want) {
				t.Errorf("error = %v, want %v", err, tt.want)
			}
			if !strings.Contains(err.Error(), tt.msg) {
				t.Errorf("error = %v, want the message %q", err, tt.msg)
			}
			if tt.code == 401 && rec.calls() != 1 {
				t.Errorf("a 401 was retried %d times", rec.calls()-1)
			}
		})
	}
}

func TestRetryDelays(t *testing.T) {
	if got := parseRetryAfter("2"); got != 2*time.Second {
		t.Errorf("Retry-After: 2 = %v", got)
	}
	if got := parseRetryAfter(time.Now().Add(3 * time.Second).UTC().Format(http.TimeFormat)); got <= 0 || got > 3*time.Second {
		t.Errorf("Retry-After date = %v, want about 3s", got)
	}
	if got := parseRetryAfter("soon"); got >= 0 {
		t.Errorf("Retry-After: soon = %v, want negative for no usable header", got)
	}
	if got := parseRetryAfter(""); got >= 0 {
		t.Errorf("no Retry-After = %v, want negative", got)
	}
	if got := parseRetryAfter("0"); got != 0 {
		t.Errorf("Retry-After: 0 = %v, want 0", got)
	}
	for attempt, want := range []time.Duration{500 * time.Millisecond, time.Second, 2 * time.Second, 4 * time.Second, 5 * time.Second} {
		got := backoff(attempt)
		if got > want || got < want*3/4 {
			t.Errorf("backoff(%d) = %v, want within a quarter under %v", attempt, got, want)
		}
	}
}

func TestListModels(t *testing.T) {
	for _, reply := range []string{
		`[{"name":"jev-1.13.0","description":"Current release","release_date":"2026-08-01"}]`,
		`{"models":[{"name":"jev-1.13.0"}]}`,
		`{"data":[{"name":"jev-1.13.0"}]}`,
	} {
		rec := &recorder{reply: reply}
		c := newClient(t, rec, Direct())
		models, err := c.listModels(t.Context())
		if err != nil {
			t.Fatalf("%s: %v", reply, err)
		}
		req, _ := rec.last(t)
		if req.Method != http.MethodGet || req.URL.Path != "/v1/models" {
			t.Errorf("request = %s %s, want GET /v1/models", req.Method, req.URL.Path)
		}
		if len(models) != 1 || models[0].Name != "jev-1.13.0" {
			t.Errorf("%s: models = %+v", reply, models)
		}
	}
	if _, err := newClient(t, &recorder{reply: `{}`}, OpenRouter()).listModels(t.Context()); !errors.Is(err, status.ErrUnimplemented) {
		t.Errorf("a gateway listing = %v, want unimplemented", err)
	}
}
