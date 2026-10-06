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

// testQuestions is one question of each kind, as they go on the wire.
var testQuestions = map[string]Question{
	"department": {
		Type:         KindChoice,
		Instructions: "Which team should handle this?",
		Criteria:     Options{{"billing", "Payments, invoicing, refunds"}, {"technical", "Bugs, outages, integrations"}},
	},
	"is_urgent": {Type: KindNoul, Instructions: "Does the ticket explicitly communicate time pressure?"},
	"frustration": {
		Type:         KindScore,
		Instructions: "How frustrated is the customer?",
		Criteria:     []any{"Calm", "Concerned but civil", "Very angry"},
		Labels:       []string{"Calm", "Concerned but civil", "Very angry"},
	},
}

// testAnswers answers testQuestions, as an endpoint returns them.
const testAnswers = `{
	"department":  {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.84, "technical": 0.16}, "confidence": 0.6},
	"is_urgent":   {"type": "noul", "noul": 0.93},
	"frustration": {"type": "score", "score": 1.3, "legend": {"0": "Calm", "1": "Concerned but civil", "2": "Very angry"}, "probabilities": {"0": 0, "1": 0.7, "2": 0.3}, "confidence": 0.54}
}`

const cannedReply = `{"model":"jev-1.13.0","answers":` + testAnswers + `,"usage":{"input_tokens":312,"output_tokens":48}}`

// testEndpoint is an endpoint that takes the native body as it is.
func testEndpoint() *Endpoint {
	return &Endpoint{Name: "test", Path: "/v1/systemone", ModelsPath: "/v1/models"}
}

func newClient(t *testing.T, rec *recorder, ep *Endpoint) *Client {
	t.Helper()
	srv := httptest.NewServer(rec)
	t.Cleanup(srv.Close)
	return &Client{HTTP: srv.Client(), BaseURL: srv.URL, APIKey: "test-key", Endpoint: ep}
}

func TestDecide(t *testing.T) {
	rec := &recorder{reply: cannedReply}
	c := newClient(t, rec, testEndpoint())
	c.Headers = http.Header{"X-Trace": {"abc"}}

	resp, err := c.Decide(t.Context(), "jev-1.13.0", &Request{State: "hi", Questions: testQuestions})
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

func TestRoute(t *testing.T) {
	rec := &recorder{reply: cannedReply}
	ep := testEndpoint()
	ep.Route = func(model string, body map[string]any) (string, any, error) {
		delete(body, "model")
		return model, map[string]any{"input": body}, nil
	}
	c := newClient(t, rec, ep)
	if _, err := c.Decide(t.Context(), "clef", &Request{State: "hi", Questions: testQuestions}); err != nil {
		t.Fatal(err)
	}
	req, body := rec.last(t)
	if req.URL.Path != "/v1/systemone/clef" {
		t.Errorf("path = %s, want the route's suffix joined to the endpoint's path", req.URL.Path)
	}
	if input, _ := body["input"].(map[string]any); input["state"] != "hi" || body["state"] != nil {
		t.Errorf("body = %v, want the route's body", body)
	}
}

func TestUnwrapErrorNamesTheEndpoint(t *testing.T) {
	ep := testEndpoint()
	ep.Unwrap = func([]byte) ([]byte, error) { return nil, errors.New("Authentication error") }
	c := newClient(t, &recorder{reply: cannedReply}, ep)
	_, err := c.Decide(t.Context(), "clef", &Request{State: "hi", Questions: testQuestions})
	if !errors.Is(err, status.ErrUnknown) || !strings.Contains(err.Error(), "test: Authentication error") {
		t.Errorf("error = %v, want UNKNOWN naming the endpoint", err)
	}
}

func TestNoKeyNoAuthorization(t *testing.T) {
	rec := &recorder{reply: cannedReply}
	c := newClient(t, rec, testEndpoint())
	c.APIKey = ""
	if _, err := c.Decide(t.Context(), "clef", &Request{State: "hi", Questions: testQuestions}); err != nil {
		t.Fatal(err)
	}
	if req, _ := rec.last(t); req.Header.Values("Authorization") != nil {
		t.Errorf("Authorization = %q, want no header without a key", req.Header.Values("Authorization"))
	}
}

func TestExtraMergesTopLevelFields(t *testing.T) {
	rec := &recorder{reply: cannedReply}
	c := newClient(t, rec, testEndpoint())
	_, err := c.Decide(t.Context(), "jev-latest", &Request{
		State:     "hi",
		Questions: testQuestions,
		Extra:     map[string]any{"session_id": "s-1", "model": "typesafe/jev-9"},
	})
	if err != nil {
		t.Fatal(err)
	}
	_, body := rec.last(t)
	if body["session_id"] != "s-1" {
		t.Errorf("session_id = %v, want s-1", body["session_id"])
	}
	if body["model"] != "jev-latest" {
		t.Errorf("model = %v, want the model called: an extra cannot replace a field the request builds", body["model"])
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
	c := newClient(t, rec, testEndpoint())
	start := time.Now()
	if _, err := c.Decide(t.Context(), "jev-latest", &Request{State: "hi", Questions: testQuestions}); err != nil {
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
	c := newClient(t, rec, testEndpoint())
	ctx, cancel := context.WithCancel(t.Context())
	time.AfterFunc(50*time.Millisecond, cancel)
	start := time.Now()
	_, err := c.Decide(ctx, "jev-latest", &Request{State: "hi", Questions: testQuestions})
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
		{403, `{"result":null,"success":false,"errors":[{"code":10000,"message":"Authentication error"}],"messages":[]}`, status.ErrPermissionDenied, "HTTP 403: Authentication error"},
		{503, `service unavailable`, status.ErrUnavailable, "service unavailable"},
		{500, ``, status.ErrInternal, "HTTP 500: Internal Server Error"},
		{501, `not implemented`, status.ErrUnimplemented, "not implemented"},
		{504, `gateway timeout`, status.ErrDeadlineExceeded, "gateway timeout"},
		{404, ``, status.ErrNotFound, "HTTP 404: Not Found"},
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
			c := newClient(t, rec, testEndpoint())
			_, err := c.Decide(t.Context(), "jev-latest", &Request{State: "hi", Questions: testQuestions})
			if !errors.Is(err, tt.want) {
				t.Errorf("error = %v, want %v", err, tt.want)
			}
			if !strings.Contains(err.Error(), tt.msg) {
				t.Errorf("error = %v, want the message %q", err, tt.msg)
			}
			if (tt.code == 401 || tt.code == 501) && rec.calls() != 1 {
				t.Errorf("a %d was retried %d times", tt.code, rec.calls()-1)
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
		// OpenRouter names a model by its id; its name is for display.
		`{"data":[{"id":"jev-1.13.0","name":"TypeSafe: Jev 1.13"}]}`,
	} {
		rec := &recorder{reply: reply}
		c := newClient(t, rec, testEndpoint())
		models, err := c.ListModels(t.Context())
		if err != nil {
			t.Fatalf("%s: %v", reply, err)
		}
		req, _ := rec.last(t)
		if req.Method != http.MethodGet || req.URL.Path != "/v1/models" {
			t.Errorf("request = %s %s, want GET /v1/models", req.Method, req.URL.Path)
		}
		if len(models) != 1 || models[0].Model() != "jev-1.13.0" {
			t.Errorf("%s: models = %+v", reply, models)
		}
	}
	if _, err := newClient(t, &recorder{reply: `{}`}, &Endpoint{Name: "gateway"}).ListModels(t.Context()); !errors.Is(err, status.ErrUnimplemented) {
		t.Errorf("a listing with no path = %v, want unimplemented", err)
	}
}
