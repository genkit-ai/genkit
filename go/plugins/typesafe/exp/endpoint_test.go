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
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/firebase/genkit/go/plugins/internal/systemone"
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

var cannedReply = `{"model":"jev-1.13.0","answers":` + triageAnswers + `,"usage":{"input_tokens":312,"output_tokens":48}}`

// triageAnswers answers the triage questions, as the API returns them.
const triageAnswers = `{
	"department":  {"type": "choice", "choice": "billing", "probabilities": {"billing": 0.84, "technical": 0.15, "other": 0.01}, "confidence": 0.6},
	"is_urgent":   {"type": "noul", "noul": 0.93},
	"frustration": {"type": "score", "score": 1.3, "legend": {"0": "Calm", "1": "Concerned but civil", "2": "Very angry"}, "probabilities": {"0": 0, "1": 0.7, "2": 0.3}, "confidence": 0.54}
}`

// triageQuestions is what the triage type puts on the wire.
func triageQuestions(t *testing.T) map[string]systemone.Question {
	t.Helper()
	questions, err := systemone.CompileQuestions(base.SchemaAsMap(base.InferJSONSchema(triage{})), "")
	if err != nil {
		t.Fatal(err)
	}
	return questions
}

func newClient(t *testing.T, rec *recorder, ep *Endpoint) *systemone.Client {
	t.Helper()
	srv := httptest.NewServer(rec)
	t.Cleanup(srv.Close)
	return &systemone.Client{HTTP: srv.Client(), BaseURL: srv.URL, APIKey: "test-key", Endpoint: ep.ep}
}

func TestDirectEndpoint(t *testing.T) {
	rec := &recorder{reply: cannedReply}
	c := newClient(t, rec, Direct())
	if _, err := c.Decide(t.Context(), "jev-1.13.0", &systemone.Request{State: "hi", Questions: triageQuestions(t)}); err != nil {
		t.Fatal(err)
	}
	req, body := rec.last(t)
	if req.URL.Path != "/v1/systemone" || req.Method != http.MethodPost {
		t.Errorf("request = %s %s, want POST /v1/systemone", req.Method, req.URL.Path)
	}
	if body["model"] != "jev-1.13.0" || body["state"] != "hi" {
		t.Errorf("body = %v: want the native body with the ID as given", body)
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
		resp, err := c.Decide(t.Context(), id, &systemone.Request{State: "hi", Questions: triageQuestions(t)})
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
	if _, err := c.Decide(t.Context(), "jev-1.13.0", &systemone.Request{State: "hi", Questions: triageQuestions(t)}); err == nil || !errors.Is(err, status.ErrInvalidArgument) || !strings.Contains(err.Error(), "jev-1.13") {
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

	resp, err := c.Decide(t.Context(), "jev-latest", &systemone.Request{State: "hi", Questions: triageQuestions(t)})
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
	if resp, err := c.Decide(t.Context(), "jev-latest", &systemone.Request{State: "hi", Questions: triageQuestions(t)}); err != nil || resp.Model != "jev-1.13.0" {
		t.Errorf("bare response: %+v, %v", resp, err)
	}

	rec.reply = `{"result":null,"success":false,"errors":[{"code":7000,"message":"No route for that URI"}]}`
	if _, err := c.Decide(t.Context(), "jev-latest", &systemone.Request{State: "hi", Questions: triageQuestions(t)}); err == nil || !strings.Contains(err.Error(), "No route") {
		t.Errorf("failed envelope error = %v", err)
	}

	calls := rec.calls()
	if _, err := c.Decide(t.Context(), "jev-1.13.0", &systemone.Request{State: "hi", Questions: triageQuestions(t)}); err == nil || !errors.Is(err, status.ErrInvalidArgument) {
		t.Errorf("pinned version on Cloudflare: error = %v, want invalid argument", err)
	}
	if rec.calls() != calls {
		t.Error("a pinned version was sent to Cloudflare instead of being refused")
	}

	// The account ID comes from the environment when none is given, and is
	// escaped into the path.
	t.Setenv("CLOUDFLARE_ACCOUNT_ID", "acct/2")
	ep, err := Cloudflare("").ep.WithAccount()
	if err != nil {
		t.Fatal(err)
	}
	c = newClient(t, rec, &Endpoint{ep})
	rec.reply = cannedReply
	if _, err := c.Decide(t.Context(), "jev-latest", &systemone.Request{State: "hi", Questions: triageQuestions(t)}); err != nil {
		t.Fatal(err)
	}
	if req, _ := rec.last(t); req.URL.EscapedPath() != "/client/v4/accounts/acct%2F2/ai/run" {
		t.Errorf("path = %s, want the account from the environment, escaped", req.URL.EscapedPath())
	}
}
