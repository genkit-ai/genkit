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
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"

	"go.opentelemetry.io/otel/trace"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal/wire"
)

// The tests here pin the client's side of the wire contract against
// hand-written server responses. The genkit/exp package's tests run the client
// against the real handler.

// remoteTestServer serves routes under /agents/a and returns the agent URL.
func remoteTestServer(t *testing.T, routes map[string]http.HandlerFunc) string {
	t.Helper()
	mux := http.NewServeMux()
	for path, h := range routes {
		mux.HandleFunc("POST /agents/a"+path, h)
	}
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv.URL + "/agents/a"
}

func writeResult(w http.ResponseWriter, v any) {
	raw, _ := json.Marshal(v)
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(wire.ResultResponse{Result: raw})
}

func writeWireError(w http.ResponseWriter, n status.Name, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(n.HTTPCode())
	json.NewEncoder(w).Encode(wire.Error{Status: n, Message: msg})
}

// decodeData decodes a request body's data into v.
func decodeData(t *testing.T, r *http.Request, v any) wire.Request {
	t.Helper()
	var req wire.Request
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		t.Errorf("decode request: %v", err)
	}
	if v != nil {
		if err := json.Unmarshal(req.Data, v); err != nil {
			t.Errorf("decode data: %v", err)
		}
	}
	return req
}

func statusOf(err error) status.Name {
	s, _ := status.Classified(err)
	return s
}

func TestNewRemoteAgent_RejectsBadInput(t *testing.T) {
	for _, tc := range []struct{ name, url string }{
		{"", "https://h/agents/a"},
		{"a", "ftp://h/agents/a"},
		{"a", "/agents/a"},
		{"a", "https://user:secret@h/agents/a"},
		{"a", "https://h/agents/a?x=1"},
		{"a", "https://h/agents/a#frag"},
	} {
		t.Run(tc.name+" "+tc.url, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Errorf("NewRemoteAgent(%q, %q) did not panic", tc.name, tc.url)
				}
			}()
			NewRemoteAgent(tc.name, tc.url)
		})
	}
}

func TestRemoteAgent_Errors(t *testing.T) {
	ctx := context.Background()
	var redirected atomic.Int32
	url := remoteTestServer(t, map[string]http.HandlerFunc{
		"/getSnapshot": func(w http.ResponseWriter, r *http.Request) {
			var req GetSnapshotRequest
			decodeData(t, r, &req)
			switch req.SnapshotID {
			case "missing":
				writeWireError(w, status.NotFound, `snapshot "missing" not found`)
			case "down":
				http.Error(w, "upstream down", http.StatusServiceUnavailable)
			case "moved":
				http.Redirect(w, r, "/agents/a/elsewhere", http.StatusTemporaryRedirect)
			case "zombie":
				writeResult(w, &SessionSnapshot[json.RawMessage]{SnapshotID: "zombie", Status: "zombie"})
			}
		},
		// The redirect target must never be reached: following it would
		// replay the request, and its credentials, to whatever the server
		// named.
		"/elsewhere": func(http.ResponseWriter, *http.Request) { redirected.Add(1) },
	})
	h := NewRemoteAgent("a", url)

	t.Run("a status from the server survives", func(t *testing.T) {
		_, err := h.GetSnapshot(ctx, "missing")
		if statusOf(err) != status.NotFound || !strings.Contains(err.Error(), `"missing" not found`) {
			t.Errorf("error = %v, want NOT_FOUND with the server's message", err)
		}
	})
	t.Run("a route the server does not publish is UNIMPLEMENTED", func(t *testing.T) {
		if _, err := h.Abort(ctx, "s"); statusOf(err) != status.Unimplemented {
			t.Errorf("error = %v, want UNIMPLEMENTED", err)
		}
	})
	t.Run("a plain HTTP error maps by its code", func(t *testing.T) {
		if _, err := h.GetSnapshot(ctx, "down"); statusOf(err) != status.Unavailable {
			t.Errorf("error = %v, want UNAVAILABLE", err)
		}
	})
	t.Run("a redirect is refused", func(t *testing.T) {
		if _, err := h.GetSnapshot(ctx, "moved"); statusOf(err) != status.FailedPrecondition {
			t.Errorf("error = %v, want FAILED_PRECONDITION", err)
		}
		if n := redirected.Load(); n != 0 {
			t.Errorf("the redirect target was reached %d times", n)
		}
	})
	t.Run("an unknown snapshot status is rejected", func(t *testing.T) {
		// Read as settled, it would be cached as a final report.
		if _, err := h.GetSnapshot(ctx, "zombie"); statusOf(err) != status.Internal {
			t.Errorf("error = %v, want INTERNAL", err)
		}
	})
}

func TestRemoteAgent_Run(t *testing.T) {
	ctx := context.Background()
	sse := func(w http.ResponseWriter, events ...any) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, e := range events {
			raw, _ := json.Marshal(e)
			fmt.Fprintf(w, "%s%s\n\n", wire.SSEDataPrefix, raw)
		}
	}
	chunk, _ := json.Marshal(&AgentStreamChunk{ModelChunk: &ai.ModelResponseChunk{Content: []*ai.Part{ai.NewTextPart("he")}}})
	output := func(out *AgentOutput[json.RawMessage]) wire.ResultResponse {
		raw, _ := json.Marshal(out)
		return wire.ResultResponse{Result: raw}
	}

	var (
		mu      sync.Mutex
		lastReq *http.Request
		lastIn  wire.Request
	)
	url := remoteTestServer(t, map[string]http.HandlerFunc{
		"": func(w http.ResponseWriter, r *http.Request) {
			var in AgentInput
			req := decodeData(t, r, &in)
			mu.Lock()
			lastReq, lastIn = r, req
			mu.Unlock()
			switch in.Message.Text() {
			case "stream":
				sse(w, wire.MessageResponse{Message: chunk},
					output(&AgentOutput[json.RawMessage]{FinishReason: AgentFinishReasonStop, Message: ai.NewModelTextMessage("hello")}))
			case "fail":
				sse(w, wire.ErrorResponse{Error: &wire.Error{Status: status.PermissionDenied, Message: "no"}})
			case "detach-without-id":
				sse(w, output(&AgentOutput[json.RawMessage]{FinishReason: AgentFinishReasonDetached}))
			case "cut":
				sse(w, wire.MessageResponse{Message: chunk})
			}
		},
	})
	h := NewRemoteAgent("a", url, WithHeaders(func(ctx context.Context) (http.Header, error) {
		return http.Header{"Authorization": {"Bearer t0k"}}, nil
	}))
	run := func(ctx context.Context, text string, cb func(context.Context, json.RawMessage) error) (*AgentOutput[json.RawMessage], error) {
		return h.transport.Run(ctx, &AgentInput{Message: ai.NewUserTextMessage(text)},
			&AgentInit[json.RawMessage]{SessionID: "sess-1"}, cb)
	}

	t.Run("streams chunks, then the output", func(t *testing.T) {
		spanCtx := trace.ContextWithSpanContext(ctx, trace.NewSpanContext(trace.SpanContextConfig{
			TraceID: trace.TraceID{1}, SpanID: trace.SpanID{2}, TraceFlags: trace.FlagsSampled,
		}))
		var chunks []string
		out, err := run(spanCtx, "stream", func(_ context.Context, raw json.RawMessage) error {
			chunks = append(chunks, string(raw))
			return nil
		})
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
		if out.Message.Text() != "hello" || len(chunks) != 1 || chunks[0] != string(chunk) {
			t.Errorf("output %q with chunks %v, want \"hello\" after one chunk", out.Message.Text(), chunks)
		}
		mu.Lock()
		defer mu.Unlock()
		if got := lastReq.Header.Get("Accept"); got != "text/event-stream" {
			t.Errorf("Accept = %q, want text/event-stream", got)
		}
		if got := lastReq.Header.Get("Authorization"); got != "Bearer t0k" {
			t.Errorf("Authorization = %q, want the WithHeaders value", got)
		}
		if got := lastReq.Header.Get("Traceparent"); !strings.HasPrefix(got, "00-01000000000000000000000000000000-0200000000000000-") {
			t.Errorf("traceparent = %q, want the caller's span", got)
		}
		if !strings.Contains(string(lastIn.Init), `"sessionId":"sess-1"`) {
			t.Errorf("init = %s, want the session source", lastIn.Init)
		}
	})
	t.Run("a stream error keeps its status", func(t *testing.T) {
		if _, err := run(ctx, "fail", nil); statusOf(err) != status.PermissionDenied {
			t.Errorf("error = %v, want PERMISSION_DENIED", err)
		}
	})
	t.Run("a detached output without a snapshot is rejected", func(t *testing.T) {
		if _, err := run(ctx, "detach-without-id", nil); statusOf(err) != status.Internal {
			t.Errorf("error = %v, want INTERNAL", err)
		}
	})
	t.Run("a stream that ends without an output is UNAVAILABLE", func(t *testing.T) {
		if _, err := run(ctx, "cut", nil); statusOf(err) != status.Unavailable {
			t.Errorf("error = %v, want UNAVAILABLE", err)
		}
	})
}

func TestRemoteAgent_WaitForSnapshot(t *testing.T) {
	ctx := context.Background()

	t.Run("asks again while the server answers in flight", func(t *testing.T) {
		// A server may cap how long one wait blocks and answer pending.
		var waits atomic.Int32
		url := remoteTestServer(t, map[string]http.HandlerFunc{
			"/waitForSnapshot": func(w http.ResponseWriter, r *http.Request) {
				st := SnapshotStatusPending
				if waits.Add(1) == 3 {
					st = SnapshotStatusCompleted
				}
				writeResult(w, &SessionSnapshot[json.RawMessage]{SnapshotID: "s", Status: st})
			},
		})
		snap, err := NewRemoteAgent("a", url).WaitForSnapshot(ctx, "s")
		if err != nil || snap.Status != SnapshotStatusCompleted || waits.Load() != 3 {
			t.Errorf("WaitForSnapshot = %+v, %v after %d waits; want completed after 3", snap, err, waits.Load())
		}
	})

	t.Run("polls when the server publishes no wait route", func(t *testing.T) {
		var (
			mu    sync.Mutex
			reads []bool // MetadataOnly of each read
		)
		url := remoteTestServer(t, map[string]http.HandlerFunc{
			"/getSnapshot": func(w http.ResponseWriter, r *http.Request) {
				var req GetSnapshotRequest
				decodeData(t, r, &req)
				mu.Lock()
				reads = append(reads, req.MetadataOnly)
				n := len(reads)
				mu.Unlock()
				st := SnapshotStatusPending
				if n >= 2 {
					st = SnapshotStatusCompleted
				}
				writeResult(w, &SessionSnapshot[json.RawMessage]{SnapshotID: "s", Status: st})
			},
		})
		snap, err := NewRemoteAgent("a", url).WaitForSnapshot(ctx, "s")
		if err != nil || snap.Status != SnapshotStatusCompleted {
			t.Fatalf("WaitForSnapshot = %+v, %v; want completed", snap, err)
		}
		mu.Lock()
		defer mu.Unlock()
		// Metadata-only reads until it settles, then one full read for the
		// state.
		if want := []bool{true, true, false}; fmt.Sprint(reads) != fmt.Sprint(want) {
			t.Errorf("reads (metadataOnly) = %v, want %v", reads, want)
		}
	})
}

func TestAgentHandle_Register(t *testing.T) {
	key := func(atype api.ActionType) string { return api.KeyFromName(atype, "a") }

	t.Run("an in-process handle registers the agent's own actions", func(t *testing.T) {
		agent := DefineCustomAgent(newTestRegistry(t), "a", noopAgentFn,
			WithSessionStore(newTestInMemStore[testState]()))
		r := newTestRegistry(t)
		agent.Handle().Register(r)
		if got := r.LookupAction(key(api.ActionTypeAgent)); got != api.Action(agent.action) {
			t.Errorf("registered agent action = %v, want the agent's own", got)
		}
		if r.LookupAction(key(api.ActionTypeAgentSnapshot)) != agent.getSnapshot {
			t.Error("the agent's getSnapshot companion was not registered")
		}
	})

	t.Run("a remote handle registers what its metadata allows", func(t *testing.T) {
		for _, tc := range []struct {
			name      string
			meta      *AgentMetadata
			companion []api.ActionType
		}{
			{"unknown", nil, []api.ActionType{api.ActionTypeAgentSnapshot, api.ActionTypeAgentWait, api.ActionTypeAgentAbort}},
			{"client-managed", &AgentMetadata{StateManagement: AgentStateManagementClient}, nil},
			{"server-managed", &AgentMetadata{StateManagement: AgentStateManagementServer}, []api.ActionType{api.ActionTypeAgentSnapshot, api.ActionTypeAgentWait}},
		} {
			t.Run(tc.name, func(t *testing.T) {
				var opts []RemoteAgentOption
				if tc.meta != nil {
					opts = append(opts, WithAgentMetadata(tc.meta))
				}
				r := newTestRegistry(t)
				NewRemoteAgent("a", "https://h/agents/a", opts...).Register(r)
				run := r.LookupAction(key(api.ActionTypeAgent))
				if run == nil || run.Desc().Metadata[wire.RemoteAgentMetadataKey] != true {
					t.Fatalf("agent action = %v, want one marked remote", run)
				}
				var got []api.ActionType
				for _, atype := range []api.ActionType{api.ActionTypeAgentSnapshot, api.ActionTypeAgentWait, api.ActionTypeAgentAbort} {
					if r.LookupAction(key(atype)) != nil {
						got = append(got, atype)
					}
				}
				if fmt.Sprint(got) != fmt.Sprint(tc.companion) {
					t.Errorf("companions = %v, want %v", got, tc.companion)
				}
			})
		}
	})
}

// noopAgentFn is an agent that returns without running a turn.
func noopAgentFn(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
	return nil, nil
}
