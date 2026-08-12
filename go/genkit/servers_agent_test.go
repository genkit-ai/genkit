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

// These tests exercise genkit.Handler serving experimental agents. Agents are
// defined via genkit/exp, which imports genkit, so the tests live in the
// external genkit_test package to avoid an import cycle.
package genkit_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/ai/exp/localstore"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/genkit/exp"
)

// agentHTTPResult mirrors the AgentOutput fields the agent handler tests
// assert on, decoded from the handler's {"result": ...} envelope.
type agentHTTPResult struct {
	FinishReason string          `json:"finishReason"`
	SessionID    string          `json:"sessionId"`
	SnapshotID   string          `json:"snapshotId"`
	Message      *ai.Message     `json:"message"`
	State        json.RawMessage `json:"state"`
	Error        *struct {
		Status  string         `json:"status"`
		Message string         `json:"message"`
		Details map[string]any `json:"details"`
	} `json:"error"`
}

// TestHandlerAgent verifies that agents, being bidi actions, serve
// one-turn-at-a-time over the standard action handler: data carries
// the turn's AgentInput, init carries the session source (state for
// client-managed agents, sessionId/snapshotId for server-managed ones), and
// the conversation resumes across requests. It also pins the error contract:
// turn-tier failures resolve as a 200 with a failed AgentOutput (so the
// caller keeps the last-good state), while init-tier failures (a rejected
// session source) are hard HTTP errors.
func TestHandlerAgent(t *testing.T) {
	g := genkit.Init(context.Background(), genkit.WithExperimental())
	var modelCalls atomic.Int64

	// Replies "echo <n>" where n is the number of messages the model saw,
	// so resumed history is observable; fails when asked to.
	genkit.DefineModel(g, "test/echo", &ai.ModelOptions{Supports: &ai.ModelSupports{Multiturn: true}},
		func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			modelCalls.Add(1)
			if len(req.Messages) == 0 {
				return nil, core.NewError(core.INTERNAL, "model saw empty messages")
			}
			last := req.Messages[len(req.Messages)-1]
			if last.Role == ai.RoleUser && strings.Contains(last.Text(), "fail") {
				return nil, core.NewError(core.RESOURCE_EXHAUSTED, "model on fire")
			}
			if cb != nil {
				cb(ctx, &ai.ModelResponseChunk{Content: []*ai.Part{ai.NewTextPart("chunk")}})
			}
			return &ai.ModelResponse{
				Message:      ai.NewModelTextMessage(fmt.Sprintf("echo %d", len(req.Messages))),
				FinishReason: ai.FinishReasonStop,
			}, nil
		})

	exp.DefineAgent[any](g, "agentClient", aix.InlinePrompt{ai.WithModelName("test/echo")})

	store, err := localstore.NewFileSessionStore[any](t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	exp.DefineAgent(g, "agentServer", aix.InlinePrompt{ai.WithModelName("test/echo")},
		aix.WithSessionStore(store),
	)

	// Agents register under their own action type, so they surface through
	// ListAgents (and not ListFlows) and Handler serves them like any other
	// action.
	for _, a := range genkit.ListFlows(g) {
		if a.Name() == "agentClient" || a.Name() == "agentServer" {
			t.Fatalf("agent %q unexpectedly listed as a flow", a.Name())
		}
	}
	handlerFor := func(t *testing.T, name string) http.HandlerFunc {
		t.Helper()
		for _, a := range exp.ListAgents(g) {
			if a.Name() == name {
				return genkit.Handler(a)
			}
		}
		t.Fatalf("agent %q not in ListAgents", name)
		return nil
	}

	post := func(t *testing.T, name, body string, stream bool) (int, string) {
		t.Helper()
		req := httptest.NewRequest("POST", "/", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		if stream {
			req.Header.Set("Accept", "text/event-stream")
		}
		w := httptest.NewRecorder()
		handlerFor(t, name)(w, req)
		respBody, _ := io.ReadAll(w.Result().Body)
		return w.Result().StatusCode, string(respBody)
	}

	parseResult := func(t *testing.T, body string) agentHTTPResult {
		t.Helper()
		var envelope struct {
			Result agentHTTPResult `json:"result"`
		}
		if err := json.Unmarshal([]byte(body), &envelope); err != nil {
			t.Fatalf("unmarshal %q: %v", body, err)
		}
		return envelope.Result
	}

	turn := func(text string) string {
		return `{"data":{"message":{"role":"user","content":[{"text":"` + text + `"}]}}}`
	}
	turnWithInit := func(text, init string) string {
		return `{"data":{"message":{"role":"user","content":[{"text":"` + text + `"}]}},"init":` + init + `}`
	}

	t.Run("client-managed conversation across requests", func(t *testing.T) {
		// Fresh turn: no init at all.
		code, body := post(t, "agentClient", turn("hello"), false)
		if code != http.StatusOK {
			t.Fatalf("status = %d, body = %s", code, body)
		}
		res := parseResult(t, body)
		if res.FinishReason != "stop" {
			t.Errorf("finishReason = %q, want %q", res.FinishReason, "stop")
		}
		if res.SessionID == "" {
			t.Error("missing sessionId")
		}
		if len(res.State) == 0 {
			t.Fatal("client-managed output must carry state")
		}
		if got := res.Message.Text(); got != "echo 1" {
			t.Errorf("message = %q, want %q", got, "echo 1")
		}

		// Resume: round-trip the returned state through init.
		code, body = post(t, "agentClient", turnWithInit("again", `{"state":`+string(res.State)+`}`), false)
		if code != http.StatusOK {
			t.Fatalf("resume status = %d, body = %s", code, body)
		}
		resumed := parseResult(t, body)
		// The model saw the prior user/model exchange plus the new message.
		if got := resumed.Message.Text(); got != "echo 3" {
			t.Errorf("resumed message = %q, want %q", got, "echo 3")
		}
		if resumed.SessionID != res.SessionID {
			t.Errorf("sessionId changed across resume: %q vs %q", resumed.SessionID, res.SessionID)
		}
	})

	t.Run("server-managed conversation across requests", func(t *testing.T) {
		code, body := post(t, "agentServer", turn("hello"), false)
		if code != http.StatusOK {
			t.Fatalf("status = %d, body = %s", code, body)
		}
		res := parseResult(t, body)
		if len(res.State) != 0 {
			t.Errorf("server-managed output must not inline state, got %s", res.State)
		}
		if res.SessionID == "" || res.SnapshotID == "" {
			t.Fatalf("missing session/snapshot ID: %+v", res)
		}

		code, body = post(t, "agentServer", turnWithInit("again", `{"sessionId":"`+res.SessionID+`"}`), false)
		if code != http.StatusOK {
			t.Fatalf("resume status = %d, body = %s", code, body)
		}
		resumed := parseResult(t, body)
		if got := resumed.Message.Text(); got != "echo 3" {
			t.Errorf("resumed message = %q, want %q", got, "echo 3")
		}
	})

	t.Run("turn failure resolves as failed output with 200", func(t *testing.T) {
		code, body := post(t, "agentClient", turn("fail"), false)
		if code != http.StatusOK {
			t.Fatalf("status = %d, want 200 (failure rides the output); body = %s", code, body)
		}
		res := parseResult(t, body)
		if res.FinishReason != "failed" {
			t.Errorf("finishReason = %q, want %q", res.FinishReason, "failed")
		}
		if res.Error == nil {
			t.Fatalf("missing error in failed output: %s", body)
		}
		if res.Error.Status != "RESOURCE_EXHAUSTED" {
			t.Errorf("error.status = %q, want RESOURCE_EXHAUSTED", res.Error.Status)
		}
		if res.Error.Message == "" {
			t.Error("missing error.message")
		}
		if _, ok := res.Error.Details["stack"]; ok {
			t.Error("error.details must not leak the in-process stack trace")
		}
		// The failed output still hands back the last-good state.
		if len(res.State) == 0 {
			t.Error("failed output must carry the last-good state")
		}
	})

	t.Run("missing turn message fails before model invocation", func(t *testing.T) {
		before := modelCalls.Load()
		code, body := post(t, "agentClient", `{"data":{}}`, false)
		if code != http.StatusOK {
			t.Fatalf("status = %d, want 200 (failure rides the output); body = %s", code, body)
		}
		res := parseResult(t, body)
		if res.FinishReason != "failed" {
			t.Errorf("finishReason = %q, want %q", res.FinishReason, "failed")
		}
		if res.Error == nil {
			t.Fatalf("missing error in failed output: %s", body)
		}
		if res.Error.Status != "INVALID_ARGUMENT" {
			t.Errorf("error.status = %q, want INVALID_ARGUMENT", res.Error.Status)
		}
		if !strings.Contains(res.Error.Message, "message") {
			t.Errorf("error.message = %q, want substring %q", res.Error.Message, "message")
		}
		if got := modelCalls.Load(); got != before {
			t.Fatalf("model calls = %d, want %d", got, before)
		}
	})

	t.Run("init-tier failures are hard HTTP errors", func(t *testing.T) {
		code, body := post(t, "agentServer", turnWithInit("hi", `{"snapshotId":"nope"}`), false)
		if code != http.StatusNotFound {
			t.Errorf("unknown snapshot: status = %d, want %d; body = %s", code, http.StatusNotFound, body)
		}

		code, body = post(t, "agentServer", turnWithInit("hi", `{"state":{"messages":[]}}`), false)
		if code != http.StatusBadRequest {
			t.Errorf("state on server-managed: status = %d, want %d; body = %s", code, http.StatusBadRequest, body)
		}

		code, body = post(t, "agentClient", turnWithInit("hi", `{"sessionId":"abc"}`), false)
		if code != http.StatusBadRequest {
			t.Errorf("sessionId on client-managed: status = %d, want %d; body = %s", code, http.StatusBadRequest, body)
		}

		// data is required for one-shot runs: only a streaming session can
		// start up and defer its first input.
		code, body = post(t, "agentClient", `{"init": {}}`, false)
		if code != http.StatusBadRequest {
			t.Errorf("init without data: status = %d, want %d; body = %s", code, http.StatusBadRequest, body)
		}
		if !strings.Contains(body, "streaming session") {
			t.Errorf("init without data: body should point the caller at streaming sessions; body = %s", body)
		}
	})

	t.Run("streaming turn delivers chunks then result", func(t *testing.T) {
		code, body := post(t, "agentClient", turn("stream"), true)
		if code != http.StatusOK {
			t.Fatalf("status = %d, body = %s", code, body)
		}
		if !strings.Contains(body, "modelChunk") {
			t.Errorf("missing modelChunk event; body = %s", body)
		}
		if !strings.Contains(body, "turnEnd") {
			t.Errorf("missing turnEnd event; body = %s", body)
		}
		if !strings.Contains(body, `"result"`) {
			t.Errorf("missing final result event; body = %s", body)
		}
	})
}

// TestHandlerAgentRef verifies that the typed agent ref is servable
// directly: the ref satisfies api.BidiAction, so Handler routes init
// through the bidi interface (session resume keeps working), and the
// companion actions plucked off the ref (GetSnapshotAction,
// AbortAction) serve the snapshot lifecycle on caller-chosen
// routes. Together they pin the detach → poll → abort story over plain
// HTTP.
func TestHandlerAgentRef(t *testing.T) {
	g := genkit.Init(context.Background(), genkit.WithExperimental())

	genkit.DefineModel(g, "test/echo", &ai.ModelOptions{Supports: &ai.ModelSupports{Multiturn: true}},
		func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			return &ai.ModelResponse{
				Message:      ai.NewModelTextMessage(fmt.Sprintf("echo %d", len(req.Messages))),
				FinishReason: ai.FinishReasonStop,
			}, nil
		})

	store, err := localstore.NewFileSessionStore[any](t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	agent := exp.DefineAgent(g, "agentRef", aix.InlinePrompt{ai.WithModelName("test/echo")},
		aix.WithSessionStore(store),
	)

	// Handlers come straight off the ref; no registry iteration involved.
	runHandler := genkit.Handler(agent)
	getSnapshotHandler := genkit.Handler(agent.GetSnapshotAction())
	abortHandler := genkit.Handler(agent.AbortAction())

	post := func(t *testing.T, h http.HandlerFunc, body string) (int, string) {
		t.Helper()
		req := httptest.NewRequest("POST", "/", strings.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		h(w, req)
		respBody, _ := io.ReadAll(w.Result().Body)
		return w.Result().StatusCode, string(respBody)
	}

	parseResult := func(t *testing.T, body string) agentHTTPResult {
		t.Helper()
		var envelope struct {
			Result agentHTTPResult `json:"result"`
		}
		if err := json.Unmarshal([]byte(body), &envelope); err != nil {
			t.Fatalf("unmarshal %q: %v", body, err)
		}
		return envelope.Result
	}

	// snapshotHTTPResult covers both companion responses: getSnapshot
	// returns the snapshot row, abort echoes {snapshotId, status}.
	type snapshotHTTPResult struct {
		SnapshotID string          `json:"snapshotId"`
		SessionID  string          `json:"sessionId"`
		Status     string          `json:"status"`
		State      json.RawMessage `json:"state"`
	}
	parseSnapshot := func(t *testing.T, body string) snapshotHTTPResult {
		t.Helper()
		var envelope struct {
			Result snapshotHTTPResult `json:"result"`
		}
		if err := json.Unmarshal([]byte(body), &envelope); err != nil {
			t.Fatalf("unmarshal %q: %v", body, err)
		}
		return envelope.Result
	}

	turn := func(text string) string {
		return `{"data":{"message":{"role":"user","content":[{"text":"` + text + `"}]}}}`
	}

	t.Run("turns resume through the ref's bidi interface", func(t *testing.T) {
		code, body := post(t, runHandler, turn("hello"))
		if code != http.StatusOK {
			t.Fatalf("status = %d, body = %s", code, body)
		}
		res := parseResult(t, body)
		if res.SessionID == "" || res.SnapshotID == "" {
			t.Fatalf("missing session/snapshot ID: %+v", res)
		}

		// Resume rides the init field, which the handler only accepts
		// because the ref satisfies api.BidiAction.
		code, body = post(t, runHandler,
			`{"data":{"message":{"role":"user","content":[{"text":"again"}]}},"init":{"sessionId":"`+res.SessionID+`"}}`)
		if code != http.StatusOK {
			t.Fatalf("resume status = %d, body = %s", code, body)
		}
		if got := parseResult(t, body).Message.Text(); got != "echo 3" {
			t.Errorf("resumed message = %q, want %q", got, "echo 3")
		}
	})

	t.Run("getSnapshot serves the persisted snapshot", func(t *testing.T) {
		code, body := post(t, runHandler, turn("snapshot me"))
		if code != http.StatusOK {
			t.Fatalf("turn status = %d, body = %s", code, body)
		}
		res := parseResult(t, body)

		code, body = post(t, getSnapshotHandler, `{"data":{"snapshotId":"`+res.SnapshotID+`"}}`)
		if code != http.StatusOK {
			t.Fatalf("getSnapshot status = %d, body = %s", code, body)
		}
		snap := parseSnapshot(t, body)
		if snap.SnapshotID != res.SnapshotID || snap.SessionID != res.SessionID {
			t.Errorf("snapshot identity = %q/%q, want %q/%q", snap.SnapshotID, snap.SessionID, res.SnapshotID, res.SessionID)
		}
		// The action normalizes the implicit empty status to "completed"
		// so remote clients don't reimplement the default.
		if snap.Status != "completed" {
			t.Errorf("status = %q, want %q", snap.Status, "completed")
		}
		if len(snap.State) == 0 {
			t.Error("snapshot must carry state")
		}
	})

	t.Run("unknown snapshot IDs map to 404", func(t *testing.T) {
		code, body := post(t, getSnapshotHandler, `{"data":{"snapshotId":"nope"}}`)
		if code != http.StatusNotFound {
			t.Errorf("getSnapshot: status = %d, want %d; body = %s", code, http.StatusNotFound, body)
		}
		code, body = post(t, abortHandler, `{"data":{"snapshotId":"nope"}}`)
		if code != http.StatusNotFound {
			t.Errorf("abort: status = %d, want %d; body = %s", code, http.StatusNotFound, body)
		}
	})

	t.Run("abort on a terminal snapshot echoes its status", func(t *testing.T) {
		code, body := post(t, runHandler, turn("terminal"))
		if code != http.StatusOK {
			t.Fatalf("turn status = %d, body = %s", code, body)
		}
		res := parseResult(t, body)

		code, body = post(t, abortHandler, `{"data":{"snapshotId":"`+res.SnapshotID+`"}}`)
		if code != http.StatusOK {
			t.Fatalf("abort status = %d, body = %s", code, body)
		}
		snap := parseSnapshot(t, body)
		if snap.Status != "completed" {
			t.Errorf("status = %q, want %q (abort of a terminal snapshot is a no-op)", snap.Status, "completed")
		}
	})

	t.Run("detached turn finalizes in the background", func(t *testing.T) {
		code, body := post(t, runHandler,
			`{"data":{"detach":true,"message":{"role":"user","content":[{"text":"work in background"}]}}}`)
		if code != http.StatusOK {
			t.Fatalf("detach status = %d, body = %s", code, body)
		}
		res := parseResult(t, body)
		if res.FinishReason != "detached" {
			t.Fatalf("finishReason = %q, want %q; body = %s", res.FinishReason, "detached", body)
		}
		if res.SnapshotID == "" {
			t.Fatal("detached output missing the pending snapshotId")
		}

		// Poll the companion route until the background turn finalizes the
		// pending row, the way a remote client would.
		deadline := time.After(10 * time.Second)
		for {
			code, body := post(t, getSnapshotHandler, `{"data":{"snapshotId":"`+res.SnapshotID+`"}}`)
			if code != http.StatusOK {
				t.Fatalf("getSnapshot status = %d, body = %s", code, body)
			}
			snap := parseSnapshot(t, body)
			if snap.Status != "pending" {
				if snap.Status != "completed" {
					t.Fatalf("final status = %q, want %q; body = %s", snap.Status, "completed", body)
				}
				if len(snap.State) == 0 {
					t.Error("finalized snapshot must carry the cumulative state")
				}
				break
			}
			select {
			case <-deadline:
				t.Fatal("snapshot still pending after 10s")
			case <-time.After(10 * time.Millisecond):
			}
		}
	})
}
