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
//
// SPDX-License-Identifier: Apache-2.0

package genkit

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/tracing"
)

// fakeManager is a test double for the CLI's reflection V2 manager. It accepts
// one WebSocket connection, records inbound messages, and lets tests drive the
// runtime by sending JSON-RPC requests / reading responses.
type fakeManager struct {
	server *httptest.Server
	url    string

	mu     sync.Mutex
	conn   *websocket.Conn
	connCh chan *websocket.Conn
}

func newFakeManager(t *testing.T) *fakeManager {
	t.Helper()
	m := &fakeManager{connCh: make(chan *websocket.Conn, 1)}

	m.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		c, err := websocket.Accept(w, r, nil)
		if err != nil {
			t.Errorf("accept: %v", err)
			return
		}
		m.mu.Lock()
		m.conn = c
		m.mu.Unlock()
		m.connCh <- c
		// Block until test closes the connection so the handler doesn't exit.
		<-r.Context().Done()
	}))
	m.url = "ws" + strings.TrimPrefix(m.server.URL, "http")
	return m
}

func (m *fakeManager) close() {
	m.mu.Lock()
	if m.conn != nil {
		m.conn.Close(websocket.StatusNormalClosure, "")
	}
	m.mu.Unlock()
	m.server.Close()
}

// waitForConnection blocks until the runtime has connected.
func (m *fakeManager) waitForConnection(t *testing.T) *websocket.Conn {
	t.Helper()
	select {
	case c := <-m.connCh:
		return c
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for runtime to connect")
		return nil
	}
}

// read reads the next JSON-RPC message from the runtime.
func (m *fakeManager) read(t *testing.T, ctx context.Context, conn *websocket.Conn) map[string]any {
	t.Helper()
	var msg map[string]any
	readCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	if err := wsjson.Read(readCtx, conn, &msg); err != nil {
		t.Fatalf("read: %v", err)
	}
	return msg
}

// write sends a JSON-RPC message to the runtime.
func (m *fakeManager) write(t *testing.T, ctx context.Context, conn *websocket.Conn, msg any) {
	t.Helper()
	if err := wsjson.Write(ctx, conn, msg); err != nil {
		t.Fatalf("write: %v", err)
	}
}

// ackRegister reads the register request from the runtime and sends back
// a minimal success response so the runtime's register goroutine completes.
// Returns the register params for assertion.
func (m *fakeManager) ackRegister(t *testing.T, ctx context.Context, conn *websocket.Conn) map[string]any {
	t.Helper()
	msg := m.read(t, ctx, conn)
	if msg["method"] != "register" {
		t.Fatalf("expected register, got method=%v", msg["method"])
	}
	id, ok := msg["id"].(string)
	if !ok || id == "" {
		t.Fatalf("register must be a request with a string id, got %v", msg["id"])
	}
	m.write(t, ctx, conn, map[string]any{
		"jsonrpc": "2.0",
		"result":  map[string]any{},
		"id":      id,
	})
	return msg
}

// startRuntime starts a reflection V2 client connected to the fake manager
// and waits for the WebSocket dial to succeed.
func startRuntime(t *testing.T, g *Genkit, m *fakeManager) (context.Context, func()) {
	t.Helper()
	tracing.WriteTelemetryImmediate(tracing.NewTestOnlyTelemetryClient())

	ctx, cancel := context.WithCancel(context.Background())
	errCh := make(chan error, 1)
	startedCh := make(chan struct{})

	go startReflectionServerV2(ctx, g, reflectionServerV2Options{URL: m.url, Name: "test-app"}, errCh, startedCh)

	select {
	case err := <-errCh:
		cancel()
		t.Fatalf("runtime failed to start: %v", err)
	case <-startedCh:
	case <-time.After(2 * time.Second):
		cancel()
		t.Fatal("timed out waiting for runtime startup")
	}

	return ctx, cancel
}

func TestReflectionServerV2_Register(t *testing.T) {
	m := newFakeManager(t)
	defer m.close()

	g := Init(context.Background())
	_, cancel := startRuntime(t, g, m)
	defer cancel()

	conn := m.waitForConnection(t)
	msg := m.read(t, context.Background(), conn)

	if msg["method"] != "register" {
		t.Fatalf("first message method = %q, want register", msg["method"])
	}
	if _, ok := msg["id"].(string); !ok {
		t.Error("register should be a request with a string id")
	}
	params, ok := msg["params"].(map[string]any)
	if !ok {
		t.Fatalf("params is not object: %v", msg["params"])
	}
	if params["name"] != "test-app" {
		t.Errorf("name = %q, want test-app", params["name"])
	}
	if params["id"] == "" || params["id"] == nil {
		t.Error("runtime id should be set")
	}
	if _, ok := params["pid"].(float64); !ok {
		t.Errorf("pid should be a number, got %T", params["pid"])
	}
	if !strings.HasPrefix(params["genkitVersion"].(string), "go/") {
		t.Errorf("genkitVersion = %q, want prefix go/", params["genkitVersion"])
	}
	if _, ok := params["reflectionApiSpecVersion"].(float64); !ok {
		t.Errorf("reflectionApiSpecVersion should be a number, got %T", params["reflectionApiSpecVersion"])
	}
	envs, ok := params["envs"].([]any)
	if !ok || len(envs) == 0 || envs[0] != "dev" {
		t.Errorf("envs = %v, want [dev]", params["envs"])
	}
}

func TestReflectionServerV2_RegisterHandshakeTelemetry(t *testing.T) {
	m := newFakeManager(t)
	defer m.close()

	g := Init(context.Background())
	_, cancel := startRuntime(t, g, m)
	defer cancel()

	conn := m.waitForConnection(t)
	msg := m.read(t, context.Background(), conn)
	id := msg["id"].(string)

	// Respond with a telemetryServerUrl; runtime should accept without error.
	m.write(t, context.Background(), conn, map[string]any{
		"jsonrpc": "2.0",
		"result":  map[string]any{"telemetryServerUrl": "http://127.0.0.1:9999"},
		"id":      id,
	})
	// Nothing more to assert over the wire; we're just exercising the response
	// path to make sure it doesn't panic or stall.
}

func TestReflectionServerV2_ListActions(t *testing.T) {
	m := newFakeManager(t)
	defer m.close()

	g := Init(context.Background())
	core.DefineAction(g.reg, "test/inc", api.ActionTypeCustom, nil, nil, inc)
	core.DefineAction(g.reg, "test/dec", api.ActionTypeCustom, nil, nil, dec)

	ctx, cancel := startRuntime(t, g, m)
	defer cancel()

	conn := m.waitForConnection(t)
	m.ackRegister(t, ctx, conn)

	m.write(t, ctx, conn, map[string]any{
		"jsonrpc": "2.0",
		"method":  "listActions",
		"id":      "1",
	})

	resp := m.read(t, ctx, conn)
	if resp["id"] != "1" {
		t.Fatalf("id = %v, want 1", resp["id"])
	}
	result, ok := resp["result"].(map[string]any)
	if !ok {
		t.Fatalf("result is not object: %v", resp["result"])
	}
	actions, ok := result["actions"].(map[string]any)
	if !ok {
		t.Fatalf("actions is not object: %v", result["actions"])
	}
	for _, key := range []string{"/custom/test/inc", "/custom/test/dec"} {
		if _, ok := actions[key]; !ok {
			t.Errorf("action %q missing from response", key)
		}
	}
}

func TestReflectionServerV2_ListValues(t *testing.T) {
	m := newFakeManager(t)
	defer m.close()

	g := Init(context.Background())
	g.reg.RegisterValue("defaultModel", "my-model")

	ctx, cancel := startRuntime(t, g, m)
	defer cancel()

	conn := m.waitForConnection(t)
	m.ackRegister(t, ctx, conn)

	m.write(t, ctx, conn, map[string]any{
		"jsonrpc": "2.0",
		"method":  "listValues",
		"params":  map[string]any{"type": "defaultModel"},
		"id":      "2",
	})

	resp := m.read(t, ctx, conn)
	if resp["id"] != "2" {
		t.Fatalf("id = %v, want 2", resp["id"])
	}
	result, ok := resp["result"].(map[string]any)
	if !ok {
		t.Fatalf("result is not object: %v", resp["result"])
	}
	values, ok := result["values"].(map[string]any)
	if !ok {
		t.Fatalf("values is not object: %v", result["values"])
	}
	if values["defaultModel"] != "my-model" {
		t.Errorf("value = %v, want my-model", values["defaultModel"])
	}
}

func TestReflectionServerV2_ListValuesRejectsUnsupportedType(t *testing.T) {
	m := newFakeManager(t)
	defer m.close()

	g := Init(context.Background())
	ctx, cancel := startRuntime(t, g, m)
	defer cancel()

	conn := m.waitForConnection(t)
	m.ackRegister(t, ctx, conn)

	m.write(t, ctx, conn, map[string]any{
		"jsonrpc": "2.0",
		"method":  "listValues",
		"params":  map[string]any{"type": "prompt"},
		"id":      "2a",
	})

	resp := m.read(t, ctx, conn)
	errObj, ok := resp["error"].(map[string]any)
	if !ok {
		t.Fatalf("expected error, got %v", resp)
	}
	if code := errObj["code"].(float64); code != float64(jsonRPCInvalidParams) {
		t.Errorf("code = %v, want %d", code, jsonRPCInvalidParams)
	}
}

func TestReflectionServerV2_RunAction(t *testing.T) {
	m := newFakeManager(t)
	defer m.close()

	g := Init(context.Background())
	core.DefineAction(g.reg, "test/inc", api.ActionTypeCustom, nil, nil, inc)

	ctx, cancel := startRuntime(t, g, m)
	defer cancel()

	conn := m.waitForConnection(t)
	m.ackRegister(t, ctx, conn)

	m.write(t, ctx, conn, map[string]any{
		"jsonrpc": "2.0",
		"method":  "runAction",
		"params": map[string]any{
			"key":   "/custom/test/inc",
			"input": 3,
		},
		"id": "3",
	})

	// Drain any runActionState notifications, then expect the final response.
	var resp map[string]any
	for {
		msg := m.read(t, ctx, conn)
		if msg["method"] == "runActionState" {
			continue
		}
		resp = msg
		break
	}
	if resp["id"] != "3" {
		t.Fatalf("id = %v, want 3", resp["id"])
	}
	if resp["error"] != nil {
		t.Fatalf("unexpected error: %v", resp["error"])
	}
	result, ok := resp["result"].(map[string]any)
	if !ok {
		t.Fatalf("result is not object: %v", resp["result"])
	}
	if got := result["result"]; got != float64(4) {
		t.Errorf("result = %v, want 4", got)
	}
	telemetry, ok := result["telemetry"].(map[string]any)
	if !ok || telemetry["traceId"] == "" {
		t.Errorf("expected non-empty traceId, got %v", result["telemetry"])
	}
}

func TestReflectionServerV2_StreamingRunAction(t *testing.T) {
	m := newFakeManager(t)
	defer m.close()

	g := Init(context.Background())
	streamInc := func(_ context.Context, x int, cb streamingCallback[json.RawMessage]) (int, error) {
		for i := range x {
			msg, _ := json.Marshal(i)
			if err := cb(context.Background(), msg); err != nil {
				return 0, err
			}
		}
		return x, nil
	}
	core.DefineStreamingAction(g.reg, "test/streaming", api.ActionTypeCustom, nil, nil, streamInc)

	ctx, cancel := startRuntime(t, g, m)
	defer cancel()

	conn := m.waitForConnection(t)
	m.ackRegister(t, ctx, conn)

	m.write(t, ctx, conn, map[string]any{
		"jsonrpc": "2.0",
		"method":  "runAction",
		"params": map[string]any{
			"key":    "/custom/test/streaming",
			"input":  3,
			"stream": true,
		},
		"id": "4",
	})

	var chunks []float64
	var final map[string]any
	for {
		msg := m.read(t, ctx, conn)
		switch msg["method"] {
		case "streamChunk":
			params := msg["params"].(map[string]any)
			if params["requestId"] != "4" {
				t.Errorf("streamChunk requestId = %v, want 4", params["requestId"])
			}
			chunks = append(chunks, params["chunk"].(float64))
			continue
		case "runActionState":
			continue
		}
		final = msg
		break
	}
	if len(chunks) != 3 {
		t.Errorf("got %d chunks, want 3", len(chunks))
	}
	for i, c := range chunks {
		if c != float64(i) {
			t.Errorf("chunk[%d] = %v, want %d", i, c, i)
		}
	}
	result := final["result"].(map[string]any)
	if result["result"] != float64(3) {
		t.Errorf("final result = %v, want 3", result["result"])
	}
}

func TestReflectionServerV2_RunActionNotFound(t *testing.T) {
	m := newFakeManager(t)
	defer m.close()

	g := Init(context.Background())
	ctx, cancel := startRuntime(t, g, m)
	defer cancel()

	conn := m.waitForConnection(t)
	m.ackRegister(t, ctx, conn)

	m.write(t, ctx, conn, map[string]any{
		"jsonrpc": "2.0",
		"method":  "runAction",
		"params":  map[string]any{"key": "/custom/does-not-exist", "input": nil},
		"id":      "5",
	})

	resp := m.read(t, ctx, conn)
	errObj, ok := resp["error"].(map[string]any)
	if !ok {
		t.Fatalf("expected error object, got %v", resp)
	}
	if code := errObj["code"].(float64); code != float64(jsonRPCServerError) {
		t.Errorf("code = %v, want %d", code, jsonRPCServerError)
	}
	data, ok := errObj["data"].(map[string]any)
	if !ok {
		t.Fatalf("expected error data, got %v", errObj["data"])
	}
	if data["code"] == nil {
		t.Error("data.code missing")
	}
	if data["message"] == nil {
		t.Error("data.message missing")
	}
}

func TestReflectionServerV2_CancelAction(t *testing.T) {
	m := newFakeManager(t)
	defer m.close()

	g := Init(context.Background())
	started := make(chan struct{})
	core.DefineAction(g.reg, "test/slow", api.ActionTypeCustom, nil, nil,
		func(ctx context.Context, _ any) (any, error) {
			close(started)
			<-ctx.Done()
			return nil, ctx.Err()
		})

	ctx, cancel := startRuntime(t, g, m)
	defer cancel()

	conn := m.waitForConnection(t)
	m.ackRegister(t, ctx, conn)

	m.write(t, ctx, conn, map[string]any{
		"jsonrpc": "2.0",
		"method":  "runAction",
		"params":  map[string]any{"key": "/custom/test/slow", "input": nil},
		"id":      "6",
	})

	<-started
	var traceID string
	for traceID == "" {
		msg := m.read(t, ctx, conn)
		if msg["method"] == "runActionState" {
			state := msg["params"].(map[string]any)["state"].(map[string]any)
			traceID = state["traceId"].(string)
		}
	}

	m.write(t, ctx, conn, map[string]any{
		"jsonrpc": "2.0",
		"method":  "cancelAction",
		"params":  map[string]any{"traceId": traceID},
		"id":      "7",
	})

	var sawCancel, sawRunErr bool
	for !sawCancel || !sawRunErr {
		msg := m.read(t, ctx, conn)
		switch msg["id"] {
		case "7":
			if result, ok := msg["result"].(map[string]any); !ok || result["message"] != "Action cancelled" {
				t.Errorf("cancel response = %v", msg)
			}
			sawCancel = true
		case "6":
			errObj, ok := msg["error"].(map[string]any)
			if !ok {
				t.Fatalf("expected runAction error, got %v", msg)
			}
			if !strings.Contains(errObj["message"].(string), "cancel") {
				t.Errorf("error message = %q, want contains 'cancel'", errObj["message"])
			}
			sawRunErr = true
		}
	}
}

func TestReflectionServerV2_BidiRunAction(t *testing.T) {
	m := newFakeManager(t)
	defer m.close()

	g := Init(context.Background())

	type initConfig struct {
		Prefix string `json:"prefix"`
	}

	core.DefineBidiAction(g.reg, "test/bidi-echo", api.ActionTypeCustom, nil,
		func(ctx context.Context, cfg initConfig, inCh <-chan string, outCh chan<- string) (string, error) {
			var n int
			for chunk := range inCh {
				n++
				outCh <- cfg.Prefix + chunk
			}
			return "processed", nil
		})

	ctx, cancel := startRuntime(t, g, m)
	defer cancel()

	conn := m.waitForConnection(t)
	m.ackRegister(t, ctx, conn)

	// Start the bidi run. `init` carries the per-session configuration;
	// streamInput signals that chunks will be sent via sendInputStreamChunk;
	// stream requests that output chunks be forwarded back (chunks are not
	// forwarded without it, matching the JS runtime).
	m.write(t, ctx, conn, map[string]any{
		"jsonrpc": "2.0",
		"method":  "runAction",
		"params": map[string]any{
			"key":         "/custom/test/bidi-echo",
			"init":        map[string]any{"prefix": "> "},
			"stream":      true,
			"streamInput": true,
		},
		"id": "bidi-1",
	})

	// Send two chunks and end the input stream.
	m.write(t, ctx, conn, map[string]any{
		"jsonrpc": "2.0",
		"method":  "sendInputStreamChunk",
		"params":  map[string]any{"requestId": "bidi-1", "chunk": "hello"},
	})
	m.write(t, ctx, conn, map[string]any{
		"jsonrpc": "2.0",
		"method":  "sendInputStreamChunk",
		"params":  map[string]any{"requestId": "bidi-1", "chunk": "world"},
	})
	m.write(t, ctx, conn, map[string]any{
		"jsonrpc": "2.0",
		"method":  "endInputStream",
		"params":  map[string]any{"requestId": "bidi-1"},
	})

	var chunks []string
	var final map[string]any
	deadline := time.After(5 * time.Second)
loop:
	for {
		select {
		case <-deadline:
			t.Fatalf("timed out; chunks=%v final=%v", chunks, final)
		default:
		}
		msg := m.read(t, ctx, conn)
		switch msg["method"] {
		case "streamChunk":
			params := msg["params"].(map[string]any)
			if params["requestId"] != "bidi-1" {
				t.Errorf("streamChunk requestId = %v, want bidi-1", params["requestId"])
			}
			chunks = append(chunks, params["chunk"].(string))
		case "runActionState":
			// early trace-id notification; ignore
		default:
			final = msg
			break loop
		}
	}

	if got, want := chunks, []string{"> hello", "> world"}; !slices.Equal(got, want) {
		t.Errorf("chunks = %v, want %v", got, want)
	}
	result, ok := final["result"].(map[string]any)
	if !ok {
		t.Fatalf("expected result object, got %v", final)
	}
	var out string
	if err := json.Unmarshal([]byte(toJSON(t, result["result"])), &out); err != nil {
		t.Fatalf("unmarshal result: %v", err)
	}
	if out != "processed" {
		t.Errorf("result = %q, want %q", out, "processed")
	}
}

func toJSON(t *testing.T, v any) string {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return string(b)
}

// TestReflectionServerV2_BidiRunActionDropsAfterEnd verifies that chunks
// arriving after endInputStream are not delivered to the action, even when
// they queue up before the underlying connection has been attached.
func TestReflectionServerV2_BidiRunActionDropsAfterEnd(t *testing.T) {
	m := newFakeManager(t)
	defer m.close()

	g := Init(context.Background())

	// The action records every chunk it sees so the test can assert that
	// chunks after endInputStream were dropped.
	var seenMu sync.Mutex
	var seen []string

	core.DefineBidiAction(g.reg, "test/bidi-record", api.ActionTypeCustom, nil,
		func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- string) (string, error) {
			for chunk := range inCh {
				seenMu.Lock()
				seen = append(seen, chunk)
				seenMu.Unlock()
			}
			return "done", nil
		})

	ctx, cancel := startRuntime(t, g, m)
	defer cancel()

	conn := m.waitForConnection(t)
	m.ackRegister(t, ctx, conn)

	// Pipeline runAction + chunks + end + extra chunk back-to-back. All
	// arrive before the handler goroutine is likely to have started the
	// session worker, so they queue inside the bidi session. The "extra"
	// chunk is enqueued after the close marker and must be dropped by the
	// worker.
	m.write(t, ctx, conn, map[string]any{
		"jsonrpc": "2.0",
		"method":  "runAction",
		"params": map[string]any{
			"key":         "/custom/test/bidi-record",
			"streamInput": true,
		},
		"id": "drop-1",
	})
	m.write(t, ctx, conn, map[string]any{
		"jsonrpc": "2.0",
		"method":  "sendInputStreamChunk",
		"params":  map[string]any{"requestId": "drop-1", "chunk": "a"},
	})
	m.write(t, ctx, conn, map[string]any{
		"jsonrpc": "2.0",
		"method":  "sendInputStreamChunk",
		"params":  map[string]any{"requestId": "drop-1", "chunk": "b"},
	})
	m.write(t, ctx, conn, map[string]any{
		"jsonrpc": "2.0",
		"method":  "endInputStream",
		"params":  map[string]any{"requestId": "drop-1"},
	})
	m.write(t, ctx, conn, map[string]any{
		"jsonrpc": "2.0",
		"method":  "sendInputStreamChunk",
		"params":  map[string]any{"requestId": "drop-1", "chunk": "after-end"},
	})

	// Drain notifications until the final response.
	deadline := time.After(5 * time.Second)
loop:
	for {
		select {
		case <-deadline:
			t.Fatalf("timed out waiting for response")
		default:
		}
		msg := m.read(t, ctx, conn)
		if _, hasResult := msg["result"]; hasResult {
			break loop
		}
		if _, hasErr := msg["error"]; hasErr {
			t.Fatalf("unexpected error response: %v", msg)
		}
	}

	seenMu.Lock()
	defer seenMu.Unlock()
	if want := []string{"a", "b"}; !slices.Equal(seen, want) {
		t.Errorf("action received %v, want %v (chunk after endInputStream should be dropped)", seen, want)
	}
}

// TestReflectionServerV2_BidiRunActionErrors verifies that an error returned
// from a bidi action surfaces as a JSON-RPC error response.
func TestReflectionServerV2_BidiRunActionErrors(t *testing.T) {
	m := newFakeManager(t)
	defer m.close()

	g := Init(context.Background())

	core.DefineBidiAction(g.reg, "test/bidi-fail", api.ActionTypeCustom, nil,
		func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- string) (string, error) {
			for range inCh {
			}
			return "", core.NewError(core.INVALID_ARGUMENT, "boom")
		})

	ctx, cancel := startRuntime(t, g, m)
	defer cancel()

	conn := m.waitForConnection(t)
	m.ackRegister(t, ctx, conn)

	m.write(t, ctx, conn, map[string]any{
		"jsonrpc": "2.0",
		"method":  "runAction",
		"params": map[string]any{
			"key":         "/custom/test/bidi-fail",
			"streamInput": true,
		},
		"id": "err-1",
	})
	m.write(t, ctx, conn, map[string]any{
		"jsonrpc": "2.0",
		"method":  "endInputStream",
		"params":  map[string]any{"requestId": "err-1"},
	})

	deadline := time.After(5 * time.Second)
	for {
		select {
		case <-deadline:
			t.Fatalf("timed out waiting for error response")
		default:
		}
		msg := m.read(t, ctx, conn)
		if _, hasResult := msg["result"]; hasResult {
			t.Fatalf("expected error, got result: %v", msg)
		}
		errObj, ok := msg["error"].(map[string]any)
		if !ok {
			continue // ignore notifications (e.g., runActionState)
		}
		if !strings.Contains(errObj["message"].(string), "boom") {
			t.Errorf("error message = %q, want substring %q", errObj["message"], "boom")
		}
		return
	}
}

// TestReflectionServerV2_BidiInvalidChunkFailsRun verifies that a chunk
// failing input-schema validation fails the whole run with the validation
// error (matching the JS runtime) rather than being silently dropped.
func TestReflectionServerV2_BidiInvalidChunkFailsRun(t *testing.T) {
	m := newFakeManager(t)
	defer m.close()

	g := Init(context.Background())

	core.DefineBidiAction(g.reg, "test/bidi-strict", api.ActionTypeCustom, nil,
		func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- string) (string, error) {
			for {
				select {
				case _, ok := <-inCh:
					if !ok {
						return "done", nil
					}
				case <-ctx.Done():
					return "", ctx.Err()
				}
			}
		})

	ctx, cancel := startRuntime(t, g, m)
	defer cancel()

	conn := m.waitForConnection(t)
	m.ackRegister(t, ctx, conn)

	m.write(t, ctx, conn, map[string]any{
		"jsonrpc": "2.0",
		"method":  "runAction",
		"params": map[string]any{
			"key":         "/custom/test/bidi-strict",
			"streamInput": true,
		},
		"id": "invalid-chunk-1",
	})
	// A number where the action expects a string: fails validation and must
	// fail the run.
	m.write(t, ctx, conn, map[string]any{
		"jsonrpc": "2.0",
		"method":  "sendInputStreamChunk",
		"params":  map[string]any{"requestId": "invalid-chunk-1", "chunk": 123},
	})

	deadline := time.After(5 * time.Second)
	for {
		select {
		case <-deadline:
			t.Fatal("timed out waiting for error response")
		default:
		}
		msg := m.read(t, ctx, conn)
		if _, hasResult := msg["result"]; hasResult {
			t.Fatalf("expected error, got result: %v", msg)
		}
		errObj, ok := msg["error"].(map[string]any)
		if !ok {
			continue // ignore notifications (e.g., runActionState)
		}
		if !strings.Contains(errObj["message"].(string), "invalid stream chunk") {
			t.Errorf("error message = %q, want substring %q", errObj["message"], "invalid stream chunk")
		}
		return
	}
}

// TestReflectionServerV2_BidiSessionOwnership verifies the ownership rules
// that protect bidi sessions against request-id reuse: a reused id stops the
// orphaned session, a stale handler's unregister cannot tear down a newer
// session under the same id, and the owner's unregister removes and stops its
// own session.
func TestReflectionServerV2_BidiSessionOwnership(t *testing.T) {
	s := &reflectionServerV2{bidiSessions: map[string]*bidiSession{}}

	s1 := newBidiSession()
	s.registerBidiSession("1", s1)

	// A reused id orphans and stops the previous session.
	s2 := newBidiSession()
	s.registerBidiSession("1", s2)
	if _, ok := s1.next(); ok {
		t.Error("orphaned session worker not stopped on id reuse")
	}

	// A stale handler unregistering with its old session must not touch the
	// newer one.
	s.unregisterBidiSession("1", s1)
	if got := s.lookupBidiSession("1"); got != s2 {
		t.Errorf("stale unregister removed the new session; lookup = %v, want s2", got)
	}

	// The owner's unregister removes and stops its session.
	s.unregisterBidiSession("1", s2)
	if got := s.lookupBidiSession("1"); got != nil {
		t.Error("session still registered after owner unregister")
	}
	if _, ok := s2.next(); ok {
		t.Error("session worker not stopped after owner unregister")
	}

	// A nil session is a no-op.
	s.unregisterBidiSession("1", nil)
}

func TestReflectionServerV2_MethodNotFound(t *testing.T) {
	m := newFakeManager(t)
	defer m.close()

	g := Init(context.Background())
	ctx, cancel := startRuntime(t, g, m)
	defer cancel()

	conn := m.waitForConnection(t)
	m.ackRegister(t, ctx, conn)

	m.write(t, ctx, conn, map[string]any{
		"jsonrpc": "2.0",
		"method":  "unknownMethod",
		"id":      "8",
	})

	resp := m.read(t, ctx, conn)
	errObj, ok := resp["error"].(map[string]any)
	if !ok {
		t.Fatalf("expected error, got %v", resp)
	}
	if code := errObj["code"].(float64); code != float64(jsonRPCMethodNotFound) {
		t.Errorf("code = %v, want %d", code, jsonRPCMethodNotFound)
	}
}

// TestReflectionServerV2_BidiRunActionNoStream verifies that output chunks
// are not forwarded when the request does not ask for output streaming
// (matching the JS runtime), while the final result still arrives.
func TestReflectionServerV2_BidiRunActionNoStream(t *testing.T) {
	m := newFakeManager(t)
	defer m.close()

	g := Init(context.Background())

	core.DefineBidiAction(g.reg, "test/bidi-quiet", api.ActionTypeCustom, nil,
		func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- string) (string, error) {
			var last string
			for chunk := range inCh {
				last = chunk
				outCh <- chunk
			}
			return "got " + last, nil
		})

	ctx, cancel := startRuntime(t, g, m)
	defer cancel()

	conn := m.waitForConnection(t)
	m.ackRegister(t, ctx, conn)

	m.write(t, ctx, conn, map[string]any{
		"jsonrpc": "2.0",
		"method":  "runAction",
		"params": map[string]any{
			"key":         "/custom/test/bidi-quiet",
			"streamInput": true,
		},
		"id": "quiet-1",
	})
	m.write(t, ctx, conn, map[string]any{
		"jsonrpc": "2.0",
		"method":  "sendInputStreamChunk",
		"params":  map[string]any{"requestId": "quiet-1", "chunk": "hello"},
	})
	m.write(t, ctx, conn, map[string]any{
		"jsonrpc": "2.0",
		"method":  "endInputStream",
		"params":  map[string]any{"requestId": "quiet-1"},
	})

	var final map[string]any
	deadline := time.After(5 * time.Second)
loop:
	for {
		select {
		case <-deadline:
			t.Fatal("timed out waiting for final response")
		default:
		}
		msg := m.read(t, ctx, conn)
		switch msg["method"] {
		case "streamChunk":
			t.Errorf("unexpected streamChunk without stream=true: %v", msg)
		case "runActionState":
			// early trace-id notification; ignore
		default:
			final = msg
			break loop
		}
	}

	result, ok := final["result"].(map[string]any)
	if !ok {
		t.Fatalf("expected result object, got %v", final)
	}
	var out string
	if err := json.Unmarshal([]byte(toJSON(t, result["result"])), &out); err != nil {
		t.Fatalf("unmarshal result: %v", err)
	}
	if out != "got hello" {
		t.Errorf("result = %q, want %q", out, "got hello")
	}
}

// TestReflectionServerV2_BidiInputClosedOnDisconnect verifies that when the
// manager connection drops mid-stream, in-flight bidi actions see their input
// stream end instead of hanging forever awaiting chunks.
func TestReflectionServerV2_BidiInputClosedOnDisconnect(t *testing.T) {
	m := newFakeManager(t)
	defer m.close()

	g := Init(context.Background())

	inputEnded := make(chan struct{})
	core.DefineBidiAction(g.reg, "test/bidi-hang", api.ActionTypeCustom, nil,
		func(ctx context.Context, _ struct{}, inCh <-chan string, outCh chan<- string) (string, error) {
			for range inCh {
			}
			close(inputEnded)
			return "done", nil
		})

	ctx, cancel := startRuntime(t, g, m)
	defer cancel()

	conn := m.waitForConnection(t)
	m.ackRegister(t, ctx, conn)

	m.write(t, ctx, conn, map[string]any{
		"jsonrpc": "2.0",
		"method":  "runAction",
		"params": map[string]any{
			"key":         "/custom/test/bidi-hang",
			"streamInput": true,
		},
		"id": "hang-1",
	})
	m.write(t, ctx, conn, map[string]any{
		"jsonrpc": "2.0",
		"method":  "sendInputStreamChunk",
		"params":  map[string]any{"requestId": "hang-1", "chunk": "hello"},
	})

	// Drop the connection without sending endInputStream. The runtime must
	// end the action's input stream so it can finish.
	m.close()

	select {
	case <-inputEnded:
	case <-time.After(5 * time.Second):
		t.Fatal("bidi action input stream was not closed on disconnect")
	}
}
