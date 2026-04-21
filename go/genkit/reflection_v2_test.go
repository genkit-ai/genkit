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
