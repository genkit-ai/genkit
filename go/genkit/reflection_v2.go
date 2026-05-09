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
	"errors"
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/tracing"
	"github.com/firebase/genkit/go/internal"
)

// JSON-RPC 2.0 error codes.
const (
	jsonRPCMethodNotFound = -32601
	jsonRPCInvalidParams  = -32602
	jsonRPCServerError    = -32000
)

// Reconnect backoff bounds. Matches the JS client (500ms base, 5s cap).
const (
	reconnectBaseDelay = 500 * time.Millisecond
	reconnectMaxDelay  = 5 * time.Second
)

// jsonRPCMessage is the union of incoming JSON-RPC 2.0 messages we handle:
// requests/notifications (identified by Method) and responses (identified
// by the presence of Result or Error).
type jsonRPCMessage struct {
	JSONRPC string          `json:"jsonrpc"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *jsonRPCError   `json:"error,omitempty"`
	ID      string          `json:"id,omitempty"`
}

// jsonRPCResponse is an outgoing JSON-RPC 2.0 response.
type jsonRPCResponse struct {
	JSONRPC string        `json:"jsonrpc"`
	Result  any           `json:"result,omitempty"`
	Error   *jsonRPCError `json:"error,omitempty"`
	ID      string        `json:"id"`
}

// jsonRPCRequestOrNotification is an outgoing request (ID set) or
// notification (ID empty) from the runtime to the manager.
type jsonRPCRequestOrNotification struct {
	JSONRPC string `json:"jsonrpc"`
	Method  string `json:"method"`
	Params  any    `json:"params,omitempty"`
	ID      string `json:"id,omitempty"`
}

// jsonRPCError is the error object in a JSON-RPC 2.0 error response.
type jsonRPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Data    any    `json:"data,omitempty"`
}

// reflectionRegisterResponse is the result payload for a register request.
// Not in the generated schema because its only field is optional and the
// JS side reads it structurally.
type reflectionRegisterResponse struct {
	TelemetryServerURL string `json:"telemetryServerUrl,omitempty"`
}

// reflectionRunActionResponse is the success payload for a runAction request.
// Not in the generated schema because only the runtime produces it.
type reflectionRunActionResponse struct {
	Result    json.RawMessage `json:"result"`
	Telemetry telemetry       `json:"telemetry"`
}

// pendingResponse is the channel used to deliver a response to a request we
// originated (e.g. register).
type pendingResponse struct {
	result json.RawMessage
	err    *jsonRPCError
}

// reflectionServerV2 is a WebSocket client that connects to the CLI's
// reflection manager and handles JSON-RPC 2.0 requests/responses.
type reflectionServerV2 struct {
	g             *Genkit
	opts          reflectionServerV2Options
	activeActions *activeActionsMap
	runtimeID     string

	writeMu sync.Mutex
	conn    *websocket.Conn

	// pending tracks responses for requests originated by the runtime (register).
	pendingMu  sync.Mutex
	pending    map[string]chan pendingResponse
	requestSeq atomic.Uint64
}

// reflectionServerV2Options configures the V2 reflection client.
type reflectionServerV2Options struct {
	Name string // App name (optional, defaults to the runtime ID).
	URL  string // WebSocket URL of the CLI manager.
}

// startReflectionServerV2 connects to the CLI's WebSocket server and spawns
// a goroutine that handles incoming reflection requests. Reconnects with
// exponential backoff on connection loss until ctx is cancelled.
func startReflectionServerV2(ctx context.Context, g *Genkit, opts reflectionServerV2Options, errCh chan<- error, serverStartCh chan<- struct{}) *reflectionServerV2 {
	if g == nil {
		errCh <- fmt.Errorf("nil Genkit provided")
		return nil
	}

	runtimeID := os.Getenv("GENKIT_RUNTIME_ID")
	if runtimeID == "" {
		runtimeID = strconv.Itoa(os.Getpid())
	}

	s := &reflectionServerV2{
		g:             g,
		opts:          opts,
		activeActions: newActiveActionsMap(),
		runtimeID:     runtimeID,
		pending:       map[string]chan pendingResponse{},
	}

	// Initial connect so startup errors surface via errCh. Reconnects after
	// this are internal and logged.
	if err := s.connect(ctx); err != nil {
		errCh <- fmt.Errorf("failed to connect to reflection V2 server at %s: %w", opts.URL, err)
		return nil
	}
	close(serverStartCh)

	go s.session(ctx)
	return s
}

// connect opens a new WebSocket connection and stores it on s. Safe to call
// only when no connection is active.
func (s *reflectionServerV2) connect(ctx context.Context) error {
	conn, _, err := websocket.Dial(ctx, s.opts.URL, nil)
	if err != nil {
		return err
	}
	s.conn = conn
	slog.Debug("reflection V2: connected", "url", s.opts.URL)
	return nil
}

// session runs the full connection lifecycle: register, read loop, reconnect.
// The first connection has already been established by startReflectionServerV2.
func (s *reflectionServerV2) session(ctx context.Context) {
	attempt := 0
	for {
		// Register runs concurrently with readLoop so the response can be
		// delivered back to the pending request channel.
		go s.register(ctx)
		s.readLoop(ctx)

		// Clean up any pending responses so callers don't block forever.
		s.drainPending(fmt.Errorf("connection closed"))

		// Close the previous connection (best-effort) before attempting reconnect.
		_ = s.conn.Close(websocket.StatusNormalClosure, "reconnecting")

		if ctx.Err() != nil {
			return
		}

		delay := reconnectBaseDelay << attempt
		if delay > reconnectMaxDelay {
			delay = reconnectMaxDelay
		}
		slog.Debug("reflection V2: scheduling reconnect", "delay", delay, "attempt", attempt+1)

		select {
		case <-ctx.Done():
			return
		case <-time.After(delay):
		}

		if err := s.connect(ctx); err != nil {
			slog.Debug("reflection V2: reconnect failed", "err", err)
			attempt++
			continue
		}
		attempt = 0
	}
}

// register sends a register request and processes the response (which may
// include a telemetry server URL). Errors are logged but do not tear down
// the connection; the manager may send configure later.
func (s *reflectionServerV2) register(ctx context.Context) {
	name := s.opts.Name
	if name == "" {
		name = s.runtimeID
	}
	params := &ReflectionRegisterParams{
		ID:                       s.runtimeID,
		PID:                      os.Getpid(),
		Name:                     name,
		GenkitVersion:            "go/" + internal.Version,
		ReflectionApiSpecVersion: internal.GENKIT_REFLECTION_API_SPEC_VERSION,
		Envs:                     []string{"dev"},
	}

	result, err := s.sendRequest(ctx, "register", params)
	if err != nil {
		slog.Error("reflection V2: register failed", "err", err)
		return
	}
	var resp reflectionRegisterResponse
	if len(result) > 0 {
		if err := json.Unmarshal(result, &resp); err != nil {
			slog.Error("reflection V2: invalid register response", "err", err)
			return
		}
	}
	if resp.TelemetryServerURL != "" {
		configureTelemetry(resp.TelemetryServerURL)
	}
}

// readLoop reads and dispatches JSON-RPC messages until the context is
// cancelled or the connection is closed.
func (s *reflectionServerV2) readLoop(ctx context.Context) {
	for {
		var msg jsonRPCMessage
		if err := wsjson.Read(ctx, s.conn, &msg); err != nil {
			if ctx.Err() == nil && websocket.CloseStatus(err) == -1 {
				slog.Debug("reflection V2: read error", "err", err)
			}
			return
		}
		if msg.JSONRPC != "2.0" {
			continue
		}
		if msg.Method != "" {
			go s.handleRequest(ctx, &msg)
		} else if msg.ID != "" {
			s.deliverResponse(&msg)
		}
	}
}

// handleRequest dispatches a JSON-RPC request to the appropriate handler.
// Each handler is responsible for sending its own response (or none, for
// notifications). Unknown methods with a request ID return "method not found";
// unknown notifications are logged and ignored.
func (s *reflectionServerV2) handleRequest(ctx context.Context, req *jsonRPCMessage) {
	switch req.Method {
	case "listActions":
		s.handleListActions(ctx, req)
	case "listValues":
		s.handleListValues(req)
	case "runAction":
		s.handleRunAction(ctx, req)
	case "cancelAction":
		s.handleCancelAction(req)
	case "configure":
		s.handleConfigure(req)
	case "sendInputStreamChunk", "endInputStream":
		// Bidirectional input streaming is not yet implemented.
		slog.Debug("reflection V2: method not implemented", "method", req.Method)
	default:
		if req.ID != "" {
			s.sendErrorResponse(req.ID, jsonRPCMethodNotFound, "method not found: "+req.Method, nil)
		} else {
			slog.Debug("reflection V2: unknown notification", "method", req.Method)
		}
	}
}

// handleListActions responds with all registered and resolvable actions.
func (s *reflectionServerV2) handleListActions(ctx context.Context, req *jsonRPCMessage) {
	if req.ID == "" {
		return
	}
	ads := listResolvableActions(ctx, s.g)
	actionsMap := make(map[string]any, len(ads))
	for _, d := range ads {
		actionsMap[d.Key] = d
	}
	s.sendResponse(req.ID, struct {
		Actions map[string]any `json:"actions"`
	}{Actions: actionsMap})
}

// handleListValues responds with registered values. The Go registry does not
// currently segment values by type, so "type" is accepted but ignored; we
// still honor the JS restriction to "defaultModel" / "middleware" so the
// error shape matches.
func (s *reflectionServerV2) handleListValues(req *jsonRPCMessage) {
	if req.ID == "" {
		return
	}
	var params ReflectionListValuesParams
	if err := json.Unmarshal(req.Params, &params); err != nil {
		s.sendErrorResponse(req.ID, jsonRPCInvalidParams, "invalid params: "+err.Error(), nil)
		return
	}
	if params.Type != "defaultModel" && params.Type != "middleware" {
		s.sendErrorResponse(req.ID, jsonRPCInvalidParams,
			fmt.Sprintf("'type' %s is not supported. Only 'defaultModel' and 'middleware' are supported", params.Type), nil)
		return
	}
	s.sendResponse(req.ID, &ReflectionListValuesResponse{Values: s.g.reg.ListValues()})
}

// handleRunAction executes an action and sends the result (with optional streaming).
func (s *reflectionServerV2) handleRunAction(ctx context.Context, req *jsonRPCMessage) {
	if req.ID == "" {
		return
	}
	var params ReflectionRunActionParams
	if err := json.Unmarshal(req.Params, &params); err != nil {
		s.sendErrorResponse(req.ID, jsonRPCInvalidParams, "invalid params: "+err.Error(), nil)
		return
	}

	slog.Debug("reflection V2: running action", "key", params.Key, "stream", params.Stream)

	actionCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	var traceIDMu sync.Mutex
	var traceID string

	telemetryCb := func(tid, _ string) {
		traceIDMu.Lock()
		traceID = tid
		traceIDMu.Unlock()

		s.activeActions.Set(tid, &activeAction{
			cancel:    cancel,
			startTime: time.Now(),
			traceID:   tid,
		})

		s.sendNotification("runActionState", &ReflectionRunActionStateParams{
			RequestID: req.ID,
			State:     &ReflectionRunActionStateParamsState{TraceID: tid},
		})
	}

	var streamCb streamingCallback[json.RawMessage]
	if params.Stream {
		streamCb = func(_ context.Context, chunk json.RawMessage) error {
			return s.sendNotification("streamChunk", &ReflectionStreamChunkParams{
				RequestID: req.ID,
				Chunk:     chunk,
			})
		}
	}

	contextMap := core.ActionContext{}
	if params.Context != nil {
		if err := json.Unmarshal(params.Context, &contextMap); err != nil {
			s.sendErrorResponse(req.ID, jsonRPCInvalidParams, "invalid context: "+err.Error(), nil)
			return
		}
	}

	actionCtx = tracing.WithTelemetryCallback(actionCtx, telemetryCb)
	resp, err := runAction(actionCtx, s.g, params.Key, params.Input, params.TelemetryLabels, streamCb, contextMap)

	traceIDMu.Lock()
	capturedTraceID := traceID
	traceIDMu.Unlock()
	if capturedTraceID != "" {
		s.activeActions.Delete(capturedTraceID)
	}

	if err != nil {
		s.sendRunActionError(req.ID, err, capturedTraceID)
		return
	}

	s.sendResponse(req.ID, &reflectionRunActionResponse{
		Result:    resp.Result,
		Telemetry: telemetry{TraceID: resp.Telemetry.TraceID},
	})
}

// sendRunActionError maps a runAction error to a JSON-RPC error response
// with a Status-shaped data field matching the JS implementation.
func (s *reflectionServerV2) sendRunActionError(id string, err error, traceID string) {
	code := core.INTERNAL
	msg := err.Error()
	if errors.Is(err, context.Canceled) {
		code = core.CANCELLED
		msg = "Action was cancelled"
	}

	details := map[string]any{}
	if traceID != "" {
		details["traceId"] = traceID
	}
	var ge *core.GenkitError
	if errors.As(err, &ge) && ge.Details != nil {
		if stack, ok := ge.Details["stack"].(string); ok {
			details["stack"] = stack
		}
	}

	data := map[string]any{
		"code":    core.StatusNameToCode[code],
		"message": msg,
	}
	if len(details) > 0 {
		data["details"] = details
	}

	s.sendErrorResponse(id, jsonRPCServerError, msg, data)
}

// handleConfigure processes a configuration notification from the manager.
func (s *reflectionServerV2) handleConfigure(req *jsonRPCMessage) {
	var params ReflectionConfigureParams
	if err := json.Unmarshal(req.Params, &params); err != nil {
		slog.Error("reflection V2: invalid configure params", "err", err)
		return
	}
	configureTelemetry(params.TelemetryServerURL)
}

// handleCancelAction cancels an in-flight action by trace ID.
func (s *reflectionServerV2) handleCancelAction(req *jsonRPCMessage) {
	if req.ID == "" {
		return
	}
	var params ReflectionCancelActionParams
	if err := json.Unmarshal(req.Params, &params); err != nil {
		s.sendErrorResponse(req.ID, jsonRPCInvalidParams, "invalid params: "+err.Error(), nil)
		return
	}
	if params.TraceID == "" {
		s.sendErrorResponse(req.ID, jsonRPCInvalidParams, "traceId is required", nil)
		return
	}

	action, ok := s.activeActions.Get(params.TraceID)
	if !ok {
		s.sendErrorResponse(req.ID, jsonRPCInvalidParams, "Action not found or already completed", nil)
		return
	}

	action.cancel()
	s.activeActions.Delete(params.TraceID)
	s.sendResponse(req.ID, &ReflectionCancelActionResponse{Message: "Action cancelled"})
}

// sendResponse sends a JSON-RPC success response. Send errors are logged:
// the read loop will detect a broken connection on its next read.
func (s *reflectionServerV2) sendResponse(id string, result any) {
	if err := s.send(&jsonRPCResponse{JSONRPC: "2.0", Result: result, ID: id}); err != nil {
		slog.Error("reflection V2: failed to send response", "err", err, "id", id)
	}
}

// sendErrorResponse sends a JSON-RPC error response.
func (s *reflectionServerV2) sendErrorResponse(id string, code int, message string, data any) {
	if err := s.send(&jsonRPCResponse{
		JSONRPC: "2.0",
		Error:   &jsonRPCError{Code: code, Message: message, Data: data},
		ID:      id,
	}); err != nil {
		slog.Error("reflection V2: failed to send error response", "err", err, "id", id)
	}
}

// sendNotification sends a JSON-RPC notification (no ID, no response expected).
func (s *reflectionServerV2) sendNotification(method string, params any) error {
	return s.send(&jsonRPCRequestOrNotification{JSONRPC: "2.0", Method: method, Params: params})
}

// sendRequest sends a JSON-RPC request and blocks until a response is
// received, the context is cancelled, or the connection drops.
func (s *reflectionServerV2) sendRequest(ctx context.Context, method string, params any) (json.RawMessage, error) {
	id := strconv.FormatUint(s.requestSeq.Add(1), 10)
	ch := make(chan pendingResponse, 1)

	s.pendingMu.Lock()
	s.pending[id] = ch
	s.pendingMu.Unlock()

	defer func() {
		s.pendingMu.Lock()
		delete(s.pending, id)
		s.pendingMu.Unlock()
	}()

	if err := s.send(&jsonRPCRequestOrNotification{JSONRPC: "2.0", Method: method, Params: params, ID: id}); err != nil {
		return nil, err
	}

	select {
	case resp := <-ch:
		if resp.err != nil {
			return nil, fmt.Errorf("jsonrpc error %d: %s", resp.err.Code, resp.err.Message)
		}
		return resp.result, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// deliverResponse routes a response message to the channel of the originating request.
func (s *reflectionServerV2) deliverResponse(msg *jsonRPCMessage) {
	s.pendingMu.Lock()
	ch, ok := s.pending[msg.ID]
	s.pendingMu.Unlock()
	if !ok {
		slog.Debug("reflection V2: response for unknown id", "id", msg.ID)
		return
	}
	ch <- pendingResponse{result: msg.Result, err: msg.Error}
}

// drainPending fails all outstanding requests. Called when the connection
// drops so callers don't block forever.
func (s *reflectionServerV2) drainPending(err error) {
	s.pendingMu.Lock()
	defer s.pendingMu.Unlock()
	errObj := &jsonRPCError{Code: jsonRPCServerError, Message: err.Error()}
	for id, ch := range s.pending {
		select {
		case ch <- pendingResponse{err: errObj}:
		default:
		}
		delete(s.pending, id)
	}
}

// send writes a JSON message to the WebSocket connection.
// It is safe for concurrent use.
func (s *reflectionServerV2) send(msg any) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return wsjson.Write(context.Background(), s.conn, msg)
}
