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
	"runtime/debug"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coder/websocket"
	"github.com/coder/websocket/wsjson"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
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

	// bidiSessions tracks in-flight bidi runAction calls so that
	// sendInputStreamChunk/endInputStream notifications can be routed to them.
	// Sessions are pre-registered in readLoop so that chunks arriving before
	// the action has finished initializing are buffered rather than dropped.
	bidiMu       sync.Mutex
	bidiSessions map[string]*bidiSession
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
		bidiSessions:  map[string]*bidiSession{},
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
	// Handler goroutines (and bidi runs surviving a disconnect) read s.conn
	// through send while the session goroutine reconnects; synchronize the
	// swap with the same lock send holds.
	s.writeMu.Lock()
	s.conn = conn
	s.writeMu.Unlock()
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
	// If the client disconnects mid-stream without sending endInputStream,
	// end the input of any in-flight bidi sessions so action bodies awaiting
	// input can terminate instead of hanging (matching the JS runtime).
	defer s.closeBidiSessions()
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
			switch msg.Method {
			case "runAction":
				// Pre-register a bidi session before dispatching the handler
				// so that later sendInputStreamChunk / endInputStream
				// notifications (which run in their own goroutines) always
				// find it.
				if msg.ID != "" {
					var peek struct {
						StreamInput bool `json:"streamInput"`
					}
					if len(msg.Params) > 0 {
						_ = json.Unmarshal(msg.Params, &peek)
					}
					if peek.StreamInput {
						s.registerBidiSession(msg.ID, newBidiSession())
					}
				}
				go s.handleRequest(ctx, &msg)
			case "sendInputStreamChunk", "endInputStream":
				// Dispatch synchronously to preserve wire ordering when
				// enqueueing onto the per-session event queue. Enqueueing
				// never blocks, so this cannot stall the read loop.
				s.handleRequest(ctx, &msg)
			default:
				go s.handleRequest(ctx, &msg)
			}
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
	// Handlers run action functions on this goroutine, where an unrecovered
	// panic would take down the whole dev process along with the failing
	// action. Report it as a server error instead, matching how the bidi
	// engine recovers its function panics.
	defer func() {
		if r := recover(); r != nil {
			slog.Error("reflection V2: handler panicked", "method", req.Method, "panic", r, "stack", string(debug.Stack()))
			if req.ID != "" {
				s.sendErrorResponse(req.ID, jsonRPCServerError, fmt.Sprintf("handler for %s panicked: %v", req.Method, r), nil)
			}
		}
	}()

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
	case "sendInputStreamChunk":
		s.handleSendInputStreamChunk(req)
	case "endInputStream":
		s.handleEndInputStream(req)
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
	// Owns cleanup for any bidi session pre-registered by readLoop, including
	// early-return paths. The session is captured up front and unregistration
	// is ownership-checked: a long-lived handler must not tear down a newer
	// session registered under a reused request id (the manager's id counter
	// restarts when the CLI restarts and reconnects).
	session := s.lookupBidiSession(req.ID)
	defer s.unregisterBidiSession(req.ID, session)

	var params ReflectionRunActionParams
	if err := json.Unmarshal(req.Params, &params); err != nil {
		s.sendErrorResponse(req.ID, jsonRPCInvalidParams, "invalid params: "+err.Error(), nil)
		return
	}

	slog.Debug("reflection V2: running action", "key", params.Key, "stream", params.Stream, "streamInput", params.StreamInput)

	if params.StreamInput {
		s.handleRunActionBidi(ctx, req, &params, session)
		return
	}

	actionCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	rt := s.newRunActionTelemetry(req.ID, cancel)

	var streamCb streamingCallback[json.RawMessage]
	if params.Stream {
		streamCb = func(_ context.Context, chunk json.RawMessage) error {
			return s.sendStreamChunk(req.ID, chunk)
		}
	}

	contextMap := core.ActionContext{}
	if params.Context != nil {
		if err := json.Unmarshal(params.Context, &contextMap); err != nil {
			s.sendErrorResponse(req.ID, jsonRPCInvalidParams, "invalid context: "+err.Error(), nil)
			return
		}
	}

	actionCtx = tracing.WithTelemetryCallback(actionCtx, rt.callback)
	resp, err := runAction(actionCtx, s.g, params.Key, params.Input, params.Init, params.TelemetryLabels, streamCb, contextMap)

	capturedTraceID := rt.finish()

	if err != nil {
		s.sendRunActionError(req.ID, err, capturedTraceID)
		return
	}

	s.sendResponse(req.ID, &reflectionRunActionResponse{
		Result:    resp.Result,
		Telemetry: telemetry{TraceID: resp.Telemetry.TraceID},
	})
}

// runActionTelemetry tracks the trace ID for one in-flight runAction request:
// the span-start callback records the id, registers the run as a cancellable
// active action, and notifies the client. finish marks the request complete
// and returns the captured trace ID; callbacks firing after finish are
// ignored, since a bidi session's span is created asynchronously and can
// start after the handler has already responded.
type runActionTelemetry struct {
	s      *reflectionServerV2
	reqID  string
	cancel context.CancelFunc

	mu       sync.Mutex
	traceID  string
	finished bool
}

func (s *reflectionServerV2) newRunActionTelemetry(reqID string, cancel context.CancelFunc) *runActionTelemetry {
	return &runActionTelemetry{s: s, reqID: reqID, cancel: cancel}
}

// callback is the tracing telemetry callback for the run's root span.
func (rt *runActionTelemetry) callback(tid, _ string) {
	rt.mu.Lock()
	if rt.finished {
		rt.mu.Unlock()
		return
	}
	rt.traceID = tid
	// Registered under the lock so finish either sees the id (and deletes
	// the entry) or this callback sees finished (and never registers).
	rt.s.activeActions.Set(tid, &activeAction{
		cancel:    rt.cancel,
		startTime: time.Now(),
		traceID:   tid,
	})
	rt.mu.Unlock()

	rt.s.sendNotification("runActionState", &ReflectionRunActionStateParams{
		RequestID: rt.reqID,
		State:     &ReflectionRunActionStateParamsState{TraceID: tid},
	})
}

// finish marks the request complete, removes the active-action entry, and
// returns the trace ID captured for the run (empty if the span never started).
func (rt *runActionTelemetry) finish() string {
	rt.mu.Lock()
	rt.finished = true
	tid := rt.traceID
	rt.mu.Unlock()
	if tid != "" {
		rt.s.activeActions.Delete(tid)
	}
	return tid
}

// sendStreamChunk forwards one output chunk to the client as a streamChunk
// notification.
func (s *reflectionServerV2) sendStreamChunk(requestID string, chunk json.RawMessage) error {
	return s.sendNotification("streamChunk", &ReflectionStreamChunkParams{
		RequestID: requestID,
		Chunk:     chunk,
	})
}

// handleRunActionBidi handles a runAction request with streamInput=true.
// It resolves the action as a bidi action, wires its input/output streams to
// the JSON-RPC connection, and waits for the final result. The session has
// already been pre-registered by readLoop.
func (s *reflectionServerV2) handleRunActionBidi(ctx context.Context, req *jsonRPCMessage, params *ReflectionRunActionParams, session *bidiSession) {
	if session == nil {
		// readLoop pre-registers a session for every streamInput run, so a
		// missing one means it was already torn down (e.g. a reused request
		// id); fail loud rather than dereference it below.
		s.sendErrorResponse(req.ID, jsonRPCServerError, "bidi session unavailable for request "+req.ID, nil)
		return
	}

	action := s.g.reg.ResolveAction(params.Key)
	if action == nil {
		s.sendErrorResponse(req.ID, jsonRPCInvalidParams, fmt.Sprintf("action not found: %s", params.Key), nil)
		return
	}
	bidi, ok := action.(api.BidiAction)
	if !ok {
		s.sendErrorResponse(req.ID, jsonRPCInvalidParams, fmt.Sprintf("action %s does not support bidirectional streaming", params.Key), nil)
		return
	}

	actionCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	rt := s.newRunActionTelemetry(req.ID, cancel)
	actionCtx = tracing.WithTelemetryCallback(actionCtx, rt.callback)

	if params.Context != nil {
		contextMap := core.ActionContext{}
		if err := json.Unmarshal(params.Context, &contextMap); err != nil {
			s.sendErrorResponse(req.ID, jsonRPCInvalidParams, "invalid context: "+err.Error(), nil)
			return
		}
		actionCtx = core.WithActionContext(actionCtx, contextMap)
	}

	conn, err := bidi.ConnectJSON(actionCtx, &api.BidiJSONOptions{Init: params.Init})
	if err != nil {
		s.sendErrorResponse(req.ID, jsonRPCServerError, err.Error(), nil)
		return
	}

	// Start consuming outgoing stream chunks before replaying buffered
	// inputs, so the action's outbound channel can drain while we feed
	// inbound chunks. Chunks are forwarded to the client only when output
	// streaming was requested (matching the JS runtime), but the stream is
	// always drained since the connection applies backpressure to the action.
	forwardDone := make(chan struct{})
	go func() {
		defer close(forwardDone)
		forward := params.Stream
		for chunk, rerr := range conn.Receive() {
			if rerr != nil {
				return
			}
			if !forward {
				continue
			}
			if err := s.sendStreamChunk(req.ID, chunk); err != nil {
				slog.Debug("reflection V2: streamChunk send failed", "err", err)
				forward = false
			}
		}
	}()

	// Hand the real connection to the pre-registered session. Any chunks
	// that arrived while we were resolving the action are replayed now.
	go session.run(conn)

	output, runErr := conn.Output()
	<-forwardDone

	capturedTraceID := rt.finish()

	if runErr != nil {
		s.sendRunActionError(req.ID, runErr, capturedTraceID)
		return
	}

	s.sendResponse(req.ID, &reflectionRunActionResponse{
		Result:    output,
		Telemetry: telemetry{TraceID: capturedTraceID},
	})
}

// handleSendInputStreamChunk routes an inbound chunk to the bidi session
// identified by RequestID. If the session has not yet attached its underlying
// connection, the chunk is buffered and replayed when it does.
func (s *reflectionServerV2) handleSendInputStreamChunk(req *jsonRPCMessage) {
	var params ReflectionSendInputStreamChunkParams
	if err := json.Unmarshal(req.Params, &params); err != nil {
		slog.Debug("reflection V2: invalid sendInputStreamChunk params", "err", err)
		return
	}
	session := s.lookupBidiSession(params.RequestID)
	if session == nil {
		slog.Debug("reflection V2: sendInputStreamChunk for unknown session", "requestId", params.RequestID)
		return
	}
	session.Send(params.Chunk)
}

// handleEndInputStream closes the input stream of the bidi session identified
// by RequestID. If the session has not yet attached its connection, the end
// signal is buffered.
func (s *reflectionServerV2) handleEndInputStream(req *jsonRPCMessage) {
	var params ReflectionEndInputStreamParams
	if err := json.Unmarshal(req.Params, &params); err != nil {
		slog.Debug("reflection V2: invalid endInputStream params", "err", err)
		return
	}
	session := s.lookupBidiSession(params.RequestID)
	if session == nil {
		slog.Debug("reflection V2: endInputStream for unknown session", "requestId", params.RequestID)
		return
	}
	session.Close()
}

func (s *reflectionServerV2) registerBidiSession(id string, session *bidiSession) {
	s.bidiMu.Lock()
	old := s.bidiSessions[id]
	s.bidiSessions[id] = session
	s.bidiMu.Unlock()
	// A reused request id orphans the previous session; stop its worker so
	// it cannot linger blocked on an empty queue.
	if old != nil {
		old.stop()
	}
}

// closeBidiSessions enqueues an end-of-input marker for every in-flight bidi
// session. Used when the connection drops, since the client can no longer
// send endInputStream; ending the input lets actions finish gracefully.
func (s *reflectionServerV2) closeBidiSessions() {
	s.bidiMu.Lock()
	sessions := make([]*bidiSession, 0, len(s.bidiSessions))
	for _, session := range s.bidiSessions {
		sessions = append(sessions, session)
	}
	s.bidiMu.Unlock()
	for _, session := range sessions {
		session.Close()
	}
}

// unregisterBidiSession removes session from the map, unless a newer session
// has been registered under the same id, and stops its worker goroutine. A
// nil session is a no-op. Safe to call multiple times for the same session.
func (s *reflectionServerV2) unregisterBidiSession(id string, session *bidiSession) {
	if session == nil {
		return
	}
	s.bidiMu.Lock()
	if s.bidiSessions[id] == session {
		delete(s.bidiSessions, id)
	}
	s.bidiMu.Unlock()
	session.stop()
}

func (s *reflectionServerV2) lookupBidiSession(id string) *bidiSession {
	s.bidiMu.Lock()
	defer s.bidiMu.Unlock()
	return s.bidiSessions[id]
}

// bidiSession is the runtime-side handle for an in-flight bidi runAction call.
// All input events (chunks plus a terminating close) are queued in arrival
// order from the read loop. The session is pre-registered before the action
// starts initializing, so events that arrive early simply accumulate; once
// the handler has created the underlying api.BidiJSONConnection it starts a
// run worker goroutine that drains the queue into the connection in order.
//
// The queue is unbounded so that enqueueing never blocks the WebSocket read
// loop, which must stay responsive to process cancelAction (the escape hatch
// for a stuck action). Memory is bounded by the client, which is the trusted
// dev tooling; the JS runtime makes the same trade.
type bidiSession struct {
	mu      sync.Mutex
	cond    *sync.Cond
	events  []bidiEvent
	stopped bool
}

// bidiEvent is one item delivered to the worker. close=true is a terminal
// marker; any chunks queued after it are dropped.
type bidiEvent struct {
	chunk json.RawMessage
	close bool
}

func newBidiSession() *bidiSession {
	s := &bidiSession{}
	s.cond = sync.NewCond(&s.mu)
	return s
}

// run forwards queued events to the connection in order. Returns after a
// close event or when the session is stopped.
func (s *bidiSession) run(conn api.BidiJSONConnection) {
	for {
		ev, ok := s.next()
		if !ok {
			return
		}
		if ev.close {
			_ = conn.Close()
			return
		}
		if err := conn.Send(ev.chunk); err != nil {
			slog.Debug("reflection V2: bidi Send failed", "err", err)
		}
	}
}

// next blocks until an event is queued or the session is stopped. Returns
// ok=false when stopped; events still queued at that point are dropped, as
// the session is being torn down.
func (s *bidiSession) next() (bidiEvent, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for len(s.events) == 0 && !s.stopped {
		s.cond.Wait()
	}
	if s.stopped {
		return bidiEvent{}, false
	}
	ev := s.events[0]
	s.events[0] = bidiEvent{} // Release the chunk for GC.
	s.events = s.events[1:]
	if len(s.events) == 0 {
		s.events = nil // Release the backing array.
	}
	return ev, true
}

// Send enqueues a chunk for delivery to the action. It never blocks.
func (s *bidiSession) Send(chunk json.RawMessage) {
	s.enqueue(bidiEvent{chunk: chunk})
}

// Close enqueues a terminal end-of-input marker. It never blocks.
func (s *bidiSession) Close() {
	s.enqueue(bidiEvent{close: true})
}

func (s *bidiSession) enqueue(ev bidiEvent) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.stopped {
		return
	}
	s.events = append(s.events, ev)
	s.cond.Signal()
}

// stop terminates the worker and drops any queued events. Safe to call
// multiple times.
func (s *bidiSession) stop() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.stopped = true
	s.cond.Broadcast()
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
