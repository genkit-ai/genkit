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
	"maps"
	"net/http"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"github.com/google/uuid"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/core/x/streaming"
)

// HandlerOption configures a Handler.
type HandlerOption interface {
	applyHandler(*handlerOptions) error
}

// handlerOptions are options for an action HTTP handler.
type handlerOptions struct {
	ContextProviders []core.ContextProvider  // Providers for action context that may be used during runtime.
	StreamManager    streaming.StreamManager // Optional manager for durable stream storage.
}

func (o *handlerOptions) applyHandler(opts *handlerOptions) error {
	if o.ContextProviders != nil {
		if opts.ContextProviders != nil {
			return errors.New("cannot set ContextProviders more than once (WithContextProviders)")
		}
		opts.ContextProviders = o.ContextProviders
	}

	if o.StreamManager != nil {
		if opts.StreamManager != nil {
			return errors.New("cannot set StreamManager more than once (WithStreamManager)")
		}
		opts.StreamManager = o.StreamManager
	}

	return nil
}

// requestID is a unique ID for each request.
var requestID atomic.Int64

// WithContextProviders adds providers for action context that may be used during runtime.
// They are called in the order added and may overwrite previous context.
func WithContextProviders(ctxProviders ...core.ContextProvider) HandlerOption {
	return &handlerOptions{ContextProviders: ctxProviders}
}

// WithStreamManager enables durable streaming with the provided StreamManager.
// When enabled, streaming responses include an x-genkit-stream-id header that clients
// can use to reconnect to in-progress or completed streams.
//
// EXPERIMENTAL: This API is subject to change.
func WithStreamManager(manager streaming.StreamManager) HandlerOption {
	return &handlerOptions{StreamManager: manager}
}

// Handler returns an HTTP handler function that serves the action with the provided options.
//
// The provided HandlerOptions are applied during construction. If any option
// fails to apply, Handler panics.
//
// Example:
//
//	genkit.Handler(
//		g,
//		genkit.WithContextProviders(func(ctx context.Context, req core.RequestData) (api.ActionContext, error) {
//			return api.ActionContext{"myKey": "myValue"}, nil
//		}))
func Handler(a api.Action, opts ...HandlerOption) http.HandlerFunc {
	return wrapHandler(HandlerFunc(a, opts...))
}

// HandlerFunc returns an HTTP handler function that executes the given action
// and returns an error instead of writing it directly to the response.
//
// It is intended for use with web frameworks that expect handlers with the
// signature:
//
//	func(http.ResponseWriter, *http.Request) error
//
// so that errors can be handled centrally (e.g., by middleware).
//
// The provided HandlerOptions are applied during construction. If any option
// fails to apply, HandlerFunc panics.
//
// Example:
//
//	genkit.HandlerFunc(
//		g,
//		genkit.WithContextProviders(func(ctx context.Context, req core.RequestData) (api.ActionContext, error) {
//			return api.ActionContext{"myKey": "myValue"}, nil
//		}),
//	)
func HandlerFunc(a api.Action, opts ...HandlerOption) func(http.ResponseWriter, *http.Request) error {
	options := &handlerOptions{}
	for _, opt := range opts {
		if err := opt.applyHandler(options); err != nil {
			panic(fmt.Errorf("genkit.HandlerFunc: error applying options: %w", err))
		}
	}

	return handler(a, options)
}

// wrapHandler wraps an HTTP handler function with common logging and error handling.
func wrapHandler(h func(http.ResponseWriter, *http.Request) error) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		log := slog.Default().With("reqId", requestID.Add(1))
		// Carry the request-scoped logger in the context so everything logged
		// while handling this request (including inside flows and tools) is
		// tagged with the same reqId.
		r = r.WithContext(logger.WithContext(r.Context(), log))
		ctx := r.Context()

		start := time.Now()
		logger.Debug(ctx, "request started", "method", r.Method, "path", r.URL.Path)

		var err error
		defer func() {
			if err != nil {
				logger.Error(ctx, "request failed",
					"method", r.Method,
					"path", r.URL.Path,
					"duration", time.Since(start).Round(time.Millisecond),
					"error", err)
			} else {
				logger.Debug(ctx, "request finished", "duration", time.Since(start).Round(time.Millisecond))
			}
		}()

		if err = h(w, r); err != nil {
			msg, code := clientError(err)
			http.Error(w, msg, code.HTTPCode())
		}
	}
}

// clientError returns the message and status to send a client for err. Both
// the HTTP code (via [status.Name.HTTPCode]) and any wire status field must
// come from this one derivation so the two can never disagree.
//
// The status always comes from the error, so an error deliberately marked
// public reaches the client with its own code rather than falling through to
// 500. The message only leaves the process when the error was built with
// [status.PublicErrorf]; anything else becomes a generic string derived from
// the status, so schema dumps, provider text, and internal identifiers stay
// server-side. The full error is still logged server-side: by wrapHandler for
// request failures, and by the streaming runners for mid-stream flow failures.
//
// GENKIT_ENV=dev is exempt: suppressing the message during local development
// only hides the failure from the developer causing it.
func clientError(err error) (string, status.Name) {
	code := status.Of(err)
	// Only reached on a failure path, so an error that classifies as OK is
	// itself the bug: the usual cause is a non-nil interface holding a nil
	// *status.Error, which would otherwise report success on a request whose
	// result was never written.
	if code == status.OK {
		code = status.Internal
	}
	msg, public := status.PublicMessage(err)
	if !public && api.CurrentEnvironment() == api.EnvironmentDev {
		msg = err.Error()
	}
	return msg, code
}

// handler returns an HTTP handler function that serves the action with the provided options.
// Streaming responses are written in server-sent events (SSE) format.
func handler(a api.Action, opts *handlerOptions) func(http.ResponseWriter, *http.Request) error {
	return func(w http.ResponseWriter, r *http.Request) error {
		if a == nil {
			return errors.New("action is nil; cannot serve")
		}

		var body struct {
			Data json.RawMessage `json:"data"`
			Init json.RawMessage `json:"init,omitempty"` // Per-session init for bidi actions; rejected otherwise.
		}
		if r.Body != nil && r.ContentLength > 0 {
			defer r.Body.Close()
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				return status.PublicErrorf(status.ErrInvalidArgument, "%w", err)
			}
		}

		// Rejected before the streaming branch commits to SSE headers, so a
		// non-bidi action receiving init fails with a proper HTTP 400 on
		// every path rather than an in-band SSE error event on a 200.
		if err := checkInitSupported(a, body.Init); err != nil {
			return err
		}

		run := func(ctx context.Context, input json.RawMessage, cb func(context.Context, json.RawMessage) error) (json.RawMessage, error) {
			r, err := runActionWithOptionalInit(ctx, a, input, body.Init, cb)
			if err != nil {
				return nil, err
			}
			return r.Result, nil
		}

		stream, err := parseBoolQueryParam(r, "stream")
		if err != nil {
			return err
		}
		stream = stream || r.Header.Get("Accept") == "text/event-stream"

		ctx, err := applyContextProviders(r.Context(), r, opts.ContextProviders, body.Data)
		if err != nil {
			return err
		}

		if stream {
			streamID := r.Header.Get("X-Genkit-Stream-Id")

			if streamID != "" && opts.StreamManager != nil {
				return subscribeToStream(ctx, w, opts.StreamManager, streamID)
			}

			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.Header().Set("Connection", "keep-alive")
			w.Header().Set("Transfer-Encoding", "chunked")

			if opts.StreamManager != nil {
				return runWithDurableStreaming(ctx, w, run, opts.StreamManager, body.Data)
			}

			return runWithStreaming(ctx, w, run, body.Data)
		}

		w.Header().Set("Content-Type", "application/json")
		out, err := run(ctx, body.Data, nil)
		if err != nil {
			return err
		}
		return writeResultResponse(w, out)
	}
}

// applyContextProviders runs the configured context providers against the
// request and folds their results into ctx, so request-derived action
// context (e.g. auth from headers) is available to the action. input is
// handed to each provider as the request's decoded input
// ([core.RequestData.Input]). A nil or empty providers slice returns ctx
// unchanged.
func applyContextProviders(ctx context.Context, r *http.Request, providers []core.ContextProvider, input json.RawMessage) (context.Context, error) {
	for _, ctxProvider := range providers {
		headers := make(map[string]string, len(r.Header))
		for k, v := range r.Header {
			headers[strings.ToLower(k)] = strings.Join(v, " ")
		}

		actionCtx, err := ctxProvider(ctx, core.RequestData{
			Method:  r.Method,
			Headers: headers,
			Input:   input,
		})
		if err != nil {
			logger.Error(ctx, "context provider rejected the request", "error", err)
			return ctx, err
		}

		if existing := core.FromContext(ctx); existing != nil {
			maps.Copy(existing, actionCtx)
			actionCtx = existing
		}
		ctx = core.WithActionContext(ctx, actionCtx)
	}
	return ctx, nil
}

// runJSONFunc abstracts over RunJSON and RunBidiJSON for the handler's
// execution paths.
type runJSONFunc = func(context.Context, json.RawMessage, func(context.Context, json.RawMessage) error) (json.RawMessage, error)

// runWithStreaming executes the action with standard HTTP streaming (no durability).
func runWithStreaming(ctx context.Context, w http.ResponseWriter, run runJSONFunc, input json.RawMessage) error {
	callback := func(ctx context.Context, msg json.RawMessage) error {
		if err := writeSSEMessage(w, msg); err != nil {
			return err
		}
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		return nil
	}

	out, err := run(ctx, input, callback)
	if err != nil {
		// The SSE frame carries only the redacted message and this function
		// returns nil, so wrapHandler never sees the error: this log is the
		// only server-side record of the real failure.
		logger.Error(ctx, "streaming flow failed", "error", err)
		if werr := writeSSEError(w, err); werr != nil {
			return werr
		}
		return nil
	}
	return writeSSEResult(w, out)
}

// runWithDurableStreaming executes the action with durable streaming support.
// Chunks are written to both the HTTP response and the stream manager for later replay.
//
// The flow execution is detached from the HTTP request context so that if the
// original client disconnects, the flow continues running and writing to durable
// storage. This allows other clients to subscribe to the stream and receive the
// remaining chunks and final result.
func runWithDurableStreaming(ctx context.Context, w http.ResponseWriter, run runJSONFunc, sm streaming.StreamManager, input json.RawMessage) error {
	streamID := uuid.New().String()

	durableStream, err := sm.Open(ctx, streamID)
	if err != nil {
		return err
	}
	defer durableStream.Close()

	w.Header().Set("X-Genkit-Stream-Id", streamID)

	// Create a detached context for flow execution. This preserves context values
	// (action context, tracing, logger) but won't be canceled when the HTTP client
	// disconnects, allowing the flow to continue streaming to durable storage.
	durableCtx := context.WithoutCancel(ctx)

	// Track whether the HTTP client is still connected.
	clientGone := ctx.Done()

	callback := func(_ context.Context, msg json.RawMessage) error {
		// Always write to durable storage regardless of client connection state.
		durableStream.Write(durableCtx, msg)

		// Only attempt HTTP writes if the client is still connected.
		select {
		case <-clientGone:
			return nil
		default:
			if err := writeSSEMessage(w, msg); err != nil {
				return nil
			}
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
		return nil
	}

	out, err := run(durableCtx, input, callback)
	if err != nil {
		// As in runWithStreaming: the wire carries only the redacted message
		// and wrapHandler never sees the error, so log the real failure here.
		// The durable record is no substitute: it expires with the stream.
		logger.Error(durableCtx, "streaming flow failed", "error", err)
		durableStream.Error(durableCtx, err)
		select {
		case <-clientGone:
			return nil
		default:
			writeSSEError(w, err)
		}
		return nil
	}

	durableStream.Done(durableCtx, out)
	select {
	case <-clientGone:
		return nil
	default:
		return writeSSEResult(w, out)
	}
}

// subscribeToStream subscribes to an existing durable stream and writes events to the HTTP response.
func subscribeToStream(ctx context.Context, w http.ResponseWriter, sm streaming.StreamManager, streamID string) error {
	events, unsubscribe, err := sm.Subscribe(ctx, streamID)
	if err != nil {
		// Subscribe's contract is any NOT_FOUND error, not the in-tree
		// streaming.ErrStreamNotFound sentinel specifically, so match on the
		// status: a third-party StreamManager returning a plain NOT_FOUND
		// gets the 204 that resuming clients key on, not a 404.
		if status.Of(err) == status.NotFound {
			w.WriteHeader(http.StatusNoContent)
			return nil
		}
		return err
	}
	defer unsubscribe()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Transfer-Encoding", "chunked")

	for event := range events {
		switch event.Type {
		case streaming.StreamEventChunk:
			if err := writeSSEMessage(w, event.Chunk); err != nil {
				return err
			}
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		case streaming.StreamEventDone:
			if err := writeSSEResult(w, event.Output); err != nil {
				return err
			}
			return nil
		case streaming.StreamEventError:
			streamErr := event.Err
			if streamErr == nil {
				streamErr = errors.New("unknown error")
			}
			if err := writeSSEError(w, streamErr); err != nil {
				return err
			}
			return nil
		}
	}

	return nil
}

// flowResultResponse wraps a final action result for JSON serialization.
type flowResultResponse struct {
	Result json.RawMessage `json:"result"`
}

// flowMessageResponse wraps a streaming chunk for JSON serialization.
type flowMessageResponse struct {
	Message json.RawMessage `json:"message"`
}

// flowErrorResponse wraps an error for JSON serialization in streaming responses.
type flowErrorResponse struct {
	Error *flowError `json:"error"`
}

// flowError represents the error payload in a streaming error response.
//
// It carries no details field: it used to hold the full err.Error() text, which
// put internal failure detail on the wire on every streamed error.
type flowError struct {
	Status  status.Name `json:"status"`
	Message string      `json:"message"`
}

// writeResultResponse writes a JSON result response for non-streaming requests.
func writeResultResponse(w http.ResponseWriter, result json.RawMessage) error {
	resp := flowResultResponse{Result: result}
	data, err := json.Marshal(resp)
	if err != nil {
		return err
	}
	_, err = w.Write(data)
	if err != nil {
		return err
	}
	_, err = w.Write([]byte("\n"))
	return err
}

// writeSSEResult writes a JSON result as a server-sent event for streaming requests.
func writeSSEResult(w http.ResponseWriter, result json.RawMessage) error {
	resp := flowResultResponse{Result: result}
	data, err := json.Marshal(resp)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "data: %s\n\n", data)
	return err
}

// writeSSEMessage writes a streaming chunk as a server-sent event.
func writeSSEMessage(w http.ResponseWriter, msg json.RawMessage) error {
	resp := flowMessageResponse{Message: msg}
	data, err := json.Marshal(resp)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "data: %s\n\n", data)
	return err
}

// writeSSEError writes an error as a server-sent event for streaming requests.
// Status and message come from the same clientError derivation, so the frame
// gets the identical redaction and OK-to-INTERNAL coercion as the HTTP path.
func writeSSEError(w http.ResponseWriter, flowErr error) error {
	msg, code := clientError(flowErr)
	resp := flowErrorResponse{
		Error: &flowError{
			Status:  code,
			Message: msg,
		},
	}
	data, err := json.Marshal(resp)
	if err != nil {
		return err
	}
	_, err = fmt.Fprintf(w, "data: %s\n\n", data)
	return err
}

func parseBoolQueryParam(r *http.Request, name string) (bool, error) {
	b := false
	if s := r.FormValue(name); s != "" {
		var err error
		b, err = strconv.ParseBool(s)
		if err != nil {
			return false, status.PublicErrorf(status.ErrInvalidArgument, "%w", err)
		}
	}
	return b, nil
}
