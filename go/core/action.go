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

package core

import (
	"context"
	"encoding/json"
	"iter"
	"reflect"
	"sync"
	"time"

	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/tracing"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/firebase/genkit/go/internal/metrics"
)

// Func is an alias for non-streaming functions with input of type In and output of type Out.
type Func[In, Out any] = func(context.Context, In) (Out, error)

// StreamingFunc is an alias for streaming functions with input of type In, output of type Out, and outgoing stream chunk of type StreamOut.
type StreamingFunc[In, Out, StreamOut any] = func(context.Context, In, StreamCallback[StreamOut]) (Out, error)

// StreamCallback is a function that is called during streaming to return the next chunk of the outgoing stream.
type StreamCallback[StreamOut any] = func(context.Context, StreamOut) error

// BidiFunc is the function signature for bidirectional streaming actions.
// It receives an initial input, reads incoming stream messages from inCh,
// and writes outgoing stream messages to outCh. It returns a final output when complete.
type BidiFunc[In, Out, StreamOut, StreamIn any] = func(ctx context.Context, in In, inCh <-chan StreamIn, outCh chan<- StreamOut) (Out, error)

// An Action is a named, observable operation that underlies all Genkit primitives.
// It consists of a function that takes an input of type In and returns an output
// of type Out, optionally streaming values of type StreamOut incrementally by
// invoking a callback. For bidirectional actions, StreamIn is the type of
// incoming stream messages.
//
// It optionally has other metadata, like a description and JSON Schemas for its input and
// output which it validates against.
//
// Each time an Action is run, it results in a new trace span.
//
// For internal use only.
type Action[In, Out, StreamOut, StreamIn any] struct {
	fn       StreamingFunc[In, Out, StreamOut]      // Function that is called during runtime. May not actually support streaming.
	bidiFn   BidiFunc[In, Out, StreamOut, StreamIn] // Non-nil for bidi actions only.
	desc     *api.ActionDesc                        // Descriptor of the action.
	registry api.Registry                           // Registry for schema resolution. Set when registered.
}

type noStream = func(context.Context, struct{}) error

// NewAction creates a new non-streaming [Action] without registering it.
// If inputSchema is nil, it is inferred from the function's input api.
func NewAction[In, Out any](
	name string,
	atype api.ActionType,
	metadata map[string]any,
	inputSchema map[string]any,
	fn Func[In, Out],
) *Action[In, Out, struct{}, struct{}] {
	return newAction[In, Out, struct{}, struct{}](name, atype, metadata, inputSchema,
		func(ctx context.Context, in In, cb noStream) (Out, error) {
			return fn(ctx, in)
		})
}

// NewStreamingAction creates a new streaming [Action] without registering it.
// If inputSchema is nil, it is inferred from the function's input api.
func NewStreamingAction[In, Out, StreamOut any](
	name string,
	atype api.ActionType,
	metadata map[string]any,
	inputSchema map[string]any,
	fn StreamingFunc[In, Out, StreamOut],
) *Action[In, Out, StreamOut, struct{}] {
	return newAction[In, Out, StreamOut, struct{}](name, atype, metadata, inputSchema, fn)
}

// ActionOptions configures a bidi action. Nil schema fields are inferred from type parameters.
type ActionOptions struct {
	Metadata        map[string]any // Arbitrary key-value data attached to the action descriptor.
	InputSchema     map[string]any // JSON schema for the action's input. Inferred from In if nil.
	OutputSchema    map[string]any // JSON schema for the action's output. Inferred from Out if nil.
	StreamOutSchema map[string]any // JSON schema for outgoing streamed chunks. Inferred from StreamOut if nil. Not used for non-streaming actions.
	StreamInSchema  map[string]any // JSON schema for incoming stream messages. Inferred from StreamIn if nil. Not used for non-bidi actions.
}

// NewBidiAction creates a new bidirectional streaming [Action] without registering it.
func NewBidiAction[In, Out, StreamOut, StreamIn any](
	name string,
	atype api.ActionType,
	opts *ActionOptions,
	fn BidiFunc[In, Out, StreamOut, StreamIn],
) *Action[In, Out, StreamOut, StreamIn] {
	if opts == nil {
		opts = &ActionOptions{}
	}

	metadata := opts.Metadata
	if metadata == nil {
		metadata = map[string]any{}
	}
	metadata["bidi"] = true

	a := newAction[In, Out, StreamOut, StreamIn](name, atype, metadata, opts.InputSchema, wrapBidiAsStreaming(fn))
	a.bidiFn = fn

	if opts.OutputSchema != nil {
		a.desc.OutputSchema = opts.OutputSchema
	}
	if opts.StreamOutSchema != nil {
		a.desc.StreamOutSchema = opts.StreamOutSchema
	}

	if opts.StreamInSchema != nil {
		a.desc.StreamInSchema = opts.StreamInSchema
	} else {
		var inStream StreamIn
		if reflect.ValueOf(inStream).Kind() != reflect.Invalid {
			a.desc.StreamInSchema = InferSchemaMap(inStream)
		}
	}

	return a
}

// DefineBidiAction creates and registers a bidirectional streaming [Action].
func DefineBidiAction[In, Out, StreamOut, StreamIn any](
	r api.Registry,
	name string,
	atype api.ActionType,
	opts *ActionOptions,
	fn BidiFunc[In, Out, StreamOut, StreamIn],
) *Action[In, Out, StreamOut, StreamIn] {
	a := NewBidiAction(name, atype, opts, fn)
	a.Register(r)
	return a
}

// DefineAction creates a new non-streaming Action and registers it.
// If inputSchema is nil, it is inferred from the function's input api.
func DefineAction[In, Out any](
	r api.Registry,
	name string,
	atype api.ActionType,
	metadata map[string]any,
	inputSchema map[string]any,
	fn Func[In, Out],
) *Action[In, Out, struct{}, struct{}] {
	return defineAction[In, Out, struct{}, struct{}](r, name, atype, metadata, inputSchema,
		func(ctx context.Context, in In, cb noStream) (Out, error) {
			return fn(ctx, in)
		})
}

// DefineStreamingAction creates a new streaming action and registers it.
// If inputSchema is nil, it is inferred from the function's input api.
func DefineStreamingAction[In, Out, StreamOut any](
	r api.Registry,
	name string,
	atype api.ActionType,
	metadata map[string]any,
	inputSchema map[string]any,
	fn StreamingFunc[In, Out, StreamOut],
) *Action[In, Out, StreamOut, struct{}] {
	return defineAction[In, Out, StreamOut, struct{}](r, name, atype, metadata, inputSchema, fn)
}

// defineAction creates an action and registers it with the given Registry.
func defineAction[In, Out, StreamOut, StreamIn any](
	r api.Registry,
	name string,
	atype api.ActionType,
	metadata map[string]any,
	inputSchema map[string]any,
	fn StreamingFunc[In, Out, StreamOut],
) *Action[In, Out, StreamOut, StreamIn] {
	a := newAction[In, Out, StreamOut, StreamIn](name, atype, metadata, inputSchema, fn)
	a.Register(r)
	return a
}

// newAction creates a new Action with the given name and arguments.
// If registry is nil, tracing state is left nil to be set later.
// If inputSchema is nil, it is inferred from In.
func newAction[In, Out, StreamOut, StreamIn any](
	name string,
	atype api.ActionType,
	metadata map[string]any,
	inputSchema map[string]any,
	fn StreamingFunc[In, Out, StreamOut],
) *Action[In, Out, StreamOut, StreamIn] {
	if inputSchema == nil {
		var i In
		if reflect.ValueOf(i).Kind() != reflect.Invalid {
			inputSchema = InferSchemaMap(i)
		}
	}

	var o Out
	var outputSchema map[string]any
	if reflect.ValueOf(o).Kind() != reflect.Invalid {
		outputSchema = InferSchemaMap(o)
	}

	var s StreamOut
	var outStreamSchema map[string]any
	if reflect.ValueOf(s).Kind() != reflect.Invalid {
		outStreamSchema = InferSchemaMap(s)
	}

	var description string
	if desc, ok := metadata["description"].(string); ok {
		description = desc
	}

	return &Action[In, Out, StreamOut, StreamIn]{
		fn: func(ctx context.Context, input In, cb StreamCallback[StreamOut]) (Out, error) {
			return fn(ctx, input, cb)
		},
		desc: &api.ActionDesc{
			Type:            atype,
			Key:             api.KeyFromName(atype, name),
			Name:            name,
			Description:     description,
			InputSchema:     inputSchema,
			OutputSchema:    outputSchema,
			StreamOutSchema: outStreamSchema,
			Metadata:        metadata,
		},
	}
}

// Name returns the Action's Name.
func (a *Action[In, Out, StreamOut, StreamIn]) Name() string { return a.desc.Name }

// Run executes the Action's function in a new trace span.
func (a *Action[In, Out, StreamOut, StreamIn]) Run(ctx context.Context, input In, cb StreamCallback[StreamOut]) (output Out, err error) {
	r, err := a.runWithTelemetry(ctx, input, cb)
	if err != nil {
		return base.Zero[Out](), err
	}
	return r.Result, nil
}

// runWithTelemetry executes the Action's function in a new trace span and returns telemetry info.
func (a *Action[In, Out, StreamOut, StreamIn]) runWithTelemetry(ctx context.Context, input In, cb StreamCallback[StreamOut]) (output api.ActionRunResult[Out], err error) {
	logger.FromContext(ctx).Debug("Action.Run", "name", a.Name())
	defer func() {
		logger.FromContext(ctx).Debug("Action.Run",
			"name", a.Name(),
			"err", err)
	}()

	// Create span metadata and inject flow name if we're in a flow context
	spanMetadata := &tracing.SpanMetadata{
		Name:     a.desc.Name,
		Type:     "action",
		Subtype:  string(a.desc.Type), // The actual action type becomes the subtype
		Metadata: make(map[string]string),
		// IsRoot will be automatically determined in tracing.go based on parent span presence
	}

	// Auto-inject flow name if we're in a flow context
	if flowName := FlowNameFromContext(ctx); flowName != "" {
		spanMetadata.Metadata["flow:name"] = flowName
	}

	var traceID string
	var spanID string
	o, err := tracing.RunInNewSpan(ctx, spanMetadata, input,
		func(ctx context.Context, input In) (output Out, err error) {
			traceInfo := tracing.SpanTraceInfo(ctx)
			traceID = traceInfo.TraceID
			spanID = traceInfo.SpanID

			start := time.Now()
			defer func() {
				latency := time.Since(start)
				if err != nil {
					metrics.WriteActionFailure(ctx, a.desc.Name, latency, err)
				} else {
					metrics.WriteActionSuccess(ctx, a.desc.Name, latency)
				}
			}()

			var inputSchema map[string]any
			inputSchema, err = ResolveSchema(a.registry, a.desc.InputSchema)
			if err != nil {
				return base.Zero[Out](), NewError(INVALID_ARGUMENT, "invalid input schema for action %q: %v", a.desc.Key, err)
			}

			var outputSchema map[string]any
			outputSchema, err = ResolveSchema(a.registry, a.desc.OutputSchema)
			if err != nil {
				return base.Zero[Out](), NewError(INVALID_ARGUMENT, "invalid output schema for action %q: %v", a.desc.Key, err)
			}

			if err = base.ValidateValue(input, inputSchema); err != nil {
				return base.Zero[Out](), NewError(INVALID_ARGUMENT, "invalid input to action %q: %v", a.desc.Key, err)
			}

			output, err = a.fn(ctx, input, cb)
			if err != nil {
				return output, err
			}
			if err = base.ValidateValue(output, outputSchema); err != nil {
				err = NewError(INTERNAL, "invalid output from action %q: %v", a.desc.Key, err)
			}

			return output, err
		},
	)

	return api.ActionRunResult[Out]{
		Result:  o,
		TraceId: traceID,
		SpanId:  spanID,
	}, err
}

// RunJSON runs the action with a JSON input, and returns a JSON result.
func (a *Action[In, Out, StreamOut, StreamIn]) RunJSON(ctx context.Context, input json.RawMessage, cb StreamCallback[json.RawMessage]) (json.RawMessage, error) {
	r, err := a.RunJSONWithTelemetry(ctx, input, cb)
	if err != nil {
		return nil, err
	}
	return r.Result, nil
}

// RunJSONWithTelemetry runs the action with a JSON input, and returns a JSON result along with telemetry info.
func (a *Action[In, Out, StreamOut, StreamIn]) RunJSONWithTelemetry(ctx context.Context, input json.RawMessage, cb StreamCallback[json.RawMessage]) (*api.ActionRunResult[json.RawMessage], error) {
	i, err := base.UnmarshalAndNormalize[In](input, a.desc.InputSchema)
	if err != nil {
		return nil, NewError(INVALID_ARGUMENT, err.Error())
	}

	var scb StreamCallback[StreamOut]
	if cb != nil {
		scb = func(ctx context.Context, s StreamOut) error {
			bytes, err := json.Marshal(s)
			if err != nil {
				return err
			}
			return cb(ctx, json.RawMessage(bytes))
		}
	}

	r, err := a.runWithTelemetry(ctx, i, scb)
	if err != nil {
		return &api.ActionRunResult[json.RawMessage]{
			TraceId: r.TraceId,
			SpanId:  r.SpanId,
		}, err
	}

	bytes, err := json.Marshal(r.Result)
	if err != nil {
		return nil, err
	}

	return &api.ActionRunResult[json.RawMessage]{
		Result:  json.RawMessage(bytes),
		TraceId: r.TraceId,
		SpanId:  r.SpanId,
	}, nil
}

// Desc returns a descriptor of the action with resolved schema references.
func (a *Action[In, Out, StreamOut, StreamIn]) Desc() api.ActionDesc {
	return *a.desc
}

// Register registers the action with the given registry.
func (a *Action[In, Out, StreamOut, StreamIn]) Register(r api.Registry) {
	a.registry = r
	r.RegisterAction(a.desc.Key, a)
}

// StreamBidi starts a bidirectional streaming connection.
// Returns an error if the action is not a bidi action.
// A trace span is created that remains open for the lifetime of the connection.
func (a *Action[In, Out, StreamOut, StreamIn]) StreamBidi(ctx context.Context, in In) (*BidiConnection[StreamIn, StreamOut, Out], error) {
	if a.bidiFn == nil {
		return nil, NewError(FAILED_PRECONDITION, "StreamBidi called on non-bidi action %q", a.desc.Name)
	}

	ctx, cancel := context.WithCancel(ctx)
	conn := &BidiConnection[StreamIn, StreamOut, Out]{
		inputCh:  make(chan StreamIn, 1),
		streamCh: make(chan StreamOut, 1),
		doneCh:   make(chan struct{}),
		ctx:      ctx,
		cancel:   cancel,
	}

	spanMetadata := &tracing.SpanMetadata{
		Name:     a.desc.Name,
		Type:     "action",
		Subtype:  string(a.desc.Type),
		Metadata: make(map[string]string),
	}
	if flowName := FlowNameFromContext(ctx); flowName != "" {
		spanMetadata.Metadata["flow:name"] = flowName
	}

	go func() {
		defer close(conn.doneCh)
		defer close(conn.streamCh)
		output, err := tracing.RunInNewSpan(conn.ctx, spanMetadata, in,
			func(ctx context.Context, in In) (Out, error) {
				start := time.Now()
				output, err := a.bidiFn(ctx, in, conn.inputCh, conn.streamCh)
				latency := time.Since(start)
				if err != nil {
					metrics.WriteActionFailure(ctx, a.desc.Name, latency, err)
				} else {
					metrics.WriteActionSuccess(ctx, a.desc.Name, latency)
				}
				return output, err
			},
		)
		conn.mu.Lock()
		conn.output = output
		conn.err = err
		conn.mu.Unlock()
	}()

	return conn, nil
}

// ResolveActionFor returns the action for the given key in the global registry,
// or nil if there is none.
// It panics if the action is of the wrong api.
func ResolveActionFor[In, Out, StreamOut, StreamIn any](r api.Registry, atype api.ActionType, name string) *Action[In, Out, StreamOut, StreamIn] {
	provider, id := api.ParseName(name)
	key := api.NewKey(atype, provider, id)
	a := r.ResolveAction(key)
	if a == nil {
		return nil
	}
	return a.(*Action[In, Out, StreamOut, StreamIn])
}

// LookupActionFor returns the action for the given key in the global registry,
// or nil if there is none.
// It panics if the action is of the wrong api.
//
// Deprecated: Use ResolveActionFor.
func LookupActionFor[In, Out, StreamOut, StreamIn any](r api.Registry, atype api.ActionType, name string) *Action[In, Out, StreamOut, StreamIn] {
	provider, id := api.ParseName(name)
	key := api.NewKey(atype, provider, id)
	a := r.LookupAction(key)
	if a == nil {
		return nil
	}
	return a.(*Action[In, Out, StreamOut, StreamIn])
}

// wrapBidiAsStreaming wraps a BidiFunc into a StreamingFunc for use with Run/RunJSON.
// The input is passed as the initial input to the bidi func, and the input stream
// channel is closed immediately (no streaming inputs). Outgoing stream chunks are
// forwarded to the callback.
func wrapBidiAsStreaming[In, Out, StreamOut, StreamIn any](fn BidiFunc[In, Out, StreamOut, StreamIn]) StreamingFunc[In, Out, StreamOut] {
	return func(ctx context.Context, input In, cb StreamCallback[StreamOut]) (Out, error) {
		inCh := make(chan StreamIn, 1)
		outCh := make(chan StreamOut, 1)
		doneCh := make(chan struct{})

		var output Out
		var fnErr error

		go func() {
			defer close(doneCh)
			defer close(outCh)
			output, fnErr = fn(ctx, input, inCh, outCh)
		}()

		// No streaming inputs when used as a non-bidi streaming action.
		close(inCh)

		// Forward streamed chunks to the callback.
		if cb != nil {
			for chunk := range outCh {
				if err := cb(ctx, chunk); err != nil {
					return base.Zero[Out](), err
				}
			}
		} else {
			// Drain the channel even without a callback.
			for range outCh {
			}
		}

		<-doneCh
		return output, fnErr
	}
}

// BidiConnection represents an active bidirectional streaming session.
type BidiConnection[StreamIn, StreamOut, Out any] struct {
	inputCh  chan StreamIn
	streamCh chan StreamOut
	doneCh   chan struct{}
	output   Out
	err      error
	ctx      context.Context
	cancel   context.CancelFunc
	mu       sync.Mutex
	closed   bool
}

// Send sends an input message to the bidi action.
// Returns an error if the connection is closed or the context is cancelled.
func (c *BidiConnection[StreamIn, StreamOut, Out]) Send(input StreamIn) (err error) {
	// Recover from "send on closed channel" panic. A check-then-send under the
	// mutex would race with Close, and holding the mutex across the send would
	// deadlock against Close when the buffer is full. Closing inputCh (rather
	// than a separate signal channel) is required so receivers can use the
	// canonical `for ... range inputCh` idiom.
	defer func() {
		if r := recover(); r != nil {
			err = NewError(FAILED_PRECONDITION, "connection is closed")
		}
	}()

	select {
	case c.inputCh <- input:
		return nil
	case <-c.ctx.Done():
		return c.ctx.Err()
	case <-c.doneCh:
		return NewError(FAILED_PRECONDITION, "action has completed")
	}
}

// Close signals that no more inputs will be sent.
func (c *BidiConnection[StreamIn, StreamOut, Out]) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.closed {
		return nil
	}
	c.closed = true
	close(c.inputCh)
	return nil
}

// Receive returns an iterator for receiving streamed response chunks.
// The iterator completes when the action finishes.
func (c *BidiConnection[StreamIn, StreamOut, Out]) Receive() iter.Seq2[StreamOut, error] {
	return func(yield func(StreamOut, error) bool) {
		for {
			select {
			case chunk, ok := <-c.streamCh:
				if !ok {
					return
				}
				if !yield(chunk, nil) {
					c.cancel()
					return
				}
			case <-c.ctx.Done():
				var zero StreamOut
				yield(zero, c.ctx.Err())
				return
			}
		}
	}
}

// Output returns the final output after the action completes.
// Blocks until done or context cancelled.
func (c *BidiConnection[StreamIn, StreamOut, Out]) Output() (Out, error) {
	select {
	case <-c.doneCh:
		c.mu.Lock()
		defer c.mu.Unlock()
		return c.output, c.err
	case <-c.ctx.Done():
		var zero Out
		return zero, c.ctx.Err()
	}
}

// Done returns a channel that is closed when the connection completes.
func (c *BidiConnection[StreamIn, StreamOut, Out]) Done() <-chan struct{} {
	return c.doneCh
}
