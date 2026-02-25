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

// StreamingFunc is an alias for streaming functions with input of type In, output of type Out, and streaming chunk of type Stream.
type StreamingFunc[In, Out, Stream any] = func(context.Context, In, StreamCallback[Stream]) (Out, error)

// StreamCallback is a function that is called during streaming to return the next chunk of the stream.
type StreamCallback[Stream any] = func(context.Context, Stream) error

// BidiFunc is the function signature for bidirectional streaming actions.
// It receives initialization data, reads inputs from inCh, and writes
// streamed outputs to outCh. It returns a final output when complete.
type BidiFunc[In, Out, Stream, Init any] = func(ctx context.Context, init Init, inCh <-chan In, outCh chan<- Stream) (Out, error)

// An Action is a named, observable operation that underlies all Genkit primitives.
// It consists of a function that takes an input of type I and returns an output
// of type O, optionally streaming values of type S incrementally by invoking a callback.
// It optionally has other metadata, like a description and JSON Schemas for its input and
// output which it validates against.
//
// Each time an Action is run, it results in a new trace span.
//
// For internal use only.
type Action[In, Out, Stream, Init any] struct {
	fn       StreamingFunc[In, Out, Stream]  // Function that is called during runtime. May not actually support streaming.
	bidiFn   BidiFunc[In, Out, Stream, Init] // Non-nil for bidi actions only.
	desc     *api.ActionDesc                 // Descriptor of the action.
	registry api.Registry                    // Registry for schema resolution. Set when registered.
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
func NewStreamingAction[In, Out, Stream any](
	name string,
	atype api.ActionType,
	metadata map[string]any,
	inputSchema map[string]any,
	fn StreamingFunc[In, Out, Stream],
) *Action[In, Out, Stream, struct{}] {
	return newAction[In, Out, Stream, struct{}](name, atype, metadata, inputSchema, fn)
}

// ActionOptions configures a bidi action. Nil schema fields are inferred from type parameters.
type ActionOptions struct {
	Metadata     map[string]any // Arbitrary key-value data attached to the action descriptor.
	InputSchema  map[string]any // JSON schema for the action's input. Inferred from In if nil.
	OutputSchema map[string]any // JSON schema for the action's output. Inferred from Out if nil.
	StreamSchema map[string]any // JSON schema for streamed chunks. Inferred from Stream if nil. Not used for non-streaming actions.
	InitSchema   map[string]any // JSON schema for bidi initialization data. Inferred from Init if nil. Not used for non-bidi actions.
}

// NewBidiAction creates a new bidirectional streaming [Action] without registering it.
func NewBidiAction[In, Out, Stream, Init any](
	name string,
	atype api.ActionType,
	opts *ActionOptions,
	fn BidiFunc[In, Out, Stream, Init],
) *Action[In, Out, Stream, Init] {
	if opts == nil {
		opts = &ActionOptions{}
	}

	metadata := opts.Metadata
	if metadata == nil {
		metadata = map[string]any{}
	}
	metadata["bidi"] = true

	a := newAction[In, Out, Stream, Init](name, atype, metadata, opts.InputSchema, wrapBidiAsStreaming(fn))
	a.bidiFn = fn

	if opts.OutputSchema != nil {
		a.desc.OutputSchema = opts.OutputSchema
	}
	if opts.StreamSchema != nil {
		a.desc.StreamSchema = opts.StreamSchema
	}

	if opts.InitSchema != nil {
		a.desc.InitSchema = opts.InitSchema
	} else {
		var init Init
		if reflect.ValueOf(init).Kind() != reflect.Invalid {
			a.desc.InitSchema = InferSchemaMap(init)
		}
	}

	return a
}

// DefineBidiAction creates and registers a bidirectional streaming [Action].
func DefineBidiAction[In, Out, Stream, Init any](
	r api.Registry,
	name string,
	atype api.ActionType,
	opts *ActionOptions,
	fn BidiFunc[In, Out, Stream, Init],
) *Action[In, Out, Stream, Init] {
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
func DefineStreamingAction[In, Out, Stream any](
	r api.Registry,
	name string,
	atype api.ActionType,
	metadata map[string]any,
	inputSchema map[string]any,
	fn StreamingFunc[In, Out, Stream],
) *Action[In, Out, Stream, struct{}] {
	return defineAction[In, Out, Stream, struct{}](r, name, atype, metadata, inputSchema, fn)
}

// defineAction creates an action and registers it with the given Registry.
func defineAction[In, Out, Stream, Init any](
	r api.Registry,
	name string,
	atype api.ActionType,
	metadata map[string]any,
	inputSchema map[string]any,
	fn StreamingFunc[In, Out, Stream],
) *Action[In, Out, Stream, Init] {
	a := newAction[In, Out, Stream, Init](name, atype, metadata, inputSchema, fn)
	a.Register(r)
	return a
}

// newAction creates a new Action with the given name and arguments.
// If registry is nil, tracing state is left nil to be set later.
// If inputSchema is nil, it is inferred from In.
func newAction[In, Out, Stream, Init any](
	name string,
	atype api.ActionType,
	metadata map[string]any,
	inputSchema map[string]any,
	fn StreamingFunc[In, Out, Stream],
) *Action[In, Out, Stream, Init] {
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

	var s Stream
	var streamSchema map[string]any
	if reflect.ValueOf(s).Kind() != reflect.Invalid {
		streamSchema = InferSchemaMap(s)
	}

	var description string
	if desc, ok := metadata["description"].(string); ok {
		description = desc
	}

	return &Action[In, Out, Stream, Init]{
		fn: func(ctx context.Context, input In, cb StreamCallback[Stream]) (Out, error) {
			return fn(ctx, input, cb)
		},
		desc: &api.ActionDesc{
			Type:         atype,
			Key:          api.KeyFromName(atype, name),
			Name:         name,
			Description:  description,
			InputSchema:  inputSchema,
			OutputSchema: outputSchema,
			StreamSchema: streamSchema,
			Metadata:     metadata,
		},
	}
}

// Name returns the Action's Name.
func (a *Action[In, Out, Stream, Init]) Name() string { return a.desc.Name }

// Run executes the Action's function in a new trace span.
func (a *Action[In, Out, Stream, Init]) Run(ctx context.Context, input In, cb StreamCallback[Stream]) (output Out, err error) {
	r, err := a.runWithTelemetry(ctx, input, cb)
	if err != nil {
		return base.Zero[Out](), err
	}
	return r.Result, nil
}

// runWithTelemetry executes the Action's function in a new trace span and returns telemetry info.
func (a *Action[In, Out, Stream, Init]) runWithTelemetry(ctx context.Context, input In, cb StreamCallback[Stream]) (output api.ActionRunResult[Out], err error) {
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
func (a *Action[In, Out, Stream, Init]) RunJSON(ctx context.Context, input json.RawMessage, cb StreamCallback[json.RawMessage]) (json.RawMessage, error) {
	r, err := a.RunJSONWithTelemetry(ctx, input, cb)
	if err != nil {
		return nil, err
	}
	return r.Result, nil
}

// RunJSONWithTelemetry runs the action with a JSON input, and returns a JSON result along with telemetry info.
func (a *Action[In, Out, Stream, Init]) RunJSONWithTelemetry(ctx context.Context, input json.RawMessage, cb StreamCallback[json.RawMessage]) (*api.ActionRunResult[json.RawMessage], error) {
	i, err := base.UnmarshalAndNormalize[In](input, a.desc.InputSchema)
	if err != nil {
		return nil, NewError(INVALID_ARGUMENT, err.Error())
	}

	var scb StreamCallback[Stream]
	if cb != nil {
		scb = func(ctx context.Context, s Stream) error {
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
func (a *Action[In, Out, Stream, Init]) Desc() api.ActionDesc {
	return *a.desc
}

// Register registers the action with the given registry.
func (a *Action[In, Out, Stream, Init]) Register(r api.Registry) {
	a.registry = r
	r.RegisterAction(a.desc.Key, a)
}

// StreamBidi starts a bidirectional streaming connection.
// Returns an error if the action is not a bidi action.
// A trace span is created that remains open for the lifetime of the connection.
func (a *Action[In, Out, Stream, Init]) StreamBidi(ctx context.Context, init Init) (*BidiConnection[In, Out, Stream], error) {
	if a.bidiFn == nil {
		return nil, NewError(FAILED_PRECONDITION, "StreamBidi called on non-bidi action %q", a.desc.Name)
	}

	ctx, cancel := context.WithCancel(ctx)
	conn := &BidiConnection[In, Out, Stream]{
		inputCh:  make(chan In, 1),
		streamCh: make(chan Stream, 1),
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
		output, err := tracing.RunInNewSpan(conn.ctx, spanMetadata, init,
			func(ctx context.Context, init Init) (Out, error) {
				start := time.Now()
				output, err := a.bidiFn(ctx, init, conn.inputCh, conn.streamCh)
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
func ResolveActionFor[In, Out, Stream, Init any](r api.Registry, atype api.ActionType, name string) *Action[In, Out, Stream, Init] {
	provider, id := api.ParseName(name)
	key := api.NewKey(atype, provider, id)
	a := r.ResolveAction(key)
	if a == nil {
		return nil
	}
	return a.(*Action[In, Out, Stream, Init])
}

// LookupActionFor returns the action for the given key in the global registry,
// or nil if there is none.
// It panics if the action is of the wrong api.
//
// Deprecated: Use ResolveActionFor.
func LookupActionFor[In, Out, Stream, Init any](r api.Registry, atype api.ActionType, name string) *Action[In, Out, Stream, Init] {
	provider, id := api.ParseName(name)
	key := api.NewKey(atype, provider, id)
	a := r.LookupAction(key)
	if a == nil {
		return nil
	}
	return a.(*Action[In, Out, Stream, Init])
}

// wrapBidiAsStreaming wraps a BidiFunc into a StreamingFunc for use with Run/RunJSON.
// The input is sent as a single message, and stream chunks are forwarded to the callback.
func wrapBidiAsStreaming[In, Out, Stream, Init any](fn BidiFunc[In, Out, Stream, Init]) StreamingFunc[In, Out, Stream] {
	return func(ctx context.Context, input In, cb StreamCallback[Stream]) (Out, error) {
		inCh := make(chan In, 1)
		outCh := make(chan Stream, 1)
		doneCh := make(chan struct{})

		var output Out
		var fnErr error

		go func() {
			defer close(doneCh)
			defer close(outCh)
			var init Init
			output, fnErr = fn(ctx, init, inCh, outCh)
		}()

		// Send the single input and close.
		inCh <- input
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
type BidiConnection[In, Out, Stream any] struct {
	inputCh  chan In
	streamCh chan Stream
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
func (c *BidiConnection[In, Out, Stream]) Send(input In) (err error) {
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
func (c *BidiConnection[In, Out, Stream]) Close() error {
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
func (c *BidiConnection[In, Out, Stream]) Receive() iter.Seq2[Stream, error] {
	return func(yield func(Stream, error) bool) {
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
				var zero Stream
				yield(zero, c.ctx.Err())
				return
			}
		}
	}
}

// Output returns the final output after the action completes.
// Blocks until done or context cancelled.
func (c *BidiConnection[In, Out, Stream]) Output() (Out, error) {
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
func (c *BidiConnection[In, Out, Stream]) Done() <-chan struct{} {
	return c.doneCh
}
