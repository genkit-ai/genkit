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
	"errors"
	"iter"
	"maps"
	"sync"
	"time"

	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/tracing"
	"github.com/firebase/genkit/go/internal/base"
)

// BidiFunc is the function signature for bidirectional streaming actions.
// It receives an initial configuration of type Init, reads incoming stream
// messages of type In from inCh, and writes outgoing stream messages of type
// Stream to outCh. It returns a final output of type Out when complete.
//
// The function must honor ctx cancellation: the framework signals shutdown
// (consumer error, invalid inbound chunk on the JSON transport, session
// cancellation) by cancelling ctx, and a function that ignores it blocks its
// session indefinitely. The framework owns closing outCh; the function must
// never close it. Writes to outCh apply backpressure: they block until the
// consumer reads earlier chunks. A panic in the function is recovered and
// reported as an INTERNAL error rather than crashing the process, since the
// function runs in a framework-owned goroutine.
//
// Experimental: bidirectional streaming is experimental and subject to change.
type BidiFunc[In, Out, Stream, Init any] = func(ctx context.Context, init Init, inCh <-chan In, outCh chan<- Stream) (Out, error)

// A BidiAction is a named, observable bidirectional streaming operation. It
// receives an initial configuration of type Init when a session starts, then
// consumes a stream of In messages while producing a stream of Stream chunks,
// and finishes with a final output of type Out.
//
// BidiAction embeds [Action], so it can also be invoked through the regular
// unary surface (Run, RunJSON): the input is delivered as a single chunk on
// the input stream with the zero Init value. Use [BidiAction.RunBidi] or
// [BidiAction.RunBidiJSON] for one-shot calls that supply init.
//
// For internal use only.
//
// Experimental: bidirectional streaming is experimental and subject to change.
type BidiAction[In, Out, Stream, Init any] struct {
	*Action[In, Out, Stream]
	bidiFn BidiFunc[In, Out, Stream, Init]
}

// BidiActionOptions configures a bidi action. Nil schema fields are inferred
// from the corresponding type parameters.
//
// Experimental: bidirectional streaming is experimental and subject to change.
type BidiActionOptions struct {
	Metadata     map[string]any // Arbitrary key-value data attached to the action descriptor.
	InputSchema  map[string]any // JSON schema for messages streamed into the action. Inferred from In if nil.
	OutputSchema map[string]any // JSON schema for the action's final output. Inferred from Out if nil.
	StreamSchema map[string]any // JSON schema for outgoing streamed chunks. Inferred from Stream if nil.
	InitSchema   map[string]any // JSON schema for the session's initial configuration. Inferred from Init if nil.
}

// NewBidiAction creates a new bidirectional streaming [BidiAction] without registering it.
//
// Experimental: bidirectional streaming is experimental and subject to change.
func NewBidiAction[In, Out, Stream, Init any](
	name string,
	atype api.ActionType,
	opts *BidiActionOptions,
	fn BidiFunc[In, Out, Stream, Init],
) *BidiAction[In, Out, Stream, Init] {
	if opts == nil {
		opts = &BidiActionOptions{}
	}

	metadata := make(map[string]any, len(opts.Metadata)+1)
	maps.Copy(metadata, opts.Metadata)
	metadata["bidi"] = true

	b := &BidiAction[In, Out, Stream, Init]{
		Action: newAction[In, Out, Stream](name, atype, metadata, opts.InputSchema),
		bidiFn: fn,
	}
	// The embedded action's fn backs the promoted unary surface (Run,
	// RunJSON): a one-shot session with the zero Init value.
	b.Action.fn = b.oneShotFn(base.Zero[Init]())

	if opts.OutputSchema != nil {
		b.desc.OutputSchema = opts.OutputSchema
	}
	if opts.StreamSchema != nil {
		b.desc.StreamSchema = opts.StreamSchema
	}

	if opts.InitSchema != nil {
		b.desc.InitSchema = opts.InitSchema
	} else if !isUnitType[Init]() {
		b.desc.InitSchema = inferSchema[Init]()
	}

	return b
}

// DefineBidiAction creates and registers a bidirectional streaming [BidiAction].
//
// Experimental: bidirectional streaming is experimental and subject to change.
func DefineBidiAction[In, Out, Stream, Init any](
	r api.Registry,
	name string,
	atype api.ActionType,
	opts *BidiActionOptions,
	fn BidiFunc[In, Out, Stream, Init],
) *BidiAction[In, Out, Stream, Init] {
	b := NewBidiAction(name, atype, opts, fn)
	b.Register(r)
	return b
}

// Register registers the bidi action with the given registry. It overrides
// the embedded Action's Register so that the registry holds the BidiAction
// itself; registry lookups must satisfy api.BidiAction.
func (b *BidiAction[In, Out, Stream, Init]) Register(r api.Registry) {
	b.Action.registry = r
	r.RegisterAction(b.desc.Key, b)
}

// oneShotFn adapts the bidi function into a single streaming call with the
// given init: the call's input becomes the only chunk on the input stream and
// outgoing chunks are forwarded to cb. The call's span and metrics come from
// the unary path (runWithTelemetry), so unlike startBidi the connection runs
// the function bare. Init is validated inside the call so that validation
// failures are recorded on the action's trace span, like input validation
// failures.
func (b *BidiAction[In, Out, Stream, Init]) oneShotFn(init Init) StreamingFunc[In, Out, Stream] {
	return func(ctx context.Context, input In, cb StreamCallback[Stream]) (Out, error) {
		if err := b.validateInit(init); err != nil {
			return base.Zero[Out](), err
		}

		conn := newBidiConnection[In, Out, Stream](ctx)
		// Released on every exit path, including a panicking callback below:
		// an unwinding panic must not strand the function goroutine blocked
		// on a stream write with no consumer. A cause recorded earlier (cb
		// error) wins; the first cancel is sticky.
		defer conn.cancel(nil)
		go conn.run(b.desc.Name, func(ctx context.Context) (Out, error) {
			return callBidiFn(ctx, b.desc.Name, b.bidiFn, init, conn.inputCh, conn.streamCh)
		})

		// inputCh is buffered, so delivering the single input cannot block.
		conn.inputCh <- input
		close(conn.inputCh)

		// Drain the stream until the function returns so its goroutine is
		// never blocked on a stream write, even after a callback failure
		// cancels the function's context.
		var cbErr error
		for chunk := range conn.streamCh {
			if cb == nil || cbErr != nil {
				continue
			}
			if err := cb(ctx, chunk); err != nil {
				cbErr = err
				conn.cancel(err)
			}
		}
		<-conn.doneCh
		if cbErr != nil {
			return base.Zero[Out](), cbErr
		}
		conn.mu.Lock()
		defer conn.mu.Unlock()
		return conn.output, conn.err
	}
}

// spanInitValue returns the value to record as the span's genkit:init
// attribute, or nil when Init is the no-init sentinel.
func (b *BidiAction[In, Out, Stream, Init]) spanInitValue(init Init) any {
	if isUnitType[Init]() {
		return nil
	}
	return init
}

// RunBidi executes the bidi action as a single one-shot call with the
// given initial configuration: input is delivered as the only chunk on the
// input stream and outgoing chunks are forwarded to cb. Returns an error if
// init fails validation against the action's InitSchema.
//
// Experimental: bidirectional streaming is experimental and subject to change.
func (b *BidiAction[In, Out, Stream, Init]) RunBidi(ctx context.Context, init Init, input In, cb StreamCallback[Stream]) (Out, error) {
	r, err := b.Action.runWithTelemetry(ctx, input, cb, b.oneShotFn(init), b.spanInitValue(init))
	if err != nil {
		return base.Zero[Out](), err
	}
	return r.Result, nil
}

// RunBidiJSON runs the bidi action as a single one-shot call: input is
// delivered as the only chunk on the input stream, outgoing chunks are
// forwarded to cb, and opts carries the session init. Returns an error if
// init fails to decode or validate.
//
// Experimental: bidirectional streaming is experimental and subject to change.
func (b *BidiAction[In, Out, Stream, Init]) RunBidiJSON(ctx context.Context, input json.RawMessage, cb StreamCallback[json.RawMessage], opts *api.BidiJSONOptions) (*api.ActionRunResult[json.RawMessage], error) {
	init, hasInit, err := b.decodeInit(opts)
	if err != nil {
		return nil, err
	}
	var spanInit any
	if hasInit {
		spanInit = init
	}
	return b.Action.runJSONWithTelemetry(ctx, input, cb, b.oneShotFn(init), spanInit)
}

// Connect starts a bidirectional streaming connection with the given
// initial configuration. For actions whose Init type is struct{} (no init),
// pass struct{}{}. Returns an error if init fails validation against the
// action's InitSchema.
// A trace span is created that remains open for the lifetime of the connection.
//
// Experimental: bidirectional streaming is experimental and subject to change.
func (b *BidiAction[In, Out, Stream, Init]) Connect(ctx context.Context, init Init) (*BidiConnection[In, Out, Stream], error) {
	if err := b.validateInit(init); err != nil {
		return nil, err
	}
	return b.startBidi(ctx, init, b.spanInitValue(init)), nil
}

// ConnectJSON starts a bidirectional streaming session using JSON-encoded
// messages. Returns an error if the init carried by opts fails to decode or
// validate.
//
// Experimental: bidirectional streaming is experimental and subject to change.
func (b *BidiAction[In, Out, Stream, Init]) ConnectJSON(ctx context.Context, opts *api.BidiJSONOptions) (api.BidiJSONConnection, error) {
	init, hasInit, err := b.decodeInit(opts)
	if err != nil {
		return nil, err
	}
	if err := b.validateInit(init); err != nil {
		return nil, err
	}
	inputSchema, err := ResolveSchema(b.registry, b.desc.InputSchema)
	if err != nil {
		return nil, NewError(INVALID_ARGUMENT, "invalid input schema for action %q: %v", b.desc.Key, err)
	}
	// Compiled once per session: Send validates every inbound chunk, and
	// recompiling the schema per chunk would dominate the streaming hot path.
	compiledInput, err := base.CompileSchema(inputSchema)
	if err != nil {
		return nil, NewError(INVALID_ARGUMENT, "invalid input schema for action %q: %v", b.desc.Key, err)
	}
	// Like RunBidiJSON, record init on the span only when the client actually
	// supplied one; the zero value from an absent init is not meaningful.
	var spanInit any
	if hasInit {
		spanInit = init
	}
	conn := b.startBidi(ctx, init, spanInit)
	return &bidiJSONConn[In, Out, Stream]{
		conn:          conn,
		key:           b.desc.Key,
		inputSchema:   inputSchema,
		compiledInput: compiledInput,
	}, nil
}

// decodeInit decodes the JSON init payload from opts into the action's Init
// type. Returns hasInit=false when opts is nil or the payload is empty or
// JSON null, so transports can pass the request's init field through
// unconditionally.
func (b *BidiAction[In, Out, Stream, Init]) decodeInit(opts *api.BidiJSONOptions) (Init, bool, error) {
	var init Init
	if opts == nil || !base.HasJSONValue(opts.Init) {
		return init, false, nil
	}
	if err := json.Unmarshal(opts.Init, &init); err != nil {
		return init, false, NewError(INVALID_ARGUMENT, "invalid init for action %q: %v", b.desc.Key, err)
	}
	return init, true, nil
}

// validateInit checks an init value against the action's InitSchema (if any),
// resolving schema $refs through the registry first. Validation runs whenever
// InitSchema is present, even for the zero init value, so a required field
// surfaces as INVALID_ARGUMENT rather than silently defaulting.
func (b *BidiAction[In, Out, Stream, Init]) validateInit(init Init) error {
	if b.desc.InitSchema == nil {
		return nil
	}
	schema, err := ResolveSchema(b.registry, b.desc.InitSchema)
	if err != nil {
		return NewError(INVALID_ARGUMENT, "invalid init schema for action %q: %v", b.desc.Key, err)
	}
	if err := base.ValidateValue(init, schema); err != nil {
		return NewError(INVALID_ARGUMENT, "invalid init for action %q: %v", b.desc.Key, err)
	}
	return nil
}

// startBidi launches the bidi function in a goroutine with the given initial
// configuration and returns a live connection for sending/receiving chunks.
// The session gets its own span (open for the connection's lifetime) and
// metrics, unlike the one-shot path, which gets both from runWithTelemetry.
// spanInit, when non-nil, is recorded as the span's genkit:init attribute.
func (b *BidiAction[In, Out, Stream, Init]) startBidi(ctx context.Context, init Init, spanInit any) *BidiConnection[In, Out, Stream] {
	conn := newBidiConnection[In, Out, Stream](ctx)

	// Init is recorded as its own span attribute (genkit:init), not as the
	// span input: the input slot describes per-call input, which a bidi
	// session receives incrementally over the connection.
	spanMetadata := b.spanMetadata(ctx, spanInit)

	go conn.run(b.desc.Name, func(ctx context.Context) (Out, error) {
		return tracing.RunInNewSpan(ctx, spanMetadata, nil,
			func(ctx context.Context, _ any) (out Out, err error) {
				start := time.Now()
				defer func() { recordActionMetrics(ctx, b.desc.Name, start, err) }()
				out, err = callBidiFn(ctx, b.desc.Name, b.bidiFn, init, conn.inputCh, conn.streamCh)
				if err != nil {
					return out, err
				}
				// Mirror the unary path: the final output is validated
				// against the action's OutputSchema.
				outputSchema, err := b.resolveOutputSchema()
				if err != nil {
					return out, err
				}
				return out, b.validateOutput(out, outputSchema)
			},
		)
	})

	return conn
}

// ResolveBidiActionFor returns the bidi action for the given name in the
// registry, or nil if there is none.
// It panics if the action is of the wrong type; plain actions resolve via
// [ResolveActionFor].
//
// Experimental: bidirectional streaming is experimental and subject to change.
func ResolveBidiActionFor[In, Out, Stream, Init any](r api.Registry, atype api.ActionType, name string) *BidiAction[In, Out, Stream, Init] {
	provider, id := api.ParseName(name)
	key := api.NewKey(atype, provider, id)
	a := r.ResolveAction(key)
	if a == nil {
		return nil
	}
	return a.(*BidiAction[In, Out, Stream, Init])
}

// callBidiFn invokes the bidi function, converting a panic into an INTERNAL
// error. The function runs in a framework-owned goroutine, so an unrecovered
// panic would crash the process rather than fail the session.
func callBidiFn[In, Out, Stream, Init any](
	ctx context.Context,
	name string,
	fn BidiFunc[In, Out, Stream, Init],
	init Init,
	inCh <-chan In,
	outCh chan<- Stream,
) (out Out, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = NewError(INTERNAL, "panic in bidi action %q: %v", name, r)
		}
	}()
	return fn(ctx, init, inCh, outCh)
}

// BidiConnection represents an active bidirectional streaming session.
//
// The connection applies backpressure: the action blocks writing a chunk
// until the consumer reads earlier ones, so a session that streams more than
// one chunk requires the caller to drain [BidiConnection.Receive] before (or
// concurrently with) waiting on [BidiConnection.Output].
//
// Experimental: bidirectional streaming is experimental and subject to change.
type BidiConnection[In, Out, Stream any] struct {
	inputCh  chan In
	streamCh chan Stream
	doneCh   chan struct{}
	output   Out
	err      error
	ctx      context.Context
	cancel   context.CancelCauseFunc
	mu       sync.Mutex
	closed   bool
}

// newBidiConnection creates an idle connection whose context derives from ctx.
// The caller must start exactly one [BidiConnection.run] goroutine to operate
// it. The context carries a cancel cause so that an abort reason (e.g. an
// invalid inbound chunk poisoning the session) survives to Send/Receive/Output
// instead of flattening to context.Canceled.
func newBidiConnection[In, Out, Stream any](ctx context.Context) *BidiConnection[In, Out, Stream] {
	ctx, cancel := context.WithCancelCause(ctx)
	return &BidiConnection[In, Out, Stream]{
		inputCh:  make(chan In, 1),
		streamCh: make(chan Stream, 1),
		doneCh:   make(chan struct{}),
		ctx:      ctx,
		cancel:   cancel,
	}
}

// ctxErr returns the reason the connection's context was cancelled, preferring
// the recorded cause over the bare context error. Only meaningful once the
// context is done.
func (c *BidiConnection[In, Out, Stream]) ctxErr() error {
	if cause := context.Cause(c.ctx); cause != nil {
		return cause
	}
	return c.ctx.Err()
}

// run executes fn, which reads c.inputCh and writes c.streamCh, then records
// its result and settles the connection. It must be called exactly once, in
// its own goroutine. fn receives the connection's context and must honor its
// cancellation; convert panics inside fn with callBidiFn.
func (c *BidiConnection[In, Out, Stream]) run(name string, fn func(context.Context) (Out, error)) {
	// Deferred calls run in reverse order: the stream channel closes first,
	// then doneCh signals completion, then the connection context is
	// released. Receive/Output rely on this ordering to prefer delivering
	// results over reporting cancellation.
	defer c.cancel(nil)
	defer close(c.doneCh)
	closingStream := false
	defer func() {
		if r := recover(); r != nil {
			c.mu.Lock()
			if c.err == nil {
				if closingStream {
					// The close below panicked: the action closed the output
					// channel itself, which the framework owns.
					c.err = NewError(INTERNAL, "bidi action %q closed its output channel; the framework owns closing it", name)
				} else {
					// A panic escaped fn's own wrapping (span, schema
					// resolution, metrics); report it as what it is rather
					// than misattributing it to the channel close.
					c.err = NewError(INTERNAL, "panic in bidi session %q: %v", name, r)
				}
			}
			c.mu.Unlock()
		}
	}()
	defer func() {
		// closingStream brackets the close so the recover above can tell a
		// double-close panic apart from one unwinding out of fn.
		closingStream = true
		close(c.streamCh)
		closingStream = false
	}()
	output, err := fn(c.ctx)
	// An abort recorded a cause (invalid inbound chunk, failed stream
	// marshal, callback error): that cause is the session's terminal error.
	// It overrides a nil error from a function that never observed the
	// cancellation and the bare Canceled it unwound with, but not a distinct
	// error the function chose to report.
	if cause := context.Cause(c.ctx); cause != nil && !errors.Is(cause, context.Canceled) {
		if err == nil || errors.Is(err, context.Canceled) {
			err = cause
		}
	}
	c.mu.Lock()
	c.output = output
	c.err = err
	c.mu.Unlock()
}

// Send sends an input message to the bidi action. It blocks until the action
// reads the message (backpressure), the connection is cancelled, or the
// action completes. Returns an error if the connection is closed or the
// context is cancelled. Typed inputs are not re-validated against the
// action's InputSchema; the JSON transport path is.
func (c *BidiConnection[In, Out, Stream]) Send(input In) (err error) {
	// Close may close inputCh concurrently with this send; sending on a
	// closed channel panics, and the recover converts that into the same
	// "connection is closed" error a pre-checked Send would return.
	defer func() {
		if r := recover(); r != nil {
			err = NewError(FAILED_PRECONDITION, "connection is closed")
		}
	}()

	// A completed or aborted connection must fail deterministically: in the
	// blocking select below all arms can be ready at once (inputCh keeps a
	// free buffer slot once the action exits) and the runtime picks one at
	// random, which would let a post-completion Send "succeed" into a
	// channel nothing reads. Like Output, completion is preferred over
	// cancellation.
	select {
	case <-c.doneCh:
		return NewError(FAILED_PRECONDITION, "action has completed")
	default:
	}
	select {
	case <-c.ctx.Done():
		return c.ctxErr()
	default:
	}

	select {
	case c.inputCh <- input:
		return nil
	case <-c.ctx.Done():
		return c.ctxErr()
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
//
// Breaking out of the loop stops consumption but does not abort the session:
// the action keeps running and later chunks remain subject to backpressure
// until Receive is iterated again or the session ends. Use
// [BidiConnection.Cancel] to abort the session. Chunks are delivered to a
// single consumer; concurrent Receive iterations split the stream between
// them.
func (c *BidiConnection[In, Out, Stream]) Receive() iter.Seq2[Stream, error] {
	return func(yield func(Stream, error) bool) {
		for {
			select {
			case chunk, ok := <-c.streamCh:
				if !ok {
					return
				}
				if !yield(chunk, nil) {
					return
				}
			case <-c.ctx.Done():
				// Completion closes the stream channel before releasing the
				// connection context, so prefer delivering chunks (and the
				// clean end of stream) over reporting cancellation.
				for {
					select {
					case chunk, ok := <-c.streamCh:
						if !ok {
							return
						}
						if !yield(chunk, nil) {
							return
						}
					default:
						var zero Stream
						yield(zero, c.ctxErr())
						return
					}
				}
			}
		}
	}
}

// Output returns the final output after the action completes.
// Blocks until done or context cancelled. If the action streams more than
// one chunk, [BidiConnection.Receive] must be drained for the action to
// finish; see the BidiConnection doc.
func (c *BidiConnection[In, Out, Stream]) Output() (Out, error) {
	select {
	case <-c.doneCh:
	case <-c.ctx.Done():
		// Completion closes doneCh before releasing the connection context,
		// but both may be ready when this select runs; prefer the result.
		select {
		case <-c.doneCh:
		default:
			var zero Out
			return zero, c.ctxErr()
		}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.output, c.err
}

// Cancel aborts the session by cancelling the connection's context: the
// action's context is cancelled, blocked Sends unblock, and Output reports
// the cancellation error unless the action already completed. Safe to call
// multiple times and after completion.
func (c *BidiConnection[In, Out, Stream]) Cancel() {
	c.cancel(nil)
}

// Done returns a channel that is closed when the connection completes.
func (c *BidiConnection[In, Out, Stream]) Done() <-chan struct{} {
	return c.doneCh
}

// bidiJSONConn adapts a typed BidiConnection to the JSON-encoded
// api.BidiJSONConnection interface.
type bidiJSONConn[In, Out, Stream any] struct {
	conn          *BidiConnection[In, Out, Stream]
	key           string               // action key, for error messages
	inputSchema   map[string]any       // resolved InputSchema, used for chunk normalization
	compiledInput *base.CompiledSchema // inputSchema compiled once; every inbound chunk validates against it
}

func (b *bidiJSONConn[In, Out, Stream]) Send(chunk json.RawMessage) error {
	// Mirrors the unary RunJSON path: normalize and validate every inbound
	// chunk against the action's input schema, since JSON transports carry
	// untrusted payloads. An explicit JSON null is validated like any other
	// payload.
	in, err := base.UnmarshalAndNormalizeWith[In](chunk, b.inputSchema, b.compiledInput)
	if err == nil && len(chunk) == 0 {
		// UnmarshalAndNormalizeWith skips validation entirely for empty
		// input; validate the zero value it produced so an absent chunk
		// payload cannot bypass a schema the zero value does not satisfy,
		// mirroring the unary path, which validates the decoded value.
		err = b.compiledInput.ValidateValue(in)
	}
	if err != nil {
		// An invalid chunk fails the session (matching the JS runtime and
		// the one-shot path, where invalid input fails the call): the error
		// poisons the connection as its cancel cause so Output reports it,
		// and is also returned for the transport to log or relay.
		err = NewError(INVALID_ARGUMENT, "invalid stream chunk for action %q: %v", b.key, err)
		b.conn.cancel(err)
		return err
	}
	return b.conn.Send(in)
}

func (b *bidiJSONConn[In, Out, Stream]) Close() error {
	return b.conn.Close()
}

func (b *bidiJSONConn[In, Out, Stream]) Receive() iter.Seq2[json.RawMessage, error] {
	return func(yield func(json.RawMessage, error) bool) {
		for chunk, err := range b.conn.Receive() {
			if err != nil {
				yield(nil, err)
				return
			}
			bytes, mErr := json.Marshal(chunk)
			if mErr != nil {
				// Later chunks of the same type would fail the same way,
				// leaving the session running with no consumer; abort it
				// with the marshal error as the cause so Output reports it.
				b.conn.cancel(mErr)
				yield(nil, mErr)
				return
			}
			if !yield(bytes, nil) {
				return
			}
		}
	}
}

func (b *bidiJSONConn[In, Out, Stream]) Output() (json.RawMessage, error) {
	out, err := b.conn.Output()
	if err != nil {
		return nil, err
	}
	return json.Marshal(out)
}

var (
	_ api.Action     = (*BidiAction[struct{}, struct{}, struct{}, struct{}])(nil)
	_ api.BidiAction = (*BidiAction[struct{}, struct{}, struct{}, struct{}])(nil)
)
