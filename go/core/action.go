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

// StreamingFunc is an alias for streaming functions with input of type In, output of type Out, and outgoing stream chunk of type Stream.
type StreamingFunc[In, Out, Stream any] = func(context.Context, In, StreamCallback[Stream]) (Out, error)

// StreamCallback is a function that is called during streaming to return the next chunk of the outgoing stream.
type StreamCallback[Stream any] = func(context.Context, Stream) error

// An Action is a named, observable operation that underlies all Genkit primitives.
// It consists of a function that takes an input of type In and returns an output
// of type Out, optionally streaming values of type Stream incrementally by
// invoking a callback.
//
// It optionally has other metadata, like a description and JSON Schemas for its input and
// output which it validates against.
//
// Each time an Action is run, it results in a new trace span.
//
// For internal use only.
type Action[In, Out, Stream any] struct {
	fn   StreamingFunc[In, Out, Stream] // Function that is called during runtime. May not actually support streaming.
	desc *api.ActionDesc                // Descriptor of the action. Immutable after construction.
	// state is the action's mutable state, held by pointer so that copies of
	// the Action value (several primitives embed it by value) share one lock
	// and one memo rather than copying the lock.
	state *actionState
}

// actionState is the mutable state behind an Action, shared by all copies of
// the Action value.
type actionState struct {
	mu       sync.Mutex
	registry api.Registry // Registry for schema resolution. Set when registered.
	// resolved memoizes successfully resolved named-schema references by role
	// ("input", "output", "stream", "init"). Sound because a registry's
	// schemas are immutable once defined (RegisterSchema panics on
	// redefinition); cleared when the action registers with a (possibly
	// different) registry.
	resolved map[string]map[string]any
}

// action is an unexported alias of [Action] used as the embedded field in the
// types built on top of it ([Flow], [BidiAction], [BackgroundAction]).
// Embedding via the alias promotes Action's methods without exporting the
// field itself, so the containment stays an internal detail: callers can't
// reach the inner Action, construct the outer type by struct literal, or
// depend on its layout, leaving us free to restructure it later without
// breaking the API.
type action[In, Out, Stream any] = Action[In, Out, Stream]

// ActionOptions configures the optional attributes of an [Action]. A nil
// options value is valid: schemas are inferred from the action's type
// parameters and the descriptor carries no metadata.
type ActionOptions struct {
	// Description is a human-readable description of the action. When empty,
	// Metadata["description"] is used if present.
	Description string
	// Metadata is arbitrary key-value data attached to the action descriptor.
	Metadata map[string]any
	// InputSchema is the JSON schema for the action's input. Inferred from In if nil.
	InputSchema map[string]any
	// OutputSchema is the JSON schema for the action's output. Inferred from Out if nil.
	OutputSchema map[string]any
	// StreamSchema is the JSON schema for outgoing stream chunks. Inferred
	// from Stream if nil; non-streaming actions advertise none.
	StreamSchema map[string]any
}

type noStream = func(context.Context, struct{}) error

// NewAction creates a new non-streaming [Action] without registering it.
func NewAction[In, Out any](
	atype api.ActionType,
	name string,
	opts *ActionOptions,
	fn Func[In, Out],
) *Action[In, Out, struct{}] {
	return NewStreamingAction(atype, name, opts,
		func(ctx context.Context, in In, _ noStream) (Out, error) {
			return fn(ctx, in)
		})

}

// NewStreamingAction creates a new streaming [Action] without registering it.
func NewStreamingAction[In, Out, Stream any](
	atype api.ActionType,
	name string,
	opts *ActionOptions,
	fn StreamingFunc[In, Out, Stream],
) *Action[In, Out, Stream] {
	a := newAction[In, Out, Stream](atype, name, opts)
	a.fn = fn
	return a
}

// DefineAction creates a new non-streaming [Action] and registers it.
func DefineAction[In, Out any](
	r api.Registry,
	atype api.ActionType,
	name string,
	opts *ActionOptions,
	fn Func[In, Out],
) *Action[In, Out, struct{}] {
	a := NewAction(atype, name, opts, fn)
	a.Register(r)
	return a
}

// DefineStreamingAction creates a new streaming [Action] and registers it.
func DefineStreamingAction[In, Out, Stream any](
	r api.Registry,
	atype api.ActionType,
	name string,
	opts *ActionOptions,
	fn StreamingFunc[In, Out, Stream],
) *Action[In, Out, Stream] {
	a := NewStreamingAction(atype, name, opts, fn)
	a.Register(r)
	return a
}

// newAction builds an Action's descriptor from opts, inferring any schema not
// explicitly provided. The caller is expected to assign a.fn.
func newAction[In, Out, Stream any](atype api.ActionType, name string, opts *ActionOptions) *Action[In, Out, Stream] {
	if opts == nil {
		opts = &ActionOptions{}
	}

	description := opts.Description
	if description == "" {
		if d, ok := opts.Metadata["description"].(string); ok {
			description = d
		}
	}

	return &Action[In, Out, Stream]{
		desc: &api.ActionDesc{
			Type:         atype,
			Key:          api.KeyFromName(atype, name),
			Name:         name,
			Description:  description,
			InputSchema:  schemaFor[In](opts.InputSchema, false),
			OutputSchema: schemaFor[Out](opts.OutputSchema, false),
			// Stream is struct{} for non-streaming actions; inferring a schema
			// from the sentinel would make every action advertise a bogus
			// streamSchema.
			StreamSchema: schemaFor[Stream](opts.StreamSchema, true),
			Metadata:     opts.Metadata,
		},
		state: &actionState{},
	}
}

// schemaFor returns the JSON schema describing values of type T: the explicit
// override when non-nil, otherwise a schema inferred from T. When
// unitMeansNone is true, the struct{} sentinel type ("no value") yields no
// schema at all; the stream and init slots use it so that actions without
// streaming or init don't advertise a schema for the sentinel.
func schemaFor[T any](override map[string]any, unitMeansNone bool) map[string]any {
	if override != nil {
		return override
	}
	if unitMeansNone && isUnitType[T]() {
		return nil
	}
	return inferSchema[T]()
}

// isUnitType reports whether T is exactly struct{}, the sentinel type
// parameter meaning "no value" (no stream chunks, no init). Named empty
// struct types do not match and are treated as real types.
func isUnitType[T any]() bool {
	return reflect.TypeFor[T]() == reflect.TypeFor[struct{}]()
}

// isNilValue reports whether v is nil or a nil pointer, map, slice, or
// interface: a value that marshals to JSON null and carries nothing to
// validate against a schema.
func isNilValue(v any) bool {
	if v == nil {
		return true
	}
	switch rv := reflect.ValueOf(v); rv.Kind() {
	case reflect.Pointer, reflect.Map, reflect.Slice, reflect.Interface:
		return rv.IsNil()
	default:
		return false
	}
}

// inferSchema returns the JSON schema inferred from T's zero value, or nil
// for interface types, whose zero value carries no type information to infer
// from.
func inferSchema[T any]() map[string]any {
	var v T
	if reflect.ValueOf(v).Kind() == reflect.Invalid {
		return nil
	}
	return InferSchemaMap(v)
}

// Name returns the Action's Name.
func (a *Action[In, Out, Stream]) Name() string { return a.desc.Name }

// Run executes the Action's function in a new trace span.
func (a *Action[In, Out, Stream]) Run(ctx context.Context, input In, cb StreamCallback[Stream]) (output Out, err error) {
	r, err := a.runWithTelemetry(ctx, input, cb, a.fn, nil)
	if err != nil {
		return base.Zero[Out](), err
	}
	return r.Result, nil
}

// runWithTelemetry executes fn in a new trace span and returns telemetry
// info. fn is a parameter (rather than always a.fn) so that BidiAction can
// inject a per-call one-shot adapter; spanInit, when non-nil, is recorded as
// the span's genkit:init attribute.
func (a *Action[In, Out, Stream]) runWithTelemetry(ctx context.Context, input In, cb StreamCallback[Stream], fn StreamingFunc[In, Out, Stream], spanInit any) (output api.ActionRunResult[Out], err error) {
	logger.FromContext(ctx).Debug("Action.Run", "name", a.Name())
	defer func() {
		logger.FromContext(ctx).Debug("Action.Run",
			"name", a.Name(),
			"err", err)
	}()

	var traceID string
	var spanID string
	o, err := tracing.RunInNewSpan(ctx, a.spanMetadata(ctx, spanInit), input,
		func(ctx context.Context, input In) (output Out, err error) {
			traceInfo := tracing.SpanTraceInfo(ctx)
			traceID = traceInfo.TraceID
			spanID = traceInfo.SpanID

			start := time.Now()
			defer func() { recordActionMetrics(ctx, a.desc.Name, start, err) }()

			inputSchema, err := a.resolveSchema("input", a.desc.InputSchema)
			if err != nil {
				return base.Zero[Out](), err
			}

			outputSchema, err := a.resolveSchema("output", a.desc.OutputSchema)
			if err != nil {
				return base.Zero[Out](), err
			}

			if err = base.ValidateValue(input, inputSchema); err != nil {
				return base.Zero[Out](), NewSchemaValidationError(a.desc.Key, err)
			}

			output, err = fn(ctx, input, cb)
			if err != nil {
				return output, err
			}
			return output, a.validateOutput(output, outputSchema)
		},
	)

	return api.ActionRunResult[Out]{
		Result:  o,
		TraceId: traceID,
		SpanId:  spanID,
	}, err
}

// spanMetadata builds the trace span metadata for one run of this action,
// injecting the flow name when ctx carries one. spanInit, when non-nil, is
// recorded as the span's genkit:init attribute. IsRoot is determined later by
// the tracing package from parent span presence.
func (a *Action[In, Out, Stream]) spanMetadata(ctx context.Context, spanInit any) *tracing.SpanMetadata {
	sm := &tracing.SpanMetadata{
		Name:            a.desc.Name,
		Type:            "action",
		Subtype:         string(a.desc.Type), // The actual action type becomes the subtype.
		Metadata:        make(map[string]string),
		TelemetryLabels: tracing.TelemetryLabelsFromContext(ctx),
		Init:            spanInit,
	}
	if flowName := FlowNameFromContext(ctx); flowName != "" {
		sm.Metadata["flow:name"] = flowName
	}
	return sm
}

// recordActionMetrics writes the success/failure metric for one action run.
func recordActionMetrics(ctx context.Context, name string, start time.Time, err error) {
	latency := time.Since(start)
	if err != nil {
		metrics.WriteActionFailure(ctx, name, latency, err)
	} else {
		metrics.WriteActionSuccess(ctx, name, latency)
	}
}

// resolveSchema resolves $refs in one of the action's schemas through its
// registry, labeling failures with the schema's role ("input", "output",
// "stream", "init"). All schema reads on the run paths go through it so that
// named schema references are never used unresolved. Successful resolutions
// are memoized per role; failures are not, so a reference to a schema that is
// defined later resolves on a subsequent call.
func (a *Action[In, Out, Stream]) resolveSchema(role string, schema map[string]any) (map[string]any, error) {
	if schemaRefName(schema) == "" {
		return schema, nil // Nil or inline; nothing to resolve.
	}

	s := a.state
	s.mu.Lock()
	if resolved, ok := s.resolved[role]; ok {
		s.mu.Unlock()
		return resolved, nil
	}
	r := s.registry
	s.mu.Unlock()

	resolved, err := ResolveSchema(r, schema)
	if err != nil {
		return nil, NewError(INVALID_ARGUMENT, "invalid %s schema for action %q: %v", role, a.desc.Key, err)
	}

	s.mu.Lock()
	if s.resolved == nil {
		s.resolved = make(map[string]map[string]any)
	}
	s.resolved[role] = resolved
	s.mu.Unlock()
	return resolved, nil
}

// validateOutput checks a final output value against the resolved output
// schema.
func (a *Action[In, Out, Stream]) validateOutput(out Out, schema map[string]any) error {
	if err := base.ValidateValue(out, schema); err != nil {
		return NewError(INTERNAL, "invalid output from action %q: %v", a.desc.Key, err)
	}
	return nil
}

// RunJSON runs the action with a JSON input and returns the JSON result along
// with trace information. Trace information is populated even when the run
// fails, so transports can report the trace of a failed run.
func (a *Action[In, Out, Stream]) RunJSON(ctx context.Context, input json.RawMessage, cb StreamCallback[json.RawMessage]) (*api.ActionRunResult[json.RawMessage], error) {
	return a.runJSON(ctx, input, cb, a.fn, nil)
}

// runJSON is the shared JSON execution path. fn and spanInit follow the same
// contract as runWithTelemetry.
func (a *Action[In, Out, Stream]) runJSON(ctx context.Context, input json.RawMessage, cb StreamCallback[json.RawMessage], fn StreamingFunc[In, Out, Stream], spanInit any) (*api.ActionRunResult[json.RawMessage], error) {
	// Resolve before normalizing: a named schema reference carries none of
	// the structural information (e.g. number widening) normalization needs.
	inputSchema, err := a.resolveSchema("input", a.desc.InputSchema)
	if err != nil {
		return nil, err
	}
	i, err := base.UnmarshalAndNormalize[In](input, inputSchema)
	if err != nil {
		return nil, NewSchemaValidationError(a.desc.Key, err)
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

	r, err := a.runWithTelemetry(ctx, i, scb, fn, spanInit)
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
// Schema references that cannot be resolved (e.g., the action is not yet registered,
// or the referenced schema has not been defined) are returned as-is.
func (a *Action[In, Out, Stream]) Desc() api.ActionDesc {
	desc := *a.desc
	for role, p := range map[string]*map[string]any{
		"input":  &desc.InputSchema,
		"output": &desc.OutputSchema,
		"stream": &desc.StreamSchema,
		"init":   &desc.InitSchema,
	} {
		if resolved, err := a.resolveSchema(role, *p); err == nil {
			*p = resolved
		}
	}
	return desc
}

// setRegistry records the registry the action resolves schemas through and
// invalidates memoized resolutions: a different registry may define a
// different schema under the same name.
func (a *Action[In, Out, Stream]) setRegistry(r api.Registry) {
	s := a.state
	s.mu.Lock()
	s.registry = r
	s.resolved = nil
	s.mu.Unlock()
}

// Register registers the action with the given registry.
func (a *Action[In, Out, Stream]) Register(r api.Registry) {
	a.setRegistry(r)
	r.RegisterAction(a.desc.Key, a)
}

// ResolveActionFor returns the action for the given key in the global registry,
// or nil if there is none.
// It panics if the action is of the wrong type. That includes bidi actions,
// which are a distinct type; resolve those via [ResolveBidiActionFor].
func ResolveActionFor[In, Out, Stream any](r api.Registry, atype api.ActionType, name string) *Action[In, Out, Stream] {
	provider, id := api.ParseName(name)
	key := api.NewKey(atype, provider, id)
	a := r.ResolveAction(key)
	if a == nil {
		return nil
	}
	return a.(*Action[In, Out, Stream])
}

var _ api.Action = (*Action[struct{}, struct{}, struct{}])(nil)
