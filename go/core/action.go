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
	fn       StreamingFunc[In, Out, Stream] // Function that is called during runtime. May not actually support streaming.
	desc     *api.ActionDesc                // Descriptor of the action.
	registry api.Registry                   // Registry for schema resolution. Set when registered.
}

// ActionDef is the previous name for [Action].
//
// Deprecated: use [Action].
type ActionDef[In, Out, Stream any] = Action[In, Out, Stream]

type noStream = func(context.Context, struct{}) error

// NewAction creates a new non-streaming [Action] without registering it.
// If inputSchema is nil, it is inferred from the function's input api.
func NewAction[In, Out any](
	name string,
	atype api.ActionType,
	metadata map[string]any,
	inputSchema map[string]any,
	fn Func[In, Out],
) *Action[In, Out, struct{}] {
	return newStreamingAction(name, atype, metadata, inputSchema,
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
) *Action[In, Out, Stream] {
	return newStreamingAction(name, atype, metadata, inputSchema, fn)
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
) *Action[In, Out, struct{}] {
	return defineStreamingAction(r, name, atype, metadata, inputSchema,
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
) *Action[In, Out, Stream] {
	return defineStreamingAction(r, name, atype, metadata, inputSchema, fn)
}

// defineStreamingAction creates a streaming action and registers it.
func defineStreamingAction[In, Out, Stream any](
	r api.Registry,
	name string,
	atype api.ActionType,
	metadata map[string]any,
	inputSchema map[string]any,
	fn StreamingFunc[In, Out, Stream],
) *Action[In, Out, Stream] {
	a := newStreamingAction(name, atype, metadata, inputSchema, fn)
	a.Register(r)
	return a
}

// newStreamingAction constructs an action with the given implementation.
// It is the common helper for NewAction, NewStreamingAction, and
// DefineStreamingAction.
func newStreamingAction[In, Out, Stream any](
	name string,
	atype api.ActionType,
	metadata map[string]any,
	inputSchema map[string]any,
	fn StreamingFunc[In, Out, Stream],
) *Action[In, Out, Stream] {
	a := newAction[In, Out, Stream](name, atype, metadata, inputSchema)
	a.fn = fn
	return a
}

// newAction populates an Action's descriptor with inferred schemas and metadata.
// The caller is expected to assign a.fn.
func newAction[In, Out, Stream any](
	name string,
	atype api.ActionType,
	metadata map[string]any,
	inputSchema map[string]any,
) *Action[In, Out, Stream] {
	if inputSchema == nil {
		inputSchema = inferSchema[In]()
	}

	outputSchema := inferSchema[Out]()

	// Stream is struct{} for non-streaming actions; inferring a schema from
	// the sentinel would make every action advertise a bogus streamSchema.
	var streamSchema map[string]any
	if !isUnitType[Stream]() {
		streamSchema = inferSchema[Stream]()
	}

	var description string
	if desc, ok := metadata["description"].(string); ok {
		description = desc
	}

	return &Action[In, Out, Stream]{
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

			var inputSchema map[string]any
			inputSchema, err = ResolveSchema(a.registry, a.desc.InputSchema)
			if err != nil {
				return base.Zero[Out](), NewError(INVALID_ARGUMENT, "invalid input schema for action %q: %v", a.desc.Key, err)
			}

			var outputSchema map[string]any
			outputSchema, err = a.resolveOutputSchema()
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

// resolveOutputSchema resolves the action's OutputSchema $refs through the
// registry.
func (a *Action[In, Out, Stream]) resolveOutputSchema() (map[string]any, error) {
	schema, err := ResolveSchema(a.registry, a.desc.OutputSchema)
	if err != nil {
		return nil, NewError(INVALID_ARGUMENT, "invalid output schema for action %q: %v", a.desc.Key, err)
	}
	return schema, nil
}

// validateOutput checks a final output value against the resolved output
// schema.
func (a *Action[In, Out, Stream]) validateOutput(out Out, schema map[string]any) error {
	if err := base.ValidateValue(out, schema); err != nil {
		return NewError(INTERNAL, "invalid output from action %q: %v", a.desc.Key, err)
	}
	return nil
}

// RunJSON runs the action with a JSON input, and returns a JSON result.
func (a *Action[In, Out, Stream]) RunJSON(ctx context.Context, input json.RawMessage, cb StreamCallback[json.RawMessage]) (json.RawMessage, error) {
	r, err := a.RunJSONWithTelemetry(ctx, input, cb)
	if err != nil {
		return nil, err
	}
	return r.Result, nil
}

// RunJSONWithTelemetry runs the action with a JSON input, and returns a JSON result along with telemetry info.
func (a *Action[In, Out, Stream]) RunJSONWithTelemetry(ctx context.Context, input json.RawMessage, cb StreamCallback[json.RawMessage]) (*api.ActionRunResult[json.RawMessage], error) {
	return a.runJSONWithTelemetry(ctx, input, cb, a.fn, nil)
}

// runJSONWithTelemetry is the shared JSON execution path. fn and spanInit
// follow the same contract as runWithTelemetry.
func (a *Action[In, Out, Stream]) runJSONWithTelemetry(ctx context.Context, input json.RawMessage, cb StreamCallback[json.RawMessage], fn StreamingFunc[In, Out, Stream], spanInit any) (*api.ActionRunResult[json.RawMessage], error) {
	i, err := base.UnmarshalAndNormalize[In](input, a.desc.InputSchema)
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
	if a.registry == nil {
		return desc
	}
	for _, p := range []*map[string]any{&desc.InputSchema, &desc.OutputSchema, &desc.StreamSchema, &desc.InitSchema} {
		if resolved, err := ResolveSchema(a.registry, *p); err == nil {
			*p = resolved
		}
	}
	return desc
}

// Register registers the action with the given registry.
func (a *Action[In, Out, Stream]) Register(r api.Registry) {
	a.registry = r
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

// LookupActionFor returns the action for the given key in the global registry,
// or nil if there is none.
// It panics if the action is of the wrong api.
//
// Deprecated: Use ResolveActionFor.
func LookupActionFor[In, Out, Stream any](r api.Registry, atype api.ActionType, name string) *Action[In, Out, Stream] {
	provider, id := api.ParseName(name)
	key := api.NewKey(atype, provider, id)
	a := r.LookupAction(key)
	if a == nil {
		return nil
	}
	return a.(*Action[In, Out, Stream])
}

var _ api.Action = (*Action[struct{}, struct{}, struct{}])(nil)
