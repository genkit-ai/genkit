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

package ai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"reflect"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal/base"
)

// ToolFunc is the function type for tool implementations.
type ToolFunc[In, Out any] = func(ctx *ToolContext, input In) (Out, error)

// MultipartToolFunc is the function type for multipart tool implementations.
// Unlike regular tools that return just an output value, multipart tools
// can return both an output value and additional content parts (like media).
type MultipartToolFunc[In any] = func(ctx *ToolContext, input In) (*MultipartToolResponse, error)

// ToolRef is a reference to a tool.
type ToolRef interface {
	Name() string
}

// ToolName is a distinct type for a tool name.
// It is meant to be passed where a ToolRef is expected but no Tool is had.
type ToolName string

// Name returns the name of the tool.
func (t ToolName) Name() string {
	return (string)(t)
}

// ToolAction is a tool backed by a registry action. It is the concrete type
// returned by [NewTool] and [NewMultipartTool].
// Internally, all tools use the v2 format (returning MultipartToolResponse).
// For regular tools, RunRaw unwraps the Output field for backward compatibility.
//
// It implements [Tool] and [api.Action], so it can be passed anywhere either
// is accepted, including the action slice a plugin returns from Init. Unlike
// the other primitives it holds its action in a named field rather than
// embedding it, so its documented methods are its whole surface.
type ToolAction[In, Out any] struct {
	action    api.Action   // The underlying action.
	multipart bool         // Whether this is a multipart-only tool.
	registry  api.Registry // Registry for schema resolution. Set when registered.
}

// Pinned here so that breaking either interface fails the build at the type
// rather than at a call site.
var (
	_ Tool       = (*ToolAction[any, any])(nil)
	_ api.Action = (*ToolAction[any, any])(nil)
)

// ToolDef is the previous name for [ToolAction]. It was renamed because it
// read as a sibling of [ToolDefinition], the wire type a tool advertises to
// the model, which it is not.
//
// Deprecated: use [ToolAction].
type ToolDef[In, Out any] = ToolAction[In, Out]

// Tool represents a tool that can be called by a model. It is the type to
// accept as an argument and to look up by name; implementations are created
// with [NewTool] or [NewMultipartTool], or their [genkit.DefineTool] and
// [genkit.DefineMultipartTool] counterparts in an application.
type Tool interface {
	// Name returns the name of the tool.
	Name() string
	// Definition returns the definition for this tool to be passed to models.
	Definition() *ToolDefinition
	// RunRaw runs this tool using the provided raw input and returns just the output.
	RunRaw(ctx context.Context, input any) (any, error)
	// RunRawMultipart runs this tool and returns the full [MultipartToolResponse].
	RunRawMultipart(ctx context.Context, input any) (*MultipartToolResponse, error)
	// Respond constructs a [Part] with a [ToolResponse] for a given interrupted tool request.
	Respond(toolReq *Part, outputData any, opts *RespondOptions) *Part
	// Restart constructs a [Part] with a new [ToolRequest] to re-trigger a tool,
	// potentially with new input and metadata.
	Restart(toolReq *Part, opts *RestartOptions) *Part
	// Register registers the tool with the given registry.
	Register(r api.Registry)
}

// IsToolInterruptError reports whether err is an interrupt raised by a tool
// call (see [github.com/firebase/genkit/go/ai/tool.Interrupt]) and returns the
// interrupt data as a map, nil for a bare interrupt. It is for code that runs a
// tool outside of [Generate], such as middleware; inside the loop, interrupts
// surface through [ModelResponse.Interrupts].
func IsToolInterruptError(err error) (bool, map[string]any) {
	var ie *base.ToolInterruptError
	if !errors.As(err, &ie) {
		return false, nil
	}
	// tool.Interrupt normalized the payload when it raised the interrupt;
	// the conversion here covers an error built with a struct directly.
	m, _ := base.ObjectPayload(ie.Data, "interrupt data")
	return true, m
}

// InterruptOptions provides configuration for tool interruption.
type InterruptOptions struct {
	Metadata map[string]any
}

// RestartOptions provides configuration options for restarting a tool.
type RestartOptions struct {
	// ReplaceInput allows replacing the existing input arguments to the tool with different ones,
	// for example if the user revised an action before confirming. When input is replaced,
	// the existing tool request will be amended in the message history.
	ReplaceInput any
	// ResumedMetadata is the metadata you want to provide to the tool to aide in reprocessing.
	// Defaults to true if none is supplied.
	ResumedMetadata any
}

// RespondOptions provides configuration options for responding to a tool request.
type RespondOptions struct {
	// Metadata is additional metadata to include in the response.
	Metadata map[string]any
}

// RespondWithOption is a functional option for [ToolAction.RespondWith].
type RespondWithOption[Out any] interface {
	applyRespondWith(*RespondOptions)
}

// applyRespondWith applies the option to the respond options. Metadata is a
// single-value slot, so the last [WithResponseMetadata] set wins.
func (o *RespondOptions) applyRespondWith(opts *RespondOptions) {
	if o.Metadata != nil {
		opts.Metadata = o.Metadata
	}
}

// WithResponseMetadata sets metadata for the response. Repeating this option
// replaces the metadata rather than merging it.
func WithResponseMetadata[Out any](meta map[string]any) RespondWithOption[Out] {
	return &RespondOptions{Metadata: meta}
}

// RestartWithOption is a functional option for [ToolAction.RestartWith].
type RestartWithOption[In any] interface {
	applyRestartWith(*RestartOptions)
}

// applyRestartWith applies the option to the restart options. The replacement
// input and the resumed metadata are independent single-value slots, so the
// last option to set each one wins.
func (o *RestartOptions) applyRestartWith(opts *RestartOptions) {
	if o.ReplaceInput != nil {
		opts.ReplaceInput = o.ReplaceInput
	}
	if o.ResumedMetadata != nil {
		opts.ResumedMetadata = o.ResumedMetadata
	}
}

// WithNewInput sets a new input value to replace the original tool request input.
// Repeating this option takes the last input set.
func WithNewInput[In any](input In) RestartWithOption[In] {
	return &RestartOptions{ReplaceInput: input}
}

// WithResumedMetadata sets metadata to pass to the resumed tool execution.
// The metadata will be available in the tool's [ToolContext.Resumed] field.
// Repeating this option replaces the metadata rather than merging it.
func WithResumedMetadata[In any](meta map[string]any) RestartWithOption[In] {
	return &RestartOptions{ResumedMetadata: meta}
}

// ToolContext provides context and utility functions for tool execution.
type ToolContext struct {
	context.Context
	// Resumed is optional metadata that can be used to resume the tool execution.
	// Map is not nil only if the tool was interrupted.
	Resumed map[string]any
	// OriginalInput is the original input to the tool if the tool was interrupted, otherwise nil.
	OriginalInput any
}

// Interrupt interrupts the tool execution and returns control to the caller
// with the total model response so far. The provided metadata is preserved
// and passed back via [ToolContext.Resumed] when the tool is restarted.
func (tc *ToolContext) Interrupt(opts *InterruptOptions) error {
	if opts == nil {
		opts = &InterruptOptions{}
	}
	return &base.ToolInterruptError{Data: opts.Metadata}
}

// InterruptWith is a convenience function to interrupt a tool with a strongly-typed metadata value.
// The metadata is converted to map[string]any via JSON marshaling.
func InterruptWith[T any](tc *ToolContext, meta T) error {
	m, err := base.StructToMap(meta)
	if err != nil {
		return fmt.Errorf("InterruptWith: failed to convert metadata: %w", err)
	}
	return &base.ToolInterruptError{Data: m}
}

// InterruptAs returns the data an interrupted tool request carries, decoded
// into T. A tool sends that data with
// [github.com/firebase/genkit/go/ai/tool.Interrupt]; it is what the tool chose
// to say about the pause, e.g. why it needs approval. Returns the zero value and false if the part is not an interrupt,
// the interrupt carries no data, or the data does not decode into T.
func InterruptAs[T any](p *Part) (T, bool) {
	var zero T
	it := p.interruptState()
	if it == nil || it.Resolved || it.Data == nil {
		return zero, false
	}
	return base.ConvertTo[T](it.Data)
}

// IsResumed returns true if this tool execution is a resumption after an interrupt.
func (tc *ToolContext) IsResumed() bool {
	return tc.Resumed != nil
}

// IsToolResumed reports whether the current context is a resumed tool execution.
// This is intended for use in middleware that needs to distinguish between
// first-time and restarted tool calls.
//
// Deprecated: Use [github.com/firebase/genkit/go/ai/tool.ResumeData], whose
// second result reports the same from any context.
func IsToolResumed(ctx context.Context) bool {
	return base.ToolResumeKey.FromContext(ctx) != nil
}

// ResumedValue retrieves a typed value from the resumed metadata on ctx.
// Returns the zero value and false if the key doesn't exist or the type doesn't match.
// Accepts either a plain [context.Context] (useful in middleware) or a [*ToolContext],
// which embeds [context.Context].
//
// Deprecated: Use [github.com/firebase/genkit/go/ai/tool.ResumeData] with a
// struct that names the fields you read, which decodes the whole payload at
// once from any context.
func ResumedValue[T any](ctx context.Context, key string) (T, bool) {
	var zero T
	m, ok := resumedMap(ctx)
	if !ok {
		return zero, false
	}
	v, ok := m[key]
	if !ok {
		return zero, false
	}
	return base.ConvertTo[T](v)
}

// resumedMap returns the resume payload of a restarted call as a map, and
// false when the call is not a resumption. The payload rides the context as
// the caller gave it, so a struct from a struct restart is converted here and
// a map is handed over untouched.
func resumedMap(ctx context.Context) (map[string]any, bool) {
	v := base.ToolResumeKey.FromContext(ctx)
	if v == nil {
		return nil, false
	}
	m, ok := base.ConvertTo[map[string]any](v)
	if !ok || m == nil {
		// Still a resumption: an empty payload keeps IsResumed true.
		return map[string]any{}, true
	}
	return m, true
}

// OriginalInputAs returns the original input typed appropriately.
// Returns the zero value and false if not resumed or type doesn't match.
//
// Deprecated: Use [github.com/firebase/genkit/go/ai/tool.OriginalInput], which
// reads the same value from any context, a [ToolContext] included.
func OriginalInputAs[T any](tc *ToolContext) (T, bool) {
	var zero T
	if tc.OriginalInput == nil {
		return zero, false
	}
	return base.ConvertTo[T](tc.OriginalInput)
}

// toolStrictKey is the metadata key under metadata["tool"] used to carry the
// per-tool strict-schema flag through the action metadata and onto
// [ToolDefinition.Metadata]. Plugins consume this key directly.
const toolStrictKey = "strict"

// applyStrictMetadata sets metadata["tool"][toolStrictKey] = *strict when
// strict is non-nil. A nil value leaves the metadata untouched.
func applyStrictMetadata(metadata map[string]any, strict *bool) {
	if strict == nil {
		return
	}
	toolMeta, _ := metadata["tool"].(map[string]any)
	if toolMeta == nil {
		toolMeta = map[string]any{}
		metadata["tool"] = toolMeta
	}
	toolMeta[toolStrictKey] = *strict
}

// applyToolOutputSchema records a custom output schema as the tool's
// advertised (original) output schema. The action's own output type stays the
// multipart envelope; [ToolAction.Definition] surfaces the original schema to the
// model and the Dev UI.
func applyToolOutputSchema(metadata map[string]any, schema map[string]any) {
	if schema != nil {
		metadata["originalOutputSchema"] = schema
	}
}

// requireAnyTypeParam panics unless the type parameter T is an interface type
// (in practice 'any'). The tool constructors call it before honoring an
// explicit schema option: the custom schema stands in for a type parameter of
// 'any', and a concrete T would silently disagree with the advertised schema.
// The requirement argument is the leading clause of the panic message, e.g.
// "WithInputSchema requires In".
func requireAnyTypeParam[T any](ctor, name, requirement string) {
	if typ := reflect.TypeFor[T](); typ.Kind() != reflect.Interface {
		panic(fmt.Errorf("%s %q: %s to be of type 'any', but got %v", ctor, name, requirement, typ))
	}
}

// NewTool creates a new [ToolAction]. It can be passed directly to [Generate].
// Use [WithInputSchema] or [WithOutputSchema] to provide custom JSON schemas
// instead of inferring them from the type parameters. Inside the function,
// [github.com/firebase/genkit/go/ai/tool.AttachParts] adds content parts (e.g.
// media) to the response and
// [github.com/firebase/genkit/go/ai/tool.SendPartial] streams progress, neither
// of which changes the signature.
func NewTool[In, Out any](name, description string, fn ToolFunc[In, Out], opts ...ToolOption) *ToolAction[In, Out] {
	toolOpts := &toolOptions{}
	for _, opt := range opts {
		opt.applyTool(toolOpts)
	}

	if toolOpts.InputSchema != nil {
		requireAnyTypeParam[In]("ai.NewTool", name, "WithInputSchema requires In")
	}
	if toolOpts.OutputSchema != nil {
		requireAnyTypeParam[Out]("ai.NewTool", name, "WithOutputSchema and WithOutputSchemaName require Out")
	}

	metadata, wrappedFn := wrapToolFunc(name, description, fn)
	metadata["dynamic"] = true
	applyToolOutputSchema(metadata, toolOpts.OutputSchema)
	applyStrictMetadata(metadata, toolOpts.StrictSchema)
	action := core.NewActionOf(api.ActionTypeToolV2, name, &core.ActionOptions{Metadata: metadata, InputSchema: toolOpts.InputSchema}, wrappedFn)
	return &ToolAction[In, Out]{action: action, multipart: false}
}

// NewToolWithInputSchema creates a new [ToolAction] with a custom input schema. It can be passed directly to [Generate].
//
// Deprecated: Use [NewTool] with [WithInputSchema] instead.
func NewToolWithInputSchema[Out any](name, description string, inputSchema map[string]any, fn ToolFunc[any, Out]) *ToolAction[any, Out] {
	return NewTool(name, description, fn, WithInputSchema(inputSchema))
}

// NewMultipartTool creates a new multipart [ToolAction]. It can be passed directly to [Generate].
// Multipart tools can return both output data and additional content parts (like media).
// Use [WithInputSchema] to provide a custom JSON schema instead of inferring from the type parameter.
// Use [WithOutputSchema] or [WithOutputSchemaName] to advertise the logical
// output the tool produces (the envelope's output field); the wire format
// stays the multipart response envelope.
//
// Deprecated: Use [NewTool] and attach content parts with
// [github.com/firebase/genkit/go/ai/tool.AttachParts], which keeps the output
// type (and therefore the advertised output schema).
func NewMultipartTool[In any](name, description string, fn MultipartToolFunc[In], opts ...ToolOption) *ToolAction[In, *MultipartToolResponse] {
	toolOpts := &toolOptions{}
	for _, opt := range opts {
		opt.applyTool(toolOpts)
	}

	// Out is fixed to the multipart envelope, so only In can disagree with an
	// explicit schema. WithOutputSchema describes the envelope's output field
	// and carries no such constraint.
	if toolOpts.InputSchema != nil {
		requireAnyTypeParam[In]("ai.NewMultipartTool", name, "WithInputSchema requires In")
	}

	metadata, wrappedFn := wrapMultipartToolFunc(name, description, fn)
	metadata["dynamic"] = true
	applyToolOutputSchema(metadata, toolOpts.OutputSchema)
	applyStrictMetadata(metadata, toolOpts.StrictSchema)
	action := core.NewActionOf(api.ActionTypeToolV2, name, &core.ActionOptions{Metadata: metadata, InputSchema: toolOpts.InputSchema}, wrappedFn)
	return &ToolAction[In, *MultipartToolResponse]{action: action, multipart: true}
}

// wrapToolFunc wraps a regular tool function to return MultipartToolResponse.
func wrapToolFunc[In, Out any](name, description string, fn ToolFunc[In, Out]) (map[string]any, func(context.Context, In) (*MultipartToolResponse, error)) {
	var o Out
	var originalOutputSchema map[string]any
	if reflect.TypeOf(o) != nil {
		originalOutputSchema = core.InferSchemaMap(o)
	}

	metadata := map[string]any{
		"type":        api.ActionTypeToolV2,
		"name":        name,
		"description": description,
		"tool":        map[string]any{"multipart": false},
	}
	if originalOutputSchema != nil {
		metadata["originalOutputSchema"] = originalOutputSchema
	}

	wrappedFn := func(ctx context.Context, input In) (*MultipartToolResponse, error) {
		return runToolFunc(ctx, name, func(ctx context.Context) (*MultipartToolResponse, error) {
			output, err := fn(newToolContext(ctx), input)
			if err != nil {
				return nil, err
			}
			return &MultipartToolResponse{Output: output}, nil
		})
	}
	return metadata, wrappedFn
}

// wrapMultipartToolFunc wraps a multipart tool function.
func wrapMultipartToolFunc[In any](name, description string, fn MultipartToolFunc[In]) (map[string]any, func(context.Context, In) (*MultipartToolResponse, error)) {
	metadata := map[string]any{
		"type":        api.ActionTypeToolV2,
		"name":        name,
		"description": description,
		"tool":        map[string]any{"multipart": true},
	}
	wrappedFn := func(ctx context.Context, input In) (*MultipartToolResponse, error) {
		return runToolFunc(ctx, name, func(ctx context.Context) (*MultipartToolResponse, error) {
			return fn(newToolContext(ctx), input)
		})
	}
	return metadata, wrappedFn
}

// newToolContext builds the [ToolContext] a tool function written against it
// receives, lifting the restart state off the context.
func newToolContext(ctx context.Context) *ToolContext {
	tc := &ToolContext{
		Context:       ctx,
		OriginalInput: base.ToolOriginalInputKey.FromContext(ctx),
	}
	if m, ok := resumedMap(ctx); ok {
		tc.Resumed = m
	}
	return tc
}

// runToolFunc runs one invocation of the tool named name. The generate loop
// installs the part sink [github.com/firebase/genkit/go/ai/tool.AttachParts]
// writes to around the whole tool call, WrapTool hooks included, folds it
// when the call returns, and marks the tool stage for the tool to claim (see
// [base.ToolCall]). Any other run is a call of its own: a
// direct run (RunRaw, the Dev UI), or a tool run under another call's
// context, such as a tool that calls another tool directly. It gets its own
// sink, folded here, and none of the enclosing call's restart state, so the
// enclosing call's attachments and resume stay with that call.
func runToolFunc(ctx context.Context, name string, run func(ctx context.Context) (*MultipartToolResponse, error)) (*MultipartToolResponse, error) {
	if base.ToolPartSinkKey.FromContext(ctx) != nil && base.ToolCallKey.FromContext(ctx).Claim(name) {
		resp, err := run(ctx)
		if err != nil {
			return nil, err
		}
		if resp == nil {
			resp = &MultipartToolResponse{}
		}
		return resp, nil
	}
	sink := &base.PartSink{}
	defer sink.Close() // On an error, too.
	ctx = base.ToolPartSinkKey.NewContext(ctx, sink)
	ctx = base.ToolCallKey.NewContext(ctx, nil)
	ctx = base.ToolRestartKey.NewContext(ctx, nil)
	ctx = base.ToolHookKey.NewContext(ctx, "")
	ctx = base.ToolResumeKey.NewContext(ctx, nil)
	ctx = base.ToolOriginalInputKey.NewContext(ctx, nil)
	ctx = base.ToolReleasedKey.NewContext(ctx, false)
	resp, err := run(ctx)
	if err != nil {
		return nil, err
	}
	return foldAttachedParts(resp, sink), nil
}

// foldAttachedParts appends the parts attached to sink during a tool call to
// resp and returns it. A multipart function may return a nil response with no
// error, which the envelope treats as an empty one, so the parts still have a
// response to land on.
func foldAttachedParts(resp *MultipartToolResponse, sink *base.PartSink) *MultipartToolResponse {
	if resp == nil {
		resp = &MultipartToolResponse{}
	}
	for _, p := range sink.Close() {
		if part, ok := p.(*Part); ok {
			resp.Content = append(resp.Content, part)
		}
	}
	return resp
}

// Name returns the name of the tool.
func (t *ToolAction[In, Out]) Name() string {
	return t.action.Name()
}

// Definition returns [ToolDefinition] for for this tool.
func (t *ToolAction[In, Out]) Definition() *ToolDefinition {
	desc := t.action.Desc()

	// Resolve the input schema if it contains a $ref.
	inputSchema := desc.InputSchema
	if t.registry != nil {
		if resolved, err := core.ResolveSchema(t.registry, inputSchema); err == nil {
			inputSchema = resolved
		}
	}

	// Every tool function is wrapped in the multipart envelope, so the action's
	// own output schema describes that envelope, never the tool's real output.
	// Advertise the schema recorded at construction time from the output type
	// (or from an explicit [WithOutputSchema]), and nothing at all when the
	// output type carries no schema, e.g. any: an unconstrained output is
	// described by no schema, not by the envelope's.
	var outputSchema map[string]any
	if origSchema, ok := desc.Metadata["originalOutputSchema"].(map[string]any); ok {
		outputSchema = origSchema
	}

	// Resolve the output schema if it contains a $ref.
	if t.registry != nil && outputSchema != nil {
		if resolved, err := core.ResolveSchema(t.registry, outputSchema); err == nil {
			outputSchema = resolved
		}
	}

	metadata := map[string]any{
		"multipart": t.multipart,
	}
	if toolMeta, ok := desc.Metadata["tool"].(map[string]any); ok {
		if s, ok := toolMeta[toolStrictKey].(bool); ok {
			metadata[toolStrictKey] = s
		}
	}

	return &ToolDefinition{
		Name:         desc.Name,
		Description:  desc.Description,
		InputSchema:  inputSchema,
		OutputSchema: outputSchema,
		Metadata:     metadata,
	}
}

// Register registers the tool with the given registry.
func (t *ToolAction[In, Out]) Register(r api.Registry) {
	t.registry = r
	t.action.Register(r)
	if !t.multipart {
		// Also register under the "tool" key for backward compatibility.
		provider, id := api.ParseName(t.action.Name())
		r.RegisterAction(api.NewKey(api.ActionTypeTool, provider, id), t.action)
	}
}

// Desc returns the tool's action descriptor: its name, schemas, and metadata.
func (t *ToolAction[In, Out]) Desc() api.ActionDesc { return t.action.Desc() }

// RunJSON runs the tool on JSON-encoded input and returns the JSON-encoded
// multipart response envelope, which is what the registry serves for this
// tool. Prefer [ToolAction.RunRaw], which unwraps the envelope's output for a
// regular tool.
func (t *ToolAction[In, Out]) RunJSON(ctx context.Context, input json.RawMessage, cb core.StreamCallback[json.RawMessage]) (json.RawMessage, error) {
	if t == nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "ai.Tool.RunJSON: tool called on a nil tool; check that all tools are defined")
	}
	return t.action.RunJSON(ctx, input, cb)
}

// RunJSONWithTelemetry is [ToolAction.RunJSON] with the run's telemetry
// returned alongside the output.
func (t *ToolAction[In, Out]) RunJSONWithTelemetry(ctx context.Context, input json.RawMessage, cb core.StreamCallback[json.RawMessage]) (*api.ActionRunResult[json.RawMessage], error) {
	if t == nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "ai.Tool.RunJSONWithTelemetry: tool called on a nil tool; check that all tools are defined")
	}
	return t.action.RunJSONWithTelemetry(ctx, input, cb)
}

// RunRaw runs this tool using the provided raw map format data (JSON parsed as map[string]any).
func (t *ToolAction[In, Out]) RunRaw(ctx context.Context, input any) (any, error) {
	resp, err := t.RunRawMultipart(ctx, input)
	if err != nil {
		return nil, err
	}
	return resp.Output, nil
}

// RunRawMultipart runs this tool using the provided raw map format data (JSON parsed as map[string]any).
// It returns the full multipart response.
func (t *ToolAction[In, Out]) RunRawMultipart(ctx context.Context, input any) (*MultipartToolResponse, error) {
	if t == nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "ai.Tool.RunRawMultipart: tool called on a nil tool; check that all tools are defined")
	}

	mi, err := json.Marshal(input)
	if err != nil {
		return nil, status.Errorf(status.ErrInvalidInput, "marshalling input for tool %q: %w", t.Name(), err)
	}
	output, err := t.action.RunJSON(ctx, mi, nil)
	if err != nil {
		return nil, fmt.Errorf("error calling tool %v: %w", t.Name(), err)
	}

	var resp MultipartToolResponse
	if err := json.Unmarshal(output, &resp); err != nil {
		return nil, status.Errorf(status.ErrInvalidOutput, "parsing output of tool %q: %w", t.Name(), err)
	}
	return &resp, nil
}

// LookupTool looks up the tool in the registry by provided name and returns it.
// It checks for "tool.v2" first, then falls back to "tool" for legacy compatibility.
// Since the types are not known at lookup time, it returns a type-erased tool.
func LookupTool(r api.Registry, name string) Tool {
	if name == "" {
		return nil
	}
	provider, id := api.ParseName(name)

	// First try tool.v2 (all new tools are registered here)
	key := api.NewKey(api.ActionTypeToolV2, provider, id)
	action := r.ResolveAction(key)

	// Fall back to tool for legacy compatibility
	if action == nil {
		key = api.NewKey(api.ActionTypeTool, provider, id)
		action = r.ResolveAction(key)
	}

	if action == nil {
		return nil
	}

	desc := action.Desc()
	multipart := false
	if toolMeta, ok := desc.Metadata["tool"].(map[string]any); ok {
		if mp, ok := toolMeta["multipart"].(bool); ok {
			multipart = mp
		}
	}

	return &ToolAction[any, any]{action: action, multipart: multipart, registry: r}
}

// IsMultipart returns true if the tool is a multipart tool (tool.v2 only).
func (t *ToolAction[In, Out]) IsMultipart() bool {
	return t.multipart
}

// Respond creates a part for [WithToolResponses] to provide a resolved response for an interrupted tool call.
// Returns nil if the part is not a tool request.
//
// Deprecated: Use [ToolAction.RespondWith] instead for strongly-typed options.
func (t *ToolAction[In, Out]) Respond(toolReq *Part, output any, opts *RespondOptions) *Part {
	if !toolReq.IsToolRequest() || toolReq.ToolRequest == nil {
		return nil
	}
	if opts == nil {
		opts = &RespondOptions{}
	}
	return newResponsePart(toolReq, output, opts.Metadata)
}

// Restart creates a part for [WithToolRestarts] to re-execute an interrupted tool call with additional context.
// Returns nil if the part is not a tool request. The resume data is carried as
// given: a value that is not a JSON object resumes the tool with an empty
// payload, the way a peer runtime's marker would.
//
// Deprecated: Use [ToolAction.RestartWith] instead for strongly-typed options.
func (t *ToolAction[In, Out]) Restart(p *Part, opts *RestartOptions) *Part {
	if !p.IsToolRequest() || p.ToolRequest == nil {
		return nil
	}
	if opts == nil {
		opts = &RestartOptions{}
	}
	// ReplaceInput is optional, so a nil of any type means it was not set.
	return buildRestartPart(p, opts.ResumedMetadata, opts.ReplaceInput, !base.IsNil(opts.ReplaceInput))
}

// RespondWith creates a part for [WithToolResponses] to provide a resolved response for an interrupted tool call.
//
// Example:
//
//	part, err := myTool.RespondWith(toolReq, output, WithResponseMetadata[MyOutput](meta))
func (t *ToolAction[In, Out]) RespondWith(toolReq *Part, output Out, opts ...RespondWithOption[Out]) (*Part, error) {
	if err := t.checkToolRequest("ai.RespondWith", toolReq); err != nil {
		return nil, err
	}
	cfg := &RespondOptions{}
	for _, opt := range opts {
		opt.applyRespondWith(cfg)
	}
	return newResponsePart(toolReq, output, cfg.Metadata), nil
}

// RestartWith creates a part for [WithToolRestarts] to re-execute an interrupted tool call with additional context.
//
// Example:
//
//	part, err := myTool.RestartWith(toolReq, WithNewInput(newInput), WithResumedMetadata[MyInput](meta))
func (t *ToolAction[In, Out]) RestartWith(toolReq *Part, opts ...RestartWithOption[In]) (*Part, error) {
	const fnName = "ai.RestartWith"
	if err := t.checkToolRequest(fnName, toolReq); err != nil {
		return nil, err
	}
	cfg := &RestartOptions{}
	for _, opt := range opts {
		opt.applyRestartWith(cfg)
	}
	if err := base.CheckObjectPayload(cfg.ResumedMetadata, "resume data"); err != nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "%s: %w", fnName, err)
	}
	// WithNewInput is optional, so a nil of any type means it was not given.
	return buildRestartPart(toolReq, cfg.ResumedMetadata, cfg.ReplaceInput, !base.IsNil(cfg.ReplaceInput)), nil
}

// checkToolRequest is the guard RespondWith and RestartWith share: toolReq
// must be a tool request for this tool. fnName names the verb in the error.
func (t *ToolAction[In, Out]) checkToolRequest(fnName string, toolReq *Part) error {
	if toolReq == nil {
		return status.Errorf(status.ErrInvalidArgument, "%s: toolReq is nil", fnName)
	}
	if !toolReq.IsToolRequest() {
		return status.Errorf(ErrInvalidPart, "%s: part is not a tool request", fnName)
	}
	if toolReq.ToolRequest == nil {
		return status.Errorf(ErrInvalidPart, "%s: tool request part has no request", fnName)
	}
	if toolReq.ToolRequest.Name != t.Name() {
		return status.Errorf(status.ErrInvalidArgument, "%s: tool request is for %q, not %q", fnName, toolReq.ToolRequest.Name, t.Name())
	}
	return nil
}

// buildRestartPart builds the tool request [Part] that re-executes an
// interrupted call. The new part keeps the interrupted part's metadata, less
// its interrupt state. resume is the payload delivered to the tool, or nil
// for a bare restart; a nil map or pointer is a bare restart too. When
// replace is set, newInput replaces the input the tool re-executes with,
// whatever its value, and the original is preserved on
// [ToolRestart.OriginalInput]; the verbs decide replacement, so that a nil
// newInput is never mistaken for "keep the input" or the other way round.
func buildRestartPart(interruptPart *Part, resume, newInput any, replace bool) *Part {
	toolReq := interruptPart.ToolRequest
	input, originalInput := toolReq.Input, any(nil)
	if replace {
		input, originalInput = newInput, input
	}

	restartPart := NewToolRequestPart(&ToolRequest{
		Name:  toolReq.Name,
		Ref:   toolReq.Ref,
		Input: input,
	})
	// The restart keeps the interrupted part's metadata, less its interrupt
	// state and less the loop's bookkeeping of a sibling's outcome
	// (pendingOutput and its companions), which describes the request in
	// history, not the restart.
	restartPart.Metadata = stripPendingKeys(stripWireKeys(maps.Clone(interruptPart.Metadata)))
	restartPart.Restart = &ToolRestart{Resume: bareIfNil(resume), OriginalInput: originalInput}
	return restartPart
}

// bareIfNil normalizes an interrupt or resume payload: an untyped nil, a nil
// map, or a nil pointer all mean a bare interrupt or restart, so they become
// an untyped nil rather than a typed nil inside the interface, which would
// serialize as JSON null instead of the bare marker.
func bareIfNil(v any) any {
	if base.IsNil(v) {
		return nil
	}
	return v
}

// newResponsePart builds the tool response [Part] that resolves an interrupted
// call with a pre-computed output. The generate loop resolves the interrupt by
// the part's place in the Respond list, matched on tool name and ref; the
// interruptResponse marker is the wire contract's mark of a caller-provided
// response, which the JS runtime writes too, and metadata, when non-nil,
// rides under it in place of the bare marker.
func newResponsePart(interruptPart *Part, output any, metadata map[string]any) *Part {
	resp := NewResponseForToolRequest(interruptPart, output)
	resp.Metadata = map[string]any{metaInterruptResponse: true}
	if metadata != nil {
		resp.Metadata[metaInterruptResponse] = metadata
	}
	return resp
}

// resolveUniqueTools resolves the list of tool refs to a list of all tool names and new tools that must be registered.
// Returns an error if there are tool refs with duplicate names.
func resolveUniqueTools(r api.Registry, toolRefs []ToolRef) (toolNames []string, newTools []Tool, err error) {
	toolMap := make(map[string]bool)

	for _, toolRef := range toolRefs {
		name := toolRef.Name()

		if toolMap[name] {
			return nil, nil, status.Errorf(status.ErrInvalidArgument, "duplicate tool %q", name)
		}
		toolMap[name] = true
		toolNames = append(toolNames, name)

		if LookupTool(r, name) == nil {
			if tool, ok := toolRef.(Tool); ok {
				newTools = append(newTools, tool)
			}
		}
	}

	return toolNames, newTools, nil
}
