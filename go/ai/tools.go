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
	"github.com/firebase/genkit/go/internal/base"
)

var resumedCtxKey = base.NewContextKey[map[string]any]()
var origInputCtxKey = base.NewContextKey[any]()

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

// ToolDef is an action with functions specific to tools.
// Internally, all tools use the v2 format (returning MultipartToolResponse).
// For regular tools, RunRaw unwraps the Output field for backward compatibility.
type ToolDef[In, Out any] struct {
	action    api.Action   // The underlying action.
	multipart bool         // Whether this is a multipart-only tool.
	registry  api.Registry // Registry for schema resolution. Set when registered.
}

// Tool represents a tool that can be called by a model.
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

// toolInterruptError represents an intentional interruption of tool execution.
type toolInterruptError struct {
	Metadata map[string]any
}

func (e *toolInterruptError) Error() string {
	if e.Metadata != nil {
		data, err := json.MarshalIndent(e.Metadata, "", "  ")
		if err == nil {
			return fmt.Sprintf("tool execution interrupted: \n\n%s", string(data))
		}
	}
	return "tool execution interrupted"
}

// IsToolInterruptError determines whether the error is an interrupt error returned by the tool.
func IsToolInterruptError(err error) (bool, map[string]any) {
	var tie *toolInterruptError
	if errors.As(err, &tie) {
		return true, tie.Metadata
	}
	return false, nil
}

// NewToolInterruptError creates a tool interrupt error with the given metadata.
// This is intended for use in middleware that needs to interrupt tool execution
// without calling the tool itself.
func NewToolInterruptError(metadata map[string]any) error {
	return &toolInterruptError{Metadata: metadata}
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

// RespondWithOption is a functional option for [ToolDef.RespondWith].
type RespondWithOption[Out any] interface {
	applyRespondWith(*RespondOptions) error
}

// applyRespondWith applies the option to the respond options.
func (o *RespondOptions) applyRespondWith(opts *RespondOptions) error {
	if o.Metadata != nil {
		if opts.Metadata != nil {
			return errors.New("cannot set metadata more than once (WithResponseMetadata)")
		}
		opts.Metadata = o.Metadata
	}
	return nil
}

// WithResponseMetadata sets metadata for the response.
func WithResponseMetadata[Out any](meta map[string]any) RespondWithOption[Out] {
	return &RespondOptions{Metadata: meta}
}

// RestartWithOption is a functional option for [ToolDef.RestartWith].
type RestartWithOption[In any] interface {
	applyRestartWith(*RestartOptions) error
}

// applyRestartWith applies the option to the restart options.
func (o *RestartOptions) applyRestartWith(opts *RestartOptions) error {
	if o.ReplaceInput != nil {
		if opts.ReplaceInput != nil {
			return errors.New("cannot set new input more than once (WithNewInput)")
		}
		opts.ReplaceInput = o.ReplaceInput
	}
	if o.ResumedMetadata != nil {
		if opts.ResumedMetadata != nil {
			return errors.New("cannot set resumed metadata more than once (WithResumedMetadata)")
		}
		opts.ResumedMetadata = o.ResumedMetadata
	}
	return nil
}

// WithNewInput sets a new input value to replace the original tool request input.
func WithNewInput[In any](input In) RestartWithOption[In] {
	return &RestartOptions{ReplaceInput: input}
}

// WithResumedMetadata sets metadata to pass to the resumed tool execution.
// The metadata will be available in the tool's [ToolContext.Resumed] field.
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
	return &toolInterruptError{
		Metadata: opts.Metadata,
	}
}

// InterruptWith is a convenience function to interrupt a tool with a strongly-typed metadata value.
// The metadata is converted to map[string]any via JSON marshaling.
func InterruptWith[T any](tc *ToolContext, meta T) error {
	m, err := base.StructToMap(meta)
	if err != nil {
		return fmt.Errorf("InterruptWith: failed to convert metadata: %w", err)
	}
	return tc.Interrupt(&InterruptOptions{Metadata: m})
}

// InterruptAs extracts strongly-typed metadata from an interrupted tool request [Part].
// Returns the zero value and false if the part is not an interrupt or the type doesn't match.
func InterruptAs[T any](p *Part) (T, bool) {
	var zero T
	if p == nil || !p.IsInterrupt() {
		return zero, false
	}
	meta, ok := p.Metadata["interrupt"].(map[string]any)
	if !ok {
		return zero, false
	}
	result, err := base.MapToStruct[T](meta)
	if err != nil {
		return zero, false
	}
	return result, true
}

// IsResumed returns true if this tool execution is a resumption after an interrupt.
func (tc *ToolContext) IsResumed() bool {
	return tc.Resumed != nil
}

// IsToolResumed reports whether the current context is a resumed tool execution.
// This is intended for use in middleware that needs to distinguish between
// first-time and restarted tool calls.
func IsToolResumed(ctx context.Context) bool {
	return resumedCtxKey.FromContext(ctx) != nil
}

// ResumedValue retrieves a typed value from the resumed metadata on ctx.
// Returns the zero value and false if the key doesn't exist or the type doesn't match.
// Accepts either a plain [context.Context] (useful in middleware) or a [*ToolContext],
// which embeds [context.Context].
func ResumedValue[T any](ctx context.Context, key string) (T, bool) {
	var zero T
	m := resumedCtxKey.FromContext(ctx)
	if m == nil {
		return zero, false
	}
	v, ok := m[key]
	if !ok {
		return zero, false
	}
	typed, ok := v.(T)
	return typed, ok
}

// OriginalInputAs returns the original input typed appropriately.
// Returns the zero value and false if not resumed or type doesn't match.
func OriginalInputAs[T any](tc *ToolContext) (T, bool) {
	var zero T
	if tc.OriginalInput == nil {
		return zero, false
	}
	// Try direct type assertion first (for when input is already typed)
	if typed, ok := tc.OriginalInput.(T); ok {
		return typed, ok
	}
	// Otherwise try to convert from map[string]any (common case from JSON)
	if m, ok := tc.OriginalInput.(map[string]any); ok {
		result, err := base.MapToStruct[T](m)
		if err != nil {
			return zero, false
		}
		return result, true
	}
	return zero, false
}

// DefineTool creates a new [ToolDef] and registers it.
// Use [WithInputSchema] to provide a custom JSON schema instead of inferring from the type parameter.
func DefineTool[In, Out any](
	r api.Registry,
	name, description string,
	fn ToolFunc[In, Out],
	opts ...ToolOption,
) *ToolDef[In, Out] {
	toolOpts := &toolOptions{}
	for _, opt := range opts {
		if err := opt.applyTool(toolOpts); err != nil {
			panic(fmt.Errorf("ai.DefineTool %q: %w", name, err))
		}
	}

	// If the user provided a custom input schema, enforce that In is 'any'
	if toolOpts.InputSchema != nil {
		typ := reflect.TypeFor[*In]()
		if typ != nil && typ.Elem().Kind() != reflect.Interface {
			panic(fmt.Errorf("ai.DefineTool %q: WithInputSchema requires In to be of type 'any', but got %v", name, typ.Elem()))
		}
	}

	metadata, wrappedFn := wrapToolFunc(name, description, fn)
	action := core.DefineAction(r, name, api.ActionTypeToolV2, metadata, toolOpts.InputSchema, wrappedFn)

	// Also register under the "tool" action type for backward compatibility.
	provider, id := api.ParseName(name)
	r.RegisterAction(api.NewKey(api.ActionTypeTool, provider, id), action)

	return &ToolDef[In, Out]{action: action, multipart: false, registry: r}
}

// DefineToolWithInputSchema creates a new [ToolDef] with a custom input schema and registers it.
//
// Deprecated: Use [DefineTool] with [WithInputSchema] instead.
func DefineToolWithInputSchema[Out any](
	r api.Registry,
	name, description string,
	inputSchema map[string]any,
	fn ToolFunc[any, Out],
) *ToolDef[any, Out] {
	return DefineTool(r, name, description, fn, WithInputSchema(inputSchema))
}

// NewTool creates a new [ToolDef]. It can be passed directly to [Generate].
// Use [WithInputSchema] to provide a custom JSON schema instead of inferring from the type parameter.
func NewTool[In, Out any](name, description string, fn ToolFunc[In, Out], opts ...ToolOption) *ToolDef[In, Out] {
	toolOpts := &toolOptions{}
	for _, opt := range opts {
		if err := opt.applyTool(toolOpts); err != nil {
			panic(fmt.Errorf("ai.NewTool %q: %w", name, err))
		}
	}

	// If the user provided a custom input schema, enforce that In is 'any'
	if toolOpts.InputSchema != nil {
		var zeroIn *In
		typ := reflect.TypeOf(zeroIn)
		if typ != nil && typ.Elem().Kind() != reflect.Interface {
			panic(fmt.Errorf("ai.NewTool %q: WithInputSchema requires In to be of type 'any', but got %v", name, typ.Elem()))
		}
	}

	metadata, wrappedFn := wrapToolFunc(name, description, fn)
	metadata["dynamic"] = true
	action := core.NewAction(name, api.ActionTypeToolV2, metadata, toolOpts.InputSchema, wrappedFn)
	return &ToolDef[In, Out]{action: action, multipart: false}
}

// NewToolWithInputSchema creates a new [ToolDef] with a custom input schema. It can be passed directly to [Generate].
//
// Deprecated: Use [NewTool] with [WithInputSchema] instead.
func NewToolWithInputSchema[Out any](name, description string, inputSchema map[string]any, fn ToolFunc[any, Out]) *ToolDef[any, Out] {
	return NewTool(name, description, fn, WithInputSchema(inputSchema))
}

// DefineMultipartTool creates a new multipart [ToolDef] and registers it.
// Multipart tools can return both output data and additional content parts (like media).
// Use [WithInputSchema] to provide a custom JSON schema instead of inferring from the type parameter.
func DefineMultipartTool[In any](
	r api.Registry,
	name, description string,
	fn MultipartToolFunc[In],
	opts ...ToolOption,
) *ToolDef[In, *MultipartToolResponse] {
	toolOpts := &toolOptions{}
	for _, opt := range opts {
		if err := opt.applyTool(toolOpts); err != nil {
			panic(fmt.Errorf("ai.DefineMultipartTool %q: %w", name, err))
		}
	}

	metadata, wrappedFn := wrapMultipartToolFunc(name, description, fn)
	action := core.DefineAction(r, name, api.ActionTypeToolV2, metadata, toolOpts.InputSchema, wrappedFn)
	return &ToolDef[In, *MultipartToolResponse]{action: action, multipart: true, registry: r}
}

// NewMultipartTool creates a new multipart [ToolDef]. It can be passed directly to [Generate].
// Multipart tools can return both output data and additional content parts (like media).
// Use [WithInputSchema] to provide a custom JSON schema instead of inferring from the type parameter.
func NewMultipartTool[In any](name, description string, fn MultipartToolFunc[In], opts ...ToolOption) *ToolDef[In, *MultipartToolResponse] {
	toolOpts := &toolOptions{}
	for _, opt := range opts {
		if err := opt.applyTool(toolOpts); err != nil {
			panic(fmt.Errorf("ai.NewMultipartTool %q: %w", name, err))
		}
	}

	metadata, wrappedFn := wrapMultipartToolFunc(name, description, fn)
	metadata["dynamic"] = true
	action := core.NewAction(name, api.ActionTypeToolV2, metadata, toolOpts.InputSchema, wrappedFn)
	return &ToolDef[In, *MultipartToolResponse]{action: action, multipart: true}
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
		toolCtx := &ToolContext{
			Context:       ctx,
			Resumed:       resumedCtxKey.FromContext(ctx),
			OriginalInput: origInputCtxKey.FromContext(ctx),
		}
		output, err := fn(toolCtx, input)
		if err != nil {
			return nil, err
		}
		return &MultipartToolResponse{Output: output}, nil
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
		toolCtx := &ToolContext{
			Context:       ctx,
			Resumed:       resumedCtxKey.FromContext(ctx),
			OriginalInput: origInputCtxKey.FromContext(ctx),
		}
		return fn(toolCtx, input)
	}
	return metadata, wrappedFn
}

// Name returns the name of the tool.
func (t *ToolDef[In, Out]) Name() string {
	return t.action.Name()
}

// Definition returns [ToolDefinition] for for this tool.
func (t *ToolDef[In, Out]) Definition() *ToolDefinition {
	desc := t.action.Desc()

	// Resolve the input schema if it contains a $ref.
	inputSchema := desc.InputSchema
	if t.registry != nil {
		if resolved, err := core.ResolveSchema(t.registry, inputSchema); err == nil {
			inputSchema = resolved
		}
	}

	// Use the original output schema if available (for non-multipart tools).
	outputSchema := desc.OutputSchema
	if origSchema, ok := desc.Metadata["originalOutputSchema"].(map[string]any); ok {
		outputSchema = origSchema
	}

	// Resolve the output schema if it contains a $ref.
	if t.registry != nil {
		if resolved, err := core.ResolveSchema(t.registry, outputSchema); err == nil {
			outputSchema = resolved
		}
	}

	return &ToolDefinition{
		Name:         desc.Name,
		Description:  desc.Description,
		InputSchema:  inputSchema,
		OutputSchema: outputSchema,
		Metadata: map[string]any{
			"multipart": t.multipart,
		},
	}
}

// Register registers the tool with the given registry.
func (t *ToolDef[In, Out]) Register(r api.Registry) {
	t.registry = r
	t.action.Register(r)
	if !t.multipart {
		// Also register under the "tool" key for backward compatibility.
		provider, id := api.ParseName(t.action.Name())
		r.RegisterAction(api.NewKey(api.ActionTypeTool, provider, id), t.action)
	}
}

// RunRaw runs this tool using the provided raw map format data (JSON parsed as map[string]any).
func (t *ToolDef[In, Out]) RunRaw(ctx context.Context, input any) (any, error) {
	resp, err := t.RunRawMultipart(ctx, input)
	if err != nil {
		return nil, err
	}
	return resp.Output, nil
}

// RunRawMultipart runs this tool using the provided raw map format data (JSON parsed as map[string]any).
// It returns the full multipart response.
func (t *ToolDef[In, Out]) RunRawMultipart(ctx context.Context, input any) (*MultipartToolResponse, error) {
	if t == nil {
		return nil, core.NewError(core.INVALID_ARGUMENT, "ai.Tool.RunRawMultipart: tool called on a nil tool; check that all tools are defined")
	}

	mi, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("error marshalling tool input for %v: %v", t.Name(), err)
	}
	output, err := t.action.RunJSON(ctx, mi, nil)
	if err != nil {
		return nil, fmt.Errorf("error calling tool %v: %w", t.Name(), err)
	}

	var resp MultipartToolResponse
	if err := json.Unmarshal(output, &resp); err != nil {
		return nil, fmt.Errorf("error parsing tool output for %v: %v", t.Name(), err)
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

	// Check if it's a multipart-only tool
	desc := action.Desc()
	multipart := false
	if toolMeta, ok := desc.Metadata["tool"].(map[string]any); ok {
		if mp, ok := toolMeta["multipart"].(bool); ok {
			multipart = mp
		}
	}

	return &ToolDef[any, any]{action: action, multipart: multipart, registry: r}
}

// IsMultipart returns true if the tool is a multipart tool (tool.v2 only).
func (t *ToolDef[In, Out]) IsMultipart() bool {
	return t.multipart
}

// Respond creates a part for [WithToolResponses] to provide a resolved response for an interrupted tool call.
// Returns nil if the part is not a tool request.
//
// Deprecated: Use [ToolDef.RespondWith] instead for strongly-typed options.
func (t *ToolDef[In, Out]) Respond(toolReq *Part, output any, opts *RespondOptions) *Part {
	if toolReq == nil || !toolReq.IsToolRequest() {
		return nil
	}

	if opts == nil {
		opts = &RespondOptions{}
	}

	newToolResp := NewResponseForToolRequest(toolReq, output)
	newToolResp.Metadata = map[string]any{
		"interruptResponse": true,
	}
	if opts.Metadata != nil {
		newToolResp.Metadata["interruptResponse"] = opts.Metadata
	}

	return newToolResp
}

// Restart creates a part for [WithToolRestarts] to re-execute an interrupted tool call with additional context.
// Returns nil if the part is not a tool request.
//
// Deprecated: Use [ToolDef.RestartWith] instead for strongly-typed options.
func (t *ToolDef[In, Out]) Restart(p *Part, opts *RestartOptions) *Part {
	if p == nil || !p.IsToolRequest() {
		return nil
	}

	if opts == nil {
		opts = &RestartOptions{}
	}

	newInput := p.ToolRequest.Input
	var originalInput any

	if opts.ReplaceInput != nil {
		originalInput = newInput
		newInput = opts.ReplaceInput
	}

	newMeta := maps.Clone(p.Metadata)
	if newMeta == nil {
		newMeta = make(map[string]any)
	}

	newMeta["resumed"] = true
	if opts.ResumedMetadata != nil {
		newMeta["resumed"] = opts.ResumedMetadata
	}

	if originalInput != nil {
		newMeta["replacedInput"] = originalInput
	}

	delete(newMeta, "interrupt")

	newToolReq := NewToolRequestPart(&ToolRequest{
		Name:  p.ToolRequest.Name,
		Ref:   p.ToolRequest.Ref,
		Input: newInput,
	})
	newToolReq.Metadata = newMeta

	return newToolReq
}

// RespondWith creates a part for [WithToolResponses] to provide a resolved response for an interrupted tool call.
//
// Example:
//
//	part, err := myTool.RespondWith(toolReq, output, WithResponseMetadata[MyOutput](meta))
func (t *ToolDef[In, Out]) RespondWith(toolReq *Part, output Out, opts ...RespondWithOption[Out]) (*Part, error) {
	if toolReq == nil {
		return nil, core.NewError(core.INVALID_ARGUMENT, "ai.RespondWith: toolReq is nil")
	}
	if !toolReq.IsToolRequest() {
		return nil, core.NewError(core.INVALID_ARGUMENT, "ai.RespondWith: part is not a tool request")
	}
	if toolReq.ToolRequest.Name != t.Name() {
		return nil, core.NewError(core.INVALID_ARGUMENT, "ai.RespondWith: tool request is for %q, not %q", toolReq.ToolRequest.Name, t.Name())
	}

	cfg := &RespondOptions{}
	for _, opt := range opts {
		if err := opt.applyRespondWith(cfg); err != nil {
			return nil, core.NewError(core.INVALID_ARGUMENT, "ai.RespondWith: %v", err)
		}
	}

	newToolResp := NewResponseForToolRequest(toolReq, output)
	newToolResp.Metadata = map[string]any{
		"interruptResponse": true,
	}
	if cfg.Metadata != nil {
		newToolResp.Metadata["interruptResponse"] = cfg.Metadata
	}

	return newToolResp, nil
}

// RestartWith creates a part for [WithToolRestarts] to re-execute an interrupted tool call with additional context.
//
// Example:
//
//	part, err := myTool.RestartWith(toolReq, WithNewInput(newInput), WithResumedMetadata[MyInput](meta))
func (t *ToolDef[In, Out]) RestartWith(toolReq *Part, opts ...RestartWithOption[In]) (*Part, error) {
	if toolReq == nil {
		return nil, core.NewError(core.INVALID_ARGUMENT, "ai.RestartWith: toolReq is nil")
	}
	if !toolReq.IsToolRequest() {
		return nil, core.NewError(core.INVALID_ARGUMENT, "ai.RestartWith: part is not a tool request")
	}
	if toolReq.ToolRequest.Name != t.Name() {
		return nil, core.NewError(core.INVALID_ARGUMENT, "ai.RestartWith: tool request is for %q, not %q", toolReq.ToolRequest.Name, t.Name())
	}

	cfg := &RestartOptions{}
	for _, opt := range opts {
		if err := opt.applyRestartWith(cfg); err != nil {
			return nil, core.NewError(core.INVALID_ARGUMENT, "ai.RestartWith: %v", err)
		}
	}

	newInput := toolReq.ToolRequest.Input
	var originalInput any

	if cfg.ReplaceInput != nil {
		originalInput = newInput
		newInput = cfg.ReplaceInput
	}

	newMeta := maps.Clone(toolReq.Metadata)
	if newMeta == nil {
		newMeta = make(map[string]any)
	}

	newMeta["resumed"] = true
	if cfg.ResumedMetadata != nil {
		newMeta["resumed"] = cfg.ResumedMetadata
	}

	if originalInput != nil {
		newMeta["replacedInput"] = originalInput
	}

	delete(newMeta, "interrupt")

	newToolReqPart := NewToolRequestPart(&ToolRequest{
		Name:  toolReq.ToolRequest.Name,
		Ref:   toolReq.ToolRequest.Ref,
		Input: newInput,
	})
	newToolReqPart.Metadata = newMeta

	return newToolReqPart, nil
}

// resolveUniqueTools resolves the list of tool refs to a list of all tool names and new tools that must be registered.
// Returns an error if there are tool refs with duplicate names.
func resolveUniqueTools(r api.Registry, toolRefs []ToolRef) (toolNames []string, newTools []Tool, err error) {
	toolMap := make(map[string]bool)

	for _, toolRef := range toolRefs {
		name := toolRef.Name()

		if toolMap[name] {
			return nil, nil, core.NewError(core.INVALID_ARGUMENT, "duplicate tool %q", name)
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
