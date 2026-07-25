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
	"sync"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal/base"
)

// ToolFunc is the function type for tool implementations. It receives a plain
// [context.Context]; use the ai/tool package helpers ([tool.AttachParts],
// [tool.SendPartial], [tool.SendChunk]) for capabilities beyond returning a
// value.
type ToolFunc[In, Out any] = func(ctx context.Context, input In) (Out, error)

// InterruptibleToolFunc is the function type for tools created with
// [DefineInterruptibleTool] and [NewInterruptibleTool]. The resume parameter
// is non-nil when the tool is being re-executed after an interrupt; it carries
// the data the caller passed when restarting.
type InterruptibleToolFunc[In, Out, Res any] = func(ctx context.Context, input In, resume *Res) (Out, error)

// ToolName is a distinct type for a tool name.
// It is meant to be passed where a [ToolArg] is expected (e.g. [WithTools])
// but no tool value is had.
type ToolName string

// Name returns the name of the tool.
func (t ToolName) Name() string {
	return (string)(t)
}

func (ToolName) toolArg() {}

// AnyTool is the type-erased handle for a tool, used where the concrete
// input/output types are not known: registry lookups, the generate loop, and
// middleware. Typed tools ([*Tool], [*InterruptibleTool]) implement it.
//
// Embedding [ToolArg] seals AnyTool to this package: tools are built with
// [NewTool], [DefineTool], and their interruptible counterparts, which is the
// only supported way to supply one. Behavior around a tool call is customized
// with a [Hooks.WrapTool] middleware hook rather than by reimplementing this
// interface.
type AnyTool interface {
	ToolArg
	// Definition returns the definition for this tool to be passed to models.
	Definition() *ToolDefinition
	// RunRaw runs this tool using the provided raw input and returns the full
	// [MultipartToolResponse]: the output plus any content parts attached via
	// [tool.AttachParts].
	RunRaw(ctx context.Context, input any) (*MultipartToolResponse, error)
	// Register registers the tool with the given registry.
	Register(r api.Registry)
}

// InterruptError is returned from a tool function (typically via
// [tool.Interrupt]) to pause execution and return control to the caller.
// Data must serialize to a JSON object (a struct or a map), since it is
// carried on the interrupted tool request part as [ToolInterrupt] data,
// which the wire protocol encodes as a JSON object in the part's metadata.
//
// Middleware may also return it from a WrapTool hook to interrupt a tool call
// without executing the tool.
type InterruptError struct {
	Data any
}

func (e *InterruptError) Error() string {
	return "tool interrupted"
}

// validateInterruptPayload checks that an interrupt or resume payload
// serializes to a JSON object, as the wire contract requires (an object, or
// bare when the payload is nil). what names the payload in the error message.
func validateInterruptPayload(data any, what string) error {
	if data == nil {
		return nil
	}
	if _, ok := data.(map[string]any); ok {
		return nil
	}
	if _, err := base.StructToMap(data); err != nil {
		return fmt.Errorf("%s must serialize to a JSON object (a struct or map), got %T: %w", what, data, err)
	}
	return nil
}

// validateInterruptedPart checks that p is an interrupted tool request
// belonging to the named tool. fn is woven into error messages.
func validateInterruptedPart(p *Part, toolName string) error {
	if !p.IsInterrupt() {
		return status.Errorf(ErrInvalidPart, "part (kind %s) is not an interrupted tool request", p.Kind)
	}
	if p.ToolRequest.Name != toolName {
		return status.Errorf(ErrInvalidPart, "tool request is for %q, want %q", p.ToolRequest.Name, toolName)
	}
	return nil
}

// toolStrictKey is the metadata key under metadata["tool"] used to carry the
// per-tool strict-schema flag through the action metadata and onto
// [ToolDefinition.Metadata]. Plugins consume this key directly.
const toolStrictKey = "strict"

// Tool is a typed tool implementation backed by a registry action.
// Create one with [DefineTool] or [NewTool].
type Tool[In, Out any] struct {
	action   api.Action   // The underlying action.
	registry api.Registry // Registry for schema resolution. Set when registered.
}

// tool is an unexported alias of [Tool] used as the embedded field in
// [InterruptibleTool]. Embedding via the alias promotes Tool's methods
// without exporting the field itself, so the containment stays an internal
// detail: callers can't reach the inner Tool, construct an InterruptibleTool
// by struct literal, or depend on its layout, leaving us free to restructure
// it later without breaking the API.
type tool[In, Out any] = Tool[In, Out]

// InterruptibleTool is a [Tool] that supports typed interrupt/resume.
// The Res type parameter is the type of data the caller sends back when
// resuming the tool after an interrupt. Create one with
// [DefineInterruptibleTool] or [NewInterruptibleTool]. Resolve an interrupt
// with [InterruptibleTool.Restart] or [InterruptibleTool.Respond].
type InterruptibleTool[In, Out, Res any] struct {
	tool[In, Out]
}

// Tools are registerable and can be passed as [ToolArg] (e.g. to [WithTools]),
// but are intentionally not [api.Action]: the raw action surface (RunJSON,
// Desc) speaks the internal multipart envelope, which [Tool.RunRaw] and
// [Tool.Definition] translate out of.
var (
	_ AnyTool          = (*Tool[any, any])(nil)
	_ AnyTool          = (*InterruptibleTool[any, any, any])(nil)
	_ api.Registerable = (*Tool[any, any])(nil)
	_ ToolArg          = (*Tool[any, any])(nil)
	_ ToolArg          = ToolName("")
)

// DefineTool creates a new [Tool] and registers it.
// Use [WithInputSchema] to provide a custom JSON schema instead of inferring
// from the type parameter. Use [tool.AttachParts] inside the function to
// return additional content parts alongside the output.
func DefineTool[In, Out any](
	r api.Registry,
	name, description string,
	fn ToolFunc[In, Out],
	opts ...ToolOption,
) *Tool[In, Out] {
	t := newTool("ai.DefineTool", name, description, fn, false, opts)
	t.Register(r)
	return t
}

// NewTool creates a new unregistered [Tool]. It can be passed directly to
// [Generate], which registers it for the duration of the call.
// Use [WithInputSchema] to provide a custom JSON schema instead of inferring
// from the type parameter. Use [tool.AttachParts] inside the function to
// return additional content parts alongside the output.
func NewTool[In, Out any](name, description string, fn ToolFunc[In, Out], opts ...ToolOption) *Tool[In, Out] {
	return newTool("ai.NewTool", name, description, fn, true, opts)
}

// DefineInterruptibleTool creates a new [InterruptibleTool] and registers it.
// The resumed parameter is non-nil when the tool is being re-executed after an
// interrupt. Use [tool.Interrupt] inside the function to pause execution and
// send data to the caller; the caller inspects it with [Part.InterruptAs] and
// restarts it with [InterruptibleTool.Restart] or resolves it directly with
// [InterruptibleTool.Respond].
func DefineInterruptibleTool[In, Out, Res any](
	r api.Registry,
	name, description string,
	fn InterruptibleToolFunc[In, Out, Res],
	opts ...ToolOption,
) *InterruptibleTool[In, Out, Res] {
	t := NewInterruptibleTool(name, description, fn, opts...)
	t.Register(r)
	return t
}

// NewInterruptibleTool creates a new unregistered [InterruptibleTool]. It
// can be passed directly to [Generate], which registers it for the duration of
// the call.
func NewInterruptibleTool[In, Out, Res any](
	name, description string,
	fn InterruptibleToolFunc[In, Out, Res],
	opts ...ToolOption,
) *InterruptibleTool[In, Out, Res] {
	t := newTool("ai.NewInterruptibleTool", name, description, func(ctx context.Context, input In) (Out, error) {
		var res *Res
		if v := base.ToolResumeKey.FromContext(ctx); v != nil {
			// ConvertToStrict rather than ConvertTo: a resume payload that
			// doesn't fit Res must fail the call with the decode error, not
			// silently yield a zero value.
			r, err := base.ConvertToStrict[Res](v)
			if err != nil {
				var zero Out
				return zero, fmt.Errorf("tool %q: failed to convert resume data: %w", name, err)
			}
			res = &r
		}
		return fn(ctx, input, res)
	}, true, opts)
	return &InterruptibleTool[In, Out, Res]{tool: *t}
}

// newTool builds the [Tool] value shared by all constructors. fnName is the
// user-facing constructor name used in panic messages. dynamic marks the
// action as created outside registration (the New* constructors); such tools
// register lazily when passed to [Generate].
func newTool[In, Out any](fnName, name, description string, fn ToolFunc[In, Out], dynamic bool, opts []ToolOption) *Tool[In, Out] {
	toolOpts := &toolOptions{}
	for _, opt := range opts {
		opt.applyTool(toolOpts)
	}

	// If the user provided a custom input schema, enforce that In is 'any'.
	if toolOpts.InputSchema != nil {
		typ := reflect.TypeFor[*In]()
		if typ.Elem().Kind() != reflect.Interface {
			panic(fmt.Errorf("%s %q: WithInputSchema requires In to be of type 'any', but got %v", fnName, name, typ.Elem()))
		}
	}

	metadata := map[string]any{
		"type":        api.ActionTypeToolV2,
		"name":        name,
		"description": description,
		"tool":        map[string]any{"multipart": true},
	}
	if dynamic {
		metadata["dynamic"] = true
	}
	// The action's own output schema is the multipart envelope; record the
	// real Out schema so Definition() can advertise it to models and the Dev
	// UI, including for type-erased tools returned by LookupTool.
	var zeroOut Out
	if reflect.TypeOf(zeroOut) != nil {
		metadata["originalOutputSchema"] = core.InferSchemaMap(zeroOut)
	}
	if toolOpts.StrictSchema != nil {
		metadata["tool"].(map[string]any)[toolStrictKey] = *toolOpts.StrictSchema
	}

	wrapped := func(ctx context.Context, input In) (*MultipartToolResponse, error) {
		// The sink may be called from goroutines the tool function spawns
		// (mirroring tool.SendPartial, which is also safe for concurrent use),
		// so guard the slice.
		var partsMu sync.Mutex
		var parts []*Part
		ctx = base.ToolPartSinkKey.NewContext(ctx, func(part any) {
			if p, ok := part.(*Part); ok {
				partsMu.Lock()
				parts = append(parts, p)
				partsMu.Unlock()
			}
		})

		output, err := fn(ctx, input)
		if err != nil {
			var ie *InterruptError
			if errors.As(err, &ie) {
				// Validate eagerly so a malformed interrupt surfaces where the
				// tool ran, not later in the generate loop.
				if vErr := validateInterruptPayload(ie.Data, "interrupt data"); vErr != nil {
					return nil, fmt.Errorf("tool %q: %w", name, vErr)
				}
			}
			return nil, err
		}

		resp := &MultipartToolResponse{Output: output}
		partsMu.Lock()
		if len(parts) > 0 {
			resp.Content = parts
		}
		partsMu.Unlock()
		return resp, nil
	}

	a := core.NewAction(api.ActionTypeToolV2, name, &core.ActionOptions{
		Description: description,
		Metadata:    metadata,
		InputSchema: toolOpts.InputSchema,
	}, wrapped)

	return &Tool[In, Out]{action: a}
}

// Name returns the name of the tool.
func (t *Tool[In, Out]) Name() string {
	return t.action.Name()
}

func (t *Tool[In, Out]) toolArg() {}

// Definition returns the [ToolDefinition] for this tool to be passed to models.
func (t *Tool[In, Out]) Definition() *ToolDefinition {
	desc := t.action.Desc()

	inputSchema := desc.InputSchema
	if t.registry != nil {
		if resolved, err := core.ResolveSchema(t.registry, inputSchema); err == nil {
			inputSchema = resolved
		}
	}

	// The action's output schema is the multipart envelope; advertise the real
	// output schema recorded at construction time instead.
	outputSchema := desc.OutputSchema
	if origSchema, ok := desc.Metadata["originalOutputSchema"].(map[string]any); ok {
		outputSchema = origSchema
	}
	if t.registry != nil {
		if resolved, err := core.ResolveSchema(t.registry, outputSchema); err == nil {
			outputSchema = resolved
		}
	}

	metadata := map[string]any{
		"multipart": true,
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
func (t *Tool[In, Out]) Register(r api.Registry) {
	t.registry = r
	t.action.Register(r)
}

// RunRaw runs this tool using the provided raw input (e.g. JSON parsed as
// map[string]any) and returns the full [MultipartToolResponse]: the output
// (in the Output field) plus any content parts attached via
// [tool.AttachParts].
func (t *Tool[In, Out]) RunRaw(ctx context.Context, input any) (*MultipartToolResponse, error) {
	if t == nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "ai.Tool.RunRaw: tool called on a nil tool; check that all tools are defined")
	}

	mi, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("error marshalling tool input for %v: %v", t.Name(), err)
	}
	res, err := t.action.RunJSON(ctx, mi, nil)
	if err != nil {
		return nil, fmt.Errorf("error calling tool %v: %w", t.Name(), err)
	}

	var resp MultipartToolResponse
	if err := json.Unmarshal(res.Result, &resp); err != nil {
		return nil, fmt.Errorf("error parsing tool output for %v: %v", t.Name(), err)
	}
	return &resp, nil
}

// Restart creates a restart [Part] for re-executing this interrupted tool,
// for use with [WithToolRestarts]. With no options, the tool simply
// re-executes; restarting is itself the approval. Use
// [InterruptibleTool.WithResume] to deliver data to the tool function's
// resume parameter, and [InterruptibleTool.WithNewInput] to provide a new
// input.
//
// [Part.ToRestart] is the type-erased equivalent for callers that don't have
// the tool value in scope; this method additionally validates that the
// interrupted part belongs to this tool.
func (t *InterruptibleTool[In, Out, Res]) Restart(interruptPart *Part, opts ...RestartOption) (*Part, error) {
	if err := validateInterruptedPart(interruptPart, t.Name()); err != nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "InterruptibleTool.Restart: %w", err)
	}
	return newRestartPart("InterruptibleTool.Restart", interruptPart, opts)
}

// WithResume returns a [RestartOption] carrying data to this tool's resume
// parameter, compile-time checked against the tool's Res type. See
// [WithResume] for the semantics.
func (t *InterruptibleTool[In, Out, Res]) WithResume(resume Res) RestartOption {
	return WithResume(resume)
}

// WithNewInput returns a [RestartOption] providing a new input for this tool
// when it re-executes, compile-time checked against the tool's In type. See
// [WithNewInput] for the semantics.
func (t *InterruptibleTool[In, Out, Res]) WithNewInput(input In) RestartOption {
	return WithNewInput(input)
}

// newRestartPart builds the tool request [Part] that re-executes an
// interrupted call, applying the given restart options. fnName is woven into
// error messages. The new part keeps the interrupted part's user metadata but
// drops its interrupt state; when a new input is provided, the original is
// preserved on [ToolRestart.OriginalInput].
func newRestartPart(fnName string, interruptPart *Part, opts []RestartOption) (*Part, error) {
	resOpts := &restartOptions{}
	for _, opt := range opts {
		opt.applyRestart(resOpts)
	}
	if resOpts.resume != nil {
		if err := validateInterruptPayload(resOpts.resume, "resume data"); err != nil {
			return nil, status.Errorf(status.ErrInvalidArgument, "%s: %w", fnName, err)
		}
	}
	toolReq := interruptPart.ToolRequest
	input := toolReq.Input
	var original any
	if resOpts.newInput != nil {
		original = input
		input = resOpts.newInput
	}
	restartedPart := NewToolRequestPart(&ToolRequest{Name: toolReq.Name, Ref: toolReq.Ref, Input: input})
	restartedPart.Metadata = maps.Clone(interruptPart.Metadata)
	restartedPart.Restart = &ToolRestart{Resume: resOpts.resume, OriginalInput: original}
	return restartedPart, nil
}

// Respond creates a tool response [Part] for this interrupted tool, for use
// with [WithToolResponses]. Instead of re-executing the tool (as [InterruptibleTool.Restart]
// does), this provides a pre-computed result directly.
//
// [Part.ToResponse] is the type-erased equivalent for callers that don't have
// the tool value in scope; this method additionally validates that the
// interrupted part belongs to this tool and accepts a strongly-typed output.
func (t *InterruptibleTool[In, Out, Res]) Respond(interruptPart *Part, output Out) (*Part, error) {
	if err := validateInterruptedPart(interruptPart, t.Name()); err != nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "InterruptibleTool.Respond: %w", err)
	}
	toolResp, err := NewResponseForToolRequest(interruptPart, output)
	if err != nil {
		return nil, err
	}
	// interruptResponse marks the part so the generate loop resolves the
	// interrupt instead of re-executing the tool.
	toolResp.Metadata = map[string]any{base.ToolMetaInterruptResponse: true}
	return toolResp, nil
}

// InterruptAs returns this interrupted tool request's interrupt data as a
// typed value, typically to decide between [Part.ToRestart] and
// [Part.ToResponse]. Returns the zero value and false if the part is not an
// interrupt, the interrupt carries no data, or the type doesn't match.
//
// This reads the Data field of the [Part.Interrupt] state.
//
//	for _, part := range resp.Interrupts() {
//		req, ok := part.InterruptAs[TransferInterrupt]()
//	}
func (p *Part) InterruptAs[T any]() (T, bool) {
	var zero T
	if p == nil || !p.IsInterrupt() || p.Interrupt.Data == nil {
		return zero, false
	}
	return base.ConvertTo[T](p.Interrupt.Data)
}

// ToRestart converts this interrupted tool request into a restart [Part] that
// re-executes the tool, for use with [WithToolRestarts]. The receiver must be
// an interrupted tool request, as received via [ModelResponse.Interrupts].
// Use [WithResume] to deliver data to the tool function's resume parameter,
// and [WithNewInput] to provide a new input.
//
// With no options the tool re-executes with a non-nil, zero-valued resume
// parameter, so restarting is itself the approval for tools that key on the
// presence of a resume. See [WithResume] for what a bare restart delivers.
// [ModelResponse.InterruptRestarts] does this for every interrupt at once.
//
// [InterruptibleTool.Restart] is the typed equivalent for callers that have
// the tool value in scope; it additionally validates that the interrupted
// part belongs to that tool.
//
//	for _, part := range resp.Interrupts() {
//		restart, err := part.ToRestart(ai.WithResume(Confirmation{Approved: true}))
//	}
func (p *Part) ToRestart(opts ...RestartOption) (*Part, error) {
	if !p.IsInterrupt() {
		return nil, status.Errorf(ErrInvalidPart, "ai.Part.ToRestart: part is not an interrupted tool request")
	}
	return newRestartPart("ai.Part.ToRestart", p, opts)
}

// ToResponse converts this interrupted tool request into a tool response
// [Part], for use with [WithToolResponses]. Instead of re-executing the tool
// (as [Part.ToRestart] does), this provides a pre-computed result directly.
//
// [InterruptibleTool.Respond] is the typed equivalent for callers that have
// the tool value in scope; it additionally validates that the interrupted
// part belongs to that tool and accepts a strongly-typed output.
func (p *Part) ToResponse(output any) (*Part, error) {
	if !p.IsInterrupt() {
		return nil, status.Errorf(status.ErrInvalidArgument, "ai.Part.ToResponse: part is not an interrupted tool request")
	}
	toolResp, err := NewResponseForToolRequest(p, output)
	if err != nil {
		return nil, err
	}
	// interruptResponse marks the part so the generate loop resolves the
	// interrupt instead of re-executing the tool.
	toolResp.Metadata = map[string]any{base.ToolMetaInterruptResponse: true}
	return toolResp, nil
}

// LookupTool looks up the tool in the registry by provided name and returns it.
// Since the types are not known at lookup time, it returns a type-erased tool.
func LookupTool(r api.Registry, name string) AnyTool {
	if name == "" {
		return nil
	}
	provider, id := api.ParseName(name)
	action := r.ResolveAction(api.NewKey(api.ActionTypeToolV2, provider, id))
	if action == nil {
		return nil
	}
	return &Tool[any, any]{action: action, registry: r}
}

// resolveUniqueTools resolves the list of tool refs to a list of all tool names and new tools that must be registered.
// Returns an error if there are tool refs with duplicate names.
func resolveUniqueTools(r api.Registry, toolArgs []ToolArg) (toolNames []string, newTools []AnyTool, err error) {
	toolMap := make(map[string]bool)

	for _, toolRef := range toolArgs {
		name := toolRef.Name()

		if toolMap[name] {
			return nil, nil, status.Errorf(status.ErrInvalidArgument, "duplicate tool %q", name)
		}
		toolMap[name] = true
		toolNames = append(toolNames, name)

		if LookupTool(r, name) == nil {
			if tool, ok := toolRef.(AnyTool); ok {
				newTools = append(newTools, tool)
			}
		}
	}

	return toolNames, newTools, nil
}
