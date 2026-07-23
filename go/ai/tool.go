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

// ToolArg is the interface for tool arguments to generate calls (e.g.
// [WithTools]). It can either be a tool value ([*Tool], [*InterruptibleTool],
// or any [AnyTool]) or a [ToolName] to be looked up.
type ToolArg interface {
	Name() string
}

// ToolName is a distinct type for a tool name.
// It is meant to be passed where a ToolArg is expected but no tool value is had.
type ToolName string

// Name returns the name of the tool.
func (t ToolName) Name() string {
	return (string)(t)
}

// AnyTool is the type-erased handle for a tool, used where the concrete
// input/output types are not known: registry lookups, the generate loop, and
// middleware. Typed tools ([*Tool], [*InterruptibleTool]) implement it.
type AnyTool interface {
	// Name returns the name of the tool.
	Name() string
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
		return core.NewError(core.INVALID_ARGUMENT, "part (kind %s) is not an interrupted tool request", p.Kind)
	}
	if p.ToolRequest.Name != toolName {
		return core.NewError(core.INVALID_ARGUMENT, "tool request is for %q, expected %q", p.ToolRequest.Name, toolName)
	}
	return nil
}

// restartOptions holds configuration for restarting an interrupted tool.
type restartOptions struct {
	resume      any
	resumeSet   bool
	newInput    any
	newInputSet bool
}

// RestartOption is an option for restarting an interrupted tool via
// [tool.Restart] or [InterruptibleTool.Restart]. Create one with [WithResume]
// or [WithNewInput], or with the compile-time checked equivalents
// [InterruptibleTool.WithResume] and [InterruptibleTool.WithNewInput].
type RestartOption interface {
	applyRestart(*restartOptions) error
}

type restartOptionFunc func(*restartOptions) error

func (f restartOptionFunc) applyRestart(cfg *restartOptions) error { return f(cfg) }

// WithResume carries data to the tool function's resume parameter when it
// re-executes, e.g. the user's answer to the question the tool interrupted
// with; middleware reads it via [tool.ResumeData]. data must serialize to a
// JSON object (a struct or a map); see [tool.Interrupt] for the rationale.
// Omit it to restart the tool without data, as an implicit approval.
// [InterruptibleTool.WithResume] is the compile-time checked equivalent.
func WithResume(resume any) RestartOption {
	return restartOptionFunc(func(cfg *restartOptions) error {
		if cfg.resumeSet {
			return errors.New("cannot set resume data more than once (WithResume)")
		}
		cfg.resume = resume
		cfg.resumeSet = true
		return nil
	})
}

// WithNewInput provides a new input for the tool when it is re-executed on
// restart, for example when the user revised an action before confirming. The
// tool can read the original input via [tool.OriginalInput]. The input is
// validated against the tool's input schema when the tool re-executes.
// [InterruptibleTool.WithNewInput] is the compile-time checked equivalent.
func WithNewInput(input any) RestartOption {
	return restartOptionFunc(func(cfg *restartOptions) error {
		if cfg.newInputSet {
			return errors.New("cannot set a new input more than once (WithNewInput)")
		}
		cfg.newInput = input
		cfg.newInputSet = true
		return nil
	})
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

var (
	_ AnyTool = (*Tool[any, any])(nil)
	_ AnyTool = (*InterruptibleTool[any, any, any])(nil)
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
// send data to the caller; the caller inspects it with [tool.InterruptData] and
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
		if err := opt.applyTool(toolOpts); err != nil {
			panic(fmt.Errorf("%s %q: %w", fnName, name, err))
		}
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

	a := core.NewAction(name, api.ActionTypeToolV2, &core.ActionOptions{
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
		return nil, core.NewError(core.INVALID_ARGUMENT, "ai.Tool.RunRaw: tool called on a nil tool; check that all tools are defined")
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
// [tool.Restart] is the type-erased equivalent for callers that don't have
// the tool value in scope; this method additionally validates that the
// interrupted part belongs to this tool.
func (t *InterruptibleTool[In, Out, Res]) Restart(interruptPart *Part, opts ...RestartOption) (*Part, error) {
	if err := validateInterruptedPart(interruptPart, t.Name()); err != nil {
		return nil, core.NewError(core.INVALID_ARGUMENT, "InterruptibleTool.Restart: %v", err)
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

// NewRestartPart creates a restart [Part] for re-executing an interrupted
// tool call, for use with [WithToolRestarts]. The interruptedPart must be an
// interrupted tool request (as received via [ModelResponse.Interrupts]).
// Configure it with [WithResume] and [WithNewInput].
//
// Most callers use [tool.Restart] or [InterruptibleTool.Restart] instead;
// this is the underlying constructor.
func NewRestartPart(interruptPart *Part, opts ...RestartOption) (*Part, error) {
	if !interruptPart.IsInterrupt() {
		return nil, core.NewError(core.INVALID_ARGUMENT, "ai.NewRestartPart: part is not an interrupted tool request")
	}
	return newRestartPart("ai.NewRestartPart", interruptPart, opts)
}

// newRestartPart builds the tool request [Part] that re-executes an
// interrupted call, applying the given restart options. fnName is woven into
// error messages. The new part keeps the interrupted part's user metadata but
// drops its interrupt state; when a new input is provided, the original is
// preserved on [ToolRestart.OriginalInput].
func newRestartPart(fnName string, interruptPart *Part, opts []RestartOption) (*Part, error) {
	resOpts := &restartOptions{}
	for _, opt := range opts {
		if err := opt.applyRestart(resOpts); err != nil {
			return nil, core.NewError(core.INVALID_ARGUMENT, "%s: %v", fnName, err)
		}
	}
	if resOpts.resume != nil {
		if err := validateInterruptPayload(resOpts.resume, "resume data"); err != nil {
			return nil, core.NewError(core.INVALID_ARGUMENT, "%s: %v", fnName, err)
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
// [tool.Respond] is the type-erased equivalent for callers that don't have
// the tool value in scope; this method additionally validates that the
// interrupted part belongs to this tool and accepts a strongly-typed output.
func (t *InterruptibleTool[In, Out, Res]) Respond(interruptPart *Part, output Out) (*Part, error) {
	if err := validateInterruptedPart(interruptPart, t.Name()); err != nil {
		return nil, core.NewError(core.INVALID_ARGUMENT, "InterruptibleTool.Respond: %v", err)
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
			return nil, nil, core.NewError(core.INVALID_ARGUMENT, "duplicate tool %q", name)
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
