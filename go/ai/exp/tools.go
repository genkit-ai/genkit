// Copyright 2026 Google LLC
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

package exp

import (
	"context"
	"errors"
	"fmt"
	"reflect"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/ai/exp/tool"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/internal/base"
)

// ToolFunc is the function signature for tools created with [NewTool].
type ToolFunc[In, Out any] = func(ctx context.Context, input In) (Out, error)

// InterruptibleToolFunc is the function signature for tools created with
// [DefineInterruptibleTool] and [NewInterruptibleTool]. The resumed parameter
// is non-nil when the tool is being re-executed after an interrupt.
type InterruptibleToolFunc[In, Out, Resume any] = func(ctx context.Context, input In, res *Resume) (Out, error)

// Tool wraps an [ai.Tool] with experimental x package features
// such as a plain [context.Context] function signature and [tool.AttachParts].
//
// DEPRECATED(breaking): With breaking changes, Tool would not wrap ai.ToolAction.
// It would be the primary tool type, backed directly by core.NewActionOf,
// eliminating the inner field and all delegation methods below.
type Tool[In, Out any] struct {
	inner *ai.ToolAction[In, *ai.MultipartToolResponse] // DEPRECATED(breaking): remove wrapper; Tool owns the action directly.
}

// DEPRECATED(breaking): The methods below exist to implement ai.Tool on top of
// the wrapped ai.ToolAction. Most are pure delegation; Definition additionally
// restores the real output schema (see its comment). With breaking changes, Tool
// would own the action directly and implement these natively, inferring the
// output schema from Out without the override.

// Name returns the name of the tool.
func (t *Tool[In, Out]) Name() string { return t.inner.Name() }

// Definition returns the [ai.ToolDefinition] for this tool.
//
// The inner tool is built on [ai.NewMultipartTool], whose function returns
// *[ai.MultipartToolResponse], so the inner definition would advertise that
// envelope as the output schema. We override OutputSchema with the schema
// inferred from the Out type parameter, making the definition equivalent to what
// [ai.NewTool] exposes (the real output type) rather than leaking the multipart
// envelope to the model and Dev UI. Genkit infers schemas with DoNotReference,
// so the result is fully inlined and needs no registry resolution.
//
// A custom schema from [ai.WithOutputSchema] or [ai.WithOutputSchemaName]
// (which require Out to be 'any') passes through from the inner tool
// untouched: with Out being 'any' there is no inferred schema to override it.
func (t *Tool[In, Out]) Definition() *ai.ToolDefinition {
	def := t.inner.Definition()
	if schema := inferOutputSchema[Out](); schema != nil {
		def.OutputSchema = schema
	}
	return def
}

// inferOutputSchema returns the inlined JSON schema for the Out type parameter,
// or nil when Out carries no schema (e.g. any), mirroring how [ai.NewTool]
// derives its output schema from the output type.
func inferOutputSchema[Out any]() map[string]any {
	var zero Out
	if reflect.TypeOf(zero) == nil {
		return nil
	}
	return core.InferSchemaMap(zero)
}

// RunRaw runs the tool with raw input.
func (t *Tool[In, Out]) RunRaw(ctx context.Context, input any) (any, error) {
	return t.inner.RunRaw(ctx, input)
}

// RunRawMultipart runs the tool with raw input and returns the full multipart response.
func (t *Tool[In, Out]) RunRawMultipart(ctx context.Context, input any) (*ai.MultipartToolResponse, error) {
	return t.inner.RunRawMultipart(ctx, input)
}

// Respond creates a tool response part for an interrupted tool request.
func (t *Tool[In, Out]) Respond(toolReq *ai.Part, outputData any, opts *ai.RespondOptions) *ai.Part {
	return t.inner.Respond(toolReq, outputData, opts)
}

// Restart creates a restart part using the legacy [ai.RestartOptions].
//
// DEPRECATED(breaking): Remove entirely. Superseded by [InterruptibleTool.Resume].
func (t *Tool[In, Out]) Restart(toolReq *ai.Part, opts *ai.RestartOptions) *ai.Part {
	return t.inner.Restart(toolReq, opts)
}

// Register registers the tool with the given registry.
func (t *Tool[In, Out]) Register(r api.Registry) { t.inner.Register(r) }

// InterruptibleTool is a [Tool] that supports typed interrupt/resume.
// The Res type parameter is the type of data the caller sends back when
// resuming the tool after an interrupt.
type InterruptibleTool[In, Out, Res any] struct {
	Tool[In, Out]
}

// Resume creates a restart part for resuming this interrupted tool with typed data.
// The data will be deserialized into the *Res parameter of the tool function
// when it is re-executed.
//
// Res must serialize to a JSON object (a struct or a map), since it is carried
// as structured metadata on the restart part; see [tool.Interrupt].
//
// Unlike [tool.Resume], this method also validates that the interrupted part
// belongs to this tool.
func (t *InterruptibleTool[In, Out, Resume]) Resume(part *ai.Part, res Resume) (*ai.Part, error) {
	if part == nil || !part.IsInterrupt() {
		return nil, fmt.Errorf("Resume: part is not an interrupted tool request")
	}
	if part.ToolRequest.Name != t.Name() {
		return nil, fmt.Errorf("Resume: tool request is for %q, not %q", part.ToolRequest.Name, t.Name())
	}
	return tool.Resume(part, res)
}

// Respond creates a tool response [ai.Part] for an interrupted tool request.
// Instead of re-executing the tool (as [Resume] does), this provides a
// pre-computed result directly.
//
// Unlike [tool.Respond], this method validates that the interrupted part
// belongs to this tool and accepts a strongly-typed output.
func (t *InterruptibleTool[In, Out, Resume]) Respond(part *ai.Part, output Out) (*ai.Part, error) {
	if part == nil || !part.IsInterrupt() {
		return nil, fmt.Errorf("Respond: part is not an interrupted tool request")
	}
	if part.ToolRequest.Name != t.Name() {
		return nil, fmt.Errorf("Respond: tool request is for %q, not %q", part.ToolRequest.Name, t.Name())
	}
	return tool.Respond(part, output)
}

// requireAnyOutForSchemaOptions panics when an output schema option is
// present and Out is a concrete type, mirroring [ai.NewTool]'s guard: the
// custom schema stands in for an Out type parameter of 'any', and a concrete
// Out would silently disagree with the advertised schema (this package's
// [Tool.Definition] prefers the schema inferred from Out). With Out being
// 'any', the option flows through the inner multipart tool untouched.
func requireAnyOutForSchemaOptions[Out any](ctor, name string, opts []ai.ToolOption) {
	for _, opt := range opts {
		if _, ok := opt.(ai.OutputSchemaOption); ok {
			if typ := reflect.TypeFor[Out](); typ.Kind() != reflect.Interface {
				panic(fmt.Errorf("%s %q: WithOutputSchema and WithOutputSchemaName require Out to be of type 'any', but got %v", ctor, name, typ))
			}
			return
		}
	}
}

// NewTool creates a new unregistered tool with a simple function signature.
// Use [tool.AttachParts] inside the function to return additional content parts.
func NewTool[In, Out any](
	name, description string,
	fn ToolFunc[In, Out],
	opts ...ai.ToolOption,
) *Tool[In, Out] {
	requireAnyOutForSchemaOptions[Out]("exp.NewTool", name, opts)
	// DEPRECATED(breaking): Call core.NewActionOf directly instead of wrapping ai.NewMultipartTool.
	inner := ai.NewMultipartTool(name, description, wrapSimpleFunc(fn), opts...)
	return &Tool[In, Out]{inner: inner}
}

// NewInterruptibleTool creates a new unregistered interruptible tool.
func NewInterruptibleTool[In, Out, Res any](
	name, description string,
	fn InterruptibleToolFunc[In, Out, Res],
	opts ...ai.ToolOption,
) *InterruptibleTool[In, Out, Res] {
	requireAnyOutForSchemaOptions[Out]("exp.NewInterruptibleTool", name, opts)
	// DEPRECATED(breaking): Call core.NewActionOf directly instead of wrapping ai.NewMultipartTool.
	inner := ai.NewMultipartTool(name, description, wrapInterruptibleFunc(fn), opts...)
	return &InterruptibleTool[In, Out, Res]{Tool: Tool[In, Out]{inner: inner}}
}

// DEPRECATED(breaking): wrapSimpleFunc exists to adapt our func(context.Context, In) (Out, error)
// to ai.MultipartToolFunc[In] (which takes *ai.ToolContext). With breaking changes,
// core.NewActionOf would accept our function signature directly, and the ToolContext
// adapter, resumed/originalInput extraction from ToolContext, and interrupt error
// conversion would all be unnecessary.
func wrapSimpleFunc[In, Out any](fn ToolFunc[In, Out]) ai.MultipartToolFunc[In] {
	return func(tc *ai.ToolContext, input In) (*ai.MultipartToolResponse, error) {
		return runToolFunc(tc, input, fn)
	}
}

// DEPRECATED(breaking): Same as wrapSimpleFunc — exists only to bridge between
// the new function signature and ai.MultipartToolFunc/ai.ToolContext.
func wrapInterruptibleFunc[In, Out, Resume any](fn InterruptibleToolFunc[In, Out, Resume]) ai.MultipartToolFunc[In] {
	return func(tc *ai.ToolContext, input In) (*ai.MultipartToolResponse, error) {
		return runToolFunc(tc, input, func(ctx context.Context, input In) (Out, error) {
			// DEPRECATED(breaking): Resumed data would come from context keys set by
			// the generate loop directly, not from ai.ToolContext.Resumed.
			var res *Resume
			if tc.Resumed != nil {
				r, err := base.MapToStruct[Resume](tc.Resumed)
				if err != nil {
					var zero Out
					return zero, fmt.Errorf("aix.wrapInterruptibleFunc: failed to convert resumed data: %w", err)
				}
				res = &r
			}
			return fn(ctx, input, res)
		})
	}
}

// runToolFunc adapts a plain func(context.Context, In) (Out, error) to one
// ai.MultipartToolFunc invocation: it sets up the per-call context (parts
// collector and, on resume, the original input), runs invoke, converts a
// tool.InterruptError into ai's interrupt signal, and folds any parts attached
// via tool.AttachParts into the response. The simple and interruptible wrappers
// differ only in how they build invoke, so they share this body.
//
// DEPRECATED(breaking): This bridging disappears once core.NewActionOf accepts
// the plain function signature directly (see wrapSimpleFunc).
func runToolFunc[In, Out any](tc *ai.ToolContext, input In, invoke ToolFunc[In, Out]) (*ai.MultipartToolResponse, error) {
	ctx := tc.Context
	ctx, collector := tool.NewPartsContext(ctx)
	if tc.OriginalInput != nil {
		ctx = tool.SetOriginalInput(ctx, tc.OriginalInput)
	}

	output, err := invoke(ctx, input)
	if err != nil {
		return nil, convertInterruptError(tc, err)
	}

	resp := &ai.MultipartToolResponse{Output: output}
	if parts := collector(); len(parts) > 0 {
		resp.Content = parts
	}
	return resp, nil
}

// DEPRECATED(breaking): convertInterruptError exists because tool.InterruptError
// must be converted to ai's unexported toolInterruptError (via tc.Interrupt) for
// the generate loop to recognize it. With breaking changes, the generate loop
// would recognize tool.InterruptError directly.
func convertInterruptError(tc *ai.ToolContext, err error) error {
	var ie *tool.InterruptError
	if errors.As(err, &ie) {
		m, mapErr := toMap(ie.Data)
		if mapErr != nil {
			return fmt.Errorf("tool.Interrupt: interrupt data must serialize to a JSON object (a struct or map), got %T: %w", ie.Data, mapErr)
		}
		return tc.Interrupt(&ai.InterruptOptions{Metadata: m})
	}
	return err
}

// DEPRECATED(breaking): toMap exists only for convertInterruptError above.
func toMap(v any) (map[string]any, error) {
	if v == nil {
		return nil, nil
	}
	if m, ok := v.(map[string]any); ok {
		return m, nil
	}
	return base.StructToMap(v)
}
