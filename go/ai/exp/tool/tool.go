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

// Package tool provides runtime helpers for use inside tool functions.
//
// APIs in this package are under active development and may change in any
// minor version release. Use with caution in production environments.
package tool

import (
	"context"
	"fmt"
	"maps"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/internal/base"
)

// --- Interrupt ---

// InterruptError is returned by [Interrupt] to signal tool interruption.
type InterruptError struct {
	Data any
}

func (e *InterruptError) Error() string {
	return "tool interrupted"
}

// Interrupt interrupts tool execution and sends data to the caller.
// The caller can read this data with [InterruptAs] and resume the tool
// with [Resume].
//
// data must serialize to a JSON object (a struct or a map), since it is
// carried as structured metadata on the interrupted tool request. A value
// that serializes to a JSON scalar or array (e.g. a string, number, or
// slice) makes the tool fail when it returns; wrap such values in a struct
// or map field instead.
func Interrupt(data any) error {
	return &InterruptError{Data: data}
}

// InterruptAs extracts typed interrupt data from an interrupted tool request [ai.Part].
// Returns the zero value and false if the part is not an interrupt or the type doesn't match.
func InterruptAs[T any](p *ai.Part) (T, bool) {
	return ai.InterruptAs[T](p)
}

// Resume creates a restart [ai.Part] for resuming an interrupted tool call.
// The interruptedPart must be an interrupted tool request (as received via
// [InterruptAs] or [ai.ModelResponse.Interrupts]). The data is delivered to
// the tool function's resume parameter when it is re-executed.
//
// This is a convenience alternative to [aix.InterruptibleTool.Resume] that
// does not require access to the tool definition.
//
// data must serialize to a JSON object (a struct or a map); see [Interrupt]
// for the rationale. A value that serializes to a JSON scalar or array
// returns an error.
func Resume[Res any](interruptedPart *ai.Part, data Res) (*ai.Part, error) {
	if interruptedPart == nil || !interruptedPart.IsInterrupt() {
		return nil, fmt.Errorf("tool.Resume: part is not an interrupted tool request")
	}

	m, err := base.StructToMap(data)
	if err != nil {
		return nil, fmt.Errorf("tool.Resume: resume data must serialize to a JSON object (a struct or map), got %T: %w", data, err)
	}

	newMeta := maps.Clone(interruptedPart.Metadata)
	if newMeta == nil {
		newMeta = make(map[string]any)
	}
	newMeta["resumed"] = m
	delete(newMeta, "interrupt")

	newPart := ai.NewToolRequestPart(&ai.ToolRequest{
		Name:  interruptedPart.ToolRequest.Name,
		Ref:   interruptedPart.ToolRequest.Ref,
		Input: interruptedPart.ToolRequest.Input,
	})
	newPart.Metadata = newMeta
	return newPart, nil
}

// --- Respond ---

// Respond creates a tool response [ai.Part] for an interrupted tool request.
// Instead of re-executing the tool (as [Resume] does), this provides a
// pre-computed result directly.
//
// This is a convenience alternative to [aix.Tool.Respond] that does not
// require access to the tool definition.
func Respond(interruptedPart *ai.Part, output any) (*ai.Part, error) {
	if interruptedPart == nil || !interruptedPart.IsInterrupt() {
		return nil, fmt.Errorf("tool.Respond: part is not an interrupted tool request")
	}

	resp := ai.NewResponseForToolRequest(interruptedPart, output)
	resp.Metadata = map[string]any{
		"interruptResponse": true,
	}
	return resp, nil
}

// --- SendPartial ---

// SendPartial streams a partial tool response during tool execution.
// The output is arbitrary structured data (e.g., progress information)
// that will be delivered to the client as a partial [ai.ToolResponse].
//
// This is best-effort: if no streaming callback is available (e.g., the
// tool is called via a non-streaming Generate), the call is a no-op.
// The tool's final return value is always the authoritative response.
//
// Example:
//
//	tool.SendPartial(ctx, map[string]any{"step": "uploading", "progress": 50})
func SendPartial(ctx context.Context, output any) {
	send := base.ToolPartialSenderKey.FromContext(ctx)
	if send == nil {
		return
	}
	send(ctx, output)
}

// --- SendChunk ---

// SendChunk streams a raw [ai.ModelResponseChunk] during tool execution.
// Unlike [SendPartial], which wraps arbitrary data in a partial tool response,
// SendChunk gives the tool full control over the chunk contents.
//
// This is best-effort: if no streaming callback is available (e.g., the
// tool is called via a non-streaming Generate), the call is a no-op.
// The tool's final return value is always the authoritative response.
func SendChunk(ctx context.Context, chunk *ai.ModelResponseChunk) {
	send := base.ToolChunkSenderKey.FromContext(ctx)
	if send == nil {
		return
	}
	send(ctx, chunk)
}

// --- AttachParts ---

// AttachParts attaches additional content parts (e.g., media) to the tool's
// response. This can be called from any tool to produce a multipart response
// without changing the function signature.
func AttachParts(ctx context.Context, parts ...*ai.Part) {
	c := partsCollectorKey.FromContext(ctx)
	if c == nil {
		return
	}
	c.parts = append(c.parts, parts...)
}

// --- OriginalInput ---

// OriginalInput returns the original input if the caller replaced it during
// restart. Returns the zero value and false if the input was not replaced
// or the tool is not being resumed.
func OriginalInput[In any](ctx context.Context) (In, bool) {
	v := originalInputKey.FromContext(ctx)
	if v == nil {
		var zero In
		return zero, false
	}
	return base.ConvertTo[In](v)
}

// --- Internal plumbing (used by the aix package, not for end users) ---

var originalInputKey = base.NewContextKey[any]()
var partsCollectorKey = base.NewContextKey[*partsCollector]()

// partsCollector accumulates content parts attached during tool execution.
type partsCollector struct {
	parts []*ai.Part
}

// SetOriginalInput stores the original input in the context.
// This is internal plumbing used by the aix package.
func SetOriginalInput(ctx context.Context, input any) context.Context {
	return originalInputKey.NewContext(ctx, input)
}

// NewPartsContext returns a context with a parts collector.
// This is internal plumbing used by the aix package.
func NewPartsContext(ctx context.Context) (context.Context, func() []*ai.Part) {
	c := &partsCollector{}
	ctx = partsCollectorKey.NewContext(ctx, c)
	return ctx, func() []*ai.Part { return c.parts }
}
