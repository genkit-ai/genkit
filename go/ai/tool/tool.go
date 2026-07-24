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

// Package tool provides the runtime verbs for tools created with
// [ai.DefineTool] and friends, called from inside a running tool function:
// [Interrupt], [AttachParts], [SendPartial], [SendChunk], [ResumeData], and
// [OriginalInput].
//
// Everything for building and wiring a tool (constructors, types, options)
// lives in package ai, as does everything that acts on a value you already
// hold: to inspect or resolve an interrupted tool request, use the [ai.Part]
// methods [ai.Part.InterruptAs], [ai.Part.ToRestart], and [ai.Part.ToResponse],
// or their typed equivalents on [ai.InterruptibleTool].
package tool

import (
	"context"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/internal/base"
)

// Interrupt interrupts tool execution and sends data to the caller as an
// [ai.InterruptError]. The caller can read this data with [InterruptData] and
// restart the tool with [Restart].
//
// data must serialize to a JSON object (a struct or a map), since it is
// carried on the interrupted tool request as [ai.ToolInterrupt] data, which
// the wire protocol encodes as a JSON object. A value that serializes to a
// JSON scalar or array (e.g. a string, number, or slice) makes the tool fail
// when it returns; wrap such values in a struct or map field instead.
func Interrupt(data any) error {
	return &ai.InterruptError{Data: data}
}

// SendPartial streams a partial tool response during tool execution.
// The output is arbitrary structured data (e.g., progress information)
// that will be delivered to the client as a partial [ai.ToolResponse].
//
// This is best-effort: if no streaming callback is available (e.g., the
// tool is called via a non-streaming Generate), the call is a no-op.
// The tool's final return value is always the authoritative response.
// Safe for concurrent use from goroutines the tool function spawns; sends
// are serialized onto the stream.
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

// SendChunk streams a raw [ai.ModelResponseChunk] during tool execution.
// Unlike [SendPartial], which wraps arbitrary data in a partial tool response,
// SendChunk gives the tool full control over the chunk contents.
//
// This is best-effort: if no streaming callback is available (e.g., the
// tool is called via a non-streaming Generate), the call is a no-op.
// The tool's final return value is always the authoritative response.
// Safe for concurrent use from goroutines the tool function spawns; sends
// are serialized onto the stream.
func SendChunk(ctx context.Context, chunk *ai.ModelResponseChunk) {
	send := base.ToolChunkSenderKey.FromContext(ctx)
	if send == nil {
		return
	}
	send(ctx, chunk)
}

// AttachParts attaches additional content parts (e.g., media) to the tool's
// response. This can be called from any tool to produce a multipart response
// without changing the function signature.
//
// Safe for concurrent use from goroutines the tool function spawns; parts are
// appended in call order per goroutine, with no ordering guarantee across
// goroutines.
func AttachParts(ctx context.Context, parts ...*ai.Part) {
	sink := base.ToolPartSinkKey.FromContext(ctx)
	if sink == nil {
		return
	}
	for _, p := range parts {
		sink(p)
	}
}

// OriginalInput extracts the typed original input if the caller provided a
// new one during restart (via [ai.WithNewInput]). Returns the zero value and
// false if no new input was provided, the tool is not being resumed, or the
// type doesn't match.
func OriginalInput[In any](ctx context.Context) (In, bool) {
	v := base.ToolOriginalInputKey.FromContext(ctx)
	if v == nil {
		var zero In
		return zero, false
	}
	return base.ConvertTo[In](v)
}

// ResumeData extracts typed resume data (sent via [ai.WithResume]) from the
// context of a restarted tool call. Returns the zero value and false if the
// call is not a resumption or the type doesn't match.
//
// Tool functions created with [ai.DefineInterruptibleTool] receive the resume
// data as a parameter and don't need this; it is primarily for middleware
// (e.g. a WrapTool hook deciding whether a call was approved on resume) and
// for plain tools resumed by generic callers.
func ResumeData[T any](ctx context.Context) (T, bool) {
	v := base.ToolResumeKey.FromContext(ctx)
	if v == nil {
		var zero T
		return zero, false
	}
	return base.ConvertTo[T](v)
}
