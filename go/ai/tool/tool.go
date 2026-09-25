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

// Package tool provides the runtime verbs called from inside a running tool
// function or WrapTool hook: [Interrupt], [AttachParts], [SendPartial],
// [SendChunk], [ResumeData], [OriginalInput], and, in a hook, [Released].
// They take a [context.Context], so they work in every tool: [ai.ToolContext]
// embeds the context they take.
//
// Everything for building and wiring a tool (constructors and types) lives in
// package [ai], and everything that acts on a value you already hold lives on
// that value: to resolve an interrupted tool request, use
// [ai.ToolAction.RestartWith] or [ai.ToolAction.RespondWith]. Read the
// interrupt data itself with [ai.InterruptAs].
package tool

import (
	"context"
	"fmt"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/internal/base"
)

// Interrupt returns the error a tool function returns to pause generation and
// send data to the caller. The interrupted tool request surfaces in
// [ai.ModelResponse.Interrupts]; the caller reads the data with
// [ai.InterruptAs] and restarts the tool with [ai.ToolAction.RestartWith] or
// answers it with [ai.ToolAction.RespondWith]. Middleware returns it from a
// WrapTool hook to hold a tool call without executing it.
//
//	func(ctx *ai.ToolContext, in TransferInput) (*TransferOutput, error) {
//		if !ctx.IsResumed() {
//			return nil, tool.Interrupt(ctx, TransferInterrupt{Reason: "large_amount", Amount: in.Amount})
//		}
//		...
//	}
//
// ctx is the context the tool function or hook received. It names the stage
// raising the interrupt, and the restart that answers it reaches that stage
// alone: for a tool function, [ai.ToolContext.Resumed] and [ResumeData]; for
// a WrapTool hook, [ResumeData] in that hook, after which the tool runs as a
// fresh call and may interrupt on its own (see [Released]).
//
// data must serialize to a JSON object (a struct or a map): it lands on the
// interrupted tool request as [ai.ToolInterrupt] data, which the wire protocol
// encodes as a JSON object, and it is converted to that object here, so the
// data has the same shape in process as after a wire hop. For a value that
// serializes to a JSON scalar or array (e.g. a string, number, or slice)
// Interrupt returns a plain error instead of an interrupt, which fails the
// tool call with a message naming the constraint; wrap such values in a
// struct or map field instead.
func Interrupt(ctx context.Context, data any) error {
	m, err := base.ObjectPayload(data, "interrupt data")
	if err != nil {
		return fmt.Errorf("tool.Interrupt: %w", err)
	}
	ie := &base.ToolInterruptError{RaisedBy: base.ToolHookKey.FromContext(ctx)}
	if m != nil {
		ie.Data = m
	}
	return ie
}

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

// AttachParts attaches additional content parts (e.g., media) to the tool's
// response. This can be called from any tool to produce a multipart response
// without changing the function signature, and from a WrapTool hook, before
// or after it runs the tool: the parts collect for the whole call and land
// on the response in call order. A nil part is ignored, so a constructor's
// failed result can be passed without a check.
//
// The parts are folded into the response when the call returns, so they must
// be attached before then: a goroutine the tool spawns may attach as long as
// the tool waits for it, and one that outlives the call attaches to nothing.
// Safe for concurrent use from such goroutines, with no ordering guarantee
// across them.
func AttachParts(ctx context.Context, parts ...*ai.Part) {
	sink := base.ToolPartSinkKey.FromContext(ctx)
	if sink == nil {
		return
	}
	for _, p := range parts {
		if p != nil {
			sink.Add(p)
		}
	}
}

// OriginalInput extracts the typed original input if the caller provided a new
// one when restarting the call (via [ai.WithNewInput]). Returns the zero value
// and false if no new input was provided, the tool is not being resumed, or the
// type doesn't match.
func OriginalInput[In any](ctx context.Context) (In, bool) {
	v := base.ToolOriginalInputKey.FromContext(ctx)
	if v == nil {
		var zero In
		return zero, false
	}
	return base.ConvertTo[In](v)
}

// ResumeData extracts typed resume data (sent via [ai.WithResumedMetadata])
// from the context of a restarted tool call, in the stage the restart
// answers: the tool function when the tool interrupted, or the WrapTool hook
// that raised the interrupt. Returns the zero value and false if the call is
// not a resumption, the restart answers another stage, or the data does not
// decode into T.
//
// Data the caller built as a T is returned as is. Any other data is read by
// its JSON shape, so a struct of another type with the same fields decodes
// the same in process as after a wire hop. It is what middleware reads
// (e.g. a WrapTool hook deciding whether the call it held was approved).
func ResumeData[T any](ctx context.Context) (T, bool) {
	var zero T
	v := base.ToolResumeKey.FromContext(ctx)
	if v == nil {
		return zero, false
	}
	if t, ok := v.(T); ok {
		return t, true
	}
	m, err := base.ObjectPayload(v, "resume data")
	if err != nil {
		return zero, false
	}
	return base.ConvertTo[T](m)
}

// Released reports, in a WrapTool hook, whether the call is a restart that
// answers a later stage, a hook after this one or the tool's own interrupt,
// and the interrupted call records that this hook let it through (see
// [ai.ToolInterrupt.ReleasedBy]). A hook that holds calls for approval passes
// such a restart on rather than holding it again. False in a fresh call, in
// the hook the restart answers (see [ResumeData]), in a hook the interrupted
// call does not record, such as one added to the chain since, in every hook
// when the restart replaced the call's input, and for an interrupt that does
// not record which stage raised it, whose answer every stage reads instead.
func Released(ctx context.Context) bool {
	return base.ToolReleasedKey.FromContext(ctx)
}
