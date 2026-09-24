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

package base

import (
	"context"
	"sync"
	"sync/atomic"
)

// A ContextKey is a unique, typed key for a value stored in a context.
type ContextKey[T any] struct {
	key *int
}

// NewContextKey returns a context key for a value of type T.
func NewContextKey[T any]() ContextKey[T] {
	return ContextKey[T]{key: new(int)}
}

// NewContext returns ctx augmented with this key and the given value.
func (k ContextKey[T]) NewContext(ctx context.Context, value T) context.Context {
	return context.WithValue(ctx, k.key, value)
}

// FromContext returns the value associated with this key in the context,
// or the internal.Zero value for T if the key is not present.
func (k ContextKey[T]) FromContext(ctx context.Context) T {
	t, _ := ctx.Value(k.key).(T)
	return t
}

// ToolPartialSenderKey is the context key for streaming partial tool responses.
// Set by ai/generate.go (handleToolRequests), read by ai/tool (SendPartial).
var ToolPartialSenderKey = NewContextKey[func(context.Context, any)]()

// ToolChunkSenderKey is the context key for streaming raw model response chunks
// from within a tool. Set by ai/generate.go (handleToolRequests), read by
// ai/tool (SendChunk). The any value is *ai.ModelResponseChunk (typed as any
// to avoid a circular import).
var ToolChunkSenderKey = NewContextKey[func(context.Context, any)]()

// ToolResumeKey is the context key holding the data a caller sent when
// restarting an interrupted tool call, as the caller gave it: a map[string]any
// after a wire hop or from a map restart, the caller's struct from a typed
// restart in process. Set by the hook chain in ai/generate.go on the one
// stage the restart answers (see [ToolRestart]); each reader converts it to
// what it returns with [ConvertTo], so a value that is already the wanted
// type is handed over untouched, with its Go types intact: ai (ToolContext.Resumed,
// IsToolResumed, ResumedValue), ai/tool (ResumeData), and the resume
// parameter of an ai.NewResumableTool tool. A bare restart stores an
// empty map, so presence of the key, not its contents, marks a call as
// resumed.
var ToolResumeKey = NewContextKey[any]()

// ToolOriginalInputKey is the context key holding a tool call's pre-replacement
// input, set when the caller restarted the call with a new input. Set by the
// hook chain in ai/generate.go on the stage the restart answers (see
// [ToolRestart]), read by ai (ToolContext.OriginalInput) and by ai/tool
// (OriginalInput).
var ToolOriginalInputKey = NewContextKey[any]()

// ToolHookKey is the context key naming the stage a tool call is running in:
// the id of the WrapTool hook whose turn it is, empty inside the tool itself.
// Set per stage by the hook chain in ai/generate.go, read by ai/tool
// (Interrupt) to attribute an interrupt to the stage that raised it.
var ToolHookKey = NewContextKey[string]()

// ToolRestart is the restart a resumed tool call answers. ai/generate.go
// (restartedToolResponse) sets it on the call's context under
// [ToolRestartKey], and the hook chain delivers it stage by stage: the stage
// that raised the interrupt (RaisedBy, empty for the tool itself) receives
// Resume and OriginalInput under [ToolResumeKey] and [ToolOriginalInputKey],
// a hook the interrupted call records in Released, which let the call through
// before, is marked under [ToolReleasedKey] unless the restart replaced the
// input, and any other stage sees a fresh call. An interrupt that records
// neither RaisedBy nor Released, as one stored before they were recorded,
// does not say which stage it came from, so every stage receives Resume and
// OriginalInput and none is released.
type ToolRestart struct {
	Resume        any
	OriginalInput any
	RaisedBy      string
	Released      []string
}

// ToolRestartKey is the context key holding the [ToolRestart] a resumed tool
// call answers, for the hook chain to deliver.
var ToolRestartKey = NewContextKey[*ToolRestart]()

// ToolReleasedKey is the context key marking a stage that let the call
// through before: the restart answers a later stage. Set per stage by the
// hook chain in ai/generate.go, read by ai/tool (Released).
var ToolReleasedKey = NewContextKey[bool]()

// ToolCall marks the context the hook chain in ai/generate.go hands the tool
// stage of one tool call. The tool function that claims it owns the call's
// restart state and part sink; any other tool run under that context, such as
// a tool calling another tool directly, finds it claimed or naming another
// tool and runs as a call of its own. A retry that runs the tool stage again
// gets a fresh mark.
type ToolCall struct {
	Name    string
	claimed atomic.Bool
}

// Claim reports whether the tool named name owns the call c marks: c names it
// and no tool function claimed c before. Nil-safe.
func (c *ToolCall) Claim(name string) bool {
	return c != nil && c.Name == name && c.claimed.CompareAndSwap(false, true)
}

// ToolCallKey is the context key holding the [ToolCall] of the tool stage.
// Set by the hook chain in ai/generate.go, read by the tool constructors in ai.
var ToolCallKey = NewContextKey[*ToolCall]()

// ToolPartSinkKey is the context key for the [PartSink] that collects content
// parts attached during one tool call. Set by ai around the whole tool call,
// the WrapTool hook chain included, so a hook can attach parts too; read by
// ai/tool (AttachParts).
var ToolPartSinkKey = NewContextKey[*PartSink]()

// PartSink collects the content parts attached during one tool call. The
// values are *ai.Part, typed as any to avoid a circular import. It is safe
// for concurrent use: a tool may attach from goroutines it waits for. The
// call closes it when it returns, and a part attached after that, from a
// goroutine that outlived the call, is dropped rather than held.
type PartSink struct {
	mu     sync.Mutex
	parts  []any
	closed bool
}

// Add appends part, or drops it once the sink is closed.
func (s *PartSink) Add(part any) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return
	}
	s.parts = append(s.parts, part)
}

// Close closes the sink and returns the parts attached before, in call
// order. Closing again returns nothing, so the parts are folded once.
func (s *PartSink) Close() []any {
	s.mu.Lock()
	defer s.mu.Unlock()
	parts := s.parts
	s.parts, s.closed = nil, true
	return parts
}
