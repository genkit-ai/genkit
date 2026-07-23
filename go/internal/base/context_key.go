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

// ToolPartSinkKey is the context key for attaching content parts to a tool's
// response. Set by the tool wrapper in ai/tool.go for the duration of one tool
// invocation, read by ai/tool (AttachParts). The any value is *ai.Part (typed
// as any to avoid a circular import).
var ToolPartSinkKey = NewContextKey[func(part any)]()

// ToolResumeKey is the context key for the resume payload of a restarted tool
// call. Set by ai/generate.go when re-executing an interrupted tool, read by
// the interruptible tool wrapper in ai/tool.go and by ai/tool (ResumeData). A
// non-nil value (an empty map for a bare restart) means the call is a
// resumption.
var ToolResumeKey = NewContextKey[any]()

// ToolOriginalInputKey is the context key for a tool's original input when the
// caller replaced it during restart. Set by ai/generate.go, read by ai/tool
// (OriginalInput).
var ToolOriginalInputKey = NewContextKey[any]()

// Part metadata keys of the interrupt/resume wire contract. In memory this
// state lives on typed ai.Part fields (Interrupt, Restart, Signature); Part
// marshaling folds it into these metadata keys and unmarshaling lifts it back
// out. The keys and values mirror the JS runtime for cross-runtime message
// compatibility.
const (
	// ToolMetaInterrupt marks a tool request part as interrupted. Holds the
	// interrupt data object, or true for a bare interrupt.
	ToolMetaInterrupt = "interrupt"
	// ToolMetaResolvedInterrupt preserves the original interrupt data on a
	// tool request part once its interrupt has been resolved.
	ToolMetaResolvedInterrupt = "resolvedInterrupt"
	// ToolMetaResumed marks a tool request part as a restart of an interrupted
	// call. Holds the resume data object, or true for a bare restart. Also
	// used on tool message metadata to carry resume metadata.
	ToolMetaResumed = "resumed"
	// ToolMetaReplacedInput preserves the original input on a restart part
	// when the caller replaced it.
	ToolMetaReplacedInput = "replacedInput"
	// ToolMetaInterruptResponse marks a caller-provided tool response part
	// that resolves an interrupt in place of re-executing the tool.
	ToolMetaInterruptResponse = "interruptResponse"
	// ToolMetaPendingOutput holds the output of a completed tool call on its
	// request part while sibling tool calls remain interrupted; consumed when
	// generation resumes.
	ToolMetaPendingOutput = "pendingOutput"
)
