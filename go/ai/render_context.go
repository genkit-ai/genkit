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

package ai

import (
	"context"

	"github.com/firebase/genkit/go/internal/base"
)

// promptHistoryKey carries the messages passed to [Prompt.Execute] so a
// prompt's own content functions can read them. A prompt is an action whose
// only input is the prompt input, so execution options cannot be threaded
// through the call directly.
var promptHistoryKey = base.NewContextKey[[]*Message]()

// promptDocsOverrideKey marks that the execution supplies its own documents,
// telling buildRequest to skip a [WithDocsFn] whose result would be discarded.
var promptDocsOverrideKey = base.NewContextKey[bool]()

// withPromptDocsOverride returns ctx marked so that rendering skips the
// prompt's docs function.
func withPromptDocsOverride(ctx context.Context) context.Context {
	return promptDocsOverrideKey.NewContext(ctx, true)
}

// promptDocsOverridden reports whether the execution supplies its own
// documents.
func promptDocsOverridden(ctx context.Context) bool {
	return promptDocsOverrideKey.FromContext(ctx)
}

// withPromptHistory returns ctx carrying the messages supplied at execution
// time, for [HistoryFromContext] to return. Nil messages are dropped on the way
// in, so every reader, rendering and a prompt's own content functions alike,
// sees the same usable conversation.
func withPromptHistory(ctx context.Context, messages []*Message) context.Context {
	return promptHistoryKey.NewContext(ctx, compactMessages(messages))
}

// NewHistoryContext returns ctx carrying a conversation for the next
// [Prompt.Render] call to place, and is what [HistoryFromContext] reads back.
//
// [Prompt.Execute] does this itself, so its callers never need it. It is for
// code that drives a prompt by hand, pairing Render with [GenerateWithRequest],
// and still wants the prompt to decide where the conversation goes. The agent
// runtime is the main example.
//
// Scope the returned context to the Render call. Generation should run on the
// original, so the conversation does not ride along into tool handlers and
// prompts executed inside the generate loop.
//
// The messages are not copied; Render clones what it places into the request.
// Nil entries are dropped.
func NewHistoryContext(ctx context.Context, messages []*Message) context.Context {
	return withPromptHistory(ctx, messages)
}

// HistoryFromContext returns the conversation history passed to
// [Prompt.Execute] via [WithMessages] or [WithMessagesFn], or nil if there is
// none.
//
// Call it from a prompt's own content functions, where it is the function-form
// counterpart of {{history}}. A prompt that declares a conversation owns the
// history: the caller's messages are not appended on top, so a function that
// wants them must read them here and return them, which is what makes
// summarizing or truncating possible. A prompt that declares no messages of its
// own has the caller's history used as the conversation directly.
//
// The returned slice is the caller's; treat it as read-only.
func HistoryFromContext(ctx context.Context) []*Message {
	return promptHistoryKey.FromContext(ctx)
}
