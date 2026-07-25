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

/*
Package status defines Genkit's canonical status codes and the error type that
carries them.

# Classifying an error

[Error] is the only error type Genkit defines. It pairs a message with a
canonical status [Name] and the [Sentinel] that classified it. Build one with
[Errorf], whose first argument is the sentinel:

	return status.Errorf(status.ErrNotFound, "model %q not found", name)

Callers branch with errors.Is rather than by matching message text:

	if errors.Is(err, ai.ErrMaxTurnsExceeded) { ... } // specific
	if errors.Is(err, status.ErrAborted) { ... }      // broad

A base sentinel exists for every status ([ErrInvalidArgument], [ErrNotFound],
[ErrAborted], ...). Packages declare domain sentinels from them with
[Sentinel.Subtype], which inherits the status and still matches the parent:

	var ErrMaxTurnsExceeded = status.ErrAborted.Subtype("max turns exceeded")

# Adding context versus reclassifying

Classify at the point where the failure mode is actually known, which is
usually deep in the call stack: the code that looked up the model is the only
code that knows a missing model is NotFound, not the HTTP handler ten frames
up. Everything above it should add context without touching the classification:

	return fmt.Errorf("agent %q: %w", name, err) // status and sentinel survive

Reclassify only at a boundary where the meaning genuinely changes, and do it
deliberately with [Errorf]. A tool's own NotFound, for instance, is not a
NotFound for the request that invoked the tool; it is an Internal failure of
that tool:

	return status.Errorf(status.ErrInternal, "tool %q failed: %w", name, err)

When several [Error] values are in one chain, errors.As finds the outermost, so
the last deliberate reclassification is the one transports report. [Of] follows
the same rule.

The pattern to avoid is restating the status on every frame as an error bubbles
up. Wrapping with %v is the usual culprit: it flattens the cause into a string,
so the sentinel, the status, and everything else in the chain are lost.

# Messages

Keep messages short and specific, and name the thing that failed: the action
key, model name, tool name, session or snapshot ID. Prefer

	status.Errorf(status.ErrNotFound, "tool %q not found", name)

over a generic "tool not found", and do not prefix messages with the name of
the unexported function that produced them. Genkit composes the surrounding
context by wrapping, so a message only needs to describe its own layer.

# Reaching clients

[Errorf] produces an error whose message stays server-side. [PublicErrorf]
marks a message as safe to return over the wire:

	return status.PublicErrorf(status.ErrInvalidArgument, "invalid %q parameter", param)

Transports call [PublicMessage], which returns the message only when the
outermost [Error] is public and a generic string derived from the status
otherwise. The status code itself is always reported.
*/
package status
