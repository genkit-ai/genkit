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
Package ai defines Genkit's AI primitives: models, prompts, tools, embedders,
retrievers, and evaluators, and the options that configure a request to them.

Applications reach these through the genkit package, whose Define* and
Generate* functions register with a [github.com/firebase/genkit/go/genkit.Genkit]
and forward here. Plugins use this package directly: New*Action builds an
unregistered primitive to return from a plugin's Init.

# Options

Options follow the standard Go functional-options pattern: pass as many as you
like, in any order, and they merge left to right. Two rules govern how repeats
combine, so composing a request from several helpers is predictable:

  - Collection options accumulate. Repeating one, or mixing its variants,
    appends in call order: WithMessages(a), WithMessages(b) sends [a, b], and
    [WithTools], [WithResources], [WithUse], [WithDocs], and [WithDataset]
    behave the same.
  - Single-value options take the last one set. [WithConfig], [WithModel],
    [WithSystem], [WithPrompt], the output-schema options, and the like each
    fill one slot, so the final call wins and earlier ones are overwritten
    rather than rejected. Every option that fills a slot shares it:
    [WithSystem], [WithSystemParts], [WithSystemFn], and [WithSystemPartsFn]
    are four ways to write one system message, so they do not merge.

A zero value does not fill a slot: WithMaxTurns(0), WithToolChoice(""), or
WithConfig(nil) is a no-op, so an earlier non-zero value cannot be un-set by a
later zero one. The rules apply within a single options list; APIs that layer
two lists (a prompt's define-time options against Execute-time options) document
their own precedence.

One combination is refused rather than merged. [WithMessagesTemplate] lays out
the whole conversation, down to where {{history}} puts the caller's, so
separately supplied messages have no position relative to it. Passing it to
DefinePrompt alongside [WithMessages] or [WithMessagesFn] panics. Repeating the
template alone is an ordinary slot.

Applying options therefore never fails on a "set more than once" conflict. What
does fail is a genuinely invalid argument (a type that [WithInputType] cannot
turn into a schema) or the refused combination above, and both panic at the call
site where the mistake is rather than deferring to the request.

# Errors

Failures are classified with the sentinels in
[github.com/firebase/genkit/go/core/status], so callers branch with errors.Is
rather than on message text:

	if errors.Is(err, ai.ErrMaxTurnsExceeded) { ... } // specific
	if errors.Is(err, status.ErrAborted) { ... }      // broad

# Writing a plugin

A provider plugin builds primitives with the New*Action constructors and returns
them from its Init. [ModelOptions] and its siblings describe what a model can
do; the constructor copies what it is given, so a table of models may share one
[ModelSupports] value. [ModelOptions.Overlay] is how a plugin lets an
application correct that description without restating it: a zero-value field
in the override keeps what the plugin already knows.
*/
package ai
