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
)

// --- AgentOption ---

// AgentOption configures an agent at definition time. It is accepted by
// [DefineAgent], [DefineCustomAgent], and [DefinePromptAgent] as a typed
// variadic, so a State mismatch fails at compile time.
//
// Every AgentOption is also a [PromptAgentOption]: the shared options
// ([WithSessionStore], [WithStateTransform], [WithStreamTransform],
// [WithDescription]) configure all three constructors. The converse does not
// hold. A prompt-source option such as [WithNamedPrompt] is a
// [PromptAgentOption] but not an AgentOption, so passing it to [DefineAgent] or
// [DefineCustomAgent] is a compile-time error.
type AgentOption[State any] interface {
	applyAgent(*agentOptions[State]) error
	applyPromptAgent(*promptAgentOptions[State]) error
}

// PromptAgentOption configures a prompt-backed agent defined via
// [DefinePromptAgent]. It is the wider set that additionally admits the
// prompt-source option [WithNamedPrompt]; every [AgentOption] satisfies it, so
// the shared agent options compose with [WithNamedPrompt] in a single variadic.
type PromptAgentOption[State any] interface {
	applyPromptAgent(*promptAgentOptions[State]) error
}

// StateTransform rewrites session state on its way out to a client: it is
// applied to the state returned by the getSnapshot companion action and to
// [AgentOutput.State] for client-managed agents, but not to state persisted in
// the store or passed to the agent function. Typical uses are PII redaction and
// stripping secrets.
//
// state is a fresh deep copy the transform owns. It returns the state to expose
// and a nil error, with two special outcomes:
//
//   - A nil state omits state from the response. This is an intentional,
//     successful outcome: the stored snapshot and the agent's own view keep the
//     data, only the outbound copy is dropped.
//   - A non-nil error fails closed: the operation being shaped aborts rather
//     than exposing anything. A snapshot read returns the error; the final
//     [AgentOutput] of an invocation resolves as a failed output carrying it.
//     Use it when redaction cannot be performed safely (e.g. an RBAC policy
//     lookup failed), where omitting (nil) would be a silent under-redaction
//     and returning raw state would leak. The error's status code (such as
//     PERMISSION_DENIED) propagates to the caller.
//
// ctx is the request or invocation context, carrying deadlines and values such
// as the caller's identity for RBAC-aware redaction.
type StateTransform[State any] = func(ctx context.Context, state *SessionState[State]) (*SessionState[State], error)

// StreamTransform rewrites an [AgentStreamChunk] on its way out to a client: it
// runs at the wire boundary on every chunk the runtime forwards (model chunks,
// artifacts, custom-state patches, and turn-end signals), after the chunk's
// in-process side effects have already been applied. So, like [StateTransform],
// it shapes only what the client sees, not session state, persisted snapshots,
// or anything passed to the agent function. It also leaves the final
// [AgentOutput] untouched; it is a stream-only hook. Typical uses are redacting
// streamed model tokens and stripping fields a client should not receive.
//
// chunk is a fresh deep copy the transform owns. It returns the chunk to
// forward and a nil error, with two special outcomes:
//
//   - A nil chunk drops it from the stream. This is an intentional, successful
//     outcome and is wire-only: the chunk's side effects (an artifact recorded
//     on the session) and the final [AgentOutput] keep the underlying data.
//     Dropping a chunk whose [AgentStreamChunk.TurnEnd] is set withholds the
//     turn-end signal a client uses to pace the conversation, so reshape such a
//     chunk rather than dropping it.
//   - A non-nil error fails closed: the whole invocation fails, so no unshaped
//     chunk reaches the client and the offending chunk's side effects do not
//     surface in the final output either. Use it when a chunk cannot be shaped
//     safely; a panic in the transform is treated the same way.
//
// ctx is the request context, carrying deadlines and values such as the
// caller's identity for RBAC-aware redaction.
//
// To redact custom state, prefer [WithStateTransform]: the runtime applies it
// before diffing, so the custom-patch stream stays internally consistent.
// Rewriting an [AgentStreamChunk.CustomPatch] delta here can desync a client
// that reconstructs custom state from the patch sequence.
//
// Unlike [StateTransform] it is not parameterized by State: a chunk carries no
// state type, so [WithStreamTransform] cannot infer State from the transform
// and takes it as an explicit type argument.
type StreamTransform = func(ctx context.Context, chunk *AgentStreamChunk) (*AgentStreamChunk, error)

type agentOptions[State any] struct {
	store           SessionStore[State]
	transform       StateTransform[State]
	streamTransform StreamTransform
	description     string
	// contextFunc decorates each invocation's context once before the turn
	// loop runs. It has no public option: the registry-level constructors set
	// it internally to seed the genkit instance (see genkitContextSeed in
	// agent.go), so callers reach the instance via genkit.FromContext.
	contextFunc func(context.Context) context.Context
}

func (o *agentOptions[State]) applyAgent(opts *agentOptions[State]) error {
	if o.store != nil {
		if opts.store != nil {
			return errors.New("cannot set session store more than once (WithSessionStore)")
		}
		opts.store = o.store
	}
	if o.transform != nil {
		if opts.transform != nil {
			return errors.New("cannot set state transform more than once (WithStateTransform)")
		}
		opts.transform = o.transform
	}
	if o.streamTransform != nil {
		if opts.streamTransform != nil {
			return errors.New("cannot set stream transform more than once (WithStreamTransform)")
		}
		opts.streamTransform = o.streamTransform
	}
	if o.description != "" {
		if opts.description != "" {
			return errors.New("cannot set description more than once (WithDescription)")
		}
		opts.description = o.description
	}
	if o.contextFunc != nil {
		// Seeded internally by the registry-level constructors
		// (genkitContextSeed), at most once per agent, so this is a plain set
		// rather than a compose of multiple decorators.
		opts.contextFunc = o.contextFunc
	}
	return nil
}

// applyPromptAgent lets a shared agent option also configure a prompt-backed
// agent: it merges the base fields into the prompt-agent accumulator's embedded
// agentOptions. Implementing it is what makes every [AgentOption] usable
// wherever a [PromptAgentOption] is expected.
func (o *agentOptions[State]) applyPromptAgent(opts *promptAgentOptions[State]) error {
	return o.applyAgent(&opts.agentOptions)
}

// promptAgentOptions accumulates a [DefinePromptAgent] configuration: the
// shared agent options plus the optional prompt source. It doubles as the
// option value returned by [WithNamedPrompt], mirroring how ai/option.go reuses
// its accumulator structs as the options that fill them.
type promptAgentOptions[State any] struct {
	agentOptions[State]
	promptName  string // referenced prompt; "" resolves to the agent's own name
	promptInput any    // render input for the referenced prompt
	promptSet   bool   // WithNamedPrompt was used (guards against duplicates)
}

// applyPromptAgent merges the base agent options and the prompt source. It
// shadows the promoted [agentOptions.applyPromptAgent] so a value carrying a
// prompt source contributes it in addition to the base fields.
func (o *promptAgentOptions[State]) applyPromptAgent(opts *promptAgentOptions[State]) error {
	if err := o.agentOptions.applyAgent(&opts.agentOptions); err != nil {
		return err
	}
	if o.promptSet {
		if opts.promptSet {
			return errors.New("cannot set prompt source more than once (WithNamedPrompt)")
		}
		opts.promptName = o.promptName
		opts.promptInput = o.promptInput
		opts.promptSet = true
	}
	return nil
}

// WithSessionStore sets the store for persisting snapshots. The store must
// implement [SnapshotReader] and [SnapshotWriter] at minimum. Detach
// support also requires [SnapshotSubscriber]; detach attempts on a store
// that lacks that interface are rejected at runtime.
func WithSessionStore[State any](store SessionStore[State]) AgentOption[State] {
	return &agentOptions[State]{store: store}
}

// WithStateTransform registers a [StateTransform] applied to session state on
// its way out to a client (via the getSnapshot companion action or
// [AgentOutput.State]). Typical uses are PII redaction and stripping secrets.
func WithStateTransform[State any](transform StateTransform[State]) AgentOption[State] {
	return &agentOptions[State]{transform: transform}
}

// WithStreamTransform registers a [StreamTransform] applied to every
// [AgentStreamChunk] on its way out to a client. Typical uses are redacting
// streamed model tokens and stripping fields a client should not receive; it is
// the stream-side counterpart to [WithStateTransform].
//
// A chunk is not parameterized by the state type, so State cannot be inferred
// from the transform and must be supplied explicitly to match the agent's, e.g.
// WithStreamTransform[MyState](fn).
func WithStreamTransform[State any](transform StreamTransform) AgentOption[State] {
	return &agentOptions[State]{streamTransform: transform}
}

// WithDescription sets a human-readable description of the agent, stored on its
// action descriptor (read back via [Agent.Desc] and shown in the Dev UI).
func WithDescription[State any](description string) AgentOption[State] {
	return &agentOptions[State]{description: description}
}

// WithNamedPrompt points a [DefinePromptAgent] at the prompt registered under
// name, rendered with input on every turn (pass nil for the prompt's own
// default input). name need not match the agent's name, so a single registered
// prompt can back many agents with different inputs; pass "" to keep the
// default same-named lookup while still supplying a custom input.
//
// Without this option a prompt agent uses the prompt registered under its own
// name. This option lets a single registered prompt back many agents, each
// rendered with its own input, and composes with the other agent options in one
// variadic.
//
// input is rendered through the prompt once at definition time as a smoke
// check, so an input that fails the prompt's schema panics there rather than on
// the first invocation.
//
// This option applies only to [DefinePromptAgent]. Passing it to [DefineAgent]
// or [DefineCustomAgent] is a compile-time error.
func WithNamedPrompt[State any](name string, input any) PromptAgentOption[State] {
	return &promptAgentOptions[State]{promptName: name, promptInput: input, promptSet: true}
}

// --- InvocationOption ---

// InvocationOption configures an agent invocation (Connect, Run, or RunText).
type InvocationOption[State any] interface {
	applyInvocation(*invocationOptions[State]) error
}

type invocationOptions[State any] struct {
	state      *SessionState[State]
	snapshotID string
	sessionID  string
	// sessionIDSet records that WithSessionID was used, independent of the
	// value: an empty session ID is rejected rather than silently ignored,
	// since it usually means an [AgentOutput.SessionID] from a client-managed
	// (storeless) invocation was piped through, and treating it as "no
	// option" would silently start a fresh conversation.
	sessionIDSet bool
}

// applyInvocation merges o into opts, rejecting duplicate options.
// Mutual exclusivity (WithState versus WithSessionID/WithSnapshotID) is
// checked once, after all options are applied, in
// [Agent.resolveOptions].
func (o *invocationOptions[State]) applyInvocation(opts *invocationOptions[State]) error {
	if o.state != nil {
		if opts.state != nil {
			return errors.New("cannot set state more than once (WithState)")
		}
		opts.state = o.state
	}
	if o.snapshotID != "" {
		if opts.snapshotID != "" {
			return errors.New("cannot set snapshot ID more than once (WithSnapshotID)")
		}
		opts.snapshotID = o.snapshotID
	}
	if o.sessionIDSet {
		if o.sessionID == "" {
			return errors.New("session ID is empty (WithSessionID); an empty AgentOutput.SessionID means the invocation had no session, check before resuming")
		}
		if opts.sessionIDSet {
			return errors.New("cannot set session ID more than once (WithSessionID)")
		}
		opts.sessionID = o.sessionID
		opts.sessionIDSet = true
	}
	return nil
}

// WithState sets the initial state for the invocation, for client-managed
// agents where the client sends state directly. The conversation's identity
// rides inside the state ([SessionState.SessionID]); the framework mints it on
// the first invocation and echoes it on the output, so resending the state
// keeps the identity. The state is deep-copied at the start, so the caller may
// reuse the object freely. Mutually exclusive with [WithSessionID] and
// [WithSnapshotID].
func WithState[State any](state *SessionState[State]) InvocationOption[State] {
	return &invocationOptions[State]{state: state}
}

// WithSnapshotID loads state from a persisted snapshot by ID.
// Use this for server-managed state where snapshots are stored.
// Combine with [WithSessionID] to assert which session the snapshot
// belongs to; a mismatch is rejected. Mutually exclusive with
// [WithState].
func WithSnapshotID[State any](id string) InvocationOption[State] {
	return &invocationOptions[State]{snapshotID: id}
}

// WithSessionID resumes the conversation with the given ID from its latest
// snapshot (see [SnapshotReader.GetLatestSnapshot]). Use it when the caller
// tracks the conversation rather than individual snapshots; the ID is assigned
// on the conversation's first invocation (see [AgentOutput.SessionID]) and
// stays stable across resumes.
//
// Valid only for server-managed agents, and so mutually exclusive with
// [WithState] (a client-managed conversation carries its identity inside the
// state). Combined with [WithSnapshotID], the snapshot picks the resume point
// and the session ID is validated against it.
//
// The resume is rejected if the latest snapshot is a failed, aborted, or
// pending dead end; a pending tip means a detached invocation is still running,
// so wait for it to finalize or abort it. If history was forked, the most
// recently updated branch wins; use [WithSnapshotID] for a specific branch. An
// empty ID is an error, not a no-op.
func WithSessionID[State any](id string) InvocationOption[State] {
	return &invocationOptions[State]{sessionID: id, sessionIDSet: true}
}
