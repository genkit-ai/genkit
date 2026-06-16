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

package exp

import (
	"context"
	"errors"
)

// --- AgentOption ---

// AgentOption configures an agent at definition time. It is accepted by
// [DefineAgent] and [DefineCustomAgent] as a typed variadic, so a State
// mismatch fails at compile time.
type AgentOption[State any] interface {
	applyAgent(*agentOptions[State]) error
}

// StateTransform rewrites session state on its way out to a client. It
// is applied to the State returned by the getSnapshot companion action
// and to [AgentOutput.State] when state is client-managed (no store).
// It is not applied to state persisted in the store or to state passed
// to the user agent function.
//
// ctx is the request or invocation context: cancellation, deadlines,
// and context-scoped values (e.g. the caller's identity for RBAC-aware
// redaction) flow through here.
//
// state is a fresh deep copy made for this call: the transform owns it
// and may mutate in place, return a new pointer, or return nil to omit
// state from the response entirely. Do not retain the pointer past the
// call; the framework drops its reference after the transform returns.
type StateTransform[State any] = func(ctx context.Context, state *SessionState[State]) *SessionState[State]

type agentOptions[State any] struct {
	store       SessionStore[State]
	callback    SnapshotCallback[State]
	transform   StateTransform[State]
	description string
}

func (o *agentOptions[State]) applyAgent(opts *agentOptions[State]) error {
	if o.store != nil {
		if opts.store != nil {
			return errors.New("cannot set session store more than once (WithSessionStore)")
		}
		opts.store = o.store
	}
	if o.callback != nil {
		if opts.callback != nil {
			return errors.New("cannot set snapshot callback more than once (WithSnapshotCallback)")
		}
		opts.callback = o.callback
	}
	if o.transform != nil {
		if opts.transform != nil {
			return errors.New("cannot set state transform more than once (WithStateTransform)")
		}
		opts.transform = o.transform
	}
	if o.description != "" {
		if opts.description != "" {
			return errors.New("cannot set description more than once (WithDescription)")
		}
		opts.description = o.description
	}
	return nil
}

// WithSessionStore sets the store for persisting snapshots. The store must
// implement [SnapshotReader] and [SnapshotWriter] at minimum. Detach
// support also requires [SnapshotAborter]; detach attempts on a store
// that lacks that interface are rejected at runtime.
func WithSessionStore[State any](store SessionStore[State]) AgentOption[State] {
	return &agentOptions[State]{store: store}
}

// WithSnapshotCallback configures when snapshots are created.
// If not provided and a store is configured, snapshots are always created.
func WithSnapshotCallback[State any](cb SnapshotCallback[State]) AgentOption[State] {
	return &agentOptions[State]{callback: cb}
}

// WithSnapshotOn configures snapshots to be created only for the specified events.
// For example, WithSnapshotOn[MyState](SnapshotEventTurnEnd) skips the
// invocation-end snapshot.
func WithSnapshotOn[State any](events ...SnapshotEvent) AgentOption[State] {
	set := make(map[SnapshotEvent]struct{}, len(events))
	for _, e := range events {
		set[e] = struct{}{}
	}
	return WithSnapshotCallback(func(_ context.Context, sc *SnapshotContext[State]) bool {
		_, ok := set[sc.Event]
		return ok
	})
}

// WithStateTransform registers a transform applied to session state on
// its way out to a client via the getSnapshot companion action or via
// [AgentOutput.State] when state is client-managed. Typical use is PII
// redaction or stripping secrets. The transform is not applied to state
// persisted in the store or to state passed to the user agent function.
func WithStateTransform[State any](transform StateTransform[State]) AgentOption[State] {
	return &agentOptions[State]{transform: transform}
}

// WithDescription sets a human-readable description of the agent. It is
// stored on the agent action's descriptor (read back via [Agent.Desc] and
// surfaced in the Dev UI's action listing), the same place every other
// primitive carries its description, so reflective tooling can render it
// without a separate field.
func WithDescription[State any](description string) AgentOption[State] {
	return &agentOptions[State]{description: description}
}

// --- InvocationOption ---

// InvocationOption configures an agent invocation (StreamBidi, Run, or RunText).
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

// WithState sets the initial state for the invocation.
// Use this for client-managed state where the client sends state directly.
// The conversation's identity rides inside the state object
// ([SessionState.SessionID]): the framework mints one on the
// conversation's first invocation and echoes it on the output state, so
// resending the state keeps the identity without tracking a separate
// field. The framework deep-copies the state when the invocation starts,
// so the caller keeps ownership of the object it passed and may reuse it
// freely. Mutually exclusive with [WithSessionID] and [WithSnapshotID].
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

// WithSessionID resumes the session (conversation) with the given ID
// from its latest snapshot: the most recently updated one that is not a
// failed/aborted dead end (see [SnapshotReader.GetLatestSnapshot]). Use
// this when the caller tracks the conversation rather than individual
// snapshots; the session ID is assigned when the conversation's first
// invocation starts (see [AgentOutput.SessionID]) and stays stable
// across resumed invocations.
//
// Only valid when the agent is server-managed (a session store is
// configured) and therefore mutually exclusive with [WithState]: a
// client-managed conversation carries its identity inside the state
// object ([SessionState.SessionID]) instead. Combined with
// [WithSnapshotID], the snapshot picks the exact resume point and the
// session ID is validated against it, so an invocation never silently
// continues a conversation other than the one named.
//
// A pending latest snapshot means a detached invocation is still
// running; the resume is rejected so it cannot race the background
// work. Wait for the snapshot to finalize, or abort it. If the
// session's history was forked (an earlier snapshot was resumed again,
// or two invocations resumed the session concurrently), the most
// recently updated branch wins; use [WithSnapshotID] to continue a
// specific branch instead.
//
// Passing an empty ID is an error rather than a no-op, so an unset
// [AgentOutput.SessionID] cannot silently start a fresh conversation.
func WithSessionID[State any](id string) InvocationOption[State] {
	return &invocationOptions[State]{sessionID: id, sessionIDSet: true}
}
