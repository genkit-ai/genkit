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
// and to [AgentResult.State] when state is client-managed (no store).
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
	store     SessionStore[State]
	callback  SnapshotCallback[State]
	transform StateTransform[State]
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
// [AgentResult.State] when state is client-managed. Typical use is PII
// redaction or stripping secrets. The transform is not applied to state
// persisted in the store or to state passed to the user agent function.
func WithStateTransform[State any](transform StateTransform[State]) AgentOption[State] {
	return &agentOptions[State]{transform: transform}
}

// --- InvocationOption ---

// InvocationOption configures an agent invocation (StreamBidi, Run, or RunText).
type InvocationOption[State any] interface {
	applyInvocation(*invocationOptions[State]) error
}

type invocationOptions[State any] struct {
	state      *SessionState[State]
	snapshotID string
}

func (o *invocationOptions[State]) applyInvocation(opts *invocationOptions[State]) error {
	if o.state != nil {
		if opts.state != nil {
			return errors.New("cannot set state more than once (WithState)")
		}
		if opts.snapshotID != "" {
			return errors.New("WithState and WithSnapshotID are mutually exclusive")
		}
		opts.state = o.state
	}
	if o.snapshotID != "" {
		if opts.snapshotID != "" {
			return errors.New("cannot set snapshot ID more than once (WithSnapshotID)")
		}
		if opts.state != nil {
			return errors.New("WithSnapshotID and WithState are mutually exclusive")
		}
		opts.snapshotID = o.snapshotID
	}
	return nil
}

// WithState sets the initial state for the invocation.
// Use this for client-managed state where the client sends state directly.
func WithState[State any](state *SessionState[State]) InvocationOption[State] {
	return &invocationOptions[State]{state: state}
}

// WithSnapshotID loads state from a persisted snapshot by ID.
// Use this for server-managed state where snapshots are stored.
func WithSnapshotID[State any](id string) InvocationOption[State] {
	return &invocationOptions[State]{snapshotID: id}
}
