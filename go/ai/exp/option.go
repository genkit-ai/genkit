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

// --- SessionFlowOption ---

// SessionFlowOption configures an SessionFlow.
type SessionFlowOption[State any] interface {
	applySessionFlow(*sessionFlowOptions[State]) error
}

type sessionFlowOptions[State any] struct {
	store    SessionStore[State]
	callback SnapshotCallback[State]
}

func (o *sessionFlowOptions[State]) applySessionFlow(opts *sessionFlowOptions[State]) error {
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
	return nil
}

// WithSessionStore sets the store for persisting snapshots.
func WithSessionStore[State any](store SessionStore[State]) SessionFlowOption[State] {
	return &sessionFlowOptions[State]{store: store}
}

// WithSnapshotCallback configures when snapshots are created.
// If not provided and a store is configured, snapshots are always created.
func WithSnapshotCallback[State any](cb SnapshotCallback[State]) SessionFlowOption[State] {
	return &sessionFlowOptions[State]{callback: cb}
}

// WithSnapshotOn configures snapshots to be created only for the specified events.
// For example, WithSnapshotOn[MyState](SnapshotEventTurnEnd) skips the
// invocation-end snapshot.
func WithSnapshotOn[State any](events ...SnapshotEvent) SessionFlowOption[State] {
	set := make(map[SnapshotEvent]struct{}, len(events))
	for _, e := range events {
		set[e] = struct{}{}
	}
	return WithSnapshotCallback[State](func(_ context.Context, sc *SnapshotContext[State]) bool {
		_, ok := set[sc.Event]
		return ok
	})
}

// --- InvocationOption ---

// InvocationOption configures an session flow invocation (StreamBidi, Run, or RunText).
type InvocationOption[State any] interface {
	applyInvocation(*invocationOptions[State]) error
}

type invocationOptions[State any] struct {
	state       *SessionState[State]
	snapshotID  string
	promptInput any
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
	if o.promptInput != nil {
		if opts.promptInput != nil {
			return errors.New("cannot set prompt input more than once (WithPromptInput)")
		}
		opts.promptInput = o.promptInput
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

// WithInputVariables overrides the default input variables for a prompt-backed session flow.
// Used with DefineSessionFlowFromPrompt to customize the input variables per invocation.
func WithInputVariables[State any](input any) InvocationOption[State] {
	return &invocationOptions[State]{promptInput: input}
}
