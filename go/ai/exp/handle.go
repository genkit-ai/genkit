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
	"encoding/json"
	"fmt"

	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/internal/base"
)

// AgentHandle is the untyped caller-side view of an agent: its run action and
// snapshot-lifecycle companion actions (getSnapshot, waitForSnapshot, abort),
// addressed with the custom-state type fixed to [json.RawMessage]. It is how
// code that does not hold the defining [Agent] value calls an agent it knows
// only by name:
// orchestrators, middleware, and tools resolve one with [LookupAgent] (or the
// genkit/exp package's LookupAgent), and a typed owner hands one out with
// [Agent.Handle].
//
// A handle adds no capability over the actions it wraps; it only removes the
// wire plumbing (JSON marshaling, companion-action lookup and dispatch) that
// name-resolved callers would otherwise repeat. Custom state on every surface
// is [json.RawMessage]: unmarshal [SessionState.Custom] into the agent's own
// state type when the caller knows it.
type AgentHandle struct {
	name string
	run  api.BidiAction
	// Companion actions; nil when the agent lacks the capability (see
	// [Agent.GetSnapshotAction], [Agent.WaitForSnapshotAction], and
	// [Agent.AbortAction]).
	getSnapshot api.Action
	wait        api.Action
	abort       api.Action
	meta        *AgentMetadata
}

// LookupAgent resolves the agent registered under name and returns its
// [AgentHandle]. It is the caller-side counterpart of [DefineAgent] and
// friends: definition hands back the typed [Agent], and a later caller that
// holds only the name recovers this untyped view, companion actions included.
// Callers holding a genkit instance should use the genkit/exp package's
// LookupAgent instead.
//
// It returns NOT_FOUND when nothing is registered under the agent key, and
// INVALID_ARGUMENT when the registered action is not an agent.
func LookupAgent(r api.Registry, name string) (*AgentHandle, error) {
	action := r.LookupAction(api.KeyFromName(api.ActionTypeAgent, name))
	if action == nil {
		return nil, status.Errorf(status.ErrNotFound, "agent %q not found in registry", name)
	}
	run, ok := action.(api.BidiAction)
	if !ok {
		return nil, status.Errorf(status.ErrInvalidArgument, "%q is registered but is not an agent", name)
	}
	return &AgentHandle{
		name:        name,
		run:         run,
		getSnapshot: r.LookupAction(api.KeyFromName(api.ActionTypeAgentSnapshot, name)),
		wait:        r.LookupAction(api.KeyFromName(api.ActionTypeAgentWait, name)),
		abort:       r.LookupAction(api.KeyFromName(api.ActionTypeAgentAbort, name)),
		meta:        AgentMetadataOf(run),
	}, nil
}

// Handle returns the agent's untyped caller-side view: the same surface
// [LookupAgent] builds for callers that hold only the agent's name. Use it to
// hand the agent to code written against [AgentHandle] (orchestration helpers,
// middleware) without exposing the typed [Agent]. It performs no registry
// lookup, so it also works for an unregistered agent (see [NewCustomAgent]).
func (a *Agent[State]) Handle() *AgentHandle {
	return &AgentHandle{
		name:        a.Name(),
		run:         a,
		getSnapshot: a.getSnapshot,
		wait:        a.wait,
		abort:       a.abort,
		meta:        AgentMetadataOf(a),
	}
}

// AgentMetadataOf extracts the [AgentMetadata] an agent's action descriptor
// carries under its "agent" metadata key, or nil when a is not an agent action
// or carries none. It handles both in-process descriptors (typed metadata) and
// descriptors decoded from JSON (map form), so callers can inspect an agent's
// capabilities ([AgentMetadata.StateManagement], [AgentMetadata.Abortable])
// without knowing where the action came from. The returned value is a copy;
// mutating it does not affect the descriptor.
func AgentMetadataOf(a api.Action) *AgentMetadata {
	if a == nil {
		return nil
	}
	switch m := a.Desc().Metadata["agent"].(type) {
	case AgentMetadata:
		return &m
	case *AgentMetadata:
		if m == nil {
			return nil
		}
		copied := *m
		return &copied
	case map[string]any:
		// Best-effort decode: encoding/json fills every well-typed field
		// before reporting the first type error, so one mistyped field in a
		// wire descriptor leaves that field zero instead of erasing the
		// capabilities that did decode.
		meta, _ := base.MapToStruct[AgentMetadata](m)
		return &meta
	}
	return nil
}

// Name returns the agent's registered name.
func (h *AgentHandle) Name() string { return h.name }

// Metadata returns the agent's capability metadata (who manages state, whether
// background work can be aborted), or nil when the agent's descriptor carries
// none. Treat the returned value as read-only; it is shared across calls.
func (h *AgentHandle) Metadata() *AgentMetadata { return h.meta }

// Run starts a single-turn invocation with the given input and returns the
// final output, with custom state as raw JSON. It is [Agent.Run] for callers
// holding only the handle; the [InvocationOption] values instantiate at
// [json.RawMessage] (e.g. WithSessionID[json.RawMessage](id)).
//
// In-band failures (e.g. a failed turn) resolve as a failed [AgentOutput]
// rather than an error, exactly like [Agent.Run]; a non-nil error means the
// invocation never started (a rejected init payload) or could not run to a
// result.
func (h *AgentHandle) Run(ctx context.Context, input *AgentInput, opts ...InvocationOption[json.RawMessage]) (*AgentOutput[json.RawMessage], error) {
	if input == nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "agent %q: input must not be nil", h.name)
	}
	init, err := resolveInvocationInit(h.name, opts)
	if err != nil {
		return nil, err
	}
	inputJSON, err := json.Marshal(input)
	if err != nil {
		return nil, fmt.Errorf("agent %q: marshal input: %w", h.name, err)
	}
	var initJSON json.RawMessage
	if init != nil {
		initJSON, err = json.Marshal(init)
		if err != nil {
			return nil, fmt.Errorf("agent %q: marshal init: %w", h.name, err)
		}
	}
	res, err := h.run.RunBidiJSON(ctx, inputJSON, nil, &api.BidiJSONOptions{Init: initJSON})
	if err != nil {
		return nil, err
	}
	var out AgentOutput[json.RawMessage]
	if err := json.Unmarshal(res.Result, &out); err != nil {
		return nil, fmt.Errorf("agent %q: unmarshal output: %w", h.name, err)
	}
	return &out, nil
}

// Start launches input as a detached (background) invocation and returns the
// task tracking it. It is the one-shot counterpart of
// [AgentConnection.Detach]: the input is delivered with [AgentInput.Detach]
// set, the agent's runtime persists a pending snapshot and keeps working on a
// context decoupled from ctx, and the returned [DetachedTask] polls, waits on,
// or aborts that snapshot. The pending snapshot is the durable record: record
// [DetachedTask.SnapshotID] and rehydrate with [AgentHandle.Task] to pick the
// work up later, including from another process.
//
// The launch is rejected with FAILED_PRECONDITION when the agent cannot
// support detach: it has no session store, or the store does not implement
// [SnapshotSubscriber]. Check [AgentMetadata.Abortable] to pre-flight. The
// rejection is decoded from the invocation's failed output, which keeps only
// the status name, so match it with [status.Of] rather than errors.Is.
//
// An agent may also settle the invocation synchronously, before the runtime
// observes the detach directive (e.g. a custom agent whose fn returns without
// consuming the input). When a turn committed, Start returns a task over its
// snapshot, which is already terminal, so Poll and Wait resolve immediately;
// when nothing was recorded, Start fails with FAILED_PRECONDITION naming the
// finish reason, since there is no durable record to track.
func (h *AgentHandle) Start(ctx context.Context, input *AgentInput, opts ...InvocationOption[json.RawMessage]) (*DetachedTask, error) {
	if input == nil {
		return nil, status.Errorf(status.ErrInvalidArgument, "agent %q: input must not be nil", h.name)
	}
	detachInput := *input
	detachInput.Detach = true
	out, err := h.Run(ctx, &detachInput, opts...)
	if err != nil {
		return nil, err
	}
	switch out.FinishReason {
	case AgentFinishReasonDetached:
		return &DetachedTask{handle: h, snapshotID: out.SnapshotID}, nil
	case AgentFinishReasonFailed:
		var cause error = out.Error
		if out.Error == nil {
			cause = status.Errorf(status.ErrInternal, "launch failed without error detail")
		}
		return nil, fmt.Errorf("agent %q: background launch failed: %w", h.name, cause)
	default:
		// The invocation settled before the runtime observed the detach
		// directive (nothing orders the intake reader ahead of an agent fn
		// that never consumes its input). A committed turn's snapshot is the
		// settled work's durable record, so hand back a task over it; Poll
		// and Wait resolve immediately with the terminal snapshot.
		if out.SnapshotID != "" {
			return &DetachedTask{handle: h, snapshotID: out.SnapshotID}, nil
		}
		return nil, status.Errorf(status.ErrFailedPrecondition,
			"agent %q: the invocation settled synchronously (finish reason %q, session %q) without recording a snapshot, so there is no background task to track",
			h.name, out.FinishReason, out.SessionID)
	}
}

// Task returns a [DetachedTask] for a snapshot ID recorded earlier (e.g. by a
// prior process that called [AgentHandle.Start]). It performs no I/O and does
// not verify the snapshot exists; the first [DetachedTask.Poll] or
// [DetachedTask.Wait] surfaces NOT_FOUND for an unknown ID.
func (h *AgentHandle) Task(snapshotID string) *DetachedTask {
	return &DetachedTask{handle: h, snapshotID: snapshotID}
}

// GetSnapshot fetches a session snapshot by ID through the agent's getSnapshot
// companion action, so the read is shaped exactly as remote callers see it:
// the configured [WithStateTransform] applies and a stale-heartbeat pending
// row surfaces as [SnapshotStatusExpired]. It is [Agent.GetSnapshot] with
// custom state as raw JSON.
//
// It returns FAILED_PRECONDITION ([ErrSessionStoreNotConfigured]) when the
// agent has no session store and INVALID_ARGUMENT when snapshotID is empty; a
// missing snapshot is NOT_FOUND.
func (h *AgentHandle) GetSnapshot(ctx context.Context, snapshotID string) (*SessionSnapshot[json.RawMessage], error) {
	if snapshotID == "" {
		return nil, status.Errorf(status.ErrInvalidArgument, "agent %q: GetSnapshot: snapshotID is required", h.name)
	}
	return h.snapshotVia(ctx, h.getSnapshot, "read", &GetSnapshotRequest{SnapshotID: snapshotID})
}

// WaitForSnapshot fetches a session snapshot by ID and blocks until it settles,
// through the agent's waitForSnapshot companion action, returning the terminal
// snapshot shaped exactly as [AgentHandle.GetSnapshot] shapes a read. An
// already-terminal snapshot returns at once. It is [Agent.WaitForSnapshot] with
// custom state as raw JSON, and it is how a caller that holds only actions
// follows a detached invocation: one dispatch for the whole wait, rather than a
// read per tick.
//
// A snapshot that failed, aborted, or expired is returned like any other, so a
// non-nil error means the wait itself could not proceed: reads failed past the
// wait's transient-retry budget, or ctx ended and its error is returned. Bound
// the wait with [context.WithTimeout], then call [AgentHandle.GetSnapshot] to
// learn where the task stands.
//
// It returns FAILED_PRECONDITION ([ErrSessionStoreNotConfigured]) when the
// agent has no session store and INVALID_ARGUMENT when snapshotID is empty; a
// missing snapshot is NOT_FOUND.
func (h *AgentHandle) WaitForSnapshot(ctx context.Context, snapshotID string) (*SessionSnapshot[json.RawMessage], error) {
	if snapshotID == "" {
		return nil, status.Errorf(status.ErrInvalidArgument, "agent %q: WaitForSnapshot: snapshotID is required", h.name)
	}
	return h.snapshotVia(ctx, h.wait, "follow", &GetSnapshotRequest{SnapshotID: snapshotID})
}

// GetLatestSnapshot fetches a session's most recently created snapshot
// (whatever its status) through the agent's getSnapshot companion action, with
// the same shaping as [AgentHandle.GetSnapshot]. It is
// [Agent.GetLatestSnapshot] with custom state as raw JSON.
//
// It returns FAILED_PRECONDITION ([ErrSessionStoreNotConfigured]) when the
// agent has no session store and INVALID_ARGUMENT when sessionID is empty; an
// unknown session is NOT_FOUND.
func (h *AgentHandle) GetLatestSnapshot(ctx context.Context, sessionID string) (*SessionSnapshot[json.RawMessage], error) {
	if sessionID == "" {
		return nil, status.Errorf(status.ErrInvalidArgument, "agent %q: GetLatestSnapshot: sessionID is required", h.name)
	}
	return h.snapshotVia(ctx, h.getSnapshot, "read", &GetSnapshotRequest{SessionID: sessionID})
}

// snapshotVia dispatches req to act, one of the two companion actions that
// answer a [GetSnapshotRequest] with a [SessionSnapshot], and decodes the
// result. verb names what the caller wanted to do with the snapshot, for the
// message reporting an agent that keeps none. The action runs in-process here,
// so its error chain stays live: sentinel matching with errors.Is works,
// subtypes included (e.g. [ErrSnapshotNotFound] is a [status.ErrNotFound]).
func (h *AgentHandle) snapshotVia(ctx context.Context, act api.Action, verb string, req *GetSnapshotRequest) (*SessionSnapshot[json.RawMessage], error) {
	if act == nil {
		return nil, status.Errorf(ErrSessionStoreNotConfigured,
			"agent %q has no session store, so it keeps no snapshots to %s", h.name, verb)
	}
	reqJSON, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("agent %q: marshal snapshot request: %w", h.name, err)
	}
	raw, err := act.RunJSON(ctx, reqJSON, nil)
	if err != nil {
		return nil, err
	}
	var snap SessionSnapshot[json.RawMessage]
	if err := json.Unmarshal(raw, &snap); err != nil {
		return nil, fmt.Errorf("agent %q: unmarshal snapshot: %w", h.name, err)
	}
	return &snap, nil
}

// Abort asks the background work behind a pending snapshot to stop, through
// the agent's abort companion action, and returns the snapshot's status after
// the attempt: [SnapshotStatusAborted] when the row was pending, or the
// existing terminal status (the abort was a no-op) when it had already
// settled. A missing snapshot is NOT_FOUND, matching the companion action
// remote callers use (unlike [Agent.Abort], which reports it as "").
//
// It returns FAILED_PRECONDITION when the agent has no session store
// ([ErrSessionStoreNotConfigured]) or the store cannot observe aborts (no
// [SnapshotSubscriber]; see [AgentMetadata.Abortable]), and INVALID_ARGUMENT
// when snapshotID is empty.
func (h *AgentHandle) Abort(ctx context.Context, snapshotID string) (SnapshotStatus, error) {
	if snapshotID == "" {
		return "", status.Errorf(status.ErrInvalidArgument, "agent %q: Abort: snapshotID is required", h.name)
	}
	if h.abort == nil {
		if h.getSnapshot == nil {
			return "", status.Errorf(ErrSessionStoreNotConfigured, "agent %q: Abort requires a session store", h.name)
		}
		return "", status.Errorf(status.ErrFailedPrecondition,
			"agent %q: the session store does not support abort (it does not implement SnapshotSubscriber)", h.name)
	}
	reqJSON, err := json.Marshal(&AgentAbortRequest{SnapshotID: snapshotID})
	if err != nil {
		return "", fmt.Errorf("agent %q: marshal abort request: %w", h.name, err)
	}
	raw, err := h.abort.RunJSON(ctx, reqJSON, nil)
	if err != nil {
		return "", err
	}
	var resp AgentAbortResponse
	if err := json.Unmarshal(raw, &resp); err != nil {
		return "", fmt.Errorf("agent %q: unmarshal abort response: %w", h.name, err)
	}
	return resp.Status, nil
}

// --- DetachedTask ---

// DetachedTask tracks a detached (background) agent invocation through its
// pending snapshot. Obtain one from [AgentHandle.Start] when launching, or
// rehydrate one from a recorded snapshot ID with [AgentHandle.Task]; the two
// are equivalent, because the snapshot is the only state a task has.
type DetachedTask struct {
	handle     *AgentHandle
	snapshotID string
}

// SnapshotID returns the ID of the snapshot tracking the task: pending while
// the background work runs, finalized in place when it settles. It is the
// task's durable identity; record it to pick the task up later with
// [AgentHandle.Task].
func (t *DetachedTask) SnapshotID() string { return t.snapshotID }

// Poll fetches the task's snapshot once, without waiting. The snapshot's
// Status says where the task stands: [SnapshotStatusPending] while the work
// runs, a terminal status once it settles (see [SnapshotStatus.Terminal]),
// including [SnapshotStatusExpired] when the worker stopped heartbeating and
// is presumed dead. A terminal snapshot carries the cumulative session state.
func (t *DetachedTask) Poll(ctx context.Context) (*SessionSnapshot[json.RawMessage], error) {
	return t.handle.GetSnapshot(ctx, t.snapshotID)
}

// Wait blocks until the task settles and returns its terminal snapshot, in one
// dispatch of the agent's waitForSnapshot companion action: the waiting happens
// server-side, next to the store that knows when the work finished, so the
// caller neither picks a cadence nor pays a dispatch per tick. A task that
// failed, aborted, or expired still returns its snapshot rather than an error
// (inspect [SessionSnapshot.Status] and [SessionSnapshot.Error]), so a non-nil
// error means the wait itself could not proceed: reads failed past the wait's
// transient-retry budget, or ctx ended and its error is returned.
//
// Use [context.WithTimeout] to bound the wait; on the deadline the wait returns
// ctx's error, and [DetachedTask.Poll] then reports where the task stands.
func (t *DetachedTask) Wait(ctx context.Context) (*SessionSnapshot[json.RawMessage], error) {
	return t.handle.WaitForSnapshot(ctx, t.snapshotID)
}

// Abort asks the task's background work to stop and returns the snapshot's
// status after the attempt; see [AgentHandle.Abort].
func (t *DetachedTask) Abort(ctx context.Context) (SnapshotStatus, error) {
	return t.handle.Abort(ctx, t.snapshotID)
}
