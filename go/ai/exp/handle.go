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
	"time"

	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
)

// defaultTaskPollInterval is how often [DetachedTask.Wait] re-reads a pending
// task's snapshot when [WithPollInterval] is not given.
const defaultTaskPollInterval = 2 * time.Second

// AgentHandle is the untyped caller-side view of an agent: its run action and
// snapshot-lifecycle companion actions (getSnapshot, abort), addressed with
// the custom-state type fixed to [json.RawMessage]. It is how code that does
// not hold the defining [Agent] value calls an agent it knows only by name:
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
	// [Agent.GetSnapshotAction] and [Agent.AbortAction]).
	getSnapshot api.Action
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
		b, err := json.Marshal(m)
		if err != nil {
			return nil
		}
		meta := &AgentMetadata{}
		if err := json.Unmarshal(b, meta); err != nil {
			return nil
		}
		return meta
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
	if init.SessionID != "" || init.SnapshotID != "" || init.State != nil {
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
		// With detach riding the first input, the runtime resolves the
		// invocation as detached or failed before any turn output can settle
		// it; any other reason means the agent did not honor the detach
		// contract.
		return nil, status.Errorf(status.ErrInternal,
			"agent %q: expected the invocation to detach, got finish reason %q", h.name, out.FinishReason)
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
	return h.readSnapshot(ctx, &GetSnapshotRequest{SnapshotID: snapshotID})
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
	return h.readSnapshot(ctx, &GetSnapshotRequest{SessionID: sessionID})
}

// readSnapshot dispatches req to the getSnapshot companion action and decodes
// the snapshot. The action runs in-process here, so its error chain stays
// live: sentinel matching with errors.Is works, subtypes included (e.g.
// [ErrSnapshotNotFound] is a [status.ErrNotFound]).
func (h *AgentHandle) readSnapshot(ctx context.Context, req *GetSnapshotRequest) (*SessionSnapshot[json.RawMessage], error) {
	if h.getSnapshot == nil {
		return nil, status.Errorf(ErrSessionStoreNotConfigured,
			"agent %q has no session store, so it keeps no snapshots to read", h.name)
	}
	reqJSON, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("agent %q: marshal snapshot request: %w", h.name, err)
	}
	raw, err := h.getSnapshot.RunJSON(ctx, reqJSON, nil)
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

// Wait blocks until the task settles and returns its terminal snapshot,
// re-reading the snapshot on an interval ([WithPollInterval] to change it). A
// task that failed, aborted, or expired still returns its snapshot rather than
// an error (inspect [SessionSnapshot.Status] and [SessionSnapshot.Error]), so
// a non-nil error means the wait itself could not proceed: a read failed, or
// ctx ended and its error is returned. Use [context.WithTimeout] to bound the
// wait.
func (t *DetachedTask) Wait(ctx context.Context, opts ...WaitOption) (*SessionSnapshot[json.RawMessage], error) {
	waitOpts := &waitOptions{}
	for _, opt := range opts {
		if err := opt.applyWait(waitOpts); err != nil {
			return nil, fmt.Errorf("agent %q: Wait: %w", t.handle.name, err)
		}
	}
	interval := waitOpts.pollInterval
	if interval == 0 {
		interval = defaultTaskPollInterval
	}
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		snap, err := t.Poll(ctx)
		if err != nil {
			return nil, err
		}
		if snap.Status.Terminal() {
			return snap, nil
		}
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-ticker.C:
		}
	}
}

// Abort asks the task's background work to stop and returns the snapshot's
// status after the attempt; see [AgentHandle.Abort].
func (t *DetachedTask) Abort(ctx context.Context) (SnapshotStatus, error) {
	return t.handle.Abort(ctx, t.snapshotID)
}
