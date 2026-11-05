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

	"github.com/firebase/genkit/go/core/status"
)

// DetachedTask tracks a detached (background) agent invocation through its
// pending snapshot, with custom state typed as State: the agent's own state
// type for a task the typed [Agent] minted, [json.RawMessage] for one an
// [AgentHandle] minted. Obtain one from [Agent.RunDetached] or
// [AgentHandle.RunDetached] when launching, or rehydrate one from a recorded
// snapshot ID with [Agent.Task] or [AgentHandle.Task]; the two
// are equivalent, because the snapshot is the only state a task has.
type DetachedTask[State any] struct {
	ops        snapshotOps[State]
	snapshotID string
}

// snapshotOps is what a [DetachedTask] needs from the surface that minted it.
// The typed [Agent] and the untyped [AgentHandle] both publish these methods,
// so a task reads, waits, and aborts exactly as its origin would, differences
// included: through a handle, aborting a missing snapshot is NOT_FOUND, where
// [Agent.Abort] reports "".
type snapshotOps[State any] interface {
	GetSnapshot(ctx context.Context, snapshotID string, opts ...SnapshotReadOption) (*SessionSnapshot[State], error)
	WaitForSnapshot(ctx context.Context, snapshotID string) (*SessionSnapshot[State], error)
	Abort(ctx context.Context, snapshotID string) (SnapshotStatus, error)
}

var (
	_ snapshotOps[any]             = (*Agent[any])(nil)
	_ snapshotOps[json.RawMessage] = (*AgentHandle)(nil)
)

// detachedInput returns a copy of input with [AgentInput.Detach] set, whatever
// the caller set it to. The copy is shallow: the message and resume payloads
// are shared, not cloned, and nothing mutates them.
func detachedInput(input *AgentInput) *AgentInput {
	detached := *input
	detached.Detach = true
	return &detached
}

// detachedTaskFrom folds the output of an invocation delivered with
// [AgentInput.Detach] into the task tracking it, or the error saying why there
// is none. ops is the surface the task reads through, and name the agent's,
// for the messages.
//
// A detached finish is the expected outcome. A failed one is the runtime
// refusing the launch (no session store, or one that cannot observe aborts),
// surfaced as an error carrying the output's status. Anything else means the
// agent settled the invocation before the runtime observed the detach
// directive (nothing orders the intake reader ahead of an agent fn that never
// consumes its input): a committed turn's snapshot is the settled work's
// durable record, so the task tracks it and Poll and Wait resolve at once; with
// nothing recorded there is nothing to track.
func detachedTaskFrom[State any](name string, ops snapshotOps[State], out *AgentOutput[State]) (*DetachedTask[State], error) {
	switch out.FinishReason {
	case AgentFinishReasonDetached:
		return &DetachedTask[State]{ops: ops, snapshotID: out.SnapshotID}, nil
	case AgentFinishReasonFailed:
		var cause error = out.Error
		if out.Error == nil {
			cause = status.Errorf(status.ErrInternal, "launch failed without error detail")
		}
		return nil, fmt.Errorf("agent %q: background launch failed: %w", name, cause)
	default:
		if out.SnapshotID != "" {
			return &DetachedTask[State]{ops: ops, snapshotID: out.SnapshotID}, nil
		}
		return nil, status.Errorf(status.ErrFailedPrecondition,
			"agent %q: the invocation settled synchronously (finish reason %q, session %q) without recording a snapshot, so there is no background task to track",
			name, out.FinishReason, out.SessionID)
	}
}

// SnapshotID returns the ID of the snapshot tracking the task: pending while
// the background work runs, finalized in place when it settles. It is the
// task's durable identity; record it to pick the task up later with
// [Agent.Task] or [AgentHandle.Task].
func (t *DetachedTask[State]) SnapshotID() string { return t.snapshotID }

// Poll fetches the task's snapshot once, without waiting. The snapshot's
// Status says where the task stands: [SnapshotStatusPending] while the work
// runs, a terminal status once it settles (see [SnapshotStatus.Terminal]),
// including [SnapshotStatusExpired] when the worker stopped heartbeating and
// is presumed dead. A terminal snapshot carries the cumulative session state;
// a poll that only dispatches on where the task stands can pass
// [WithMetadataOnly] to skip that payload.
func (t *DetachedTask[State]) Poll(ctx context.Context, opts ...SnapshotReadOption) (*SessionSnapshot[State], error) {
	return t.ops.GetSnapshot(ctx, t.snapshotID, opts...)
}

// Wait blocks until the task settles and returns its terminal snapshot: the
// waiting happens next to the store that knows when the work finished, so the
// caller neither picks a cadence nor pays a read per tick. A task that failed,
// aborted, or expired still returns its snapshot rather than an error (inspect
// [SessionSnapshot.Status] and [SessionSnapshot.Error]), so a non-nil error
// means the wait itself could not proceed: reads failed past the wait's
// transient-retry budget, or ctx ended and its error is returned.
//
// Use [context.WithTimeout] to bound the wait; on the deadline the wait returns
// ctx's error, and [DetachedTask.Poll] then reports where the task stands.
func (t *DetachedTask[State]) Wait(ctx context.Context) (*SessionSnapshot[State], error) {
	return t.ops.WaitForSnapshot(ctx, t.snapshotID)
}

// Abort asks the task's background work to stop and returns the snapshot's
// status after the attempt, as [Agent.Abort] or [AgentHandle.Abort] reports it.
func (t *DetachedTask[State]) Abort(ctx context.Context) (SnapshotStatus, error) {
	return t.ops.Abort(ctx, t.snapshotID)
}
