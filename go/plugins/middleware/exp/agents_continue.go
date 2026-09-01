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

// Task continuation for the [Agents] middleware.
//
// A delegation that settled leaves a handle behind ("<agent>:<snapshotId>",
// see delegationResult.TaskID), and the sub-agent runtime makes the snapshot
// behind it a resume point: a failed or aborted run holds the state through
// its last committed turn, and a completed run holds the whole conversation.
// The shared continue tool spends that handle: it retries a failed or aborted
// task from its saved progress (an empty instructions field re-attempts the
// turn as committed; a non-empty one steers the retry), and follows up on a
// completed task inside the sub-agent's own session, so the orchestrator can
// press on without re-buying work that already happened, and without the
// sub-agent's conversation ever entering its own context window.
//
// Only server-managed sub-agents (those with a session store) are continuable.
// A client-managed delegation settles inline and leaves nothing durable a
// handle could name, so its result carries no taskId, and redoing its work
// means delegating again.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/status"
)

// continueTaskToolName is the well-known name of the shared continue tool,
// namespaced by an explicitly set [Agents.ToolPrefix] like the
// background-task tools (see backgroundToolNames for the rationale).
const continueTaskToolName = "continue_task"

// continueInput is the continue tool's input.
type continueInput struct {
	TaskID       string `json:"taskId" jsonschema_description:"The task handle to continue (\"<agent>:<snapshotId>\"), from a delegation result or a background-task report."`
	Instructions string `json:"instructions,omitempty" jsonschema_description:"Optional guidance delivered to the sub-agent as it continues. Omit it to retry a failed or aborted task exactly as it stood; required when following up on a completed task."`
}

// asyncContinueInput is the continue tool's input when [Agents.Async] is set: the
// plain input plus the background flag.
type asyncContinueInput struct {
	continueInput
	Background bool `json:"background,omitempty" jsonschema_description:"Continue the task in the background. The tool returns immediately with a new taskId; collect the result later with the background-task tools."`
}

// continueToolName returns the continue tool's name for this configuration,
// following the background-task tools' prefix rule.
func (a *Agents) continueToolName() string {
	prefix := ""
	if a.ToolPrefix != nil {
		prefix = *a.ToolPrefix
	}
	return makeToolName(prefix, continueTaskToolName)
}

// continueToolDescription renders the continue tool's model-facing description.
func continueToolDescription() string {
	return "Continues a sub-agent task by its taskId: a failed or aborted task picks up from its last saved progress (omit instructions to retry it as it stood, or pass instructions to steer it), and a completed task accepts follow-up instructions inside its own session. A task that stopped on an interrupt cannot be continued."
}

// continueTask builds the continue tool function ([Agents.Async] unset).
func (a *Agents) continueTask(st *agentsState) func(context.Context, continueInput) (delegationResult, error) {
	return func(ctx context.Context, in continueInput) (delegationResult, error) {
		return a.runContinue(ctx, st, in, false)
	}
}

// continueTaskAsync is the continue tool function when [Agents.Async] is set,
// accepting the extra background flag.
func (a *Agents) continueTaskAsync(st *agentsState) func(context.Context, asyncContinueInput) (delegationResult, error) {
	return func(ctx context.Context, in asyncContinueInput) (delegationResult, error) {
		return a.runContinue(ctx, st, in.continueInput, in.Background)
	}
}

// runContinue is the continue tool body. It spends a delegation slot like any
// delegation (a continuation is a real sub-agent run, and an always-failing
// task continued forever is exactly the runaway MaxDelegations bounds),
// resolves the handle, and continues the snapshot behind it. Only
// server-managed sub-agents are continuable: a client-managed delegation
// settles inline and leaves nothing durable a handle could name.
//
// Refusal slot policy follows launchDelegation's precedent: a refusal that
// ran no sub-agent work and names a corrected retry that can succeed (wrong
// background flag, missing instructions, task still running, a transient
// read failure) returns its slot; a dead end (unknown handle, unresolvable
// or client-managed agent, no saved progress) keeps it, because its retry
// fails identically and refunding it would leave the cap unable to bite.
func (a *Agents) runContinue(ctx context.Context, st *agentsState, in continueInput, background bool) (delegationResult, error) {
	ref, snapshotID, err := a.resolveTaskID(in.TaskID)
	if err != nil {
		return delegationResult{Response: "Error: " + err.Error()}, nil
	}
	invocationNum, _, agent, refusal := a.beginDelegation(ctx, ref, st)
	if refusal != nil {
		return *refusal, nil
	}
	if isClientManaged(agent) {
		// Nothing durable exists behind a client-managed delegation, so no
		// handle can name a resume point; a dead end keeps its slot.
		return delegationResult{Response: fmt.Sprintf(
			"Error: agent %q manages its state on the client and its delegations cannot be continued; delegate the task again. Only sub-agents with a session store leave continuable task handles.", ref.Name)}, nil
	}
	return a.continueFromStore(ctx, ref, st, agent, invocationNum, in, snapshotID, background)
}

// continueFromStore continues a server-managed task from the snapshot behind its
// handle. The read goes through the sub-agent's companion action, so the
// runtime's shaping applies: a pending row whose heartbeat went stale reads
// as expired here rather than as forever-running. Every pre-read on this
// path is metadata-only ([aix.WithMetadataOnly]): the flow
// dispatches on status and finish reason alone, and the run itself loads the
// state it resumes, so nothing here needs the conversation serialized.
func (a *Agents) continueFromStore(ctx context.Context, ref aix.AgentRef, st *agentsState, agent *aix.AgentHandle, invocationNum int, in continueInput, snapshotID string, background bool) (delegationResult, error) {
	snap, err := agent.GetSnapshot(ctx, snapshotID, aix.WithMetadataOnly())
	if err != nil {
		logger.Debug(ctx, "continue read failed", "agent", ref.Name, "taskId", in.TaskID, "error", err)
		switch {
		case errors.Is(err, status.ErrNotFound):
			return delegationResult{Response: fmt.Sprintf(
				"Error: no record of task %q exists (%v). Delegate the task again if the work is still needed.", in.TaskID, err)}, nil
		case errors.Is(err, status.ErrFailedPrecondition), errors.Is(err, status.ErrInvalidArgument):
			return delegationResult{Response: fmt.Sprintf("Error continuing task %q: %v", in.TaskID, err)}, nil
		default:
			a.releaseDelegation(st)
			return delegationResult{Response: fmt.Sprintf(
				"Error: could not read task %q (%v). Try again later.", in.TaskID, err)}, nil
		}
	}

	switch snap.Status {
	case aix.SnapshotStatusPending:
		a.releaseDelegation(st)
		hint := ""
		if a.Async {
			names := a.backgroundToolNames()
			hint = fmt.Sprintf(" Collect it with %s or %s, or stop it with %s first.", names.check, names.wait, names.abort)
		}
		return delegationResult{Response: fmt.Sprintf("Task %q is still running; only a settled task can be continued.%s", in.TaskID, hint)}, nil

	case aix.SnapshotStatusAborting:
		// The stop landed and the worker is draining toward the finalize that
		// makes the row a continuation point; the same ID is the thing to
		// retry, and the refund reflects that.
		a.releaseDelegation(st)
		hint := ""
		if a.Async {
			hint = fmt.Sprintf(" Collect its settled state with %s, then continue that.", a.backgroundToolNames().wait)
		}
		return delegationResult{Response: fmt.Sprintf(
			"Task %q is winding down after a stop signal; its progress is being saved. Retry this taskId once it settles.%s", in.TaskID, hint)}, nil

	case aix.SnapshotStatusCompleted, aix.SnapshotStatusFailed, aix.SnapshotStatusAborted:
		return a.continueSettled(ctx, ref, st, agent, invocationNum, in, snapshotID, snap, background)

	case aix.SnapshotStatusExpired:
		return a.continueExpired(ctx, ref, st, agent, invocationNum, in, snapshotID, background)
	}
	a.releaseDelegation(st)
	return delegationResult{Response: fmt.Sprintf("Error: task %q is in an unexpected state (%q) and cannot be continued.", in.TaskID, snap.Status)}, nil
}

// continueSettled continues a handle whose shaped row is settled (completed,
// failed, or aborted), applying the interrupt refusal and the completed-task
// instructions gate before the run. An aborted row seen here carries state:
// the runtime's finalize lands both, a row still winding down reads as
// aborting, and a dead worker's flipped row reads as expired, so neither of
// those reaches this path.
func (a *Agents) continueSettled(ctx context.Context, ref aix.AgentRef, st *agentsState, agent *aix.AgentHandle, invocationNum int, in continueInput, snapshotID string, snap *aix.SessionSnapshot[json.RawMessage], background bool) (delegationResult, error) {
	if snap.FinishReason == aix.AgentFinishReasonInterrupted {
		// Continuing past an interrupt means answering it, which the
		// orchestrator cannot do; a retry fails identically, so this dead end
		// keeps its slot.
		return delegationResult{Response: fmt.Sprintf(
			"Error: task %q stopped on an interrupt (a tool request that needs an answer from outside the sub-agent), and continuing interrupted tasks is not supported. Delegate a more self-contained task instead.", in.TaskID)}, nil
	}
	if refusal := a.refuseEmptyFollowUp(st, snap, in, fmt.Sprintf(
		"Task %q already completed. To follow up in the sub-agent's session, call this tool again with instructions; re-running it without instructions would only repeat the finished work.", in.TaskID)); refusal != nil {
		return *refusal, nil
	}
	return a.runContinueFromSnapshot(ctx, ref, st, agent, invocationNum, in, snapshotID, background)
}

// continueExpired recovers a task whose worker is presumed dead. "Presumed" is
// the word: a slow worker may still be alive and beating late, and continuing
// past it would fork the run against it. The row is aborted first as a fence
// (a live worker observes the flip and stops; a dead one is unaffected), and
// one shaped re-read then decides the recovery point:
//
//   - A settled row (completed, failed, or aborted): a finalize landed in the
//     window, so the worker was alive after all and the row itself holds the
//     run's full state. It continues like any settled handle, completed rows
//     gated the same way.
//   - An aborting row: the fence reached a live worker, whose beats keep the
//     row reading as aborting because its finalize is coming. That settled
//     row will be the continuation point, so the refusal names the retry and
//     returns the slot.
//   - Still expired: the worker really is gone, and the recovery falls back
//     to the dead row's parent, the last snapshot committed before the
//     detach.
//
// The fence is the one write standing between the recovery and a live
// worker, so a fence that fails is a refusal, not a log line: proceeding
// without it risks two live branches of one session. Fence and re-read
// failures are transient (the retried call can succeed), so both refusals
// return their slot.
func (a *Agents) continueExpired(ctx context.Context, ref aix.AgentRef, st *agentsState, agent *aix.AgentHandle, invocationNum int, in continueInput, snapshotID string, background bool) (delegationResult, error) {
	if _, err := agent.Abort(ctx, snapshotID); err != nil {
		logger.Debug(ctx, "continue fence abort failed", "agent", ref.Name, "taskId", in.TaskID, "error", err)
		a.releaseDelegation(st)
		return delegationResult{Response: fmt.Sprintf(
			"Error: could not fence task %q before recovering it (%v). Try again later.", in.TaskID, err)}, nil
	}
	cur, err := agent.GetSnapshot(ctx, snapshotID, aix.WithMetadataOnly())
	if err != nil {
		logger.Debug(ctx, "continue fence re-read failed", "agent", ref.Name, "taskId", in.TaskID, "error", err)
		a.releaseDelegation(st)
		return delegationResult{Response: fmt.Sprintf(
			"Error: could not read task %q after fencing it (%v). Try again later.", in.TaskID, err)}, nil
	}
	switch cur.Status {
	case aix.SnapshotStatusCompleted, aix.SnapshotStatusFailed, aix.SnapshotStatusAborted:
		return a.continueSettled(ctx, ref, st, agent, invocationNum, in, snapshotID, cur, background)
	case aix.SnapshotStatusPending, aix.SnapshotStatusAborting:
		// The fence reached a live worker: it beat after the flip, so its
		// finalize is coming and the parent must not be raced.
		a.releaseDelegation(st)
		hint := ""
		if a.Async {
			hint = fmt.Sprintf(" Collect its settled state with %s, then continue that.", a.backgroundToolNames().wait)
		}
		return delegationResult{Response: fmt.Sprintf(
			"Task %q is still winding down after the stop signal reached it; its progress is being saved. Retry this taskId once it settles.%s", in.TaskID, hint)}, nil
	}
	return a.continueFromParent(ctx, ref, st, agent, invocationNum, in, cur.ParentID, background)
}

// refuseEmptyFollowUp refuses an instructions-less continuation of a snapshot whose
// last committed turn finished (completed, with a result-carrying reason). An
// empty input re-attempts the last committed turn, which is the right retry
// for a run that stopped short and pure duplicate work for one that finished;
// the refusal delivers msg (which names the fix) and returns the slot, since
// the corrected call is a real run that can succeed. The release happens
// here, next to the refusal it belongs to, so every gate call site inherits
// the refund instead of each having to remember it.
func (a *Agents) refuseEmptyFollowUp(st *agentsState, snap *aix.SessionSnapshot[json.RawMessage], in continueInput, msg string) *delegationResult {
	if snap.Status != aix.SnapshotStatusCompleted || !snap.FinishReason.CarriesResult() || in.Instructions != "" {
		return nil
	}
	a.releaseDelegation(st)
	return &delegationResult{Response: msg}
}

// runContinueFromSnapshot runs the sub-agent from the named snapshot and folds
// the outcome like a synchronous delegation (or launches it in the
// background, like a background delegation).
func (a *Agents) runContinueFromSnapshot(ctx context.Context, ref aix.AgentRef, st *agentsState, agent *aix.AgentHandle, invocationNum int, in continueInput, snapshotID string, background bool) (delegationResult, error) {
	return a.runContinueWith(ctx, ref, st, agent, invocationNum, in, background, aix.WithSnapshotID[json.RawMessage](snapshotID))
}

// continueFromParent recovers a dead task from its pending row's parent: the
// last snapshot committed before the detach. The session's latest row cannot
// serve here, because it is the dead pending row itself (minted at detach,
// newer than every committed turn), so the parent pointer is the one durable
// path back to the committed work.
//
// A background delegation detaches at turn zero and has no parent, which is
// the honest nothing-was-saved case. The parent is read first so a finished
// parent turn gets the same instructions gate as a completed task: an empty
// input would re-run that finished turn rather than continue the dead work.
func (a *Agents) continueFromParent(ctx context.Context, ref aix.AgentRef, st *agentsState, agent *aix.AgentHandle, invocationNum int, in continueInput, parentID string, background bool) (delegationResult, error) {
	if parentID == "" {
		return delegationResult{Response: fmt.Sprintf(
			"Error: task %q saved no progress to continue from (it detached at the start of the run, and its worker died before finalizing). Delegate the task again if the work is still needed.", in.TaskID)}, nil
	}
	parent, err := agent.GetSnapshot(ctx, parentID, aix.WithMetadataOnly())
	if err != nil {
		if !errors.Is(err, status.ErrNotFound) && !errors.Is(err, status.ErrFailedPrecondition) && !errors.Is(err, status.ErrInvalidArgument) {
			// Transient, and the refusal names a retry that can succeed, so
			// the slot comes back, the same refund continueFromStore applies to
			// its own transient read failures. The dead-end classes (a
			// deleted parent, a rejected request) keep their slot.
			a.releaseDelegation(st)
		}
		return delegationResult{Response: fmt.Sprintf(
			"Error: task %q kept its progress in snapshot %q, which could not be read (%v). Try again later.", in.TaskID, parentID, err)}, nil
	}
	if refusal := a.refuseEmptyFollowUp(st, parent, in, fmt.Sprintf(
		"Task %q kept progress only up to its last finished turn (from before the background work started). Call this tool again with instructions to continue from there; an empty retry would only re-run that finished turn.", in.TaskID)); refusal != nil {
		return *refusal, nil
	}
	return a.runContinueFromSnapshot(ctx, ref, st, agent, invocationNum, in, parentID, background)
}

// runContinueWith is the shared tail of every store-backed continuation: the optional
// background pre-flight, the run itself, and the folding of its outcome.
func (a *Agents) runContinueWith(ctx context.Context, ref aix.AgentRef, st *agentsState, agent *aix.AgentHandle, invocationNum int, in continueInput, background bool, opt aix.InvocationOption[json.RawMessage]) (delegationResult, error) {
	if background {
		// Same pre-flight as launchDelegation: a genkit-defined agent that
		// cannot detach is refused deterministically, and the refusal names
		// the synchronous retry, so it returns its slot.
		if meta := agent.Metadata(); meta != nil && !meta.Abortable {
			a.releaseDelegation(st)
			return delegationResult{Response: fmt.Sprintf(
				"Error continuing task %q: this agent lacks a session store that supports background work. Continue it without \"background\" instead.", in.TaskID)}, nil
		}
	}

	logger.Debug(ctx, "continuing sub-agent task",
		"agent", ref.Name, "taskId", in.TaskID, "invocation", invocationNum, "background", background)
	out, err := runContinuedSubAgent(ctx, agent, in.Instructions, background, opt)
	if err != nil {
		logger.Warn(ctx, "sub-agent continuation failed", "agent", ref.Name, "taskId", in.TaskID, "error", err)
		if errors.Is(err, status.ErrFailedPrecondition) {
			// The runtime rejected the resume point itself (nothing behind
			// it, or a still-live worker); its message says which.
			return delegationResult{Response: fmt.Sprintf(
				"Error continuing task %q: %v. If no progress was saved, delegate the task again.", in.TaskID, err)}, nil
		}
		return delegationResult{Response: fmt.Sprintf("Error continuing task %q: %v", in.TaskID, err)}, nil
	}

	if background && out.FinishReason == aix.AgentFinishReasonFailed && maybeDetachRejection(agent, out) {
		// launchDelegation's hedge, mirrored: only a metadata-less agent
		// reaches the runtime's detach rejection (the pre-flight above
		// refused a genkit-defined agent that cannot detach), and the
		// refusal names the synchronous retry, so it returns its slot.
		a.releaseDelegation(st)
		return delegationResult{Response: fmt.Sprintf(
			"Error continuing task %q: %s If this agent lacks a session store that supports background work, continue it without \"background\" instead.",
			in.TaskID, subAgentFailureMessage(out.FinishReason, out.Error, out.Message))}, nil
	}

	if out.FinishReason == aix.AgentFinishReasonDetached {
		taskID := formatTaskID(ref.Name, out.SnapshotID)
		names := a.backgroundToolNames()
		logger.Debug(ctx, "background continuation started",
			"agent", ref.Name, "taskId", taskID, "continuedFrom", in.TaskID, "sessionId", out.SessionID)
		result := delegationResult{
			TaskID: taskID,
			Status: string(aix.SnapshotStatusPending),
			Response: fmt.Sprintf(
				"Task %s continued in the background as %s. Collect the result with %s or %s, or stop it with %s.",
				in.TaskID, taskID, names.check, names.wait, names.abort),
		}
		a.labelTask(st, &result, a.taskLabel(st, in.TaskID))
		return result, nil
	}
	result := a.foldDelegationOutput(ctx, ref, out, invocationID(ref, out, invocationNum))
	// The continuation is the same undertaking; its label follows the handle.
	a.labelTask(st, &result, a.taskLabel(st, in.TaskID))
	return result, nil
}

// runContinuedSubAgent runs the agent with the resume init option and the
// instructions as the turn's user message. No instructions means no message:
// the runtime re-attempts the conversation as committed, which is the retry
// semantics for a run that stopped short.
//
// The exception is a background retry. A detached input with no payload of
// its own is a pure detach signal to the runtime and runs no turn (see
// hasInputPayload in ai/exp), so an empty background continuation would finalize
// the loaded state untouched instead of retrying it. It gets the smallest
// honest payload instead: a continue message, which also documents in the
// sub-agent's transcript why the run picked back up.
func runContinuedSubAgent(ctx context.Context, agent *aix.AgentHandle, instructions string, detach bool, opts ...aix.InvocationOption[json.RawMessage]) (*aix.AgentOutput[json.RawMessage], error) {
	if detach && instructions == "" {
		instructions = "Continue the task from where it stopped."
	}
	input := &aix.AgentInput{Detach: detach}
	if instructions != "" {
		input.Message = ai.NewUserTextMessage(instructions)
	}
	return agent.Run(ctx, input, opts...)
}
