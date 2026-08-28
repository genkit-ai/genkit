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

// Task resume for the [Agents] middleware.
//
// A delegation that settled leaves a handle behind ("<agent>:<snapshotId>",
// see delegationResult.TaskID), and the sub-agent runtime makes the snapshot
// behind it a resume point: a failed or aborted run holds the state through
// its last committed turn, and a completed run holds the whole conversation.
// The shared resume tool spends that handle: it retries a failed or aborted
// task from its saved progress (an empty instructions field re-attempts the
// turn as committed; a non-empty one steers the retry), and follows up on a
// completed task inside the sub-agent's own session, so the orchestrator can
// press on without re-buying work that already happened, and without the
// sub-agent's conversation ever entering its own context window.
//
// Client-managed sub-agents have no store, so their settled state is held in
// the middleware's per-call stash instead and addressed by an in-memory
// handle ("<agent>:mem-<n>"). The stash lives exactly as long as the generate
// call that minted it: long enough for the orchestrator to hear about a
// failure and decide, gone once the turn ends. Durable, cross-turn resume is
// what a session store provides; the refusal for a stale in-memory handle
// says as much.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/status"
)

// resumeSubagentToolName is the well-known name of the shared resume tool,
// namespaced by an explicitly set [Agents.ToolPrefix] like the
// background-task tools (see backgroundToolNames for the rationale).
const resumeSubagentToolName = "resume_subagent"

// memHandlePrefix marks the snapshot-ID position of an in-memory handle
// ("<agent>:mem-<n>"). The agent runtime mints snapshot IDs as UUIDs, which
// never carry this prefix, so the two handle kinds cannot collide.
const memHandlePrefix = "mem-"

// resumeInput is the resume tool's input.
type resumeInput struct {
	TaskID       string `json:"taskId" jsonschema_description:"The task handle to resume (\"<agent>:<snapshotId>\"), from a delegation result or a background-task report."`
	Instructions string `json:"instructions,omitempty" jsonschema_description:"Optional guidance delivered to the sub-agent as it resumes. Omit it to retry a failed or aborted task exactly as it stood; required when following up on a completed task."`
}

// asyncResumeInput is the resume tool's input when [Agents.Async] is set: the
// plain input plus the background flag.
type asyncResumeInput struct {
	resumeInput
	Background bool `json:"background,omitempty" jsonschema_description:"Resume the task in the background. The tool returns immediately with a new taskId; collect the result later with the background-task tools."`
}

// memStash is one settled client-managed delegation's final state, held for
// the rest of the generate call under its minted in-memory handle.
type memStash struct {
	// state is the sub-agent's final session state as the runtime returned
	// it: already a private copy (the runtime deep-copies at its boundary),
	// and deep-copied again by the runtime on the way back in, so holding
	// and re-running it needs no copies here.
	state *aix.SessionState[json.RawMessage]
	// status is the settled outcome the stash was minted for; "completed"
	// gates the instructions requirement exactly as it does for a completed
	// snapshot.
	status string
}

// resumeToolName returns the resume tool's name for this configuration,
// following the background-task tools' prefix rule.
func (a *Agents) resumeToolName() string {
	prefix := ""
	if a.ToolPrefix != nil {
		prefix = *a.ToolPrefix
	}
	return makeToolName(prefix, resumeSubagentToolName)
}

// resumeToolDescription renders the resume tool's model-facing description.
func resumeToolDescription() string {
	return "Resumes a sub-agent task by its taskId: a failed or aborted task continues from its last saved progress (omit instructions to retry it as it stood, or pass instructions to steer it), and a completed task accepts follow-up instructions inside its own session."
}

// resume builds the resume tool function ([Agents.Async] unset).
func (a *Agents) resume(st *agentsState) func(context.Context, resumeInput) (delegationResult, error) {
	return func(ctx context.Context, in resumeInput) (delegationResult, error) {
		return a.runResume(ctx, st, in, false)
	}
}

// resumeAsync is the resume tool function when [Agents.Async] is set,
// accepting the extra background flag.
func (a *Agents) resumeAsync(st *agentsState) func(context.Context, asyncResumeInput) (delegationResult, error) {
	return func(ctx context.Context, in asyncResumeInput) (delegationResult, error) {
		return a.runResume(ctx, st, in.resumeInput, in.Background)
	}
}

// runResume is the resume tool body. It spends a delegation slot like any
// delegation (a resume is a real sub-agent run, and an always-failing task
// resumed forever is exactly the runaway MaxDelegations bounds), resolves the
// handle, and dispatches on its kind: in-memory handles replay the stashed
// client-managed state, store handles resume the snapshot behind them.
//
// Refusal slot policy follows launchDelegation's precedent: a refusal that
// ran no sub-agent work and names a corrected retry that can succeed (wrong
// background flag, missing instructions, task still running) returns its
// slot; a dead end (unknown handle, unresolvable agent, no saved progress)
// keeps it, because its retry fails identically and refunding it would leave
// the cap unable to bite.
func (a *Agents) runResume(ctx context.Context, st *agentsState, in resumeInput, background bool) (delegationResult, error) {
	ref, rest, err := a.resolveTaskID(in.TaskID)
	if err != nil {
		return delegationResult{Response: "Error: " + err.Error()}, nil
	}
	invocationNum, _, agent, refusal := a.beginDelegation(ctx, ref, st)
	if refusal != nil {
		return *refusal, nil
	}
	if strings.HasPrefix(rest, memHandlePrefix) {
		return a.resumeFromStash(ctx, ref, st, agent, invocationNum, in, background)
	}
	return a.resumeFromStore(ctx, ref, st, agent, invocationNum, in, rest, background)
}

// resumeFromStash resumes a client-managed delegation from the per-call
// stash: the stashed state rides back in as init state, exactly as the
// original delegation's history did, plus the instructions as the new user
// message (or no message at all, which re-attempts the committed turn).
func (a *Agents) resumeFromStash(ctx context.Context, ref aix.AgentRef, st *agentsState, agent *aix.AgentHandle, invocationNum int, in resumeInput, background bool) (delegationResult, error) {
	st.mu.Lock()
	stash, ok := st.stashes[in.TaskID]
	st.mu.Unlock()
	if !ok {
		// Unlike a store handle, nothing durable sits behind an in-memory
		// one, so a miss is a dead end (slot kept), not a retry hint.
		return delegationResult{Response: fmt.Sprintf(
			"Error: no in-memory state is held for %q. In-memory handles (\"<agent>:%s<n>\") live only until the turn that minted them ends; delegate the task again. For handles that survive turns, give the sub-agent a session store.",
			in.TaskID, memHandlePrefix)}, nil
	}
	if background {
		a.releaseDelegation(st)
		return delegationResult{Response: fmt.Sprintf(
			"Error: in-memory handle %q cannot be resumed in the background (background work requires the sub-agent to have a session store). Resume it without \"background\".", in.TaskID)}, nil
	}
	if refusal := a.requireFollowUpInstructions(stash.status, in); refusal != nil {
		return *refusal, nil
	}

	logger.Debug(ctx, "resuming sub-agent from stashed state",
		"agent", ref.Name, "taskId", in.TaskID, "invocation", invocationNum)
	out, err := runResumedSubAgent(ctx, agent, in.Instructions, false, aix.WithState(stash.state))
	if err != nil {
		logger.Warn(ctx, "sub-agent resume failed", "agent", ref.Name, "taskId", in.TaskID, "error", err)
		return delegationResult{Response: fmt.Sprintf("Error resuming task %q: %v", in.TaskID, err)}, nil
	}
	result := a.foldDelegationOutput(ctx, ref, out, fmt.Sprintf("%s_%d", ref.Name, invocationNum))
	// Re-stash the continued conversation under a fresh handle so the next
	// failure or follow-up starts from this run's end, not the original's.
	a.stashClientState(st, ref, out, &result)
	return result, nil
}

// resumeFromStore resumes a server-managed task from the snapshot behind its
// handle. The read goes through the sub-agent's companion action, so the
// runtime's shaping applies: a pending row whose heartbeat went stale reads
// as expired here rather than as forever-running.
func (a *Agents) resumeFromStore(ctx context.Context, ref aix.AgentRef, st *agentsState, agent *aix.AgentHandle, invocationNum int, in resumeInput, snapshotID string, background bool) (delegationResult, error) {
	snap, err := agent.GetSnapshot(ctx, snapshotID)
	if err != nil {
		logger.Debug(ctx, "resume read failed", "agent", ref.Name, "taskId", in.TaskID, "error", err)
		switch {
		case errors.Is(err, status.ErrNotFound):
			return delegationResult{Response: fmt.Sprintf(
				"Error: no record of task %q exists (%v). Delegate the task again if the work is still needed.", in.TaskID, err)}, nil
		case errors.Is(err, status.ErrFailedPrecondition), errors.Is(err, status.ErrInvalidArgument):
			return delegationResult{Response: fmt.Sprintf("Error resuming task %q: %v", in.TaskID, err)}, nil
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
		return delegationResult{Response: fmt.Sprintf("Task %q is still running; only a settled task can be resumed.%s", in.TaskID, hint)}, nil

	case aix.SnapshotStatusCompleted, aix.SnapshotStatusFailed:
		if snap.Status == aix.SnapshotStatusCompleted && snap.FinishReason.CarriesResult() {
			if refusal := a.requireFollowUpInstructions(string(aix.SnapshotStatusCompleted), in); refusal != nil {
				return *refusal, nil
			}
		}
		return a.runResumeFromSnapshot(ctx, ref, st, agent, invocationNum, in, snapshotID, background)

	case aix.SnapshotStatusAborted:
		if snap.State != nil {
			return a.runResumeFromSnapshot(ctx, ref, st, agent, invocationNum, in, snapshotID, background)
		}
		// Aborted with no state: the row is caught between the abort flip and
		// the finalize, and only the runtime's heartbeat heuristic can say
		// whether that finalize is still coming. Resume the row itself and
		// let the runtime adjudicate: a live worker's rejection says to retry
		// this same ID shortly, a dead one's names the parent snapshot to
		// resume instead, and either message reaches the model through the
		// error text.
		return a.runResumeFromSnapshot(ctx, ref, st, agent, invocationNum, in, snapshotID, background)

	case aix.SnapshotStatusExpired:
		// The worker is presumed dead, but "presumed" is the word: a slow
		// worker may still be alive and beating late, and resuming past it
		// would fork the run against it. Abort the row first as a fence (a
		// live worker observes the flip and stops; a dead one is
		// unaffected), then resume from whatever the run durably holds.
		if _, err := agent.Abort(ctx, snapshotID); err != nil {
			logger.Debug(ctx, "resume fence abort failed", "agent", ref.Name, "taskId", in.TaskID, "error", err)
		}
		// Re-read once: if a finalize landed in the window (the worker was
		// alive after all), the row now holds the run's full state and is
		// itself the resume point.
		if snap, err := agent.GetSnapshot(ctx, snapshotID); err == nil && snap.State != nil && snap.Status != aix.SnapshotStatusPending {
			return a.runResumeFromSnapshot(ctx, ref, st, agent, invocationNum, in, snapshotID, background)
		}
		return a.resumeFromParent(ctx, ref, st, agent, invocationNum, in, snap.ParentID, background)
	}
	a.releaseDelegation(st)
	return delegationResult{Response: fmt.Sprintf("Error: task %q is in an unexpected state (%q) and cannot be resumed.", in.TaskID, snap.Status)}, nil
}

// requireFollowUpInstructions refuses an instructions-less resume of a
// completed task. An empty input re-attempts the last committed turn, which
// is the right retry for a run that stopped short and pure duplicate work for
// one that finished; the refusal names the fix and returns the slot, since
// the corrected call is a real run that can succeed.
func (a *Agents) requireFollowUpInstructions(settled string, in resumeInput) *delegationResult {
	if settled != string(aix.SnapshotStatusCompleted) || in.Instructions != "" {
		return nil
	}
	return &delegationResult{Response: fmt.Sprintf(
		"Task %q already completed. To follow up in the sub-agent's session, call this tool again with instructions; re-running it without instructions would only repeat the finished work.", in.TaskID)}
}

// runResumeFromSnapshot runs the sub-agent from the named snapshot and folds
// the outcome like a synchronous delegation (or launches it in the
// background, like a background delegation).
func (a *Agents) runResumeFromSnapshot(ctx context.Context, ref aix.AgentRef, st *agentsState, agent *aix.AgentHandle, invocationNum int, in resumeInput, snapshotID string, background bool) (delegationResult, error) {
	return a.runResumeWith(ctx, ref, st, agent, invocationNum, in, background, aix.WithSnapshotID[json.RawMessage](snapshotID))
}

// resumeFromParent recovers a dead task from its pending row's parent: the
// last snapshot committed before the detach. The session's latest row cannot
// serve here, because it is the dead pending row itself (minted at detach,
// newer than every committed turn), so the parent pointer is the one durable
// path back to the committed work.
//
// A background delegation detaches at turn zero and has no parent, which is
// the honest nothing-was-saved case. The parent is read first so a finished
// parent turn gets the same instructions gate as a completed task: an empty
// input would re-run that finished turn rather than continue the dead work.
func (a *Agents) resumeFromParent(ctx context.Context, ref aix.AgentRef, st *agentsState, agent *aix.AgentHandle, invocationNum int, in resumeInput, parentID string, background bool) (delegationResult, error) {
	if parentID == "" {
		return delegationResult{Response: fmt.Sprintf(
			"Error: task %q saved no resumable progress (it detached at the start of the run, and its worker died before finalizing). Delegate the task again if the work is still needed.", in.TaskID)}, nil
	}
	parent, err := agent.GetSnapshot(ctx, parentID)
	if err != nil {
		return delegationResult{Response: fmt.Sprintf(
			"Error: task %q kept its progress in snapshot %q, which could not be read (%v). Try again later.", in.TaskID, parentID, err)}, nil
	}
	if parent.Status == aix.SnapshotStatusCompleted && parent.FinishReason.CarriesResult() && in.Instructions == "" {
		return delegationResult{Response: fmt.Sprintf(
			"Task %q kept progress only up to its last finished turn (from before the background work started). Call this tool again with instructions to continue from there; an empty retry would only re-run that finished turn.", in.TaskID)}, nil
	}
	return a.runResumeFromSnapshot(ctx, ref, st, agent, invocationNum, in, parentID, background)
}

// runResumeWith is the shared tail of every store-backed resume: the optional
// background pre-flight, the run itself, and the folding of its outcome.
func (a *Agents) runResumeWith(ctx context.Context, ref aix.AgentRef, st *agentsState, agent *aix.AgentHandle, invocationNum int, in resumeInput, background bool, opt aix.InvocationOption[json.RawMessage]) (delegationResult, error) {
	if background {
		// Same pre-flight as launchDelegation: a genkit-defined agent that
		// cannot detach is refused deterministically, and the refusal names
		// the synchronous retry, so it returns its slot.
		if meta := agent.Metadata(); meta != nil && !meta.Abortable {
			a.releaseDelegation(st)
			return delegationResult{Response: fmt.Sprintf(
				"Error resuming task %q: this agent lacks a session store that supports background work. Resume it without \"background\" instead.", in.TaskID)}, nil
		}
	}

	logger.Debug(ctx, "resuming sub-agent task",
		"agent", ref.Name, "taskId", in.TaskID, "invocation", invocationNum, "background", background)
	out, err := runResumedSubAgent(ctx, agent, in.Instructions, background, opt)
	if err != nil {
		logger.Warn(ctx, "sub-agent resume failed", "agent", ref.Name, "taskId", in.TaskID, "error", err)
		if errors.Is(err, status.ErrFailedPrecondition) {
			// The runtime rejected the resume point itself (nothing behind
			// it, or a still-live worker); its message says which.
			return delegationResult{Response: fmt.Sprintf(
				"Error resuming task %q: %v. If no progress was saved, delegate the task again.", in.TaskID, err)}, nil
		}
		return delegationResult{Response: fmt.Sprintf("Error resuming task %q: %v", in.TaskID, err)}, nil
	}

	if out.FinishReason == aix.AgentFinishReasonDetached {
		taskID := formatTaskID(ref.Name, out.SnapshotID)
		names := a.backgroundToolNames()
		logger.Debug(ctx, "background resume started",
			"agent", ref.Name, "taskId", taskID, "resumedFrom", in.TaskID, "sessionId", out.SessionID)
		return delegationResult{
			TaskID: taskID,
			Status: string(aix.SnapshotStatusPending),
			Response: fmt.Sprintf(
				"Task %s resumed in the background as %s. Collect the result with %s or %s, or stop it with %s.",
				in.TaskID, taskID, names.check, names.wait, names.abort),
		}, nil
	}
	return a.foldDelegationOutput(ctx, ref, out, fmt.Sprintf("%s_%d", ref.Name, invocationNum)), nil
}

// runResumedSubAgent runs the agent with the resume init option and the
// instructions as the turn's user message. No instructions means no message:
// the runtime re-attempts the conversation as committed, which is the retry
// semantics for a run that stopped short.
//
// The exception is a background retry. A detached input with no payload of
// its own is a pure detach signal to the runtime and runs no turn (see
// hasInputPayload in ai/exp), so an empty background resume would finalize
// the loaded state untouched instead of retrying it. It gets the smallest
// honest payload instead: a continue message, which also documents in the
// sub-agent's transcript why the run picked back up.
func runResumedSubAgent(ctx context.Context, agent *aix.AgentHandle, instructions string, detach bool, opts ...aix.InvocationOption[json.RawMessage]) (*aix.AgentOutput[json.RawMessage], error) {
	if detach && instructions == "" {
		instructions = "Continue the task from where it stopped."
	}
	input := &aix.AgentInput{Detach: detach}
	if instructions != "" {
		input.Message = ai.NewUserTextMessage(instructions)
	}
	return agent.Run(ctx, input, opts...)
}

// stashClientState holds a settled client-managed delegation's final state in
// the per-call stash and stamps the minted in-memory handle (plus the settled
// outcome) on the result, giving client-managed delegations the same
// addressable currency as server-managed ones for as long as the state
// exists. Interrupts stay unaddressable (resuming one means answering it),
// and a run that returned no state has nothing to hold.
func (a *Agents) stashClientState(st *agentsState, ref aix.AgentRef, out *aix.AgentOutput[json.RawMessage], result *delegationResult) {
	if out.State == nil || out.FinishReason == aix.AgentFinishReasonInterrupted || out.FinishReason == aix.AgentFinishReasonDetached {
		return
	}
	status := settledStatus(out.FinishReason)
	st.mu.Lock()
	st.memSeq++
	handle := formatTaskID(ref.Name, fmt.Sprintf("%s%d", memHandlePrefix, st.memSeq))
	st.stashes[handle] = &memStash{state: out.State, status: status}
	st.mu.Unlock()
	result.TaskID = handle
	result.Status = status
}
