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

// Background delegation for the [Agents] middleware ([Agents.Async]).
//
// A background delegation launches the sub-agent with [aix.AgentInput.Detach]:
// the sub-agent's runtime persists a pending snapshot, returns its ID at once,
// and keeps running the turn on a context decoupled from this tool call. The
// pending snapshot is the durable record of the task: it is heartbeated while
// the worker lives, finalized in place with the cumulative session state when
// the work settles, and surfaced as expired by readers once the heartbeat goes
// stale (worker presumed dead).
//
// The middleware itself keeps no task registry. The task handle
// ("<agent>:<snapshotId>") is self-contained and rides in the delegation tool
// result, so it is recorded in the orchestrator's conversation history; a
// re-instantiated orchestrator resumes tracking from the IDs in its history.
// Status goes through the sub-agent's [aix.AgentHandle] (resolved via
// resolveAgent, i.e. genkit/exp.LookupAgent, the sanctioned path for
// third-party middleware): the check tool dispatches the agent's getSnapshot
// companion action, the wait tool its waitForSnapshot counterpart, which
// blocks next to the store rather than making this middleware re-read on a
// timer, and the abort tool its abort counterpart, which flips a pending row
// to aborted so the sub-agent's runtime cancels the work.

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync"
	"time"

	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/genkit"
)

// Names of the shared background-task tools added when [Agents.Async] is set.
const (
	checkBackgroundTasksToolName = "check_background_tasks"
	waitBackgroundTasksToolName  = "wait_for_background_tasks"
	abortBackgroundTasksToolName = "abort_background_tasks"
)

// taskStatusUnknown is the report status for a task that could not be resolved
// (malformed ID, unconfigured agent, missing snapshot, or read error). It is
// terminal for waiting purposes: pending is the only status that can still
// change on its own.
const taskStatusUnknown = "unknown"

// noTaskIDsNote is the guidance returned when a background-task tool is called
// without task IDs.
const noTaskIDsNote = "No task IDs given. Pass the taskId values returned by background delegations."

// asyncDelegateInput is the delegation tool input when [Agents.Async] is set:
// the plain task plus the background flag.
type asyncDelegateInput struct {
	Task       string `json:"task" jsonschema_description:"A clear, self-contained description of the task to delegate."`
	Background bool   `json:"background,omitempty" jsonschema_description:"Run the delegation in the background. The tool returns immediately with a taskId; collect the result later with check_background_tasks or wait_for_background_tasks."`
}

// backgroundTasksInput is the input of the check and abort tools: a bare list
// of handles, the whole of what either needs.
//
// taskIds is omitempty so the schema does not mark it required. A model that
// calls one of these tools with no arguments is making a recoverable mistake,
// and the tools answer it with noTaskIDsNote; a required field would instead
// fail decoding, which surfaces as a tool error that fails the whole generate
// call rather than a turn the model can correct.
type backgroundTasksInput struct {
	TaskIDs []string `json:"taskIds,omitempty" jsonschema_description:"Task IDs returned by background delegations (form \"<agent>:<snapshotId>\")."`
}

// waitBackgroundTasksInput is the input of the wait tool: the shared handle
// list plus a bound on how long to block. The embedded struct keeps one
// description of taskIds, so the two tools cannot teach the model two
// different handle formats.
type waitBackgroundTasksInput struct {
	backgroundTasksInput
	TimeoutSeconds int `json:"timeoutSeconds,omitempty" jsonschema_description:"Maximum seconds to wait before returning the current statuses. 0 or omitted waits until every task settles; a negative value returns the current statuses immediately. Values too large to represent are treated as unbounded."`
}

// backgroundTaskReport is the per-task entry returned by the check and wait
// tools.
type backgroundTaskReport struct {
	// TaskID is the handle the report describes.
	TaskID string `json:"taskId"`
	// Agent is the sub-agent running the task.
	Agent string `json:"agent,omitempty"`
	// Status is the task's lifecycle state: "pending", "completed", "failed",
	// "aborted", "expired" (worker presumed dead), or "unknown" (the ID could
	// not be resolved; see Error). It answers what the reader must act on
	// rather than mirroring the stored row, so a task that committed without
	// producing an answer reports "failed" and explains itself in Error.
	// "completed" always carries a Response.
	Status string `json:"status"`
	// Response is the sub-agent's final text response, for completed tasks.
	Response string `json:"response,omitempty"`
	// Artifacts are the completed task's artifacts. Content is populated only
	// under ArtifactStrategyInline.
	Artifacts []delegatedArtifact `json:"artifacts,omitempty"`
	// Error describes why no response is available (failure, abort, expiry, or
	// an unresolvable task ID).
	Error string `json:"error,omitempty"`
}

// backgroundTasksResult is the output of the check and wait tools.
type backgroundTasksResult struct {
	Tasks []backgroundTaskReport `json:"tasks,omitempty"`
	// TimedOut is set when the wait returned because timeoutSeconds elapsed
	// while some tasks were still pending.
	TimedOut bool `json:"timedOut,omitempty"`
	// Note carries usage guidance when the call itself was unusable (e.g. no
	// task IDs given).
	Note string `json:"note,omitempty"`
}

// delegateAsync is the async-enabled delegation tool function: it behaves like
// [Agents.delegate] unless the model sets background, in which case it launches
// the task and returns its handle without waiting.
func (a *Agents) delegateAsync(ref aix.AgentRef, st *agentsState) func(context.Context, asyncDelegateInput) (delegationResult, error) {
	return func(ctx context.Context, in asyncDelegateInput) (delegationResult, error) {
		if !in.Background {
			return a.runDelegation(ctx, ref, st, in.Task)
		}
		return a.launchDelegation(ctx, ref, st, in.Task)
	}
}

// launchDelegation starts a background delegation through the sub-agent's
// detach support and returns the task handle without waiting for the work.
// Launches count against MaxDelegations like synchronous delegations, except
// for a launch the sub-agent cannot support at all: that refusal returns its
// slot, so the synchronous fallback it hints at is not refused by a cap the
// refusal consumed. History is never forwarded: detach requires a
// server-managed sub-agent, and server-managed init rejects seeded state.
func (a *Agents) launchDelegation(ctx context.Context, ref aix.AgentRef, st *agentsState, task string) (delegationResult, error) {
	invocationNum, _, agent, refusal := a.beginDelegation(ctx, ref, st)
	if refusal != nil {
		return *refusal, nil
	}

	// Pre-flight detach capability from the agent's own metadata: Abortable
	// is derived at definition time from exactly the store conditions the
	// runtime's detach check enforces, so a genkit-defined agent that cannot
	// detach is rejected here deterministically, without a wasted invocation
	// and without the hedged post-hoc wording below (which remains only for
	// metadata-less agents).
	if meta := agent.Metadata(); meta != nil && !meta.Abortable {
		a.releaseDelegation(st)
		logger.Warn(ctx, "background launch refused, agent cannot detach", "agent", ref.Name)
		return delegationResult{Response: fmt.Sprintf(
			"Error calling agent %q: this agent lacks a session store that supports background work, so it cannot run tasks in the background. Delegate to it without \"background\" instead.",
			ref.Name)}, nil
	}

	out, err := runSubAgent(ctx, agent, task, nil, true)
	if err != nil {
		logger.Warn(ctx, "background launch failed", "agent", ref.Name, "error", err)
		return delegationResult{Response: fmt.Sprintf("Error calling agent %q: %v", ref.Name, err)}, nil
	}

	switch out.FinishReason {
	case aix.AgentFinishReasonDetached:
		taskID := formatTaskID(ref.Name, out.SnapshotID)
		names := a.backgroundToolNames()
		logger.Debug(ctx, "background task started",
			"agent", ref.Name, "taskId", taskID, "sessionId", out.SessionID)
		return delegationResult{
			TaskID: taskID,
			Status: string(aix.SnapshotStatusPending),
			Response: fmt.Sprintf(
				"Background task %s started for agent %q. Collect the result with %s or %s, or stop it with %s.",
				taskID, ref.Name, names.check, names.wait, names.abort),
		}, nil
	case aix.AgentFinishReasonFailed:
		// FAILED_PRECONDITION is how the runtime rejects a detach-incapable
		// agent (no session store, or one without subscriber support). The
		// error was decoded from the wire, which keeps only the status name
		// (never the sentinel), and the status is the runtime's general
		// precondition category, so the hint is phrased conditionally rather
		// than asserting the cause. It applies only to metadata-less agents:
		// a genkit-defined agent that cannot detach was already rejected by
		// the pre-flight above, so its other FAILED_PRECONDITIONs (e.g. from
		// the agent fn itself) must not carry a misleading capability hint.
		msg := subAgentFailureMessage(out.FinishReason, out.Error, out.Message)
		errStatus := ""
		if out.Error != nil {
			errStatus = string(out.Error.Status)
		}
		logger.Warn(ctx, "background launch rejected",
			"agent", ref.Name, "status", errStatus, "error", msg)
		if agent.Metadata() == nil && out.Error != nil && out.Error.Status == status.FailedPrecondition {
			// Only this failure earns its slot back, and only for an agent
			// that published no metadata. The retry it points at is the
			// synchronous delegation, so the cap must not turn that retry
			// away. Every other failure is the sub-agent's own, and it ran to
			// produce it, so it counts against the cap or an agent that always
			// fails could be delegated to forever.
			//
			// The metadata check is what keeps the two apart. An agent that
			// publishes metadata was already refused by the pre-flight above
			// if it cannot detach, so a FAILED_PRECONDITION arriving here is
			// by definition not the detach rejection: it came from the agent's
			// own turn, and refunding it would leave the cap unable to bite.
			a.releaseDelegation(st)
			msg += " If this agent lacks a session store that supports background work, delegate to it without \"background\" instead."
		}
		return delegationResult{Response: fmt.Sprintf("Error calling agent %q: %s", ref.Name, msg)}, nil
	default:
		// The invocation settled before the detach landed; fold it like a
		// synchronous delegation.
		logger.Debug(ctx, "background launch settled synchronously",
			"agent", ref.Name, "finishReason", string(out.FinishReason))
		return a.foldDelegationOutput(ctx, ref, out, fmt.Sprintf("%s_%d", ref.Name, invocationNum)), nil
	}
}

// backgroundTaskTools builds the shared background-task tools added when
// [Agents.Async] is set, one per control the orchestrator has over a launched
// task, named per this configuration (see [Agents.backgroundToolNames]). st
// carries the per-call cache of terminal reports.
func (a *Agents) backgroundTaskTools(st *agentsState) []ai.Tool {
	names := a.backgroundToolNames()
	return []ai.Tool{
		aix.NewTool(names.check,
			"Returns the current status of background sub-agent tasks without waiting, including results for tasks that finished.",
			a.taskReportTool(st, readSnapshotOnce)),
		aix.NewTool(names.wait,
			"Waits until the given background sub-agent tasks finish and returns their results. Set timeoutSeconds to bound the wait; on timeout the current statuses are returned.",
			a.waitForBackgroundTasks(st)),
		aix.NewTool(names.abort,
			"Stops background sub-agent tasks whose results are no longer needed, and returns where that left each one. A task that had already finished is unaffected and reports its result.",
			a.taskReportTool(st, abortSnapshot)),
	}
}

// taskReportTool builds a non-blocking background-task tool: one companion
// dispatch per task, then a report of where that left it. The dispatch is the
// only difference between the two tools built this way. The check tool reads a
// task's row; the abort tool stops the task first and reports the row it left
// behind.
func (a *Agents) taskReportTool(st *agentsState, fetch snapshotFetch) func(context.Context, backgroundTasksInput) (backgroundTasksResult, error) {
	return func(ctx context.Context, in backgroundTasksInput) (backgroundTasksResult, error) {
		if len(in.TaskIDs) == 0 {
			return backgroundTasksResult{Note: noTaskIDsNote}, nil
		}
		return a.reportTasks(ctx, st, in.TaskIDs, fetch)
	}
}

// waitForBackgroundTasks is the blocking status tool: it follows every task to
// its end, or returns the current statuses when the optional timeout elapses.
// Each task is followed by the sub-agent's waitForSnapshot companion action, so
// the waiting happens next to the store that knows when the work finished
// rather than as a snapshot read per tick here: one action dispatch per task
// for the whole wait, which is one span each in a trace instead of a stream of
// them, and a settlement is observed as it happens rather than on the next
// tick. The waits run concurrently, so the slowest task sets the wall clock.
//
// A settled task's report is cached for the rest of the generate call (no
// snapshot re-reads or artifact re-merges when the model checks again), and
// transient store blips are ridden out inside the companion action itself, so
// a read error that reaches here is already a dead end. A timeout returns the
// current statuses rather than an error so the
// orchestrator can do other work and come back; cancellation of the calling
// context propagates as an error.
func (a *Agents) waitForBackgroundTasks(st *agentsState) func(context.Context, waitBackgroundTasksInput) (backgroundTasksResult, error) {
	return func(ctx context.Context, in waitBackgroundTasksInput) (backgroundTasksResult, error) {
		if len(in.TaskIDs) == 0 {
			return backgroundTasksResult{Note: noTaskIDsNote}, nil
		}
		// A negative timeout means "don't wait": report the current statuses.
		if in.TimeoutSeconds < 0 {
			return a.reportTasks(ctx, st, in.TaskIDs, readSnapshotOnce)
		}

		waitCtx := ctx
		if in.TimeoutSeconds > 0 {
			// A model-supplied value large enough to overflow the nanosecond
			// multiplication (about 292 years) is effectively unbounded, the
			// same as 0; without the clamp it wraps negative and the wait
			// would return instantly with an already-expired context.
			const maxWaitSeconds = math.MaxInt64 / int64(time.Second)
			if int64(in.TimeoutSeconds) <= maxWaitSeconds {
				var cancel context.CancelFunc
				waitCtx, cancel = context.WithTimeout(ctx, time.Duration(in.TimeoutSeconds)*time.Second)
				defer cancel()
			}
		}

		start := time.Now()
		logger.Debug(ctx, "waiting for background tasks",
			"tasks", len(in.TaskIDs), "timeoutSeconds", in.TimeoutSeconds)

		g := genkit.FromContext(ctx)
		reports := collectReports(in.TaskIDs, func(taskID string) backgroundTaskReport {
			return a.awaitTask(waitCtx, g, st, taskID, start)
		})

		// The calling context ending is cancellation, not a timeout; let it
		// fail the tool call rather than dressing it up as a settled result.
		if ctx.Err() != nil {
			return backgroundTasksResult{}, ctx.Err()
		}

		res := backgroundTasksResult{Tasks: reports}
		pending := 0
		for _, report := range reports {
			// Terminal() is the same rule the runtime applies, so a status
			// added there cannot leave this loop counting it as settled.
			// taskStatusUnknown reads terminal, which is right: it only
			// arrives once a read failure was classified unhelpable.
			if !aix.SnapshotStatus(report.Status).Terminal() {
				pending++
			}
		}
		if pending > 0 {
			logger.Debug(ctx, "wait for background tasks timed out",
				"pending", pending, "elapsedMs", time.Since(start).Milliseconds())
			res.TimedOut = true
			res.Note = "Stopped waiting; the pending tasks are still running. Check them again later."
			return res, nil
		}
		logger.Debug(ctx, "wait for background tasks finished",
			"tasks", len(in.TaskIDs), "elapsedMs", time.Since(start).Milliseconds())
		return res, nil
	}
}

// awaitTask follows one task to its end and returns its report. Transient
// store blips are ridden out inside the waitForSnapshot dispatch itself, so a
// fetch error that reaches this level is a dead end worth reporting. The one
// exception is ctx ending (the wait's own timeout, or the caller's
// cancellation): the task is still running by definition, so a follow cut
// short that way is reported as pending rather than as a read that did not
// finish. That is decided from the failure itself, not from ctx alone: a
// handle that never resolved, or an agent that is not registered, is not ctx's
// doing and keeps its error however the wait ended, or the model would be told
// to keep re-checking an ID that can never settle. start is the wait's start,
// for the settlement log line.
func (a *Agents) awaitTask(ctx context.Context, g *genkit.Genkit, st *agentsState, taskID string, start time.Time) backgroundTaskReport {
	report, err := a.reportTask(ctx, g, st, taskID, awaitSnapshot)
	if err != nil && ctx.Err() != nil && errors.Is(err, ctx.Err()) {
		report.Status = string(aix.SnapshotStatusPending)
		report.Error = ""
		return report
	}
	logger.Debug(ctx, "background task settled",
		"taskId", taskID, "status", report.Status,
		"elapsedMs", time.Since(start).Milliseconds())
	return report
}

// reportTasks builds one report per task ID with a single fetch each, for the
// check and abort tools and the wait tool's don't-wait path.
func (a *Agents) reportTasks(ctx context.Context, st *agentsState, taskIDs []string, fetch snapshotFetch) (backgroundTasksResult, error) {
	g := genkit.FromContext(ctx)
	res := backgroundTasksResult{Tasks: collectReports(taskIDs, func(taskID string) backgroundTaskReport {
		// None of these callers wait, so a failure is the report; the raw
		// error is only of interest to the wait tool, which classifies it.
		report, _ := a.reportTask(ctx, g, st, taskID, fetch)
		return report
	})}
	// A dead caller context fails every dispatch, and each failure reads back
	// as "could not read this task, check again later". Reported together that
	// is a settled-looking answer claiming the caller's live tasks are
	// unreadable, so let the cancellation be the result instead. The rule
	// lives here, with the fan-out, so a tool added later cannot forget it.
	if err := ctx.Err(); err != nil {
		return backgroundTasksResult{}, err
	}
	return res, nil
}

// collectReports builds one report per entry of taskIDs, fetching each
// distinct ID once and copying its report to every duplicate (the IDs are
// model-authored, so repeats happen). The fetches run concurrently: each
// dispatches a store-backed companion action, so the slowest distinct task
// sets the wall clock rather than the sum. Failures stay isolated per task
// (an unresolvable ID never fails the whole call), so one bad handle cannot
// hide the status of the others.
func collectReports(taskIDs []string, report func(taskID string) backgroundTaskReport) []backgroundTaskReport {
	// One slot per distinct ID, each written by its own goroutine and read
	// back after Wait, so duplicates cost one fetch and share its answer.
	fetched := make(map[string]*backgroundTaskReport, len(taskIDs))
	for _, id := range taskIDs {
		if _, ok := fetched[id]; !ok {
			fetched[id] = new(backgroundTaskReport)
		}
	}
	var wg sync.WaitGroup
	for id, slot := range fetched {
		wg.Go(func() { *slot = report(id) })
	}
	wg.Wait()

	reports := make([]backgroundTaskReport, len(taskIDs))
	for i, id := range taskIDs {
		reports[i] = *fetched[id]
	}
	return reports
}

// snapshotFetch is how a task's snapshot is obtained: read once for the check
// tool, waited for by the wait tool, aborted first by the abort tool. All three
// dispatch companion actions of the sub-agent, so all three apply the runtime's
// read shaping and all three keep the error chain live for classification.
// Ending on the row, rather than on each tool's own idea of an outcome, is what
// lets one report path serve every tool.
type snapshotFetch func(context.Context, *aix.AgentHandle, string) (*aix.SessionSnapshot[json.RawMessage], error)

var (
	readSnapshotOnce snapshotFetch = func(ctx context.Context, agent *aix.AgentHandle, snapshotID string) (*aix.SessionSnapshot[json.RawMessage], error) {
		return agent.GetSnapshot(ctx, snapshotID)
	}
	awaitSnapshot snapshotFetch = func(ctx context.Context, agent *aix.AgentHandle, snapshotID string) (*aix.SessionSnapshot[json.RawMessage], error) {
		return agent.WaitForSnapshot(ctx, snapshotID)
	}
	// abortSnapshot reads before it stops anything, because there are rows an
	// abort must not touch. Expiry is decided on read, not stored: a worker
	// that stopped heartbeating leaves a row that is still pending in the
	// store and reads as expired. Aborting that row overwrites the one signal
	// telling the model the work is gone and should be delegated again, and
	// the report caches as terminal, so nothing later can recover it.
	//
	// The read is not an extra cost. A task that already settled needs no
	// abort at all and is answered from the row alone, which is one dispatch
	// where aborting first then reading took two. Only a genuinely live task
	// pays for both, and it is the one the caller asked to stop.
	abortSnapshot snapshotFetch = func(ctx context.Context, agent *aix.AgentHandle, snapshotID string) (*aix.SessionSnapshot[json.RawMessage], error) {
		cur, err := agent.GetSnapshot(ctx, snapshotID)
		if err != nil {
			return nil, err
		}
		if cur.Status.Terminal() {
			// Nothing to stop. Expired, completed, failed and already-aborted
			// rows each report themselves, which is what the caller needs to
			// know about a task that outlived the request to cancel it.
			return cur, nil
		}
		st, err := agent.Abort(ctx, snapshotID)
		if err != nil {
			return nil, err
		}
		if st == aix.SnapshotStatusAborted {
			// The abort settled the row, and the row just read is the same one
			// with a new status. Restamping it beats rebuilding a partial
			// snapshot from the few fields this path happens to need, and it
			// keeps the abort's own outcome out of reach of a re-read that
			// could fail on its own.
			cur.Status = st
			return cur, nil
		}
		// The task settled between the read and the abort, so the abort was a
		// no-op on a terminal row. Re-read for the answer it now carries.
		return agent.GetSnapshot(ctx, snapshotID)
	}
)

// reportTask resolves one task handle, obtains its snapshot through fetch, and
// shapes the result into a report. Completed tasks surface the sub-agent's
// final response and artifacts; terminal non-success statuses surface an
// explanatory error instead.
//
// Reports for completed, failed, and aborted tasks are cached on st for the
// rest of the generate call: those rows never change, so a re-check skips the
// snapshot fetch and artifact re-merge (and cannot clobber a merged artifact
// the orchestrator has since edited). Pending, expired, and unresolvable
// reports can still change on their own and are never cached.
func (a *Agents) reportTask(ctx context.Context, g *genkit.Genkit, st *agentsState, taskID string, fetch snapshotFetch) (backgroundTaskReport, error) {
	st.mu.Lock()
	cached, ok := st.settledReports[taskID]
	st.mu.Unlock()
	if ok {
		return cached, nil
	}

	ref, snapshotID, err := a.resolveTaskID(taskID)
	if err != nil {
		logger.Debug(ctx, "background task id did not resolve", "taskId", taskID, "error", err)
		return backgroundTaskReport{TaskID: taskID, Status: taskStatusUnknown, Error: err.Error()}, err
	}

	report := backgroundTaskReport{TaskID: taskID, Agent: ref.Name}
	// Both fetches dispatch a companion action of the sub-agent, which applies
	// the runtime's read shaping: a pending row whose heartbeat went stale is
	// surfaced as expired.
	// Resolving the agent and reading its snapshot fail for unrelated reasons,
	// so they are reported separately. Chained, both arrive as NOT_FOUND and
	// an unregistered agent gets the missing-snapshot advice, which tells the
	// model to delegate again into a delegation tool that fails identically.
	agent, err := resolveAgent(g, ref)
	if err != nil {
		logger.Debug(ctx, "background task agent did not resolve",
			"taskId", taskID, "agent", ref.Name, "error", err)
		report.Status = taskStatusUnknown
		report.Error = fmt.Sprintf("%v. This task cannot be collected here; report it as unavailable rather than delegating it again.", err)
		return report, err
	}

	snap, err := fetch(ctx, agent, snapshotID)
	if err != nil {
		logger.Debug(ctx, "background task read failed",
			"taskId", taskID, "agent", ref.Name, "error", err)
		report.Status = taskStatusUnknown
		// The handle dispatches the companion action in-process, so the error
		// chain is live and status matching works, subtypes included
		// (aix.ErrSnapshotNotFound is an ErrNotFound,
		// aix.ErrSessionStoreNotConfigured an ErrFailedPrecondition). A wait
		// fetch has already ridden out transient store blips inside the
		// companion action, so whatever surfaces here is worth reporting. The
		// agent resolved above, so NOT_FOUND here is the snapshot and nothing
		// else, and re-delegating is genuinely the way to get the work done.
		switch {
		case errors.Is(err, status.ErrNotFound):
			report.Error = fmt.Sprintf("No record of this task exists (%v). Delegate the task again if the result is still needed.", err)
		case errors.Is(err, status.ErrFailedPrecondition), errors.Is(err, status.ErrInvalidArgument):
			report.Error = err.Error()
		default:
			report.Error = fmt.Sprintf("Could not read the task's status: %v. Check again later.", err)
		}
		return report, err
	}
	report.Status = string(snap.Status)

	switch snap.Status {
	case aix.SnapshotStatusPending:
		// Still running; nothing to report yet.
	case aix.SnapshotStatusCompleted:
		// Fold the settled snapshot exactly as a synchronous delegation folds
		// its output, so a delegation reports the same answer and the same
		// artifacts whether it ran in the background or not, with one caveat:
		// this reads the snapshot through the agent's own companion action, so
		// a sub-agent configured WithStateTransform has already shaped what is
		// read here, while the synchronous path sees the output unshaped. The response is
		// the persisted conversation's tip, which is the literal final message
		// the sub-agent returned (SessionRunner.Result), rather than older text
		// it spoke mid-tool-loop; an interrupt carries the same limitation as
		// the synchronous path, since it cannot be resumed from here.
		tip := snapshotTip(snap)
		var arts []*aix.Artifact
		if snap.State != nil {
			arts = snap.State.Artifacts
		}
		// Deterministic namespace (unlike the sync path's per-call counter):
		// AddArtifacts replaces by name, so a re-check after the orchestrator
		// restarts overwrites the same artifact names instead of duplicating
		// them.
		folded := a.foldDelegationOutput(ctx, ref, &aix.AgentOutput[json.RawMessage]{
			FinishReason: snap.FinishReason,
			Message:      tip,
			Artifacts:    arts,
		}, fmt.Sprintf("%s_%s", ref.Name, shortSnapshotID(snapshotID)))
		if snap.FinishReason.CarriesResult() {
			report.Response, report.Artifacts = folded.Response, folded.Artifacts
		} else {
			// The row committed, so the stored status is completed, but the
			// agent declared a reason that carries no answer (it can do so
			// without erroring, which is why the two disagree). Report the
			// outcome the reader has to act on, not the row's bookkeeping: a
			// model that sees "completed" moves on and never reads the error.
			// Which reason it was, and what the agent last said, is in Error.
			report.Status = string(aix.SnapshotStatusFailed)
			report.Error = folded.Response
		}
	case aix.SnapshotStatusFailed:
		report.Error = subAgentFailureMessage(snap.FinishReason, snap.Error, snapshotTip(snap))
	case aix.SnapshotStatusAborted:
		report.Error = "The task was aborted before it finished."
	case aix.SnapshotStatusExpired:
		report.Error = "The background worker stopped reporting progress and is presumed dead. Delegate the task again if the result is still needed."
	}

	// Expired is the one terminal read that can still change its mind: the
	// worker may be alive and merely slow to beat, so a later read can find it
	// settled properly. Everything else terminal is final and worth caching.
	if snap.Status.Terminal() && snap.Status != aix.SnapshotStatusExpired {
		st.mu.Lock()
		st.settledReports[taskID] = report
		st.mu.Unlock()
	}
	return report, nil
}

// snapshotTip returns the persisted conversation's last message, which is the
// literal final message the sub-agent returned (SessionRunner.Result) rather
// than older text it spoke mid-tool-loop. Nil when the row carries no state or
// no messages.
//
// It serves both the completed and failed arms. A failed detach-finalize
// writes the full final state, so the tip is there too, and for a row whose
// Error is empty it is the only account of what the agent managed to do.
func snapshotTip(snap *aix.SessionSnapshot[json.RawMessage]) *ai.Message {
	if snap.State == nil {
		return nil
	}
	if n := len(snap.State.Messages); n > 0 {
		return snap.State.Messages[n-1]
	}
	return nil
}

// formatTaskID builds the model-facing handle of a background delegation. The
// handle is self-contained ("<agent>:<snapshotId>") so it can be parsed back
// after the orchestrator is re-instantiated with nothing but its conversation
// history.
func formatTaskID(agentName, snapshotID string) string {
	return agentName + ":" + snapshotID
}

// resolveTaskID parses a task handle by matching it against the configured
// agents, taking the longest matching name so a configured name containing ':'
// cannot have its tasks claimed by a shorter configured prefix of it. The
// agent runtime mints the detach pending row's ID itself (a UUID, never
// containing ':'; see the detach handler in ai/exp), so the longest configured
// prefix is always the launching agent; anchoring the parse on the finite set
// of configured names, rather than on the ID's shape, is also what keeps it
// robust and confines the background-task tools to the agents this middleware
// was configured with.
func (a *Agents) resolveTaskID(taskID string) (aix.AgentRef, string, error) {
	var (
		best    aix.AgentRef
		bestLen int
	)
	for _, ref := range a.Agents {
		prefix := ref.Name + ":"
		if len(taskID) > len(prefix) && strings.HasPrefix(taskID, prefix) && len(prefix) > bestLen {
			best, bestLen = ref, len(prefix)
		}
	}
	if bestLen == 0 {
		return aix.AgentRef{}, "", status.Errorf(status.ErrInvalidArgument,
			"task ID %q does not match any configured agent (expected \"<agent>:<snapshotId>\")", taskID)
	}
	return best, taskID[bestLen:], nil
}

// shortSnapshotID trims a snapshot ID to a compact artifact-namespace
// component.
func shortSnapshotID(id string) string {
	if len(id) > 8 {
		return id[:8]
	}
	return id
}
