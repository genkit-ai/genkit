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
// companion action, and the wait tool its waitForSnapshot counterpart, which
// blocks next to the store rather than making this middleware re-read on a
// timer.

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
)

// transientReadRetries is the number of attempts the wait tool makes on one
// task before it gives up and surfaces the read error. It rides out isolated
// store blips while keeping the wait bounded on a persistently broken store. A
// wait cut short by its own deadline is not a failure and does not count.
const transientReadRetries = 3

// transientRetryDelay is how long the wait tool pauses before it tries a task
// again after a failed read, so a broken store is retried rather than spun on.
var transientRetryDelay = time.Second

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

// backgroundTasksInput is the input of the check tool.
type backgroundTasksInput struct {
	TaskIDs []string `json:"taskIds" jsonschema_description:"Task IDs returned by background delegations (form \"<agent>:<snapshotId>\")."`
}

// waitBackgroundTasksInput is the input of the wait tool.
type waitBackgroundTasksInput struct {
	TaskIDs        []string `json:"taskIds" jsonschema_description:"Task IDs returned by background delegations (form \"<agent>:<snapshotId>\")."`
	TimeoutSeconds int      `json:"timeoutSeconds,omitempty" jsonschema_description:"Maximum seconds to wait before returning the current statuses. 0 or omitted waits until every task settles; a negative value returns the current statuses immediately. Values too large to represent are treated as unbounded."`
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
	// not be resolved; see Error).
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
// Launches count against MaxDelegations like synchronous delegations, but a
// launch rejected before any work ran returns its slot, so a retry (e.g. the
// synchronous fallback hinted below) is not refused by a cap the rejection
// consumed. History is never forwarded: detach requires a server-managed
// sub-agent, and server-managed init rejects seeded state.
func (a *Agents) launchDelegation(ctx context.Context, ref aix.AgentRef, st *agentsState, task string) (delegationResult, error) {
	invocationNum, _, ok := a.reserveDelegation(st)
	if !ok {
		logger.Warn(ctx, "delegation refused, limit reached", "agent", ref.Name, "limit", a.MaxDelegations)
		return delegationLimitResult(a.MaxDelegations), nil
	}

	agent, err := resolveAgent(genkit.FromContext(ctx), ref)
	if err != nil {
		a.releaseDelegation(st)
		logger.Warn(ctx, "sub-agent resolution failed", "agent", ref.Name, "error", err)
		return delegationResult{Response: "Error: " + err.Error()}, nil
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
		a.releaseDelegation(st)
		logger.Warn(ctx, "background launch failed", "agent", ref.Name, "error", err)
		return delegationResult{Response: fmt.Sprintf("Error calling agent %q: %v", ref.Name, err)}, nil
	}

	switch out.FinishReason {
	case aix.AgentFinishReasonDetached:
		taskID := formatTaskID(ref.Name, out.SnapshotID)
		checkName, waitName := a.backgroundToolNames()
		logger.Debug(ctx, "background task started",
			"agent", ref.Name, "taskId", taskID, "sessionId", out.SessionID)
		return delegationResult{
			TaskID: taskID,
			Status: string(aix.SnapshotStatusPending),
			Response: fmt.Sprintf(
				"Background task %s started for agent %q. Collect the result with %s or %s.",
				taskID, ref.Name, checkName, waitName),
		}, nil
	case aix.AgentFinishReasonFailed:
		// A failed launch is a pre-detach rejection: with detach on the first
		// input, the invocation either detaches or fails before a turn runs,
		// so no sub-agent work happened and the reserved slot goes back.
		a.releaseDelegation(st)
		// FAILED_PRECONDITION is how the runtime rejects a detach-incapable
		// agent (no session store, or one without subscriber support). The
		// error was decoded from the wire, which keeps only the status name
		// (never the sentinel), and the status is the runtime's general
		// precondition category, so the hint is phrased conditionally rather
		// than asserting the cause. It applies only to metadata-less agents:
		// a genkit-defined agent that cannot detach was already rejected by
		// the pre-flight above, so its other FAILED_PRECONDITIONs (e.g. from
		// the agent fn itself) must not carry a misleading capability hint.
		msg := subAgentFailureMessage(out.Error)
		errStatus := ""
		if out.Error != nil {
			errStatus = string(out.Error.Status)
		}
		logger.Warn(ctx, "background launch rejected",
			"agent", ref.Name, "status", errStatus, "error", msg)
		if out.Error != nil && out.Error.Status == status.FailedPrecondition && agent.Metadata() == nil {
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

// backgroundTaskTools builds the shared status tools added when [Agents.Async]
// is set, named per this configuration (see [Agents.backgroundToolNames]).
// st carries the per-call cache of terminal reports.
func (a *Agents) backgroundTaskTools(st *agentsState) []ai.Tool {
	checkName, waitName := a.backgroundToolNames()
	return []ai.Tool{
		aix.NewTool(checkName,
			"Returns the current status of background sub-agent tasks without waiting, including results for tasks that finished.",
			a.checkBackgroundTasks(st)),
		aix.NewTool(waitName,
			"Waits until the given background sub-agent tasks finish and returns their results. Set timeoutSeconds to bound the wait; on timeout the current statuses are returned.",
			a.waitForBackgroundTasks(st)),
	}
}

// checkBackgroundTasks is the non-blocking status tool: one snapshot read per
// task, no waiting.
func (a *Agents) checkBackgroundTasks(st *agentsState) func(context.Context, backgroundTasksInput) (backgroundTasksResult, error) {
	return func(ctx context.Context, in backgroundTasksInput) (backgroundTasksResult, error) {
		if len(in.TaskIDs) == 0 {
			return backgroundTasksResult{Note: noTaskIDsNote}, nil
		}
		return a.reportTasks(ctx, st, in.TaskIDs), nil
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
// snapshot re-reads or artifact re-merges when the model checks again), and a
// failed read is retried up to transientReadRetries times before the error is
// surfaced. A timeout returns the current statuses rather than an error so the
// orchestrator can do other work and come back; cancellation of the calling
// context propagates as an error.
func (a *Agents) waitForBackgroundTasks(st *agentsState) func(context.Context, waitBackgroundTasksInput) (backgroundTasksResult, error) {
	return func(ctx context.Context, in waitBackgroundTasksInput) (backgroundTasksResult, error) {
		if len(in.TaskIDs) == 0 {
			return backgroundTasksResult{Note: noTaskIDsNote}, nil
		}
		// A negative timeout means "don't wait": report the current statuses.
		if in.TimeoutSeconds < 0 {
			return a.reportTasks(ctx, st, in.TaskIDs), nil
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
		reports := make([]backgroundTaskReport, len(in.TaskIDs))
		var wg sync.WaitGroup
		for i, id := range in.TaskIDs {
			wg.Add(1)
			go func() {
				defer wg.Done()
				reports[i] = a.awaitTask(waitCtx, g, st, id, start)
			}()
		}
		wg.Wait()

		// The calling context ending is cancellation, not a timeout; let it
		// fail the tool call rather than dressing it up as a settled result.
		if ctx.Err() != nil {
			return backgroundTasksResult{}, ctx.Err()
		}

		res := backgroundTasksResult{Tasks: reports}
		pending := 0
		for _, report := range reports {
			// taskStatusUnknown deliberately counts as terminal here: it only
			// arrives after a read failure was classified as a dead end, so
			// there is nothing left to wait for.
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

// awaitTask follows one task to its end and returns its report, retrying a
// failed read a few times before it surfaces the error. When ctx ends first
// (the wait's own timeout, or the caller's cancellation) the task is still
// running by definition, so it is reported as pending rather than as a read
// that did not finish; start is the wait's start, for the settlement log line.
func (a *Agents) awaitTask(ctx context.Context, g *genkit.Genkit, st *agentsState, taskID string, start time.Time) backgroundTaskReport {
	for attempt := 1; ; attempt++ {
		report, transient := a.reportTask(ctx, g, st, taskID, awaitSnapshot)
		if ctx.Err() != nil {
			report.Status = string(aix.SnapshotStatusPending)
			report.Error = ""
			return report
		}
		if !transient {
			logger.Debug(ctx, "background task settled",
				"taskId", taskID, "status", report.Status,
				"elapsedMs", time.Since(start).Milliseconds())
			return report
		}
		if attempt >= transientReadRetries {
			logger.Warn(ctx, "background task read retries exhausted",
				"taskId", taskID, "attempts", attempt)
			return report
		}
		select {
		case <-ctx.Done():
			report.Status = string(aix.SnapshotStatusPending)
			report.Error = ""
			return report
		case <-time.After(transientRetryDelay):
		}
	}
}

// reportTasks builds one report per task ID. Failures are isolated per task
// (an unresolvable ID never fails the whole call), so one bad handle cannot
// hide the status of the others.
func (a *Agents) reportTasks(ctx context.Context, st *agentsState, taskIDs []string) backgroundTasksResult {
	g := genkit.FromContext(ctx)
	res := backgroundTasksResult{Tasks: make([]backgroundTaskReport, 0, len(taskIDs))}
	for _, id := range taskIDs {
		report, _ := a.reportTask(ctx, g, st, id, readSnapshotOnce)
		res.Tasks = append(res.Tasks, report)
	}
	return res
}

// snapshotFetch is how a task's snapshot is obtained: read once for the check
// tool, waited for by the wait tool. Both dispatch a companion action of the
// sub-agent, so both apply the runtime's read shaping and both keep the error
// chain live for classification.
type snapshotFetch func(context.Context, *aix.AgentHandle, string) (*aix.SessionSnapshot[json.RawMessage], error)

var (
	readSnapshotOnce snapshotFetch = func(ctx context.Context, agent *aix.AgentHandle, snapshotID string) (*aix.SessionSnapshot[json.RawMessage], error) {
		return agent.GetSnapshot(ctx, snapshotID)
	}
	awaitSnapshot snapshotFetch = func(ctx context.Context, agent *aix.AgentHandle, snapshotID string) (*aix.SessionSnapshot[json.RawMessage], error) {
		return agent.WaitForSnapshot(ctx, snapshotID)
	}
)

// reportTask resolves one task handle, obtains its snapshot through fetch, and
// shapes the result into a report. Completed tasks surface the sub-agent's
// final response and artifacts; terminal non-success statuses surface an
// explanatory error instead. The second return reports whether a failed fetch
// looked transient (a store or transport error rather than a
// sentinel-classified dead end); the wait tool retries transient fetches a few
// times before surfacing them.
//
// Reports for completed, failed, and aborted tasks are cached on st for the
// rest of the generate call: those rows never change, so a re-check skips the
// snapshot fetch and artifact re-merge (and cannot clobber a merged artifact
// the orchestrator has since edited). Pending, expired, and unresolvable
// reports can still change on their own and are never cached.
func (a *Agents) reportTask(ctx context.Context, g *genkit.Genkit, st *agentsState, taskID string, fetch snapshotFetch) (backgroundTaskReport, bool) {
	st.mu.Lock()
	cached, ok := st.settledReports[taskID]
	st.mu.Unlock()
	if ok {
		return cached, false
	}

	ref, snapshotID, err := a.resolveTaskID(taskID)
	if err != nil {
		logger.Debug(ctx, "background task id did not resolve", "taskId", taskID, "error", err)
		return backgroundTaskReport{TaskID: taskID, Status: taskStatusUnknown, Error: err.Error()}, false
	}

	report := backgroundTaskReport{TaskID: taskID, Agent: ref.Name}
	// Both fetches dispatch a companion action of the sub-agent, which applies
	// the runtime's read shaping: a pending row whose heartbeat went stale is
	// surfaced as expired.
	agent, err := resolveAgent(g, ref)
	var snap *aix.SessionSnapshot[json.RawMessage]
	if err == nil {
		snap, err = fetch(ctx, agent, snapshotID)
	}
	if err != nil {
		logger.Debug(ctx, "background task read failed",
			"taskId", taskID, "agent", ref.Name, "error", err)
		report.Status = taskStatusUnknown
		// The handle dispatches the companion action in-process, so the error
		// chain is live and status matching works, subtypes included
		// (aix.ErrSnapshotNotFound is an ErrNotFound,
		// aix.ErrSessionStoreNotConfigured an ErrFailedPrecondition).
		// Classified dead ends stop retries; anything else is presumed
		// transient. NOT_FOUND covers a missing snapshot and an unregistered
		// agent alike; the wrapped cause names which.
		switch {
		case errors.Is(err, status.ErrNotFound):
			report.Error = fmt.Sprintf("No record of this task exists (%v). Delegate the task again if the result is still needed.", err)
		case errors.Is(err, status.ErrFailedPrecondition), errors.Is(err, status.ErrInvalidArgument):
			report.Error = err.Error()
		default:
			report.Error = fmt.Sprintf("Could not read the task's status: %v. Check again later.", err)
			return report, true
		}
		return report, false
	}
	report.Status = string(snap.Status)

	switch snap.Status {
	case aix.SnapshotStatusPending:
		// Still running; nothing to report yet.
	case aix.SnapshotStatusCompleted:
		if snap.FinishReason == aix.AgentFinishReasonInterrupted {
			// Same limitation as synchronous delegation: the interrupt cannot
			// be resumed from here.
			report.Response = interruptedResponse(ref.Name)
			break
		}
		// Mirror the synchronous path, whose response is the literal final
		// message (SessionRunner.Result): the persisted conversation's tip is
		// that same message, so a delegation reports the same answer whether
		// it ran in the background or not, rather than walking back to older
		// text the model spoke mid-tool-loop.
		var tip *ai.Message
		var arts []*aix.Artifact
		if snap.State != nil {
			if n := len(snap.State.Messages); n > 0 {
				tip = snap.State.Messages[n-1]
			}
			arts = snap.State.Artifacts
		}
		report.Response = messageText(tip)
		if report.Response == "" {
			report.Response = "(no response)"
		}
		if sub := namedArtifacts(arts); len(sub) > 0 {
			// Deterministic namespace (unlike the sync path's per-call
			// counter): AddArtifacts replaces by name, so a re-check after the
			// orchestrator restarts overwrites the same artifact names instead
			// of duplicating them.
			invocationID := fmt.Sprintf("%s_%s", ref.Name, shortSnapshotID(snapshotID))
			mergeArtifacts(ctx, ref.Name, invocationID, sub)
			report.Artifacts = delegatedArtifacts(invocationID, sub, a.strategy())
		}
	case aix.SnapshotStatusFailed:
		report.Error = subAgentFailureMessage(snap.Error)
	case aix.SnapshotStatusAborted:
		report.Error = "The task was aborted before it finished."
	case aix.SnapshotStatusExpired:
		report.Error = "The background worker stopped reporting progress and is presumed dead. Delegate the task again if the result is still needed."
	}

	switch snap.Status {
	case aix.SnapshotStatusCompleted, aix.SnapshotStatusFailed, aix.SnapshotStatusAborted:
		st.mu.Lock()
		st.settledReports[taskID] = report
		st.mu.Unlock()
	}
	return report, false
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
