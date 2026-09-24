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
	"errors"
	"fmt"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/ai/exp/localstore"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/genkit"
	genkitx "github.com/firebase/genkit/go/genkit/exp"
)

// lenientDelegation decodes a delegation tool output without failing the test,
// for use inside model functions, where t.Fatalf would fire on the wrong
// goroutine.
func lenientDelegation(v any) delegationResult {
	var dr delegationResult
	if b, err := json.Marshal(v); err == nil {
		_ = json.Unmarshal(b, &dr)
	}
	return dr
}

// TestAgentsAsyncDelegationLifecycle drives the full background flow in one
// generate call: launch with background=true, observe pending via the check
// tool while the sub-agent is still gated, then release the gate and collect
// the completed result (response and inline artifact) via the wait tool.
func TestAgentsAsyncDelegationLifecycle(t *testing.T) {
	g := newTestGenkit(t)

	// Gate the sub-agent so the task stays pending until the orchestrator has
	// observed the pending status.
	gate := make(chan struct{})
	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(gate) }) }
	t.Cleanup(release)

	genkitx.DefineCustomAgent[any](g, "researcher",
		func(ctx context.Context, resp aix.Responder, sess *aix.SessionRunner[any]) (*aix.AgentResult, error) {
			var last *ai.Message
			err := sess.Run(ctx, func(ctx context.Context, input *aix.AgentInput) (*aix.TurnResult, error) {
				select {
				case <-gate:
				case <-ctx.Done():
					return nil, ctx.Err()
				}
				resp.SendArtifact(&aix.Artifact{
					Name:  "findings.md",
					Parts: []*ai.Part{ai.NewTextPart("the findings body")},
				})
				last = ai.NewModelTextMessage("research complete")
				sess.AddMessages(last)
				return &aix.TurnResult{FinishReason: aix.AgentFinishReasonStop}, nil
			})
			if err != nil {
				return nil, err
			}
			return &aix.AgentResult{Message: last, Artifacts: sess.Artifacts()}, nil
		},
		aix.WithSessionStore[any](localstore.NewInMemorySessionStore[any]()),
	)

	// Scripted orchestrator: launch in background, check, release the gate,
	// wait, then finish. Each step keys off the tool responses accumulated in
	// the request so far.
	var capturedSystem string
	orch := toolModel(t, g, "test/orch-async", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		if sys := findSystem(req.Messages); sys != nil {
			capturedSystem = systemText(sys)
		}
		launches := toolOutputs(req.Messages, "delegate_to_researcher")
		checks := toolOutputs(req.Messages, checkBackgroundTasksToolName)
		waits := toolOutputs(req.Messages, waitBackgroundTasksToolName)
		switch {
		case len(launches) == 0:
			return toolReqResp(req, &ai.ToolRequest{
				Name:  "delegate_to_researcher",
				Input: map[string]any{"task": "dig into X", "background": true},
			}), nil
		case len(checks) == 0:
			return toolReqResp(req, &ai.ToolRequest{
				Name:  checkBackgroundTasksToolName,
				Input: map[string]any{"taskIds": []string{lenientDelegation(launches[0]).TaskID}},
			}), nil
		case len(waits) == 0:
			release()
			return toolReqResp(req, &ai.ToolRequest{
				Name:  waitBackgroundTasksToolName,
				Input: map[string]any{"taskIds": []string{lenientDelegation(launches[0]).TaskID}},
			}), nil
		default:
			return textResp(req, "done"), nil
		}
	})

	mw := &Agents{Agents: []aix.AgentRef{{Name: "researcher"}}, Async: true}
	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("research X"), ai.WithUse(mw))
	if err != nil {
		t.Fatal(err)
	}
	history := resp.History()

	for _, want := range []string{"background", checkBackgroundTasksToolName, waitBackgroundTasksToolName} {
		if !strings.Contains(capturedSystem, want) {
			t.Errorf("async system prompt missing %q; got:\n%s", want, capturedSystem)
		}
	}

	launches := delegationResponses(t, history, "delegate_to_researcher")
	if len(launches) != 1 {
		t.Fatalf("expected 1 launch response, got %d", len(launches))
	}
	launch := launches[0]
	if launch.Status != "pending" || !strings.HasPrefix(launch.TaskID, "researcher:") {
		t.Fatalf("unexpected launch result: %+v", launch)
	}

	checkOuts := toolOutputs(history, checkBackgroundTasksToolName)
	if len(checkOuts) != 1 {
		t.Fatalf("expected 1 check response, got %d", len(checkOuts))
	}
	check := decodeToolOutput[backgroundTasksResult](t, checkOuts[0])
	if len(check.Tasks) != 1 || check.Tasks[0].Status != "pending" {
		t.Errorf("check while gated: want 1 pending task, got %+v", check.Tasks)
	}

	waitOuts := toolOutputs(history, waitBackgroundTasksToolName)
	if len(waitOuts) != 1 {
		t.Fatalf("expected 1 wait response, got %d", len(waitOuts))
	}
	wait := decodeToolOutput[backgroundTasksResult](t, waitOuts[0])
	if len(wait.Tasks) != 1 {
		t.Fatalf("expected 1 waited task, got %+v", wait.Tasks)
	}
	task := wait.Tasks[0]
	if task.Status != "completed" || task.Response != "research complete" || task.Agent != "researcher" {
		t.Errorf("unexpected completed report: %+v", task)
	}
	snapshotID := strings.TrimPrefix(launch.TaskID, "researcher:")
	wantArtifact := "researcher_" + shortSnapshotID(snapshotID) + "/findings.md"
	if len(task.Artifacts) != 1 || task.Artifacts[0].Name != wantArtifact ||
		!strings.Contains(task.Artifacts[0].Content, "the findings body") {
		t.Errorf("unexpected artifacts (want %q with inline content): %+v", wantArtifact, task.Artifacts)
	}
}

// TestAgentsBackgroundTasksPickUpAcrossInstantiations launches a background
// task in one generate call and collects it in a second call through a fresh
// middleware instance, using only the task ID recorded in the first call's
// history: the pickup path a re-instantiated orchestrator relies on. The wait
// also carries two unusable handles to verify per-task error isolation.
func TestAgentsBackgroundTasksPickUpAcrossInstantiations(t *testing.T) {
	g := newTestGenkit(t)

	genkitx.DefineAgent[any](g, "researcher",
		aix.InlinePrompt{ai.WithModel(toolModel(t, g, "test/researcher-bg", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			return textResp(req, "background answer"), nil
		}))},
		aix.WithSessionStore[any](localstore.NewInMemorySessionStore[any]()),
	)

	// First call: launch in the background and stop without waiting.
	launcher := toolModel(t, g, "test/orch-launch", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		if hasToolResponse(req.Messages) {
			return textResp(req, "launched"), nil
		}
		return toolReqResp(req, &ai.ToolRequest{
			Name:  "delegate_to_researcher",
			Input: map[string]any{"task": "long dig", "background": true},
		}), nil
	})
	resp1, err := genkit.Generate(ctx, g, ai.WithModel(launcher), ai.WithPrompt("go"),
		ai.WithUse(&Agents{Agents: []aix.AgentRef{{Name: "researcher"}}, Async: true}))
	if err != nil {
		t.Fatal(err)
	}
	launches := delegationResponses(t, resp1.History(), "delegate_to_researcher")
	if len(launches) != 1 || launches[0].TaskID == "" {
		t.Fatalf("expected a launch with a task ID, got %+v", launches)
	}
	taskID := launches[0].TaskID

	// Second call, fresh middleware instance: wait on the recorded task ID
	// plus a missing snapshot and an unconfigured agent.
	badSnapshot := "researcher:no-such-snapshot"
	badAgent := "ghost:whatever"
	waiter := toolModel(t, g, "test/orch-wait", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		if hasToolResponse(req.Messages) {
			return textResp(req, "collected"), nil
		}
		return toolReqResp(req, &ai.ToolRequest{
			Name:  waitBackgroundTasksToolName,
			Input: map[string]any{"taskIds": []string{taskID, badSnapshot, badAgent}},
		}), nil
	})
	resp2, err := genkit.Generate(ctx, g, ai.WithModel(waiter), ai.WithPrompt("collect"),
		ai.WithUse(&Agents{Agents: []aix.AgentRef{{Name: "researcher"}}, Async: true}))
	if err != nil {
		t.Fatal(err)
	}

	waitOuts := toolOutputs(resp2.History(), waitBackgroundTasksToolName)
	if len(waitOuts) != 1 {
		t.Fatalf("expected 1 wait response, got %d", len(waitOuts))
	}
	res := decodeToolOutput[backgroundTasksResult](t, waitOuts[0])
	if len(res.Tasks) != 3 {
		t.Fatalf("expected 3 task reports, got %+v", res.Tasks)
	}
	if got := res.Tasks[0]; got.Status != "completed" || got.Response != "background answer" {
		t.Errorf("picked-up task: want completed with the sub-agent's answer, got %+v", got)
	}
	// "Delegate the task again" is the not-found sentinel branch: it proves
	// errors.Is matched aix.ErrSnapshotNotFound through the companion action,
	// rather than the generic read-failure branch relaying the same text.
	if got := res.Tasks[1]; got.Status != taskStatusUnknown ||
		!strings.Contains(got.Error, "not found") || !strings.Contains(got.Error, "Delegate the task again") {
		t.Errorf("missing snapshot: want unknown with the not-found guidance, got %+v", got)
	}
	if got := res.Tasks[2]; got.Status != taskStatusUnknown || !strings.Contains(got.Error, "does not match any configured agent") {
		t.Errorf("unconfigured agent: want unknown with a no-match error, got %+v", got)
	}
	if res.TimedOut {
		t.Errorf("wait should settle without timing out, got %+v", res)
	}
}

// TestAgentsBackgroundCompletedWithoutAnswerReportsFailed covers the case where
// the stored row and the agent's own verdict disagree: the turn committed, so
// the snapshot is "completed", but the agent declared a finish reason that
// carries no answer. The report has to say what the reader must act on, since a
// model that sees "completed" moves on and never reads the error.
func TestAgentsBackgroundCompletedWithoutAnswerReportsFailed(t *testing.T) {
	g := newTestGenkit(t)

	// Gate the turn so the detach lands before the agent settles; the fn then
	// declares failure without returning an error, so the runtime commits the
	// row as completed with a finish reason of failed.
	gate := make(chan struct{})
	var releaseOnce sync.Once
	release := func() { releaseOnce.Do(func() { close(gate) }) }
	t.Cleanup(release)

	genkitx.DefineCustomAgent[any](g, "researcher",
		func(ctx context.Context, resp aix.Responder, sess *aix.SessionRunner[any]) (*aix.AgentResult, error) {
			err := sess.Run(ctx, func(ctx context.Context, input *aix.AgentInput) (*aix.TurnResult, error) {
				select {
				case <-gate:
				case <-ctx.Done():
					return nil, ctx.Err()
				}
				sess.AddMessages(ai.NewModelTextMessage("partial notes"))
				return &aix.TurnResult{FinishReason: aix.AgentFinishReasonFailed}, nil
			})
			if err != nil {
				return nil, err
			}
			return &aix.AgentResult{}, nil
		},
		aix.WithSessionStore[any](localstore.NewInMemorySessionStore[any]()),
	)

	orch := toolModel(t, g, "test/orch-no-answer", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		launches := toolOutputs(req.Messages, "delegate_to_researcher")
		waits := toolOutputs(req.Messages, waitBackgroundTasksToolName)
		switch {
		case len(launches) == 0:
			return toolReqResp(req, &ai.ToolRequest{
				Name:  "delegate_to_researcher",
				Input: map[string]any{"task": "dig into X", "background": true},
			}), nil
		case len(waits) == 0:
			release()
			return toolReqResp(req, &ai.ToolRequest{
				Name:  waitBackgroundTasksToolName,
				Input: map[string]any{"taskIds": []string{lenientDelegation(launches[0]).TaskID}},
			}), nil
		default:
			return textResp(req, "done"), nil
		}
	})

	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("research X"),
		ai.WithUse(&Agents{Agents: []aix.AgentRef{{Name: "researcher"}}, Async: true}))
	if err != nil {
		t.Fatal(err)
	}

	waitOuts := toolOutputs(resp.History(), waitBackgroundTasksToolName)
	if len(waitOuts) != 1 {
		t.Fatalf("expected 1 wait response, got %d", len(waitOuts))
	}
	res := decodeToolOutput[backgroundTasksResult](t, waitOuts[0])
	if len(res.Tasks) != 1 {
		t.Fatalf("expected 1 task report, got %+v", res.Tasks)
	}
	got := res.Tasks[0]
	if got.Status != string(aix.SnapshotStatusFailed) {
		t.Errorf("Status = %q, want %q: the task produced no answer", got.Status, aix.SnapshotStatusFailed)
	}
	if got.Error == "" {
		t.Error("Error is empty, want the reason the task carries no answer")
	}
	if got.Response != "" {
		t.Errorf("Response = %q, want empty: there is no answer to report", got.Response)
	}
}

// TestAgentsWaitTimeoutKeepsUnresolvableErrors covers the two halves of what a
// timed-out wait must report. A task that is genuinely still running comes back
// pending, because the wait ran out of time rather than learning anything. A
// handle that can never resolve keeps its error, because nothing about the
// deadline makes it more likely to settle later; reporting it as pending would
// send the orchestrator back to re-check it forever.
func TestAgentsWaitTimeoutKeepsUnresolvableErrors(t *testing.T) {
	g := newTestGenkit(t)

	// The sub-agent never finishes, so its task is pending for the whole wait.
	gate := make(chan struct{})
	t.Cleanup(func() { close(gate) })
	genkitx.DefineCustomAgent[any](g, "researcher",
		func(ctx context.Context, resp aix.Responder, sess *aix.SessionRunner[any]) (*aix.AgentResult, error) {
			err := sess.Run(ctx, func(ctx context.Context, input *aix.AgentInput) (*aix.TurnResult, error) {
				select {
				case <-gate:
				case <-ctx.Done():
				}
				return nil, ctx.Err()
			})
			if err != nil {
				return nil, err
			}
			return &aix.AgentResult{}, nil
		},
		aix.WithSessionStore[any](localstore.NewInMemorySessionStore[any]()),
	)

	orch := toolModel(t, g, "test/orch-wait-timeout", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		launches := toolOutputs(req.Messages, "delegate_to_researcher")
		waits := toolOutputs(req.Messages, waitBackgroundTasksToolName)
		switch {
		case len(launches) == 0:
			return toolReqResp(req, &ai.ToolRequest{
				Name:  "delegate_to_researcher",
				Input: map[string]any{"task": "dig into X", "background": true},
			}), nil
		case len(waits) == 0:
			return toolReqResp(req, &ai.ToolRequest{
				Name: waitBackgroundTasksToolName,
				Input: map[string]any{
					"taskIds":        []string{lenientDelegation(launches[0]).TaskID, "ghost:whatever"},
					"timeoutSeconds": float64(1),
				},
			}), nil
		default:
			return textResp(req, "done"), nil
		}
	})

	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("research X"),
		ai.WithUse(&Agents{Agents: []aix.AgentRef{{Name: "researcher"}}, Async: true}))
	if err != nil {
		t.Fatal(err)
	}

	waitOuts := toolOutputs(resp.History(), waitBackgroundTasksToolName)
	if len(waitOuts) != 1 {
		t.Fatalf("expected 1 wait response, got %d", len(waitOuts))
	}
	res := decodeToolOutput[backgroundTasksResult](t, waitOuts[0])
	if len(res.Tasks) != 2 {
		t.Fatalf("expected 2 task reports, got %+v", res.Tasks)
	}
	if !res.TimedOut {
		t.Errorf("TimedOut = false, want true with a task still running: %+v", res)
	}
	if got := res.Tasks[0]; got.Status != string(aix.SnapshotStatusPending) || got.Error != "" {
		t.Errorf("running task: want pending with no error, got %+v", got)
	}
	if got := res.Tasks[1]; got.Status != taskStatusUnknown ||
		!strings.Contains(got.Error, "does not match any configured agent") {
		t.Errorf("unconfigured agent: want unknown with its error kept past the deadline, got %+v", got)
	}
}

// TestAwaitTaskKeepsUnresolvableErrorPastCancellation isolates the half of the
// timed-out wait that a full generate call cannot stage reliably: a report that
// failed for a reason ctx had nothing to do with, produced while ctx is already
// over. Deciding "still pending" from ctx alone would blank the error here and
// send the orchestrator back to re-check a handle that can never settle.
func TestAwaitTaskKeepsUnresolvableErrorPastCancellation(t *testing.T) {
	a := &Agents{Agents: []aix.AgentRef{{Name: "researcher"}}}
	st := &agentsState{settledReports: map[string]backgroundTaskReport{}}

	over, cancel := context.WithCancel(context.Background())
	cancel()

	// The handle names no configured agent, so it fails before any I/O and
	// stays unresolvable however long anyone waits.
	got := a.awaitTask(over, nil, st, "ghost:whatever", time.Now())
	if got.Status != taskStatusUnknown {
		t.Errorf("Status = %q, want %q", got.Status, taskStatusUnknown)
	}
	if !strings.Contains(got.Error, "does not match any configured agent") {
		t.Errorf("Error = %q, want the resolution failure kept", got.Error)
	}
}

// TestResolveTaskIDLongestPrefix pins the longest-prefix rule: a configured
// name containing ':' keeps its tasks even when another configured name is a
// prefix of it, regardless of configuration order.
func TestResolveTaskIDLongestPrefix(t *testing.T) {
	a := &Agents{Agents: []aix.AgentRef{{Name: "a"}, {Name: "a:b"}}}
	ref, snap, err := a.resolveTaskID("a:b:1234")
	if err != nil || ref.Name != "a:b" || snap != "1234" {
		t.Fatalf("resolveTaskID(a:b:1234) = %q, %q, %v; want a:b, 1234", ref.Name, snap, err)
	}
	ref, snap, err = a.resolveTaskID("a:5678")
	if err != nil || ref.Name != "a" || snap != "5678" {
		t.Fatalf("resolveTaskID(a:5678) = %q, %q, %v; want a, 5678", ref.Name, snap, err)
	}
	if _, _, err := a.resolveTaskID("ghost:1"); err == nil {
		t.Error("expected an error for an unconfigured agent")
	}
	if _, _, err := a.resolveTaskID("a:"); err == nil {
		t.Error("expected an error for an empty snapshot ID")
	}
}

// TestAgentsBackgroundLaunchRejectedWithoutStore verifies that launching a
// background delegation on a sub-agent that cannot detach (no session store)
// reports the runtime's rejection plus the synchronous-fallback hint. The hint
// hangs off the FAILED_PRECONDITION status of the wire-decoded error: the
// sentinel itself does not survive JSON, so the status name is the contract.
func TestAgentsBackgroundLaunchRejectedWithoutStore(t *testing.T) {
	g := newTestGenkit(t)

	genkitx.DefineAgent[any](g, "researcher",
		aix.InlinePrompt{ai.WithModel(toolModel(t, g, "test/researcher-nostore", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			return textResp(req, "should never be needed"), nil
		}))},
	)

	orch := toolModel(t, g, "test/orch-nostore", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		if hasToolResponse(req.Messages) {
			return textResp(req, "done"), nil
		}
		return toolReqResp(req, &ai.ToolRequest{
			Name:  "delegate_to_researcher",
			Input: map[string]any{"task": "dig", "background": true},
		}), nil
	})

	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("go"),
		ai.WithUse(&Agents{Agents: []aix.AgentRef{{Name: "researcher"}}, Async: true}))
	if err != nil {
		t.Fatal(err)
	}
	got := delegationResponses(t, resp.History(), "delegate_to_researcher")
	if len(got) != 1 {
		t.Fatalf("expected 1 delegation response, got %d", len(got))
	}
	if got[0].TaskID != "" {
		t.Errorf("rejected launch must not hand out a task ID, got %+v", got[0])
	}
	for _, want := range []string{"Error calling agent", "session store that supports background work", "without \"background\""} {
		if !strings.Contains(got[0].Response, want) {
			t.Errorf("rejection response missing %q; got %q", want, got[0].Response)
		}
	}
}

// TestAgentsAsyncInstancesCoexistWithPrefixes pins that two Async middleware
// instances with distinct explicit prefixes can share one generate call: the
// background-task tools are namespaced per instance, so the request is not
// rejected as carrying duplicate tools.
func TestAgentsAsyncInstancesCoexistWithPrefixes(t *testing.T) {
	g := newTestGenkit(t)

	genkitx.DefineAgent[any](g, "researcher",
		aix.InlinePrompt{ai.WithModel(toolModel(t, g, "test/researcher-coexist", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			return textResp(req, "unused"), nil
		}))},
		aix.WithSessionStore[any](localstore.NewInMemorySessionStore[any]()),
	)

	model := toolModel(t, g, "test/orch-coexist", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		return textResp(req, "done"), nil
	})

	research, code := "research", "code"
	resp, err := genkit.Generate(ctx, g, ai.WithModel(model), ai.WithPrompt("go"),
		ai.WithUse(
			&Agents{Agents: []aix.AgentRef{{Name: "researcher"}}, ToolPrefix: &research, Async: true},
			&Agents{Agents: []aix.AgentRef{{Name: "researcher"}}, ToolPrefix: &code, Async: true},
		))
	if err != nil {
		t.Fatalf("two prefixed Async instances should coexist, got %v", err)
	}
	if resp.Text() != "done" {
		t.Fatalf("unexpected response: %q", resp.Text())
	}
}

// TestAgentsWaitTimeoutOverflowIsUnbounded pins the timeoutSeconds overflow
// clamp: a value too large for the nanosecond multiplication is treated as
// unbounded rather than wrapping negative into an already-expired context, so
// the wait still settles its tasks normally instead of returning instantly
// with timedOut and unresolved statuses.
func TestAgentsWaitTimeoutOverflowIsUnbounded(t *testing.T) {
	g := newTestGenkit(t)

	genkitx.DefineAgent[any](g, "researcher",
		aix.InlinePrompt{ai.WithModel(toolModel(t, g, "test/researcher-overflow", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			return textResp(req, "unused"), nil
		}))},
		aix.WithSessionStore[any](localstore.NewInMemorySessionStore[any]()),
	)

	// A missing snapshot on a configured agent settles on the first pass
	// (NOT_FOUND is a dead end), so the wait returns without waiting out the
	// absurd timeout; before the clamp, the dead context instead failed every
	// read and the result came back timedOut with a read error.
	waiter := toolModel(t, g, "test/orch-overflow", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		if hasToolResponse(req.Messages) {
			return textResp(req, "collected"), nil
		}
		return toolReqResp(req, &ai.ToolRequest{
			Name: waitBackgroundTasksToolName,
			Input: map[string]any{
				"taskIds":        []string{"researcher:no-such-snapshot"},
				"timeoutSeconds": float64(10000000000),
			},
		}), nil
	})
	resp, err := genkit.Generate(ctx, g, ai.WithModel(waiter), ai.WithPrompt("collect"),
		ai.WithUse(&Agents{Agents: []aix.AgentRef{{Name: "researcher"}}, Async: true}))
	if err != nil {
		t.Fatal(err)
	}

	waitOuts := toolOutputs(resp.History(), waitBackgroundTasksToolName)
	if len(waitOuts) != 1 {
		t.Fatalf("expected 1 wait response, got %d", len(waitOuts))
	}
	res := decodeToolOutput[backgroundTasksResult](t, waitOuts[0])
	if res.TimedOut {
		t.Errorf("overflowed timeout must behave as unbounded, got timedOut result: %+v", res)
	}
	if len(res.Tasks) != 1 || res.Tasks[0].Status != taskStatusUnknown ||
		!strings.Contains(res.Tasks[0].Error, "not found") {
		t.Errorf("expected the missing snapshot to settle as unknown/not-found, got %+v", res.Tasks)
	}
}

// TestAgentsWaitForFirstSettled pins the wait tool's race join: with
// waitFor "first" the tool returns as soon as any listed task settles, the
// still-running tasks report as pending, and the return is not a timeout.
func TestAgentsWaitForFirstSettled(t *testing.T) {
	g := newTestGenkit(t)

	genkitx.DefineAgent[any](g, "quick",
		aix.InlinePrompt{ai.WithModel(toolModel(t, g, "test/quick", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			return textResp(req, "quick answer"), nil
		}))},
		aix.WithSessionStore[any](localstore.NewInMemorySessionStore[any]()),
	)
	// The slow sub-agent finishes only when released, so the race can only be
	// won by the quick one.
	release := make(chan struct{})
	t.Cleanup(func() { close(release) })
	genkitx.DefineCustomAgent[any](g, "slow",
		func(ctx context.Context, resp aix.Responder, sess *aix.SessionRunner[any]) (*aix.AgentResult, error) {
			err := sess.Run(ctx, func(ctx context.Context, input *aix.AgentInput) (*aix.TurnResult, error) {
				select {
				case <-release:
				case <-ctx.Done():
				}
				return nil, errors.New("released")
			})
			if err != nil {
				return nil, err
			}
			return &aix.AgentResult{}, nil
		},
		aix.WithSessionStore[any](localstore.NewInMemorySessionStore[any]()),
	)

	orch := toolModel(t, g, "test/orch-race", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		slowLaunches := toolOutputs(req.Messages, "delegate_to_slow")
		quickLaunches := toolOutputs(req.Messages, "delegate_to_quick")
		switch {
		case len(slowLaunches) == 0:
			return toolReqResp(req, &ai.ToolRequest{
				Name:  "delegate_to_slow",
				Input: map[string]any{"task": "dig forever", "background": true},
			}), nil
		case len(quickLaunches) == 0:
			return toolReqResp(req, &ai.ToolRequest{
				Name:  "delegate_to_quick",
				Input: map[string]any{"task": "answer fast", "background": true},
			}), nil
		case len(toolOutputs(req.Messages, waitBackgroundTasksToolName)) == 0:
			// The slow task first in the list, so a settled result in slot 1
			// proves the join raced instead of following input order.
			return toolReqResp(req, &ai.ToolRequest{
				Name: waitBackgroundTasksToolName,
				Input: map[string]any{
					"taskIds": []string{
						lenientDelegation(slowLaunches[0]).TaskID,
						lenientDelegation(quickLaunches[0]).TaskID,
					},
					"waitFor": "first",
				},
			}), nil
		default:
			return textResp(req, "done"), nil
		}
	})

	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("go"),
		ai.WithUse(&Agents{Agents: []aix.AgentRef{{Name: "quick"}, {Name: "slow"}}, Async: true}))
	if err != nil {
		t.Fatal(err)
	}
	waits := toolOutputs(resp.History(), waitBackgroundTasksToolName)
	if len(waits) != 1 {
		t.Fatalf("expected 1 wait result, got %d", len(waits))
	}
	res := decodeToolOutput[backgroundTasksResult](t, waits[0])
	if res.TimedOut {
		t.Errorf("a won race must not report TimedOut: %+v", res)
	}
	if len(res.Tasks) != 2 {
		t.Fatalf("expected 2 reports, got %+v", res.Tasks)
	}
	if got := res.Tasks[0]; got.Status != string(aix.SnapshotStatusPending) {
		t.Errorf("slow task report = %+v, want pending", got)
	}
	if got := res.Tasks[1]; got.Status != string(aix.SnapshotStatusCompleted) || got.Response != "quick answer" {
		t.Errorf("quick task report = %+v, want the settled answer", got)
	}
	if !strings.Contains(res.Note, "first settled") {
		t.Errorf("expected the race note, got %q", res.Note)
	}
}

// TestAgentsWaitForUnknownJoinRefused pins the recoverable guidance for a
// waitFor value the tool does not know.
func TestAgentsWaitForUnknownJoinRefused(t *testing.T) {
	g := newTestGenkit(t)
	genkitx.DefineAgent[any](g, "quick",
		aix.InlinePrompt{ai.WithModel(toolModel(t, g, "test/quick2", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			return textResp(req, "unused"), nil
		}))},
		aix.WithSessionStore[any](localstore.NewInMemorySessionStore[any]()),
	)
	orch := toolModel(t, g, "test/orch-badjoin", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		if len(toolOutputs(req.Messages, waitBackgroundTasksToolName)) > 0 {
			return textResp(req, "done"), nil
		}
		return toolReqResp(req, &ai.ToolRequest{
			Name:  waitBackgroundTasksToolName,
			Input: map[string]any{"taskIds": []string{"quick:whatever"}, "waitFor": "any"},
		}), nil
	})
	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("go"),
		ai.WithUse(&Agents{Agents: []aix.AgentRef{{Name: "quick"}}, Async: true}))
	if err != nil {
		t.Fatal(err)
	}
	res := decodeToolOutput[backgroundTasksResult](t, toolOutputs(resp.History(), waitBackgroundTasksToolName)[0])
	if !strings.Contains(res.Note, `"first"`) || len(res.Tasks) != 0 {
		t.Fatalf("expected join guidance and no reports, got %+v", res)
	}
}

// TestAgentsAbortBackgroundTask drives the abort control end to end. The task
// is stopped mid-flight, so the abort has to reach the work and not just the
// row. The abort itself never waits: it answers "did the stop land?", either
// with the settled row when the worker's finalize won the race or with
// "aborting" while the task winds down, and the settled, resumable aborted
// row is collected through the wait tool, which is the flow the tools teach.
func TestAgentsAbortBackgroundTask(t *testing.T) {
	g := newTestGenkit(t)

	// The sub-agent never finishes on its own, so the abort is the only thing
	// that can end this task. It signals its own cancellation, which is how
	// the test tells a stopped task from a merely rewritten row.
	stopped := make(chan struct{})
	var stopOnce sync.Once
	genkitx.DefineCustomAgent[any](g, "researcher",
		func(ctx context.Context, resp aix.Responder, sess *aix.SessionRunner[any]) (*aix.AgentResult, error) {
			err := sess.Run(ctx, func(ctx context.Context, input *aix.AgentInput) (*aix.TurnResult, error) {
				<-ctx.Done()
				stopOnce.Do(func() { close(stopped) })
				return nil, ctx.Err()
			})
			if err != nil {
				return nil, err
			}
			return &aix.AgentResult{}, nil
		},
		aix.WithSessionStore[any](localstore.NewInMemorySessionStore[any]()),
	)

	// Scripted orchestrator: launch in background, abort, then collect the
	// settled row through the wait tool.
	var capturedSystem string
	orch := toolModel(t, g, "test/orch-abort", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		if sys := findSystem(req.Messages); sys != nil {
			capturedSystem = systemText(sys)
		}
		launches := toolOutputs(req.Messages, "delegate_to_researcher")
		aborts := toolOutputs(req.Messages, abortBackgroundTasksToolName)
		waits := toolOutputs(req.Messages, waitBackgroundTasksToolName)
		switch {
		case len(launches) == 0:
			return toolReqResp(req, &ai.ToolRequest{
				Name:  "delegate_to_researcher",
				Input: map[string]any{"task": "dig into X", "background": true},
			}), nil
		case len(aborts) == 0:
			return toolReqResp(req, &ai.ToolRequest{
				Name:  abortBackgroundTasksToolName,
				Input: map[string]any{"taskIds": []string{lenientDelegation(launches[0]).TaskID}},
			}), nil
		case len(waits) == 0:
			return toolReqResp(req, &ai.ToolRequest{
				Name:  waitBackgroundTasksToolName,
				Input: map[string]any{"taskIds": []string{lenientDelegation(launches[0]).TaskID}},
			}), nil
		default:
			return textResp(req, "done"), nil
		}
	})

	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("research X"),
		ai.WithUse(&Agents{Agents: []aix.AgentRef{{Name: "researcher"}}, Async: true}))
	if err != nil {
		t.Fatal(err)
	}
	history := resp.History()

	if !strings.Contains(capturedSystem, abortBackgroundTasksToolName) {
		t.Errorf("async system prompt missing %q; got:\n%s", abortBackgroundTasksToolName, capturedSystem)
	}

	abortOuts := toolOutputs(history, abortBackgroundTasksToolName)
	if len(abortOuts) != 1 {
		t.Fatalf("expected 1 abort response, got %d", len(abortOuts))
	}
	aborted := decodeToolOutput[backgroundTasksResult](t, abortOuts[0])
	if len(aborted.Tasks) != 1 {
		t.Fatalf("expected 1 aborted task, got %+v", aborted.Tasks)
	}
	// The flip's own return decides the report, so stopping a live task
	// answers "aborting" deterministically: no re-read races the worker's
	// finalize, and the raw mid-flip window is never exposed.
	if got := aborted.Tasks[0]; got.Status != string(aix.SnapshotStatusAborting) || got.Error == "" {
		t.Errorf("unexpected abort report: %+v", got)
	}

	// The row said aborted; the runtime observes that flip and cancels the
	// work, which is the half a status write alone would not prove.
	select {
	case <-stopped:
	case <-time.After(5 * time.Second):
		t.Error("the sub-agent was never cancelled: the abort reached the row but not the work")
	}

	waitOuts := toolOutputs(history, waitBackgroundTasksToolName)
	if len(waitOuts) != 1 {
		t.Fatalf("expected 1 wait response, got %d", len(waitOuts))
	}
	waited := decodeToolOutput[backgroundTasksResult](t, waitOuts[0])
	if len(waited.Tasks) != 1 || waited.Tasks[0].Status != string(aix.SnapshotStatusAborted) {
		t.Fatalf("wait after abort: want 1 settled aborted task, got %+v", waited.Tasks)
	}
	if got := waited.Tasks[0].Error; !strings.Contains(got, continueTaskToolName) {
		t.Errorf("settled aborted report should carry the resume hint, got %q", got)
	}
}

// TestAgentsBackgroundLaunchEchoesLabel pins the label plumbing: a
// caller-chosen name rides the launch result and background-task reports next
// to the taskId, purely as a reading aid.
func TestAgentsBackgroundLaunchEchoesLabel(t *testing.T) {
	g := newTestGenkit(t)
	genkitx.DefineAgent[any](g, "quick",
		aix.InlinePrompt{ai.WithModel(toolModel(t, g, "test/quick-label", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			return textResp(req, "quick answer"), nil
		}))},
		aix.WithSessionStore[any](localstore.NewInMemorySessionStore[any]()),
	)
	orch := toolModel(t, g, "test/orch-label", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		launches := toolOutputs(req.Messages, "delegate_to_quick")
		switch {
		case len(launches) == 0:
			return toolReqResp(req, &ai.ToolRequest{
				Name:  "delegate_to_quick",
				Input: map[string]any{"task": "answer fast", "background": true, "name": "fast-lane"},
			}), nil
		case len(toolOutputs(req.Messages, waitBackgroundTasksToolName)) == 0:
			return toolReqResp(req, &ai.ToolRequest{
				Name:  waitBackgroundTasksToolName,
				Input: map[string]any{"taskIds": []string{lenientDelegation(launches[0]).TaskID}},
			}), nil
		default:
			return textResp(req, "done"), nil
		}
	})
	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("go"),
		ai.WithUse(&Agents{Agents: []aix.AgentRef{{Name: "quick"}}, Async: true}))
	if err != nil {
		t.Fatal(err)
	}
	launch := lenientDelegation(toolOutputs(resp.History(), "delegate_to_quick")[0])
	if launch.Name != "fast-lane" {
		t.Errorf("launch result Name = %q, want %q", launch.Name, "fast-lane")
	}
	waited := decodeToolOutput[backgroundTasksResult](t, toolOutputs(resp.History(), waitBackgroundTasksToolName)[0])
	if len(waited.Tasks) != 1 || waited.Tasks[0].Name != "fast-lane" {
		t.Errorf("report did not echo the label: %+v", waited.Tasks)
	}
}

// TestAgentsAbortReportsAbortingWhileWindingDown pins the abort report's
// honesty without any waiting inside the abort: aborted is a promise the row
// is settled and resumable, so a worker that has not finalized reports the
// "aborting" (the stop was delivered, the row is winding down),
// and the settled aborted row arrives through the wait tool once the worker
// lets go. That the wait settles at all also proves "aborting" was never
// cached: a cached report would be returned without following the row.
func TestAgentsAbortReportsAbortingWhileWindingDown(t *testing.T) {
	g := newTestGenkit(t)

	// The sub-agent deliberately ignores its cancellation until released, so
	// the abort's single re-read deterministically finds the row unsettled.
	release := make(chan struct{})
	var releaseOnce sync.Once
	unblock := func() { releaseOnce.Do(func() { close(release) }) }
	t.Cleanup(unblock)
	genkitx.DefineCustomAgent[any](g, "stubborn",
		func(ctx context.Context, resp aix.Responder, sess *aix.SessionRunner[any]) (*aix.AgentResult, error) {
			err := sess.Run(ctx, func(ctx context.Context, input *aix.AgentInput) (*aix.TurnResult, error) {
				<-release
				return nil, errors.New("released")
			})
			if err != nil {
				return nil, err
			}
			return &aix.AgentResult{}, nil
		},
		aix.WithSessionStore[any](localstore.NewInMemorySessionStore[any]()),
	)

	orch := toolModel(t, g, "test/orch-stubborn", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		launches := toolOutputs(req.Messages, "delegate_to_stubborn")
		switch {
		case len(launches) == 0:
			return toolReqResp(req, &ai.ToolRequest{
				Name:  "delegate_to_stubborn",
				Input: map[string]any{"task": "dig in", "background": true},
			}), nil
		case len(toolOutputs(req.Messages, abortBackgroundTasksToolName)) == 0:
			return toolReqResp(req, &ai.ToolRequest{
				Name:  abortBackgroundTasksToolName,
				Input: map[string]any{"taskIds": []string{lenientDelegation(launches[0]).TaskID}},
			}), nil
		case len(toolOutputs(req.Messages, waitBackgroundTasksToolName)) == 0:
			// The abort has reported; let the worker wind down and follow the
			// row to its settled state.
			unblock()
			return toolReqResp(req, &ai.ToolRequest{
				Name:  waitBackgroundTasksToolName,
				Input: map[string]any{"taskIds": []string{lenientDelegation(launches[0]).TaskID}},
			}), nil
		default:
			return textResp(req, "done"), nil
		}
	})

	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("go"),
		ai.WithUse(&Agents{Agents: []aix.AgentRef{{Name: "stubborn"}}, Async: true}))
	if err != nil {
		t.Fatal(err)
	}
	abortOuts := toolOutputs(resp.History(), abortBackgroundTasksToolName)
	if len(abortOuts) != 1 {
		t.Fatalf("expected 1 abort response, got %d", len(abortOuts))
	}
	aborted := decodeToolOutput[backgroundTasksResult](t, abortOuts[0])
	if len(aborted.Tasks) != 1 || aborted.Tasks[0].Status != string(aix.SnapshotStatusAborting) {
		t.Errorf("abort of an unfinalized task: want an %q report, got %+v", string(aix.SnapshotStatusAborting), aborted.Tasks)
	} else if got := aborted.Tasks[0].Error; !strings.Contains(got, waitBackgroundTasksToolName) {
		t.Errorf("the aborting report should point at the wait tool, got %q", got)
	}
	waited := decodeToolOutput[backgroundTasksResult](t, toolOutputs(resp.History(), waitBackgroundTasksToolName)[0])
	if waited.TimedOut || len(waited.Tasks) != 1 || waited.Tasks[0].Status != string(aix.SnapshotStatusAborted) {
		t.Errorf("wait after abort: want 1 settled aborted task, got %+v", waited)
	}
}

// TestAgentsAbortAfterCompletionReportsTheResult pins what an abort owes a
// caller when there was nothing left to stop. The abort action returns a bare
// status, so reporting that alone would answer "completed" and strand the
// answer the sub-agent had already produced; the abort reads the row it left
// behind instead and folds it like any other report.
func TestAgentsAbortAfterCompletionReportsTheResult(t *testing.T) {
	g := newTestGenkit(t)
	genkitx.DefineCustomAgent[any](g, "researcher",
		func(ctx context.Context, resp aix.Responder, sess *aix.SessionRunner[any]) (*aix.AgentResult, error) {
			var last *ai.Message
			err := sess.Run(ctx, func(ctx context.Context, input *aix.AgentInput) (*aix.TurnResult, error) {
				resp.SendArtifact(&aix.Artifact{
					Name:  "findings.md",
					Parts: []*ai.Part{ai.NewTextPart("the findings body")},
				})
				last = ai.NewModelTextMessage("research complete")
				sess.AddMessages(last)
				return &aix.TurnResult{FinishReason: aix.AgentFinishReasonStop}, nil
			})
			if err != nil {
				return nil, err
			}
			return &aix.AgentResult{Message: last, Artifacts: sess.Artifacts()}, nil
		},
		aix.WithSessionStore[any](localstore.NewInMemorySessionStore[any]()),
	)

	// Launch and settle the task outside the middleware. Collecting it through
	// the tools first would cache its report, and the abort would then answer
	// from that cache without ever dispatching the call under test.
	h := genkitx.LookupAgent(g, "researcher")
	task, err := h.RunDetached(ctx, &aix.AgentInput{Message: ai.NewUserTextMessage("dig into X")})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := h.WaitForSnapshot(ctx, task.SnapshotID()); err != nil {
		t.Fatal(err)
	}

	a := &Agents{Agents: []aix.AgentRef{{Name: "researcher"}}, Async: true}
	st := &agentsState{settledReports: map[string]backgroundTaskReport{}}
	got, err := a.reportTask(ctx, g, st, formatTaskID("researcher", task.SnapshotID()), a.abortSnapshot())
	if err != nil {
		t.Fatal(err)
	}
	if got.Status != string(aix.SnapshotStatusCompleted) {
		t.Errorf("Status = %q, want %q: the abort had nothing to stop", got.Status, aix.SnapshotStatusCompleted)
	}
	if got.Response != "research complete" {
		t.Errorf("Response = %q, want the answer the task had already produced", got.Response)
	}
	wantArtifact := "researcher_" + shortSnapshotID(task.SnapshotID()) + "/findings.md"
	if len(got.Artifacts) != 1 || got.Artifacts[0].Name != wantArtifact ||
		!strings.Contains(got.Artifacts[0].Content, "the findings body") {
		t.Errorf("unexpected artifacts (want %q with inline content): %+v", wantArtifact, got.Artifacts)
	}
}

// TestFoldDelegationNonAnswerReasons pins what an orchestrator is told when a
// turn ends without an answer. Every reason that carries no result has to read
// as a failure and has to say what actually happened: reporting the agent's
// last words as the response hands partial work over as if it were final, and
// reporting a bare placeholder discards the only account of the outcome a
// completed-but-failed row ever carries.
func TestFoldDelegationNonAnswerReasons(t *testing.T) {
	a := &Agents{Agents: []aix.AgentRef{{Name: "researcher"}}}
	ref := aix.AgentRef{Name: "researcher"}
	tip := ai.NewModelTextMessage("partial notes: found 3 of 5 sources")

	for _, reason := range []aix.AgentFinishReason{
		aix.AgentFinishReasonFailed,
		aix.AgentFinishReasonBlocked,
		aix.AgentFinishReasonLength,
		aix.AgentFinishReasonAborted,
	} {
		t.Run(string(reason), func(t *testing.T) {
			got := a.foldDelegationOutput(t.Context(), ref,
				&aix.AgentOutput[json.RawMessage]{FinishReason: reason, Message: tip}, 0)
			if !strings.Contains(got.Response, "Error calling agent") {
				t.Errorf("Response = %q, want it reported as a failure", got.Response)
			}
			if !strings.Contains(got.Response, string(reason)) {
				t.Errorf("Response = %q, want it to name the finish reason %q", got.Response, reason)
			}
			// The agent's last words explain the outcome; losing them leaves
			// the model with nothing it can act on.
			if !strings.Contains(got.Response, "found 3 of 5 sources") {
				t.Errorf("Response = %q, want the agent's last message kept", got.Response)
			}
		})
	}

	// A structured failure still wins: it is the better explanation.
	got := a.foldDelegationOutput(t.Context(), ref, &aix.AgentOutput[json.RawMessage]{
		FinishReason: aix.AgentFinishReasonFailed,
		Message:      tip,
		Error:        &status.Error{Status: status.Internal, Message: "upstream model refused"},
	}, 0)
	if !strings.Contains(got.Response, "upstream model refused") {
		t.Errorf("Response = %q, want the structured failure preferred", got.Response)
	}

	// A reason that does carry a result is untouched.
	got = a.foldDelegationOutput(t.Context(), ref,
		&aix.AgentOutput[json.RawMessage]{FinishReason: aix.AgentFinishReasonStop, Message: tip}, 0)
	if got.Response != "partial notes: found 3 of 5 sources" {
		t.Errorf("Response = %q, want the message reported as the answer", got.Response)
	}
}

func TestFoldDelegationNoFinalMessage(t *testing.T) {
	a := &Agents{Agents: []aix.AgentRef{{Name: "researcher"}}}
	ref := aix.AgentRef{Name: "researcher"}
	toolOnly := &ai.Message{Role: ai.RoleModel, Content: []*ai.Part{
		ai.NewToolRequestPart(&ai.ToolRequest{Name: "search", Input: map[string]any{"q": "x"}}),
	}}
	arts := func(n int) []*aix.Artifact {
		var out []*aix.Artifact
		for i := range n {
			out = append(out, &aix.Artifact{Name: fmt.Sprintf("a%d.md", i), Parts: []*ai.Part{ai.NewTextPart("body")}})
		}
		return out
	}

	// A silent success must read as a success, and say where the result is.
	got := a.foldDelegationOutput(t.Context(), ref,
		&aix.AgentOutput[json.RawMessage]{FinishReason: aix.AgentFinishReasonStop}, 0)
	if !strings.Contains(got.Response, "completed") || !strings.Contains(got.Response, "no final message") || !strings.Contains(got.Response, "no artifacts") {
		t.Errorf("Response = %q, want a completed-without-message notice naming the missing artifacts", got.Response)
	}
	got = a.foldDelegationOutput(t.Context(), ref,
		&aix.AgentOutput[json.RawMessage]{FinishReason: aix.AgentFinishReasonStop, Message: toolOnly, Artifacts: arts(2)}, 0)
	if !strings.Contains(got.Response, "no final message") || !strings.Contains(got.Response, "2 artifacts") {
		t.Errorf("Response = %q, want the notice to point at the 2 artifacts", got.Response)
	}
	if len(got.Artifacts) != 2 {
		t.Errorf("Artifacts = %d, want 2 surfaced alongside the notice", len(got.Artifacts))
	}
	got = a.foldDelegationOutput(t.Context(), ref,
		&aix.AgentOutput[json.RawMessage]{FinishReason: aix.AgentFinishReasonStop, Artifacts: arts(1)}, 0)
	if !strings.Contains(got.Response, "one artifact") {
		t.Errorf("Response = %q, want the notice to point at the one artifact", got.Response)
	}
}

func TestLastModelMessage(t *testing.T) {
	said := ai.NewModelTextMessage("working on it")
	toolMsg := &ai.Message{Role: ai.RoleTool, Content: []*ai.Part{
		ai.NewToolResponsePart(&ai.ToolResponse{Name: "search", Output: "raw results"}),
	}}
	snap := func(msgs ...*ai.Message) *aix.SessionSnapshot[json.RawMessage] {
		return &aix.SessionSnapshot[json.RawMessage]{State: &aix.SessionState[json.RawMessage]{Messages: msgs}}
	}

	// The transcript's tip is a tool response; the answer is what the model
	// said before it.
	if got := lastModelMessage(snap(ai.NewUserTextMessage("go"), said, toolMsg)); got != said {
		t.Errorf("lastModelMessage = %+v, want the model message before the tool response", got)
	}
	if got := lastModelMessage(snap(ai.NewUserTextMessage("go"), toolMsg)); got != nil {
		t.Errorf("lastModelMessage = %+v, want nil when no model message exists", got)
	}
	if got := lastModelMessage(&aix.SessionSnapshot[json.RawMessage]{}); got != nil {
		t.Errorf("lastModelMessage = %+v, want nil for a stateless row", got)
	}
}

// TestAgentsBackgroundReportUsesLastModelMessage pins the report's answer to
// what the sub-agent said, through a run whose transcript ends on something
// else. The first case ends on a tool response after the model spoke, so the
// report carries the model's words, not the tool's; the second ends on a model
// message holding only a tool request, so there is nothing the model said and
// the report says so, pointing at the artifact that holds the result.
func TestAgentsBackgroundReportUsesLastModelMessage(t *testing.T) {
	toolMsg := &ai.Message{Role: ai.RoleTool, Content: []*ai.Part{
		ai.NewToolResponsePart(&ai.ToolResponse{Name: "search", Output: "raw results"}),
	}}
	toolOnly := &ai.Message{Role: ai.RoleModel, Content: []*ai.Part{
		ai.NewToolRequestPart(&ai.ToolRequest{Name: "search", Input: map[string]any{"q": "x"}}),
	}}
	cases := []struct {
		name         string
		turn         func(resp aix.Responder, sess *aix.SessionRunner[any])
		wantResponse string
	}{
		{
			name: "tool response ends the transcript",
			turn: func(_ aix.Responder, sess *aix.SessionRunner[any]) {
				sess.AddMessages(ai.NewModelTextMessage("working on it"), toolMsg)
			},
			wantResponse: "working on it",
		},
		{
			name: "model ends on a bare tool request",
			turn: func(resp aix.Responder, sess *aix.SessionRunner[any]) {
				resp.SendArtifact(&aix.Artifact{Name: "report.md", Parts: []*ai.Part{ai.NewTextPart("the report body")}})
				sess.AddMessages(toolOnly)
			},
			wantResponse: noFinalMessageResponse(1),
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g := newTestGenkit(t)
			gate := make(chan struct{})
			var releaseOnce sync.Once
			release := func() { releaseOnce.Do(func() { close(gate) }) }
			t.Cleanup(release)

			genkitx.DefineCustomAgent[any](g, "researcher",
				func(ctx context.Context, resp aix.Responder, sess *aix.SessionRunner[any]) (*aix.AgentResult, error) {
					err := sess.Run(ctx, func(ctx context.Context, input *aix.AgentInput) (*aix.TurnResult, error) {
						select {
						case <-gate:
						case <-ctx.Done():
							return nil, ctx.Err()
						}
						tc.turn(resp, sess)
						return &aix.TurnResult{FinishReason: aix.AgentFinishReasonStop}, nil
					})
					if err != nil {
						return nil, err
					}
					return &aix.AgentResult{Artifacts: sess.Artifacts()}, nil
				},
				aix.WithSessionStore[any](localstore.NewInMemorySessionStore[any]()),
			)

			orch := toolModel(t, g, "test/orch-last-model-"+tc.name, func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
				launches := toolOutputs(req.Messages, "delegate_to_researcher")
				waits := toolOutputs(req.Messages, waitBackgroundTasksToolName)
				switch {
				case len(launches) == 0:
					return toolReqResp(req, &ai.ToolRequest{
						Name:  "delegate_to_researcher",
						Input: map[string]any{"task": "dig into X", "background": true},
					}), nil
				case len(waits) == 0:
					release()
					return toolReqResp(req, &ai.ToolRequest{
						Name:  waitBackgroundTasksToolName,
						Input: map[string]any{"taskIds": []string{lenientDelegation(launches[0]).TaskID}},
					}), nil
				default:
					return textResp(req, "done"), nil
				}
			})

			resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("research X"),
				ai.WithUse(&Agents{Agents: []aix.AgentRef{{Name: "researcher"}}, Async: true}))
			if err != nil {
				t.Fatal(err)
			}
			waitOuts := toolOutputs(resp.History(), waitBackgroundTasksToolName)
			if len(waitOuts) != 1 {
				t.Fatalf("expected 1 wait response, got %d", len(waitOuts))
			}
			res := decodeToolOutput[backgroundTasksResult](t, waitOuts[0])
			if len(res.Tasks) != 1 {
				t.Fatalf("expected 1 task report, got %+v", res.Tasks)
			}
			got := res.Tasks[0]
			if got.Status != string(aix.SnapshotStatusCompleted) {
				t.Errorf("Status = %q, want %q", got.Status, aix.SnapshotStatusCompleted)
			}
			if got.Response != tc.wantResponse {
				t.Errorf("Response = %q, want %q", got.Response, tc.wantResponse)
			}
			if got.Error != "" {
				t.Errorf("Error = %q, want empty: the task succeeded", got.Error)
			}
		})
	}
}

// TestBackgroundTaskToolsAcceptNoArguments covers the call a model makes by
// mistake. taskIds must be omissible: a required field fails decoding, and a
// tool-input decode failure is not a turn the model can correct, it fails the
// whole generate call.
func TestBackgroundTaskToolsAcceptNoArguments(t *testing.T) {
	g := newTestGenkit(t)
	genkitx.DefineCustomAgent[any](g, "researcher",
		func(ctx context.Context, resp aix.Responder, sess *aix.SessionRunner[any]) (*aix.AgentResult, error) {
			return &aix.AgentResult{}, nil
		},
		aix.WithSessionStore[any](localstore.NewInMemorySessionStore[any]()),
	)

	mw := &Agents{Agents: []aix.AgentRef{{Name: "researcher"}}, Async: true}
	hooks, err := mw.New(ctx)
	if err != nil {
		t.Fatal(err)
	}
	names := mw.backgroundToolNames().all()
	byName := map[string]ai.Tool{}
	for _, tool := range hooks.Tools {
		byName[tool.Name()] = tool
	}
	for _, name := range names {
		t.Run(name, func(t *testing.T) {
			tool, ok := byName[name]
			if !ok {
				t.Fatalf("tool %q not registered", name)
			}
			// The schema must not require taskIds. Asserting on that field
			// rather than on an empty required list keeps this passing if a
			// genuinely required field is added later.
			if req, _ := tool.Definition().InputSchema["required"].([]any); slices.Contains(req, any("taskIds")) {
				t.Errorf("input schema requires %v; an omitted taskIds must decode", req)
			}
			out, err := tool.RunRaw(ctx, map[string]any{})
			if err != nil {
				t.Fatalf("calling %q with no arguments returned an error, which fails the whole generate: %v", name, err)
			}
			res := decodeToolOutput[backgroundTasksResult](t, out)
			if res.Note == "" {
				t.Errorf("Note is empty; want the guidance that tells the model what to pass")
			}
		})
	}
}

func TestAgentsSyncTaskCheckDoesNotDuplicateArtifacts(t *testing.T) {
	// A synchronous delegation to a server-managed sub-agent merges its
	// artifacts under the run's snapshot-based namespace, the same
	// deterministic one the background-task report path folds under, so
	// checking the sync result's taskId re-merges over the identical names
	// instead of duplicating the artifacts in the parent session.
	g := newTestGenkit(t)

	genkitx.DefineCustomAgent[any](g, "writer",
		func(ctx context.Context, resp aix.Responder, sess *aix.SessionRunner[any]) (*aix.AgentResult, error) {
			err := sess.Run(ctx, func(ctx context.Context, input *aix.AgentInput) (*aix.TurnResult, error) {
				resp.SendArtifact(&aix.Artifact{
					Name:  "report.md",
					Parts: []*ai.Part{ai.NewTextPart("the report body")},
				})
				sess.AddMessages(ai.NewModelTextMessage("wrote the report"))
				return &aix.TurnResult{FinishReason: aix.AgentFinishReasonStop}, nil
			})
			if err != nil {
				return nil, err
			}
			return &aix.AgentResult{
				Message:   ai.NewModelTextMessage("wrote the report"),
				Artifacts: sess.Artifacts(),
			}, nil
		},
		aix.WithSessionStore[any](localstore.NewInMemorySessionStore[any]()),
	)

	delegating := toolModel(t, g, "test/orch-sync-check", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		if len(toolOutputs(req.Messages, checkBackgroundTasksToolName)) > 0 {
			return textResp(req, "done"), nil
		}
		if res, ok := lastDelegationOutput(req.Messages, "delegate_to_writer"); ok {
			return toolReqResp(req, &ai.ToolRequest{
				Name:  checkBackgroundTasksToolName,
				Input: map[string]any{"taskIds": []string{res.TaskID}},
			}), nil
		}
		return toolReqResp(req, &ai.ToolRequest{Name: "delegate_to_writer", Input: map[string]any{"task": "write a report"}}), nil
	})

	// The orchestrator is itself an agent, so the delegation runs within a
	// session the artifacts can merge into.
	orchestrator := genkitx.DefineCustomAgent[any](g, "orchestrator",
		func(ctx context.Context, resp aix.Responder, sess *aix.SessionRunner[any]) (*aix.AgentResult, error) {
			var last *ai.Message
			err := sess.Run(ctx, func(ctx context.Context, input *aix.AgentInput) (*aix.TurnResult, error) {
				r, err := genkit.Generate(ctx, g,
					ai.WithModel(delegating),
					ai.WithMessages(input.Message),
					ai.WithUse(&Agents{Agents: []aix.AgentRef{{Name: "writer"}}, Async: true}),
				)
				if err != nil {
					return nil, err
				}
				last = r.Message
				return &aix.TurnResult{FinishReason: aix.AgentFinishReasonStop}, nil
			})
			if err != nil {
				return nil, err
			}
			return &aix.AgentResult{Message: last, Artifacts: sess.Artifacts()}, nil
		},
	)

	out, err := orchestrator.RunText(ctx, "please produce a report")
	if err != nil {
		t.Fatal(err)
	}
	if len(out.Artifacts) != 1 {
		t.Fatalf("expected exactly 1 merged artifact after check, got %v", artifactNames(out.Artifacts))
	}
	name := out.Artifacts[0].Name
	if !strings.HasPrefix(name, "writer_") || !strings.HasSuffix(name, "/report.md") || name == "writer_1/report.md" {
		t.Errorf("artifact name = %q, want the snapshot-based \"writer_<snap>/report.md\" namespace", name)
	}
}
