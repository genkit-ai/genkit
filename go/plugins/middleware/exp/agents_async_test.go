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
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/ai/exp/localstore"
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
