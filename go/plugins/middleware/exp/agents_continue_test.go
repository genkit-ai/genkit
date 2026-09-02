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
	"errors"
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

// flakyStore wraps an in-memory session store so a test can fail or rewrite
// individual snapshot reads and writes by ID, exercising the resume flow's
// transient-error and race arms. The embedded store carries the interface's
// remaining methods, subscriber support included.
type flakyStore struct {
	*localstore.InMemorySessionStore[any]
	mu sync.Mutex
	// failSave fails SaveSnapshot for exactly the given snapshot IDs.
	failSave map[string]error
	// getHook, when set, may replace the outcome of any GetSnapshot.
	getHook func(id string, snap *aix.SessionSnapshot[any], err error) (*aix.SessionSnapshot[any], error)
}

func newFlakyStore() *flakyStore {
	return &flakyStore{
		InMemorySessionStore: localstore.NewInMemorySessionStore[any](),
		failSave:             map[string]error{},
	}
}

func (s *flakyStore) setSaveFailure(id string, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if err == nil {
		delete(s.failSave, id)
		return
	}
	s.failSave[id] = err
}

func (s *flakyStore) setGetHook(hook func(id string, snap *aix.SessionSnapshot[any], err error) (*aix.SessionSnapshot[any], error)) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.getHook = hook
}

func (s *flakyStore) SaveSnapshot(ctx context.Context, snapshotID string, fn func(*aix.SessionSnapshot[any]) (*aix.SessionSnapshot[any], error)) (*aix.SessionSnapshot[any], error) {
	s.mu.Lock()
	err := s.failSave[snapshotID]
	s.mu.Unlock()
	if err != nil {
		return nil, err
	}
	return s.InMemorySessionStore.SaveSnapshot(ctx, snapshotID, fn)
}

// GetSnapshotMetadata routes through GetSnapshot so the hook shapes metadata
// reads exactly as it shapes full ones, then drops the state. The embedded
// store's own capability is promoted and would otherwise answer the
// runtime's metadata-only reads around the hook.
func (s *flakyStore) GetSnapshotMetadata(ctx context.Context, snapshotID string) (*aix.SessionSnapshot[any], error) {
	snap, err := s.GetSnapshot(ctx, snapshotID)
	if snap != nil {
		snap.State = nil
	}
	return snap, err
}

func (s *flakyStore) GetSnapshot(ctx context.Context, snapshotID string) (*aix.SessionSnapshot[any], error) {
	snap, err := s.InMemorySessionStore.GetSnapshot(ctx, snapshotID)
	s.mu.Lock()
	hook := s.getHook
	s.mu.Unlock()
	if hook != nil {
		return hook(snapshotID, snap, err)
	}
	return snap, err
}

// seedDeadKeeperTask defines a server-managed "keeper" sub-agent on store,
// runs one delegation to commit a conversation, and plants a dead worker's
// pending row on top of it. It returns the dead task's handle and the
// committed (parent) snapshot ID.
func seedDeadKeeperTask(t *testing.T, g *genkit.Genkit, store aix.SessionStore[any], keeperModel ai.Model) (deadTask, committedID string) {
	t.Helper()
	genkitx.DefineAgent[any](g, "keeper",
		aix.InlinePrompt{ai.WithModel(keeperModel)},
		aix.WithSessionStore[any](store),
	)
	first, err := genkit.Generate(ctx, g,
		ai.WithModel(delegateOnceModel(t, g, "test/seed", "delegate_to_keeper", "start X")),
		ai.WithPrompt("go"),
		ai.WithUse(&Agents{Agents: []aix.AgentRef{{Name: "keeper"}}}))
	if err != nil {
		t.Fatal(err)
	}
	seeded := delegationResponses(t, first.History(), "delegate_to_keeper")
	if len(seeded) != 1 || seeded[0].TaskID == "" {
		t.Fatalf("expected a seeded delegation with a handle, got %+v", seeded)
	}
	committedID = strings.TrimPrefix(seeded[0].TaskID, "keeper:")
	committed, err := store.GetSnapshot(ctx, committedID)
	if err != nil || committed == nil {
		t.Fatalf("read committed snapshot %q: %v", committedID, err)
	}
	pending, err := saveDeadPendingRow(store, committed.SessionID, committed.SnapshotID)
	if err != nil {
		t.Fatalf("SaveSnapshot pending row: %v", err)
	}
	return "keeper:" + pending.SnapshotID, committedID
}

// lastDelegationOutput decodes the newest tool response for toolName in msgs.
// It is for model functions (no *testing.T in scope): ok is false when there
// is none, which fails the scripted expectation that follows.
func lastDelegationOutput(msgs []*ai.Message, toolName string) (delegationResult, bool) {
	outs := toolOutputs(msgs, toolName)
	if len(outs) == 0 {
		return delegationResult{}, false
	}
	return lenientDelegation(outs[len(outs)-1]), true
}

// failNTimesModel returns a model that errors its first n calls and then
// answers with successText, recording each request's messages in *seen.
func failNTimesModel(t *testing.T, g *genkit.Genkit, name string, n int, successText string, seen *[][]*ai.Message) ai.Model {
	t.Helper()
	calls := 0
	return toolModel(t, g, name, func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		if seen != nil {
			*seen = append(*seen, req.Messages)
		}
		calls++
		if calls <= n {
			return nil, errors.New("model melted")
		}
		return textResp(req, successText), nil
	})
}

// delegateThenResumeModel returns an orchestrator that delegates once, then
// answers every failed delegation by resuming its taskId (with instructions
// when given), and says "done" once a resume settles without a taskable
// failure.
func delegateThenResumeModel(t *testing.T, g *genkit.Genkit, name, delegateTool, resumeTool, task, instructions string) ai.Model {
	t.Helper()
	return toolModel(t, g, name, func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		if res, ok := lastDelegationOutput(req.Messages, resumeTool); ok {
			return textResp(req, "done after resume: "+res.Response), nil
		}
		if res, ok := lastDelegationOutput(req.Messages, delegateTool); ok {
			input := map[string]any{"taskId": res.TaskID}
			if instructions != "" {
				input["instructions"] = instructions
			}
			return toolReqResp(req, &ai.ToolRequest{Name: resumeTool, Input: input}), nil
		}
		return toolReqResp(req, &ai.ToolRequest{Name: delegateTool, Input: map[string]any{"task": task}}), nil
	})
}

func TestAgentsContinueRetriesFailedTask(t *testing.T) {
	// A server-managed sub-agent fails, the failure result carries its task
	// handle, and resuming that handle re-attempts the run from the failed
	// snapshot: the sub-agent's second call sees the same conversation and
	// answers, and the resume result folds like a synchronous delegation,
	// handle included.
	g := newTestGenkit(t)

	var seen [][]*ai.Message
	genkitx.DefineAgent[any](g, "flaky",
		aix.InlinePrompt{ai.WithModel(failNTimesModel(t, g, "test/flaky", 1, "recovered", &seen))},
		aix.WithSessionStore[any](localstore.NewInMemorySessionStore[any]()),
	)

	orch := delegateThenResumeModel(t, g, "test/orch", "delegate_to_flaky", "continue_task", "try X", "")
	mw := &Agents{Agents: []aix.AgentRef{{Name: "flaky"}}}

	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("go"), ai.WithUse(mw))
	if err != nil {
		t.Fatal(err)
	}

	failures := delegationResponses(t, resp.History(), "delegate_to_flaky")
	if len(failures) != 1 || failures[0].Status != string(aix.SnapshotStatusFailed) {
		t.Fatalf("expected 1 failed delegation, got %+v", failures)
	}
	if !strings.Contains(failures[0].Response, "continue_task") {
		t.Errorf("failure response does not advertise the continue tool: %q", failures[0].Response)
	}

	resumes := delegationResponses(t, resp.History(), "continue_task")
	if len(resumes) != 1 {
		t.Fatalf("expected 1 resume response, got %d", len(resumes))
	}
	if resumes[0].Response != "recovered" {
		t.Errorf("resume Response = %q, want %q", resumes[0].Response, "recovered")
	}
	if resumes[0].Status != string(aix.SnapshotStatusCompleted) || !strings.HasPrefix(resumes[0].TaskID, "flaky:") {
		t.Errorf("resume result not stamped as a completed task: %+v", resumes[0])
	}

	// The retry ran on the committed conversation: same task, no new input.
	if len(seen) != 2 {
		t.Fatalf("expected the sub-agent model to run twice, ran %d times", len(seen))
	}
	retry := seen[1]
	if len(retry) == 0 || !strings.Contains(retry[len(retry)-1].Text(), "try X") {
		t.Errorf("expected the retry to re-attempt the committed task, messages: %v", retry)
	}
}

func TestAgentsContinueWithInstructionsSteersRetry(t *testing.T) {
	// Instructions ride into the resumed run as a fresh user message on top of
	// the committed conversation, steering the retry instead of repeating it.
	g := newTestGenkit(t)

	var seen [][]*ai.Message
	genkitx.DefineAgent[any](g, "flaky",
		aix.InlinePrompt{ai.WithModel(failNTimesModel(t, g, "test/flaky", 1, "steered", &seen))},
		aix.WithSessionStore[any](localstore.NewInMemorySessionStore[any]()),
	)

	orch := delegateThenResumeModel(t, g, "test/orch", "delegate_to_flaky", "continue_task", "try X", "skip the flaky source")
	mw := &Agents{Agents: []aix.AgentRef{{Name: "flaky"}}}

	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("go"), ai.WithUse(mw))
	if err != nil {
		t.Fatal(err)
	}
	resumes := delegationResponses(t, resp.History(), "continue_task")
	if len(resumes) != 1 || resumes[0].Response != "steered" {
		t.Fatalf("expected a steered resume, got %+v", resumes)
	}
	if len(seen) != 2 {
		t.Fatalf("expected the sub-agent model to run twice, ran %d times", len(seen))
	}
	retry := seen[1]
	last := retry[len(retry)-1]
	if last.Role != ai.RoleUser || !strings.Contains(last.Text(), "skip the flaky source") {
		t.Errorf("expected the instructions as the retry's new user message, got role=%s text=%q", last.Role, last.Text())
	}
	joined := ""
	for _, m := range retry {
		joined += m.Text() + "\n"
	}
	if !strings.Contains(joined, "try X") {
		t.Errorf("expected the committed task to remain in the retry's context, messages: %q", joined)
	}
}

func TestAgentsContinueCompletedTask(t *testing.T) {
	// A completed task refuses an instructions-less resume (an empty input
	// would re-run the finished turn) and accepts a follow-up with
	// instructions inside the sub-agent's own session.
	g := newTestGenkit(t)

	var seen [][]*ai.Message
	genkitx.DefineAgent[any](g, "helper",
		aix.InlinePrompt{ai.WithModel(failNTimesModel(t, g, "test/helper", 0, "answered", &seen))},
		aix.WithSessionStore[any](localstore.NewInMemorySessionStore[any]()),
	)

	// Delegate, try an instructions-less resume (refused), then follow up.
	orch := toolModel(t, g, "test/orch", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		resumes := 0
		var lastResume delegationResult
		for _, v := range toolOutputs(req.Messages, "continue_task") {
			lastResume = lenientDelegation(v)
			resumes++
		}
		delegated, ok := lastDelegationOutput(req.Messages, "delegate_to_helper")
		switch {
		case !ok:
			return toolReqResp(req, &ai.ToolRequest{Name: "delegate_to_helper", Input: map[string]any{"task": "answer X"}}), nil
		case resumes == 0:
			return toolReqResp(req, &ai.ToolRequest{Name: "continue_task", Input: map[string]any{"taskId": delegated.TaskID}}), nil
		case resumes == 1:
			return toolReqResp(req, &ai.ToolRequest{Name: "continue_task", Input: map[string]any{"taskId": delegated.TaskID, "instructions": "now also cover Y"}}), nil
		default:
			return textResp(req, "done: "+lastResume.Response), nil
		}
	})
	// MaxDelegations pins the refusal's slot refund: the delegation and the
	// corrected follow-up spend the two slots, so the instructions-less
	// refusal in between must return the one it reserved.
	mw := &Agents{Agents: []aix.AgentRef{{Name: "helper"}}, MaxDelegations: 2}

	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("go"), ai.WithUse(mw))
	if err != nil {
		t.Fatal(err)
	}
	resumes := delegationResponses(t, resp.History(), "continue_task")
	if len(resumes) != 2 {
		t.Fatalf("expected 2 resume responses, got %d", len(resumes))
	}
	if !strings.Contains(resumes[0].Response, "already completed") {
		t.Errorf("expected the instructions-less resume to be refused, got %q", resumes[0].Response)
	}
	if resumes[1].Response != "answered" || resumes[1].Status != string(aix.SnapshotStatusCompleted) {
		t.Errorf("expected the follow-up to settle, got %+v", resumes[1])
	}
	if len(seen) != 2 {
		t.Fatalf("expected the sub-agent model to run twice, ran %d times", len(seen))
	}
	followUp := seen[1]
	joined := ""
	for _, m := range followUp {
		joined += m.Text() + "\n"
	}
	if !strings.Contains(joined, "answer X") || !strings.Contains(joined, "now also cover Y") {
		t.Errorf("expected the follow-up to run inside the original session, messages: %q", joined)
	}
}

func TestAgentsContinueCarriesLabel(t *testing.T) {
	// A resumed task is the same undertaking, so the caller-chosen label
	// follows the handle onto the continuation's result.
	g := newTestGenkit(t)
	genkitx.DefineAgent[any](g, "flaky",
		aix.InlinePrompt{ai.WithModel(failNTimesModel(t, g, "test/flaky-label", 1, "recovered", nil))},
		aix.WithSessionStore[any](localstore.NewInMemorySessionStore[any]()),
	)
	orch := toolModel(t, g, "test/orch-label2", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		if _, ok := lastDelegationOutput(req.Messages, "continue_task"); ok {
			return textResp(req, "done"), nil
		}
		if res, ok := lastDelegationOutput(req.Messages, "delegate_to_flaky"); ok {
			return toolReqResp(req, &ai.ToolRequest{Name: "continue_task", Input: map[string]any{"taskId": res.TaskID}}), nil
		}
		return toolReqResp(req, &ai.ToolRequest{Name: "delegate_to_flaky",
			Input: map[string]any{"task": "try X", "name": "second-try"}}), nil
	})
	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("go"),
		ai.WithUse(&Agents{Agents: []aix.AgentRef{{Name: "flaky"}}}))
	if err != nil {
		t.Fatal(err)
	}
	failures := delegationResponses(t, resp.History(), "delegate_to_flaky")
	if len(failures) != 1 || failures[0].Name != "second-try" {
		t.Fatalf("expected the labeled failure, got %+v", failures)
	}
	resumes := delegationResponses(t, resp.History(), "continue_task")
	if len(resumes) != 1 || resumes[0].Response != "recovered" || resumes[0].Name != "second-try" {
		t.Fatalf("expected the label to follow the resume, got %+v", resumes)
	}
}

func TestAgentsContinueCountsAgainstCap(t *testing.T) {
	// A resume is a real sub-agent run and spends a delegation slot; with the
	// cap exhausted by the delegation itself, the resume is refused.
	g := newTestGenkit(t)

	genkitx.DefineAgent[any](g, "flaky",
		aix.InlinePrompt{ai.WithModel(failNTimesModel(t, g, "test/flaky", 99, "", nil))},
		aix.WithSessionStore[any](localstore.NewInMemorySessionStore[any]()),
	)

	orch := delegateThenResumeModel(t, g, "test/orch", "delegate_to_flaky", "continue_task", "try X", "")
	mw := &Agents{Agents: []aix.AgentRef{{Name: "flaky"}}, MaxDelegations: 1}

	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("go"), ai.WithUse(mw))
	if err != nil {
		t.Fatal(err)
	}
	resumes := delegationResponses(t, resp.History(), "continue_task")
	if len(resumes) != 1 || !strings.Contains(resumes[0].Response, "Delegation limit reached") {
		t.Fatalf("expected the resume to be refused by the cap, got %+v", resumes)
	}
}

func TestAgentsContinueExpiredRecoversCommittedProgress(t *testing.T) {
	// An expired handle (a dead worker's pending row) is fenced with an abort
	// and resumed from the session's latest committed snapshot, recovering
	// whatever the run persisted before it detached.
	g := newTestGenkit(t)

	// A committed conversation to recover, then a dead worker's pending row
	// as the session tip.
	store := localstore.NewInMemorySessionStore[any]()
	var seen [][]*ai.Message
	deadTask, _ := seedDeadKeeperTask(t, g, store, failNTimesModel(t, g, "test/keeper", 0, "kept going", &seen))
	pendingID := strings.TrimPrefix(deadTask, "keeper:")

	orch := toolModel(t, g, "test/orch", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		if res, ok := lastDelegationOutput(req.Messages, "continue_task"); ok {
			return textResp(req, "done: "+res.Response), nil
		}
		return toolReqResp(req, &ai.ToolRequest{Name: "continue_task",
			Input: map[string]any{"taskId": deadTask, "instructions": "continue"}}), nil
	})
	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("go"),
		ai.WithUse(&Agents{Agents: []aix.AgentRef{{Name: "keeper"}}}))
	if err != nil {
		t.Fatal(err)
	}
	resumes := delegationResponses(t, resp.History(), "continue_task")
	if len(resumes) != 1 || resumes[0].Response != "kept going" {
		t.Fatalf("expected the expired task to resume from committed progress, got %+v", resumes)
	}

	// The fence flipped the dead pending row so a slow worker cannot race the
	// recovered session. No worker is behind it to finalize, so the raw row
	// stays at the flip.
	fenced, err := store.GetSnapshot(ctx, pendingID)
	if err != nil || fenced == nil {
		t.Fatalf("read fenced row: %v", err)
	}
	if fenced.Status != aix.SnapshotStatusAborting {
		t.Errorf("fenced row status = %q, want %q", fenced.Status, aix.SnapshotStatusAborting)
	}

	// The recovered run saw the committed conversation plus the instructions.
	last := seen[len(seen)-1]
	joined := ""
	for _, m := range last {
		joined += m.Text() + "\n"
	}
	if !strings.Contains(joined, "start X") || !strings.Contains(joined, "continue") {
		t.Errorf("expected recovery from the committed conversation, messages: %q", joined)
	}
}

func TestAgentsContinueExpiredWithNothingSavedRefused(t *testing.T) {
	// A dead worker that never committed anything (a background launch dies
	// before finalize) left nothing to recover; the refusal says to delegate
	// again instead of pretending to resume.
	g := newTestGenkit(t)

	store := localstore.NewInMemorySessionStore[any]()
	genkitx.DefineAgent[any](g, "keeper",
		aix.InlinePrompt{ai.WithModel(toolModel(t, g, "test/keeper", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			return textResp(req, "unused"), nil
		}))},
		aix.WithSessionStore[any](store),
	)

	pending, err := saveDeadPendingRow(store, "sess-dead", "")
	if err != nil {
		t.Fatalf("SaveSnapshot pending row: %v", err)
	}

	deadTask := "keeper:" + pending.SnapshotID
	orch := toolModel(t, g, "test/orch", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		if _, ok := lastDelegationOutput(req.Messages, "continue_task"); ok {
			return textResp(req, "done"), nil
		}
		return toolReqResp(req, &ai.ToolRequest{Name: "continue_task", Input: map[string]any{"taskId": deadTask}}), nil
	})
	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("go"),
		ai.WithUse(&Agents{Agents: []aix.AgentRef{{Name: "keeper"}}}))
	if err != nil {
		t.Fatal(err)
	}
	resumes := delegationResponses(t, resp.History(), "continue_task")
	if len(resumes) != 1 {
		t.Fatalf("expected 1 resume response, got %d", len(resumes))
	}
	if !strings.Contains(resumes[0].Response, "saved no progress to continue from") ||
		!strings.Contains(resumes[0].Response, "Delegate the task again") {
		t.Errorf("expected an honest nothing-saved refusal, got %q", resumes[0].Response)
	}
}

// saveDeadPendingRow writes a dead worker's pending row the way the detach
// handler mints one: created now (newer than every committed row in the
// session) with a heartbeat that went stale.
func saveDeadPendingRow(store aix.SnapshotWriter[any], sessionID, parentID string) (*aix.SessionSnapshot[any], error) {
	now := time.Now()
	stale := now.Add(-10 * time.Minute)
	return store.SaveSnapshot(ctx, "", func(_ *aix.SessionSnapshot[any]) (*aix.SessionSnapshot[any], error) {
		return &aix.SessionSnapshot[any]{
			SessionID:   sessionID,
			ParentID:    parentID,
			Status:      aix.SnapshotStatusPending,
			CreatedAt:   now,
			UpdatedAt:   now,
			HeartbeatAt: &stale,
		}, nil
	})
}

func TestAgentsContinueExpiredFinishedParentRequiresInstructions(t *testing.T) {
	// The parent behind a dead task can be a finished turn; continuing past
	// it gets the same instructions gate as a completed task, since an empty
	// input would re-run the finished turn instead of continuing the work.
	g := newTestGenkit(t)

	deadTask, _ := seedDeadKeeperTask(t, g, localstore.NewInMemorySessionStore[any](),
		failNTimesModel(t, g, "test/keeper", 0, "kept", nil))
	orch := toolModel(t, g, "test/orch", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		if _, ok := lastDelegationOutput(req.Messages, "continue_task"); ok {
			return textResp(req, "done"), nil
		}
		return toolReqResp(req, &ai.ToolRequest{Name: "continue_task", Input: map[string]any{"taskId": deadTask}}), nil
	})
	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("go"),
		ai.WithUse(&Agents{Agents: []aix.AgentRef{{Name: "keeper"}}}))
	if err != nil {
		t.Fatal(err)
	}
	resumes := delegationResponses(t, resp.History(), "continue_task")
	if len(resumes) != 1 || !strings.Contains(resumes[0].Response, "last finished turn") {
		t.Fatalf("expected the finished-parent instructions gate, got %+v", resumes)
	}
}

func TestAgentsContinueInBackground(t *testing.T) {
	// With Async set, a failed task can be resumed in the background: the tool
	// returns a fresh pending handle in the same session, and the wait tool
	// collects the retried result.
	g := newTestGenkit(t)

	genkitx.DefineAgent[any](g, "flaky",
		aix.InlinePrompt{ai.WithModel(failNTimesModel(t, g, "test/flaky", 1, "recovered later", nil))},
		aix.WithSessionStore[any](localstore.NewInMemorySessionStore[any]()),
	)

	orch := toolModel(t, g, "test/orch", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		if outs := toolOutputs(req.Messages, waitBackgroundTasksToolName); len(outs) > 0 {
			return textResp(req, "done"), nil
		}
		if res, ok := lastDelegationOutput(req.Messages, "continue_task"); ok {
			return toolReqResp(req, &ai.ToolRequest{Name: waitBackgroundTasksToolName,
				Input: map[string]any{"taskIds": []string{res.TaskID}}}), nil
		}
		if res, ok := lastDelegationOutput(req.Messages, "delegate_to_flaky"); ok {
			return toolReqResp(req, &ai.ToolRequest{Name: "continue_task",
				Input: map[string]any{"taskId": res.TaskID, "background": true}}), nil
		}
		return toolReqResp(req, &ai.ToolRequest{Name: "delegate_to_flaky", Input: map[string]any{"task": "try X"}}), nil
	})
	mw := &Agents{Agents: []aix.AgentRef{{Name: "flaky"}}, Async: true}

	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("go"), ai.WithUse(mw))
	if err != nil {
		t.Fatal(err)
	}
	resumes := delegationResponses(t, resp.History(), "continue_task")
	if len(resumes) != 1 || resumes[0].Status != string(aix.SnapshotStatusPending) {
		t.Fatalf("expected a pending background resume, got %+v", resumes)
	}
	if !strings.HasPrefix(resumes[0].TaskID, "flaky:") {
		t.Fatalf("expected a fresh task handle, got %q", resumes[0].TaskID)
	}

	var waited backgroundTasksResult
	outs := toolOutputs(resp.History(), waitBackgroundTasksToolName)
	if len(outs) != 1 {
		t.Fatalf("expected 1 wait result, got %d", len(outs))
	}
	waited = decodeToolOutput[backgroundTasksResult](t, outs[0])
	if len(waited.Tasks) != 1 || waited.Tasks[0].Status != string(aix.SnapshotStatusCompleted) {
		t.Fatalf("expected the background resume to complete, got %+v", waited.Tasks)
	}
	if waited.Tasks[0].Response != "recovered later" {
		t.Errorf("background resume response = %q, want %q", waited.Tasks[0].Response, "recovered later")
	}
}

func TestAgentsContinueExpiredFenceFailureRefusesAndRefunds(t *testing.T) {
	// The fence is the one write standing between the recovery and a live
	// worker: a fence that fails must refuse rather than recover unfenced,
	// and the refusal names a retry that can succeed, so it returns its slot.
	// MaxDelegations of 1 pins the refund: the retried resume only fits if
	// the failed fence gave its slot back.
	g := newTestGenkit(t)
	store := newFlakyStore()
	deadTask, _ := seedDeadKeeperTask(t, g, store, failNTimesModel(t, g, "test/keeper", 0, "kept going", nil))
	pendingID := strings.TrimPrefix(deadTask, "keeper:")
	store.setSaveFailure(pendingID, errors.New("store blip"))

	orch := toolModel(t, g, "test/orch", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		resume := &ai.ToolRequest{Name: "continue_task",
			Input: map[string]any{"taskId": deadTask, "instructions": "continue"}}
		switch len(toolOutputs(req.Messages, "continue_task")) {
		case 0:
			return toolReqResp(req, resume), nil
		case 1:
			// The blip clears; the model retries the same handle.
			store.setSaveFailure(pendingID, nil)
			return toolReqResp(req, resume), nil
		default:
			return textResp(req, "done"), nil
		}
	})
	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("go"),
		ai.WithUse(&Agents{Agents: []aix.AgentRef{{Name: "keeper"}}, MaxDelegations: 1}))
	if err != nil {
		t.Fatal(err)
	}
	resumes := delegationResponses(t, resp.History(), "continue_task")
	if len(resumes) != 2 {
		t.Fatalf("expected 2 resume responses, got %+v", resumes)
	}
	if !strings.Contains(resumes[0].Response, "could not fence") {
		t.Errorf("expected the fence-failure refusal, got %q", resumes[0].Response)
	}
	if resumes[1].Response != "kept going" {
		t.Errorf("expected the retried resume to recover, got %+v", resumes[1])
	}
}

func TestAgentsContinueExpiredDeadEndsKeepTheSlot(t *testing.T) {
	// A fence or post-fence re-read that fails with a classified dead end
	// refuses without a retry hint and keeps its slot, like every other read
	// on the ladder: the retry fails identically, so a refund would let the
	// model spend forever on a call that cannot succeed. MaxDelegations of 1
	// pins the kept slot: the retried call trips the cap.
	for _, tc := range []struct {
		name    string
		arm     func(store *flakyStore, pendingID string)
		refusal string
	}{
		{"fence refused", func(store *flakyStore, pendingID string) {
			store.setSaveFailure(pendingID, status.Errorf(status.ErrFailedPrecondition, "store cannot fence"))
		}, "could not fence"},
		{"row gone after the fence", func(store *flakyStore, pendingID string) {
			// The pre-read sees the dead pending row; the fence flips it to
			// aborting; the re-read then finds it gone, which the companion
			// action reports as NOT_FOUND.
			store.setGetHook(func(id string, snap *aix.SessionSnapshot[any], err error) (*aix.SessionSnapshot[any], error) {
				if id == pendingID && snap != nil && snap.Status == aix.SnapshotStatusAborting {
					return nil, nil
				}
				return snap, err
			})
		}, "could not read"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			g := newTestGenkit(t)
			store := newFlakyStore()
			deadTask, _ := seedDeadKeeperTask(t, g, store, failNTimesModel(t, g, "test/keeper", 0, "kept going", nil))
			tc.arm(store, strings.TrimPrefix(deadTask, "keeper:"))

			orch := toolModel(t, g, "test/orch", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
				resume := &ai.ToolRequest{Name: "continue_task",
					Input: map[string]any{"taskId": deadTask, "instructions": "continue"}}
				if len(toolOutputs(req.Messages, "continue_task")) < 2 {
					return toolReqResp(req, resume), nil
				}
				return textResp(req, "done"), nil
			})
			resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("go"),
				ai.WithUse(&Agents{Agents: []aix.AgentRef{{Name: "keeper"}}, MaxDelegations: 1}))
			if err != nil {
				t.Fatal(err)
			}
			resumes := delegationResponses(t, resp.History(), "continue_task")
			if len(resumes) != 2 {
				t.Fatalf("expected 2 resume responses, got %+v", resumes)
			}
			if !strings.Contains(resumes[0].Response, tc.refusal) || strings.Contains(resumes[0].Response, "Try again later") {
				t.Errorf("expected a dead-end refusal without a retry hint, got %q", resumes[0].Response)
			}
			if !strings.Contains(resumes[1].Response, "Delegation limit reached") {
				t.Errorf("expected the kept slot to trip the cap on retry, got %q", resumes[1].Response)
			}
		})
	}
}

func TestAgentsContinueParentReadBlipRefundsSlot(t *testing.T) {
	// A transient failure reading the dead task's parent snapshot refuses
	// with a retry hint and returns its slot, exactly as continueFromStore's
	// own transient read arm does; MaxDelegations of 1 pins the refund.
	g := newTestGenkit(t)
	store := newFlakyStore()
	deadTask, committedID := seedDeadKeeperTask(t, g, store, failNTimesModel(t, g, "test/keeper", 0, "kept going", nil))
	store.setGetHook(func(id string, snap *aix.SessionSnapshot[any], err error) (*aix.SessionSnapshot[any], error) {
		if id == committedID {
			return nil, errors.New("parent read blip")
		}
		return snap, err
	})

	orch := toolModel(t, g, "test/orch", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		resume := &ai.ToolRequest{Name: "continue_task",
			Input: map[string]any{"taskId": deadTask, "instructions": "continue"}}
		switch len(toolOutputs(req.Messages, "continue_task")) {
		case 0:
			return toolReqResp(req, resume), nil
		case 1:
			store.setGetHook(nil)
			return toolReqResp(req, resume), nil
		default:
			return textResp(req, "done"), nil
		}
	})
	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("go"),
		ai.WithUse(&Agents{Agents: []aix.AgentRef{{Name: "keeper"}}, MaxDelegations: 1}))
	if err != nil {
		t.Fatal(err)
	}
	resumes := delegationResponses(t, resp.History(), "continue_task")
	if len(resumes) != 2 {
		t.Fatalf("expected 2 resume responses, got %+v", resumes)
	}
	if !strings.Contains(resumes[0].Response, "could not be read") || !strings.Contains(resumes[0].Response, "Try again later") {
		t.Errorf("expected the transient parent-read refusal, got %q", resumes[0].Response)
	}
	if resumes[1].Response != "kept going" {
		t.Errorf("expected the retried resume to recover, got %+v", resumes[1])
	}
}

func TestAgentsContinueExpiredWindingDownRefusesAndRefunds(t *testing.T) {
	// A worker that is alive after all observes the fence and keeps beating
	// while it drains, so the post-fence re-read shapes the row as aborting.
	// The resume must not fall back to the parent (that would race the
	// finalize and re-buy committed turns): it refuses, names the retry, and
	// returns its slot. Once the row settles the same handle resumes.
	g := newTestGenkit(t)
	store := newFlakyStore()
	deadTask, _ := seedDeadKeeperTask(t, g, store, failNTimesModel(t, g, "test/keeper", 0, "kept going", nil))
	pendingID := strings.TrimPrefix(deadTask, "keeper:")
	// While the hook is set, any read of the fenced row reports a fresh
	// heartbeat: the aborting row as a live worker's beats keep it.
	store.setGetHook(func(id string, snap *aix.SessionSnapshot[any], err error) (*aix.SessionSnapshot[any], error) {
		if id == pendingID && err == nil && snap != nil && snap.Status == aix.SnapshotStatusAborting {
			fresh := *snap
			now := time.Now()
			fresh.HeartbeatAt = &now
			return &fresh, nil
		}
		return snap, err
	})

	orch := toolModel(t, g, "test/orch", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		resume := &ai.ToolRequest{Name: "continue_task",
			Input: map[string]any{"taskId": deadTask, "instructions": "continue"}}
		switch len(toolOutputs(req.Messages, "continue_task")) {
		case 0:
			return toolReqResp(req, resume), nil
		case 1:
			// The worker "dies" without finalizing: beats stop, the row goes
			// stale, and the same handle recovers through the parent.
			store.setGetHook(nil)
			return toolReqResp(req, resume), nil
		default:
			return textResp(req, "done"), nil
		}
	})
	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("go"),
		ai.WithUse(&Agents{Agents: []aix.AgentRef{{Name: "keeper"}}, MaxDelegations: 1}))
	if err != nil {
		t.Fatal(err)
	}
	resumes := delegationResponses(t, resp.History(), "continue_task")
	if len(resumes) != 2 {
		t.Fatalf("expected 2 resume responses, got %+v", resumes)
	}
	if !strings.Contains(resumes[0].Response, "winding down") {
		t.Errorf("expected the winding-down refusal, got %q", resumes[0].Response)
	}
	if resumes[1].Response != "kept going" {
		t.Errorf("expected the settled handle to resume, got %+v", resumes[1])
	}
}

func TestAgentsContinueExpiredCompletedFinalizeGetsInstructionsGate(t *testing.T) {
	// The worker the fence targeted was alive and its COMPLETED finalize won
	// the race: the post-fence re-read finds a finished row, and an
	// instructions-less resume of it must hit the same gate as any completed
	// task instead of silently re-running the finished turn.
	g := newTestGenkit(t)
	store := newFlakyStore()
	deadTask, _ := seedDeadKeeperTask(t, g, store, failNTimesModel(t, g, "test/keeper", 0, "kept going", nil))
	pendingID := strings.TrimPrefix(deadTask, "keeper:")
	// After the fence flips the raw row, reads report the worker's completed
	// finalize having landed instead.
	store.setGetHook(func(id string, snap *aix.SessionSnapshot[any], err error) (*aix.SessionSnapshot[any], error) {
		if id == pendingID && err == nil && snap != nil && snap.Status == aix.SnapshotStatusAborting {
			done := *snap
			done.Status = aix.SnapshotStatusCompleted
			done.FinishReason = aix.AgentFinishReasonStop
			done.State = &aix.SessionState[any]{Messages: []*ai.Message{ai.NewUserTextMessage("finished")}}
			done.HeartbeatAt = nil
			return &done, nil
		}
		return snap, err
	})

	orch := toolModel(t, g, "test/orch", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		if len(toolOutputs(req.Messages, "continue_task")) == 0 {
			return toolReqResp(req, &ai.ToolRequest{Name: "continue_task",
				Input: map[string]any{"taskId": deadTask}}), nil
		}
		return textResp(req, "done"), nil
	})
	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("go"),
		ai.WithUse(&Agents{Agents: []aix.AgentRef{{Name: "keeper"}}}))
	if err != nil {
		t.Fatal(err)
	}
	resumes := delegationResponses(t, resp.History(), "continue_task")
	if len(resumes) != 1 || !strings.Contains(resumes[0].Response, "already completed") {
		t.Fatalf("expected the completed-task instructions gate, got %+v", resumes)
	}
}

func TestAgentsAbortingTaskReportsAndRefusesContinue(t *testing.T) {
	// An aborting row is a live wind-down: the check tool reports it as such,
	// and the continue tool refuses it with a refund, since the same handle
	// continues once the worker's finalize lands.
	g := newTestGenkit(t)
	store := newFlakyStore()
	_, committedID := seedDeadKeeperTask(t, g, store, failNTimesModel(t, g, "test/keeper", 0, "kept going", nil))
	committed, err := store.GetSnapshot(ctx, committedID)
	if err != nil || committed == nil {
		t.Fatalf("read committed snapshot: %v", err)
	}
	now := time.Now()
	aborting, err := store.SaveSnapshot(ctx, "", func(_ *aix.SessionSnapshot[any]) (*aix.SessionSnapshot[any], error) {
		return &aix.SessionSnapshot[any]{
			SessionID:   committed.SessionID,
			ParentID:    committed.SnapshotID,
			Status:      aix.SnapshotStatusAborting,
			CreatedAt:   now,
			UpdatedAt:   now,
			HeartbeatAt: &now,
		}, nil
	})
	if err != nil {
		t.Fatalf("SaveSnapshot aborting row: %v", err)
	}
	task := "keeper:" + aborting.SnapshotID

	orch := toolModel(t, g, "test/orch", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		checks := toolOutputs(req.Messages, checkBackgroundTasksToolName)
		continues := toolOutputs(req.Messages, "continue_task")
		delegations := toolOutputs(req.Messages, "delegate_to_keeper")
		switch {
		case len(checks) == 0:
			return toolReqResp(req, &ai.ToolRequest{Name: checkBackgroundTasksToolName,
				Input: map[string]any{"taskIds": []string{task}}}), nil
		case len(continues) == 0:
			return toolReqResp(req, &ai.ToolRequest{Name: "continue_task",
				Input: map[string]any{"taskId": task, "instructions": "go on"}}), nil
		case len(delegations) == 0:
			return toolReqResp(req, &ai.ToolRequest{Name: "delegate_to_keeper",
				Input: map[string]any{"task": "more"}}), nil
		default:
			return textResp(req, "done"), nil
		}
	})
	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("go"),
		ai.WithUse(&Agents{Agents: []aix.AgentRef{{Name: "keeper"}}, Async: true, MaxDelegations: 1}))
	if err != nil {
		t.Fatal(err)
	}
	checks := toolOutputs(resp.History(), checkBackgroundTasksToolName)
	if len(checks) != 1 {
		t.Fatalf("expected 1 check response, got %d", len(checks))
	}
	res := decodeToolOutput[backgroundTasksResult](t, checks[0])
	if len(res.Tasks) != 1 || res.Tasks[0].Status != string(aix.SnapshotStatusAborting) || !strings.Contains(res.Tasks[0].Error, "winding down") {
		t.Errorf("check report = %+v, want status %q with the winding-down explanation", res.Tasks, aix.SnapshotStatusAborting)
	}
	continues := delegationResponses(t, resp.History(), "continue_task")
	if len(continues) != 1 || !strings.Contains(continues[0].Response, "winding down") {
		t.Errorf("expected the winding-down refusal, got %+v", continues)
	}
	// The refusal returned its slot: under a cap of one, a delegation still
	// runs afterwards.
	delegations := delegationResponses(t, resp.History(), "delegate_to_keeper")
	if len(delegations) != 1 || delegations[0].Response != "kept going" {
		t.Errorf("expected the refused continuation to refund its slot, got %+v", delegations)
	}
}

func TestAgentsInterruptedTaskNotContinuable(t *testing.T) {
	// A background task that stops on an interrupt gets a handle like any
	// settled task, but nothing can continue it: the report says so without
	// a continue hint, and the continue tool refuses the handle.
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
				sess.AddMessages(ai.NewModelTextMessage("need a human"))
				return &aix.TurnResult{FinishReason: aix.AgentFinishReasonInterrupted}, nil
			})
			if err != nil {
				return nil, err
			}
			return &aix.AgentResult{}, nil
		},
		aix.WithSessionStore[any](localstore.NewInMemorySessionStore[any]()),
	)

	orch := toolModel(t, g, "test/orch-interrupt", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		launches := toolOutputs(req.Messages, "delegate_to_researcher")
		waits := toolOutputs(req.Messages, waitBackgroundTasksToolName)
		continues := toolOutputs(req.Messages, "continue_task")
		switch {
		case len(launches) == 0:
			return toolReqResp(req, &ai.ToolRequest{Name: "delegate_to_researcher",
				Input: map[string]any{"task": "dig into X", "background": true}}), nil
		case len(waits) == 0:
			release()
			return toolReqResp(req, &ai.ToolRequest{Name: waitBackgroundTasksToolName,
				Input: map[string]any{"taskIds": []string{lenientDelegation(launches[0]).TaskID}}}), nil
		case len(continues) == 0:
			return toolReqResp(req, &ai.ToolRequest{Name: "continue_task",
				Input: map[string]any{"taskId": lenientDelegation(launches[0]).TaskID, "instructions": "the answer is 42"}}), nil
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
	if got.Status != string(aix.SnapshotStatusFailed) || !strings.Contains(got.Error, "interrupted") {
		t.Errorf("report = %+v, want a failed status explained as an interrupt", got)
	}
	if strings.Contains(got.Error, "continue it with") {
		t.Errorf("report advertises continuation of an interrupted task: %q", got.Error)
	}
	continues := delegationResponses(t, resp.History(), "continue_task")
	if len(continues) != 1 || !strings.Contains(continues[0].Response, "stopped on an interrupt") {
		t.Errorf("expected the interrupt refusal, got %+v", continues)
	}
}
