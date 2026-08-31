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
	"strings"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/ai/exp/localstore"
	"github.com/firebase/genkit/go/genkit"
	genkitx "github.com/firebase/genkit/go/genkit/exp"
)

// lastDelegationOutput decodes the newest tool response for toolName in msgs.
// It is for model functions (no *testing.T in scope); decode problems surface
// as ok=false and fail the scripted expectation that follows.
func lastDelegationOutput(msgs []*ai.Message, toolName string) (delegationResult, bool) {
	var out delegationResult
	found := false
	for _, m := range msgs {
		for _, p := range m.Content {
			if p.IsToolResponse() && p.ToolResponse != nil && p.ToolResponse.Name == toolName {
				b, err := json.Marshal(p.ToolResponse.Output)
				if err != nil {
					continue
				}
				var decoded delegationResult
				if json.Unmarshal(b, &decoded) == nil {
					out, found = decoded, true
				}
			}
		}
	}
	return out, found
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

func TestAgentsResumeRetriesFailedTask(t *testing.T) {
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

	orch := delegateThenResumeModel(t, g, "test/orch", "delegate_to_flaky", "resume_subagent", "try X", "")
	mw := &Agents{Agents: []aix.AgentRef{{Name: "flaky"}}}

	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("go"), ai.WithUse(mw))
	if err != nil {
		t.Fatal(err)
	}

	failures := delegationResponses(t, resp.History(), "delegate_to_flaky")
	if len(failures) != 1 || failures[0].Status != string(aix.SnapshotStatusFailed) {
		t.Fatalf("expected 1 failed delegation, got %+v", failures)
	}
	if !strings.Contains(failures[0].Response, "resume_subagent") {
		t.Errorf("failure response does not advertise the resume tool: %q", failures[0].Response)
	}

	resumes := delegationResponses(t, resp.History(), "resume_subagent")
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

func TestAgentsResumeWithInstructionsSteersRetry(t *testing.T) {
	// Instructions ride into the resumed run as a fresh user message on top of
	// the committed conversation, steering the retry instead of repeating it.
	g := newTestGenkit(t)

	var seen [][]*ai.Message
	genkitx.DefineAgent[any](g, "flaky",
		aix.InlinePrompt{ai.WithModel(failNTimesModel(t, g, "test/flaky", 1, "steered", &seen))},
		aix.WithSessionStore[any](localstore.NewInMemorySessionStore[any]()),
	)

	orch := delegateThenResumeModel(t, g, "test/orch", "delegate_to_flaky", "resume_subagent", "try X", "skip the flaky source")
	mw := &Agents{Agents: []aix.AgentRef{{Name: "flaky"}}}

	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("go"), ai.WithUse(mw))
	if err != nil {
		t.Fatal(err)
	}
	resumes := delegationResponses(t, resp.History(), "resume_subagent")
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

func TestAgentsResumeCompletedTask(t *testing.T) {
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
		for _, v := range toolOutputs(req.Messages, "resume_subagent") {
			b, _ := json.Marshal(v)
			var r delegationResult
			if json.Unmarshal(b, &r) == nil {
				lastResume = r
				resumes++
			}
		}
		delegated, ok := lastDelegationOutput(req.Messages, "delegate_to_helper")
		switch {
		case !ok:
			return toolReqResp(req, &ai.ToolRequest{Name: "delegate_to_helper", Input: map[string]any{"task": "answer X"}}), nil
		case resumes == 0:
			return toolReqResp(req, &ai.ToolRequest{Name: "resume_subagent", Input: map[string]any{"taskId": delegated.TaskID}}), nil
		case resumes == 1:
			return toolReqResp(req, &ai.ToolRequest{Name: "resume_subagent", Input: map[string]any{"taskId": delegated.TaskID, "instructions": "now also cover Y"}}), nil
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
	resumes := delegationResponses(t, resp.History(), "resume_subagent")
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

func TestAgentsResumeClientManagedFromStash(t *testing.T) {
	// A failed client-managed delegation is resumable within the same call
	// through the stash behind its in-memory handle: the retry sees the held
	// conversation, and the settled continuation is re-stashed under a fresh
	// handle.
	g := newTestGenkit(t)

	var seen [][]*ai.Message
	genkitx.DefineAgent[any](g, "eph",
		aix.InlinePrompt{ai.WithModel(failNTimesModel(t, g, "test/eph", 1, "recovered", &seen))},
	)

	orch := delegateThenResumeModel(t, g, "test/orch", "delegate_to_eph", "resume_subagent", "try X", "")
	mw := &Agents{Agents: []aix.AgentRef{{Name: "eph"}}}

	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("go"), ai.WithUse(mw))
	if err != nil {
		t.Fatal(err)
	}
	failures := delegationResponses(t, resp.History(), "delegate_to_eph")
	if len(failures) != 1 || !strings.HasPrefix(failures[0].TaskID, "eph:"+memHandlePrefix) {
		t.Fatalf("expected a failed delegation with an in-memory handle, got %+v", failures)
	}
	resumes := delegationResponses(t, resp.History(), "resume_subagent")
	if len(resumes) != 1 || resumes[0].Response != "recovered" {
		t.Fatalf("expected the stash resume to settle, got %+v", resumes)
	}
	if !strings.HasPrefix(resumes[0].TaskID, "eph:"+memHandlePrefix) || resumes[0].TaskID == failures[0].TaskID {
		t.Errorf("expected a fresh in-memory handle on the continuation, got %q (was %q)", resumes[0].TaskID, failures[0].TaskID)
	}
	if len(seen) != 2 {
		t.Fatalf("expected the sub-agent model to run twice, ran %d times", len(seen))
	}
	retry := seen[1]
	if len(retry) == 0 || !strings.Contains(retry[len(retry)-1].Text(), "try X") {
		t.Errorf("expected the retry to re-attempt the held conversation, messages: %v", retry)
	}
}

func TestAgentsResumeCarriesLabel(t *testing.T) {
	// A resumed task is the same undertaking, so the caller-chosen label
	// follows the handle onto the continuation's result.
	g := newTestGenkit(t)
	genkitx.DefineAgent[any](g, "flaky",
		aix.InlinePrompt{ai.WithModel(failNTimesModel(t, g, "test/flaky-label", 1, "recovered", nil))},
		aix.WithSessionStore[any](localstore.NewInMemorySessionStore[any]()),
	)
	orch := toolModel(t, g, "test/orch-label2", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		if _, ok := lastDelegationOutput(req.Messages, "resume_subagent"); ok {
			return textResp(req, "done"), nil
		}
		if res, ok := lastDelegationOutput(req.Messages, "delegate_to_flaky"); ok {
			return toolReqResp(req, &ai.ToolRequest{Name: "resume_subagent", Input: map[string]any{"taskId": res.TaskID}}), nil
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
	resumes := delegationResponses(t, resp.History(), "resume_subagent")
	if len(resumes) != 1 || resumes[0].Response != "recovered" || resumes[0].Name != "second-try" {
		t.Fatalf("expected the label to follow the resume, got %+v", resumes)
	}
}

func TestAgentsResumeUnknownMemHandleRefused(t *testing.T) {
	// An in-memory handle from another call (or a made-up one) has no stash
	// behind it; the refusal explains the handle's lifetime and the durable
	// alternative.
	g := newTestGenkit(t)

	genkitx.DefineAgent[any](g, "eph",
		aix.InlinePrompt{ai.WithModel(toolModel(t, g, "test/eph", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			return textResp(req, "unused"), nil
		}))},
	)

	orch := toolModel(t, g, "test/orch", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		if _, ok := lastDelegationOutput(req.Messages, "resume_subagent"); ok {
			return textResp(req, "done"), nil
		}
		return toolReqResp(req, &ai.ToolRequest{Name: "resume_subagent", Input: map[string]any{"taskId": "eph:mem-99"}}), nil
	})
	mw := &Agents{Agents: []aix.AgentRef{{Name: "eph"}}}

	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("go"), ai.WithUse(mw))
	if err != nil {
		t.Fatal(err)
	}
	resumes := delegationResponses(t, resp.History(), "resume_subagent")
	if len(resumes) != 1 || !strings.Contains(resumes[0].Response, "no in-memory state") {
		t.Fatalf("expected a stash-miss refusal, got %+v", resumes)
	}
}

func TestAgentsResumeCountsAgainstCap(t *testing.T) {
	// A resume is a real sub-agent run and spends a delegation slot; with the
	// cap exhausted by the delegation itself, the resume is refused.
	g := newTestGenkit(t)

	genkitx.DefineAgent[any](g, "flaky",
		aix.InlinePrompt{ai.WithModel(failNTimesModel(t, g, "test/flaky", 99, "", nil))},
		aix.WithSessionStore[any](localstore.NewInMemorySessionStore[any]()),
	)

	orch := delegateThenResumeModel(t, g, "test/orch", "delegate_to_flaky", "resume_subagent", "try X", "")
	mw := &Agents{Agents: []aix.AgentRef{{Name: "flaky"}}, MaxDelegations: 1}

	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("go"), ai.WithUse(mw))
	if err != nil {
		t.Fatal(err)
	}
	resumes := delegationResponses(t, resp.History(), "resume_subagent")
	if len(resumes) != 1 || !strings.Contains(resumes[0].Response, "Delegation limit reached") {
		t.Fatalf("expected the resume to be refused by the cap, got %+v", resumes)
	}
}

func TestAgentsResumeExpiredRecoversCommittedProgress(t *testing.T) {
	// An expired handle (a dead worker's pending row) is fenced with an abort
	// and resumed from the session's latest committed snapshot, recovering
	// whatever the run persisted before it detached.
	g := newTestGenkit(t)

	store := localstore.NewInMemorySessionStore[any]()
	var seen [][]*ai.Message
	genkitx.DefineAgent[any](g, "keeper",
		aix.InlinePrompt{ai.WithModel(failNTimesModel(t, g, "test/keeper", 0, "kept going", &seen))},
		aix.WithSessionStore[any](store),
	)

	// A committed conversation to recover, then a dead worker's pending row
	// as the session tip.
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
	committedID := strings.TrimPrefix(seeded[0].TaskID, "keeper:")
	committed, err := store.GetSnapshot(ctx, committedID)
	if err != nil || committed == nil {
		t.Fatalf("read committed snapshot %q: %v", committedID, err)
	}
	pending, err := saveDeadPendingRow(store, committed.SessionID, committed.SnapshotID)
	if err != nil {
		t.Fatalf("SaveSnapshot pending row: %v", err)
	}

	deadTask := "keeper:" + pending.SnapshotID
	orch := toolModel(t, g, "test/orch", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		if res, ok := lastDelegationOutput(req.Messages, "resume_subagent"); ok {
			return textResp(req, "done: "+res.Response), nil
		}
		return toolReqResp(req, &ai.ToolRequest{Name: "resume_subagent",
			Input: map[string]any{"taskId": deadTask, "instructions": "continue"}}), nil
	})
	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("go"),
		ai.WithUse(&Agents{Agents: []aix.AgentRef{{Name: "keeper"}}}))
	if err != nil {
		t.Fatal(err)
	}
	resumes := delegationResponses(t, resp.History(), "resume_subagent")
	if len(resumes) != 1 || resumes[0].Response != "kept going" {
		t.Fatalf("expected the expired task to resume from committed progress, got %+v", resumes)
	}

	// The fence flipped the dead pending row so a slow worker cannot race the
	// recovered session.
	fenced, err := store.GetSnapshot(ctx, pending.SnapshotID)
	if err != nil || fenced == nil {
		t.Fatalf("read fenced row: %v", err)
	}
	if fenced.Status != aix.SnapshotStatusAborted {
		t.Errorf("fenced row status = %q, want %q", fenced.Status, aix.SnapshotStatusAborted)
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

func TestAgentsResumeExpiredWithNothingSavedRefused(t *testing.T) {
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
		if _, ok := lastDelegationOutput(req.Messages, "resume_subagent"); ok {
			return textResp(req, "done"), nil
		}
		return toolReqResp(req, &ai.ToolRequest{Name: "resume_subagent", Input: map[string]any{"taskId": deadTask}}), nil
	})
	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("go"),
		ai.WithUse(&Agents{Agents: []aix.AgentRef{{Name: "keeper"}}}))
	if err != nil {
		t.Fatal(err)
	}
	resumes := delegationResponses(t, resp.History(), "resume_subagent")
	if len(resumes) != 1 {
		t.Fatalf("expected 1 resume response, got %d", len(resumes))
	}
	if !strings.Contains(resumes[0].Response, "saved no resumable progress") ||
		!strings.Contains(resumes[0].Response, "Delegate the task again") {
		t.Errorf("expected an honest nothing-saved refusal, got %q", resumes[0].Response)
	}
}

// saveDeadPendingRow writes a dead worker's pending row the way the detach
// handler mints one: created now (newer than every committed row in the
// session) with a heartbeat that went stale.
func saveDeadPendingRow(store *localstore.InMemorySessionStore[any], sessionID, parentID string) (*aix.SessionSnapshot[any], error) {
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

func TestAgentsResumeExpiredFinishedParentRequiresInstructions(t *testing.T) {
	// The parent behind a dead task can be a finished turn; continuing past
	// it gets the same instructions gate as a completed task, since an empty
	// input would re-run the finished turn instead of continuing the work.
	g := newTestGenkit(t)

	store := localstore.NewInMemorySessionStore[any]()
	genkitx.DefineAgent[any](g, "keeper",
		aix.InlinePrompt{ai.WithModel(failNTimesModel(t, g, "test/keeper", 0, "kept", nil))},
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
	committedID := strings.TrimPrefix(seeded[0].TaskID, "keeper:")
	committed, err := store.GetSnapshot(ctx, committedID)
	if err != nil || committed == nil {
		t.Fatalf("read committed snapshot %q: %v", committedID, err)
	}
	pending, err := saveDeadPendingRow(store, committed.SessionID, committed.SnapshotID)
	if err != nil {
		t.Fatalf("SaveSnapshot pending row: %v", err)
	}

	deadTask := "keeper:" + pending.SnapshotID
	orch := toolModel(t, g, "test/orch", func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
		if _, ok := lastDelegationOutput(req.Messages, "resume_subagent"); ok {
			return textResp(req, "done"), nil
		}
		return toolReqResp(req, &ai.ToolRequest{Name: "resume_subagent", Input: map[string]any{"taskId": deadTask}}), nil
	})
	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("go"),
		ai.WithUse(&Agents{Agents: []aix.AgentRef{{Name: "keeper"}}}))
	if err != nil {
		t.Fatal(err)
	}
	resumes := delegationResponses(t, resp.History(), "resume_subagent")
	if len(resumes) != 1 || !strings.Contains(resumes[0].Response, "last finished turn") {
		t.Fatalf("expected the finished-parent instructions gate, got %+v", resumes)
	}
}

func TestAgentsResumeInBackground(t *testing.T) {
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
		if res, ok := lastDelegationOutput(req.Messages, "resume_subagent"); ok {
			return toolReqResp(req, &ai.ToolRequest{Name: waitBackgroundTasksToolName,
				Input: map[string]any{"taskIds": []string{res.TaskID}}}), nil
		}
		if res, ok := lastDelegationOutput(req.Messages, "delegate_to_flaky"); ok {
			return toolReqResp(req, &ai.ToolRequest{Name: "resume_subagent",
				Input: map[string]any{"taskId": res.TaskID, "background": true}}), nil
		}
		return toolReqResp(req, &ai.ToolRequest{Name: "delegate_to_flaky", Input: map[string]any{"task": "try X"}}), nil
	})
	mw := &Agents{Agents: []aix.AgentRef{{Name: "flaky"}}, Async: true}

	resp, err := genkit.Generate(ctx, g, ai.WithModel(orch), ai.WithPrompt("go"), ai.WithUse(mw))
	if err != nil {
		t.Fatal(err)
	}
	resumes := delegationResponses(t, resp.History(), "resume_subagent")
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
