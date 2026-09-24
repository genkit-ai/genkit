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
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
)

// defineEchoAgent registers a client-managed custom agent that answers each
// turn with "echo:<text> n:<history length>", so tests can observe both the
// turn input and any history seeded through init state.
func defineEchoAgent(t *testing.T, reg api.Registry, name string) *Agent[any] {
	t.Helper()
	return DefineCustomAgent(reg, name,
		func(ctx context.Context, resp Responder, sess *SessionRunner[any]) (*AgentResult, error) {
			if err := sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				sess.AddMessages(ai.NewModelTextMessage(
					fmt.Sprintf("echo:%s n:%d", input.Message.Text(), len(sess.Messages()))))
				return nil, nil
			}); err != nil {
				return nil, err
			}
			return sess.Result(), nil
		})
}

// defineGatedAgent registers a server-managed agent whose single turn parks
// between two channels: it signals entered, waits for release, then commits a
// model message and custom state. It is the background-work stand-in for the
// detach lifecycle tests.
func defineGatedAgent(t *testing.T, reg api.Registry, name string, store *testInMemStore[testState]) (agent *Agent[testState], entered, release chan struct{}) {
	t.Helper()
	entered = make(chan struct{})
	release = make(chan struct{})
	agent = DefineCustomAgent(reg, name,
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				select {
				case entered <- struct{}{}:
				case <-ctx.Done():
					return nil, ctx.Err()
				}
				select {
				case <-release:
				case <-ctx.Done():
					return nil, ctx.Err()
				}
				sess.AddMessages(ai.NewModelTextMessage("finished"))
				sess.UpdateCustom(func(s testState) testState {
					s.Counter = 42
					return s
				})
				return nil, nil
			})
		},
		WithSessionStore(store),
	)
	return agent, entered, release
}

func TestLookupAgent(t *testing.T) {
	t.Run("resolves a registered agent with its companions", func(t *testing.T) {
		reg := newTestRegistry(t)
		store := newTestInMemStore[testState]()
		defineGatedAgent(t, reg, "researcher", store)

		h := LookupAgent(reg, "researcher")
		if got := h.Name(); got != "researcher" {
			t.Errorf("Name() = %q, want %q", got, "researcher")
		}
		meta := h.Metadata()
		if meta == nil {
			t.Fatal("Metadata() = nil, want agent metadata")
		}
		if meta.StateManagement != AgentStateManagementServer {
			t.Errorf("StateManagement = %q, want %q", meta.StateManagement, AgentStateManagementServer)
		}
		if !meta.Abortable {
			t.Error("Abortable = false, want true (store implements SnapshotSubscriber)")
		}
	})

	// Both misses answer nil, as every other Lookup in the framework does. The
	// caller knows the name it asked for, so telling the two apart would add a
	// distinction with one remedy.
	t.Run("misses report nil", func(t *testing.T) {
		reg := newTestRegistry(t)
		if h := LookupAgent(reg, "ghost"); h != nil {
			t.Errorf("LookupAgent(unregistered) = %+v, want nil", h)
		}

		core.NewActionOf(api.ActionTypeAgent, "impostor", nil,
			func(ctx context.Context, in string) (string, error) { return in, nil },
		).Register(reg)
		if h := LookupAgent(reg, "impostor"); h != nil {
			t.Errorf("LookupAgent(non-agent action) = %+v, want nil", h)
		}
	})

	// A nil handle names its own cause rather than panicking, which is what a
	// caller that skipped the nil check actually needs to read.
	t.Run("nil handle reports itself", func(t *testing.T) {
		var h *AgentHandle
		_, err := h.Run(context.Background(), &AgentInput{Message: ai.NewUserTextMessage("hi")})
		if !errors.Is(err, status.ErrInvalidArgument) || !strings.Contains(err.Error(), "nil handle") {
			t.Errorf("Run on a nil handle = %v, want INVALID_ARGUMENT naming the nil handle", err)
		}
		if _, err := h.RunText(context.Background(), "hi"); !errors.Is(err, status.ErrInvalidArgument) {
			t.Errorf("RunText on a nil handle = %v, want INVALID_ARGUMENT", err)
		}
		if _, err := h.RunDetached(context.Background(), &AgentInput{}); !errors.Is(err, status.ErrInvalidArgument) {
			t.Errorf("RunDetached on a nil handle = %v, want INVALID_ARGUMENT", err)
		}
	})
}

// fakeDescAction is an api.Action stub carrying only a descriptor, for
// agentMetadataOf's decode paths. The embedded nil interface covers the
// methods agentMetadataOf never calls.
type fakeDescAction struct {
	api.Action
	desc api.ActionDesc
}

func (f fakeDescAction) Desc() api.ActionDesc { return f.desc }

func TestAgentMetadataOf(t *testing.T) {
	t.Run("nil action", func(t *testing.T) {
		if got := agentMetadataOf(nil); got != nil {
			t.Fatalf("agentMetadataOf(nil) = %+v, want nil", got)
		}
	})

	t.Run("typed value from a live agent", func(t *testing.T) {
		reg := newTestRegistry(t)
		agent := defineEchoAgent(t, reg, "typed")
		meta := agentMetadataOf(agent)
		if meta == nil {
			t.Fatal("agentMetadataOf = nil, want metadata")
		}
		if meta.StateManagement != AgentStateManagementClient {
			t.Errorf("StateManagement = %q, want %q", meta.StateManagement, AgentStateManagementClient)
		}
		if meta.Abortable {
			t.Error("Abortable = true, want false (no store)")
		}
	})

	t.Run("pointer form", func(t *testing.T) {
		a := fakeDescAction{desc: api.ActionDesc{Metadata: map[string]any{
			"agent": &AgentMetadata{StateManagement: AgentStateManagementServer, Abortable: true},
		}}}
		meta := agentMetadataOf(a)
		if meta == nil || meta.StateManagement != AgentStateManagementServer || !meta.Abortable {
			t.Fatalf("agentMetadataOf = %+v, want server-managed abortable", meta)
		}
	})

	t.Run("map form as decoded from a JSON descriptor", func(t *testing.T) {
		// Round-trip a real agent's metadata through JSON so the map shape is
		// exactly what a serialized descriptor yields.
		reg := newTestRegistry(t)
		agent := defineEchoAgent(t, reg, "wire")
		b, err := json.Marshal(agent.Desc().Metadata)
		if err != nil {
			t.Fatalf("marshal metadata: %v", err)
		}
		var decoded map[string]any
		if err := json.Unmarshal(b, &decoded); err != nil {
			t.Fatalf("unmarshal metadata: %v", err)
		}
		meta := agentMetadataOf(fakeDescAction{desc: api.ActionDesc{Metadata: decoded}})
		if meta == nil {
			t.Fatal("agentMetadataOf = nil, want metadata decoded from map")
		}
		if meta.StateManagement != AgentStateManagementClient {
			t.Errorf("StateManagement = %q, want %q", meta.StateManagement, AgentStateManagementClient)
		}
	})

	t.Run("map form that does not decode reports nothing, not a partial", func(t *testing.T) {
		// Every field here is a capability a caller gates on, so a partial
		// decode is worse than none: the mistyped abortable below would read
		// as a definite "cannot run in the background" and get the agent
		// refused work it can do. Callers treat nil as "unknown" and ask the
		// runtime instead.
		a := fakeDescAction{desc: api.ActionDesc{Metadata: map[string]any{
			"agent": map[string]any{"stateManagement": "client", "abortable": "true"},
		}}}
		if meta := agentMetadataOf(a); meta != nil {
			t.Fatalf("agentMetadataOf = %+v, want nil for a descriptor that did not decode", meta)
		}
	})

	t.Run("returned metadata does not alias the descriptor", func(t *testing.T) {
		// The copy claim has to cover StateSchema too: it is a map, so a
		// struct copy alone shares it with every other reader of a descriptor
		// documented as immutable.
		desc := api.ActionDesc{Metadata: map[string]any{
			"agent": AgentMetadata{StateSchema: map[string]any{"type": "object"}},
		}}
		meta := agentMetadataOf(fakeDescAction{desc: desc})
		if meta == nil {
			t.Fatal("agentMetadataOf = nil, want typed metadata")
		}
		meta.StateSchema["type"] = "tampered"
		original := desc.Metadata["agent"].(AgentMetadata)
		if got := original.StateSchema["type"]; got != "object" {
			t.Errorf("descriptor schema type = %v, want %q untouched", got, "object")
		}
	})

	t.Run("action without agent metadata", func(t *testing.T) {
		a := fakeDescAction{desc: api.ActionDesc{Metadata: map[string]any{"other": true}}}
		if got := agentMetadataOf(a); got != nil {
			t.Fatalf("agentMetadataOf = %+v, want nil", got)
		}
	})
}

func TestAgentHandle_Run(t *testing.T) {
	reg := newTestRegistry(t)
	defineEchoAgent(t, reg, "echo")
	h := LookupAgent(reg, "echo")

	t.Run("plain turn", func(t *testing.T) {
		out, err := h.Run(context.Background(), &AgentInput{Message: ai.NewUserTextMessage("hi")})
		if err != nil {
			t.Fatalf("Run: %v", err)
		}
		// History at respond time is the user message plus nothing else.
		if got, want := out.Message.Text(), "echo:hi n:1"; got != want {
			t.Errorf("Message.Text() = %q, want %q", got, want)
		}
		if out.State == nil || out.State.SessionID == "" {
			t.Errorf("client-managed output should carry state with a session ID, got %+v", out.State)
		}
	})

	t.Run("options seed init state", func(t *testing.T) {
		seeded := &SessionState[json.RawMessage]{Messages: []*ai.Message{
			ai.NewUserTextMessage("earlier question"),
			ai.NewModelTextMessage("earlier answer"),
		}}
		out, err := h.Run(context.Background(),
			&AgentInput{Message: ai.NewUserTextMessage("again")},
			WithState(seeded))
		if err != nil {
			t.Fatalf("Run with WithState: %v", err)
		}
		// Two seeded messages plus this turn's user message.
		if got, want := out.Message.Text(), "echo:again n:3"; got != want {
			t.Errorf("Message.Text() = %q, want %q", got, want)
		}
	})

	t.Run("RunText delivers a user text message", func(t *testing.T) {
		out, err := h.RunText(context.Background(), "hi")
		if err != nil {
			t.Fatalf("RunText: %v", err)
		}
		if got, want := out.Message.Text(), "echo:hi n:1"; got != want {
			t.Errorf("Message.Text() = %q, want %q", got, want)
		}
		out, err = h.RunText(context.Background(), "again",
			WithState(&SessionState[json.RawMessage]{Messages: []*ai.Message{
				ai.NewUserTextMessage("earlier question"),
				ai.NewModelTextMessage("earlier answer"),
			}}))
		if err != nil {
			t.Fatalf("RunText with WithState: %v", err)
		}
		if got, want := out.Message.Text(), "echo:again n:3"; got != want {
			t.Errorf("Message.Text() = %q, want %q", got, want)
		}
	})

	t.Run("rejects nil input", func(t *testing.T) {
		_, err := h.Run(context.Background(), nil)
		if !errors.Is(err, status.ErrInvalidArgument) {
			t.Fatalf("Run(nil) error = %v, want INVALID_ARGUMENT", err)
		}
	})

	t.Run("rejects duplicate options", func(t *testing.T) {
		_, err := h.Run(context.Background(), &AgentInput{Message: ai.NewUserTextMessage("x")},
			WithSessionID[json.RawMessage]("a"), WithSessionID[json.RawMessage]("b"))
		if err == nil || !strings.Contains(err.Error(), "more than once") {
			t.Fatalf("duplicate WithSessionID error = %v, want duplicate-option rejection", err)
		}
	})

	t.Run("rejects mutually exclusive options", func(t *testing.T) {
		_, err := h.Run(context.Background(), &AgentInput{Message: ai.NewUserTextMessage("x")},
			WithState(&SessionState[json.RawMessage]{}), WithSessionID[json.RawMessage]("a"))
		if err == nil || !strings.Contains(err.Error(), "mutually exclusive") {
			t.Fatalf("WithState+WithSessionID error = %v, want mutual-exclusion rejection", err)
		}
	})
}

func TestAgentHandle_RunDetachedPollWaitRehydrate(t *testing.T) {
	// Full background lifecycle through the handle: launch, observe pending,
	// rehydrate the task from nothing but its snapshot ID, and wait for the
	// finalized snapshot.
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()
	_, entered, release := defineGatedAgent(t, reg, "worker", store)

	h := LookupAgent(reg, "worker")

	task, err := h.RunDetached(context.Background(), &AgentInput{Message: ai.NewUserTextMessage("go")})
	if err != nil {
		t.Fatalf("RunDetached: %v", err)
	}
	if task.SnapshotID() == "" {
		t.Fatal("RunDetached returned a task with no snapshot ID")
	}

	select {
	case <-entered:
	case <-time.After(2 * time.Second):
		t.Fatal("background work did not start")
	}

	snap, err := task.Poll(context.Background())
	if err != nil {
		t.Fatalf("Poll: %v", err)
	}
	if snap.Status != SnapshotStatusPending {
		t.Fatalf("Poll status = %q, want %q", snap.Status, SnapshotStatusPending)
	}
	if snap.Status.Terminal() {
		t.Error("pending status reported as terminal")
	}

	// Rehydrate from the recorded ID alone, as a later process would.
	h2 := LookupAgent(reg, "worker")
	rehydrated := h2.Task(task.SnapshotID())

	close(release)
	final, err := rehydrated.Wait(context.Background())
	if err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if final.Status != SnapshotStatusCompleted {
		t.Fatalf("Wait status = %q, want %q", final.Status, SnapshotStatusCompleted)
	}
	if final.State == nil {
		t.Fatal("finalized snapshot carries no state")
	}
	var custom testState
	if err := json.Unmarshal(final.State.Custom, &custom); err != nil {
		t.Fatalf("unmarshal custom state: %v", err)
	}
	if custom.Counter != 42 {
		t.Errorf("custom counter = %d, want 42", custom.Counter)
	}
	if got, want := tipText(t, final.State), "finished"; got != want {
		t.Errorf("final message = %q, want %q", got, want)
	}
}

func TestAgentHandle_RunDetachedRejected(t *testing.T) {
	t.Run("client-managed agent cannot detach", func(t *testing.T) {
		reg := newTestRegistry(t)
		defineEchoAgent(t, reg, "storeless")
		h := LookupAgent(reg, "storeless")
		_, err := h.RunDetached(context.Background(), &AgentInput{Message: ai.NewUserTextMessage("go")})
		if err == nil {
			t.Fatal("RunDetached succeeded on a storeless agent, want rejection")
		}
		// The rejection is decoded from the invocation's failed output, so only
		// the status name survives; match by status, not sentinel.
		if got := status.Of(err); got != status.FailedPrecondition {
			t.Fatalf("status.Of(err) = %v, want FAILED_PRECONDITION (err: %v)", got, err)
		}
	})

	t.Run("store without subscriber cannot abort", func(t *testing.T) {
		reg := newTestRegistry(t)
		DefineCustomAgent(reg, "unabortable",
			func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
				return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
					return nil, nil
				})
			},
			WithSessionStore[testState](minimalStore[testState]{}),
		)
		h := LookupAgent(reg, "unabortable")
		if meta := h.Metadata(); meta == nil || meta.Abortable {
			t.Errorf("Metadata() = %+v, want non-abortable", meta)
		}
		_, err := h.Abort(context.Background(), "some-id")
		if !errors.Is(err, status.ErrFailedPrecondition) || !strings.Contains(err.Error(), "SnapshotSubscriber") {
			t.Fatalf("Abort error = %v, want FAILED_PRECONDITION naming SnapshotSubscriber", err)
		}
	})
}

func TestAgentHandle_AbortLifecycle(t *testing.T) {
	// Launch through the typed bridge (Agent.Handle), abort the task, and
	// observe the aborted snapshot through Wait; a second abort is a no-op
	// reporting the settled status.
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()
	agent, entered, _ := defineGatedAgent(t, reg, "abortable", store)

	h := agent.Handle()
	task, err := h.RunDetached(context.Background(), &AgentInput{Message: ai.NewUserTextMessage("go")})
	if err != nil {
		t.Fatalf("RunDetached: %v", err)
	}
	select {
	case <-entered:
	case <-time.After(2 * time.Second):
		t.Fatal("background work did not start")
	}

	got, err := task.Abort(context.Background())
	if err != nil {
		t.Fatalf("Abort: %v", err)
	}
	if got != SnapshotStatusAborting {
		t.Fatalf("Abort status = %q, want %q: the stop landed and the row settles on the finalize", got, SnapshotStatusAborting)
	}

	final, err := task.Wait(context.Background())
	if err != nil {
		t.Fatalf("Wait after abort: %v", err)
	}
	if final.Status != SnapshotStatusAborted {
		t.Fatalf("Wait status = %q, want %q", final.Status, SnapshotStatusAborted)
	}

	// Aborting a settled task reports the existing terminal status.
	again, err := task.Abort(context.Background())
	if err != nil {
		t.Fatalf("second Abort: %v", err)
	}
	if again != SnapshotStatusAborted {
		t.Fatalf("second Abort status = %q, want %q", again, SnapshotStatusAborted)
	}

	// Aborting an unknown snapshot is NOT_FOUND, matching the companion
	// action; the in-process chain keeps the sentinel matchable.
	if _, err := h.Abort(context.Background(), "does-not-exist"); !errors.Is(err, ErrSnapshotNotFound) {
		t.Fatalf("Abort(unknown) error = %v, want ErrSnapshotNotFound", err)
	}
}

func TestAgentHandle_GetSnapshotMetadataOnly(t *testing.T) {
	// The metadata read shapes exactly as the full read and drops only the
	// state payload: a settled row reports its status with a nil State, and a
	// row inside the abort wind-down window reads as aborting through both.
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()
	af := defineLastGoodTestAgent(reg, "metaRead", WithSessionStore(store))

	out, err := af.RunText(context.Background(), "first")
	if err != nil {
		t.Fatalf("RunText: %v", err)
	}
	h := LookupAgent(reg, "metaRead")

	full, err := h.GetSnapshot(context.Background(), out.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}
	if full.State == nil {
		t.Fatal("full read returned no state")
	}
	meta, err := h.GetSnapshot(context.Background(), out.SnapshotID, WithMetadataOnly())
	if err != nil {
		t.Fatalf("GetSnapshot(WithMetadataOnly): %v", err)
	}
	if meta.State != nil {
		t.Errorf("meta read returned state: %+v", meta.State)
	}
	if meta.Status != full.Status || meta.SessionID != full.SessionID || meta.FinishReason != full.FinishReason {
		t.Errorf("meta read shaped differently from the full read: meta=%+v full=%+v", meta, full)
	}

	// The option rides every read surface: latest-by-session, and a task poll.
	latestMeta, err := h.GetLatestSnapshot(context.Background(), out.SessionID, WithMetadataOnly())
	if err != nil {
		t.Fatalf("GetLatestSnapshot(WithMetadataOnly): %v", err)
	}
	if latestMeta.State != nil {
		t.Errorf("latest meta read returned state: %+v", latestMeta.State)
	}
	polled, err := h.Task(out.SnapshotID).Poll(context.Background(), WithMetadataOnly())
	if err != nil {
		t.Fatalf("Poll(WithMetadataOnly): %v", err)
	}
	if polled.State != nil {
		t.Errorf("meta poll returned state: %+v", polled.State)
	}

	// Shaping needs only the metadata, so a winding-down row tells the same
	// status story through both reads: aborting while its beat is live, and
	// expired once the beat is stale.
	beat := time.Now()
	saveAborting := func(beat time.Time) string {
		t.Helper()
		row, err := store.SaveSnapshot(context.Background(), "",
			func(_ *SessionSnapshot[testState]) (*SessionSnapshot[testState], error) {
				return &SessionSnapshot[testState]{
					SessionID:   "sess-window",
					Status:      SnapshotStatusAborting,
					CreatedAt:   beat,
					UpdatedAt:   beat,
					HeartbeatAt: &beat,
				}, nil
			})
		if err != nil {
			t.Fatalf("SaveSnapshot aborting row: %v", err)
		}
		return row.SnapshotID
	}
	for _, tc := range []struct {
		name string
		beat time.Time
		want SnapshotStatus
	}{
		{"live beat", beat, SnapshotStatusAborting},
		{"stale beat", beat.Add(-2 * defaultHeartbeatTimeout), SnapshotStatusExpired},
	} {
		id := saveAborting(tc.beat)
		metaRow, err := h.GetSnapshot(context.Background(), id, WithMetadataOnly())
		if err != nil {
			t.Fatalf("GetSnapshot(%s, WithMetadataOnly): %v", tc.name, err)
		}
		fullRow, err := h.GetSnapshot(context.Background(), id)
		if err != nil {
			t.Fatalf("GetSnapshot(%s): %v", tc.name, err)
		}
		if metaRow.Status != tc.want || fullRow.Status != tc.want {
			t.Errorf("%s: meta status = %q, full status = %q, want both %q", tc.name, metaRow.Status, fullRow.Status, tc.want)
		}
	}
}

func TestAgentHandle_SnapshotReadErrors(t *testing.T) {
	t.Run("no session store", func(t *testing.T) {
		reg := newTestRegistry(t)
		defineEchoAgent(t, reg, "bare")
		h := LookupAgent(reg, "bare")
		if _, err := h.GetSnapshot(context.Background(), "any"); !errors.Is(err, ErrSessionStoreNotConfigured) {
			t.Fatalf("GetSnapshot error = %v, want ErrSessionStoreNotConfigured", err)
		}
		if _, err := h.GetLatestSnapshot(context.Background(), "sess"); !errors.Is(err, ErrSessionStoreNotConfigured) {
			t.Fatalf("GetLatestSnapshot error = %v, want ErrSessionStoreNotConfigured", err)
		}
		if _, err := h.Abort(context.Background(), "any"); !errors.Is(err, ErrSessionStoreNotConfigured) {
			t.Fatalf("Abort error = %v, want ErrSessionStoreNotConfigured", err)
		}
	})

	t.Run("empty IDs are invalid", func(t *testing.T) {
		reg := newTestRegistry(t)
		store := newTestInMemStore[testState]()
		defineGatedAgent(t, reg, "ids", store)
		h := LookupAgent(reg, "ids")
		if _, err := h.GetSnapshot(context.Background(), ""); !errors.Is(err, status.ErrInvalidArgument) {
			t.Fatalf("GetSnapshot(\"\") error = %v, want INVALID_ARGUMENT", err)
		}
		if _, err := h.GetLatestSnapshot(context.Background(), ""); !errors.Is(err, status.ErrInvalidArgument) {
			t.Fatalf("GetLatestSnapshot(\"\") error = %v, want INVALID_ARGUMENT", err)
		}
		if _, err := h.Abort(context.Background(), ""); !errors.Is(err, status.ErrInvalidArgument) {
			t.Fatalf("Abort(\"\") error = %v, want INVALID_ARGUMENT", err)
		}
	})

	t.Run("missing snapshot is NOT_FOUND with a live sentinel", func(t *testing.T) {
		reg := newTestRegistry(t)
		store := newTestInMemStore[testState]()
		defineGatedAgent(t, reg, "misses", store)
		h := LookupAgent(reg, "misses")
		if _, err := h.GetSnapshot(context.Background(), "nope"); !errors.Is(err, ErrSnapshotNotFound) {
			t.Fatalf("GetSnapshot(missing) error = %v, want ErrSnapshotNotFound", err)
		}
		if _, err := h.Task("nope").Poll(context.Background()); !errors.Is(err, status.ErrNotFound) {
			t.Fatalf("Poll(missing) error = %v, want NOT_FOUND", err)
		}
	})
}

func TestAgentHandle_RunDetachedSettledSynchronously(t *testing.T) {
	// An agent whose fn returns without consuming its input can settle the
	// invocation before the runtime observes the detach directive. RunDetached
	// must treat that as a first-class outcome: a task over the committed
	// snapshot when one exists, or FAILED_PRECONDITION when nothing was
	// recorded; never an INTERNAL contract-violation error. The race is
	// scheduling-dependent, so accept either outcome on each iteration and pin
	// only the contract.
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()
	agent := DefineCustomAgent(reg, "eager",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, nil // settles immediately, no turn, no snapshot
		},
		WithSessionStore(store),
	)
	h := agent.Handle()
	for i := 0; i < 20; i++ {
		task, err := h.RunDetached(context.Background(), &AgentInput{Message: ai.NewUserTextMessage("go")})
		if err == nil {
			// The detach won the race; the launch is a normal task.
			if task == nil || task.SnapshotID() == "" {
				t.Fatalf("iteration %d: nil or empty task without error", i)
			}
			continue
		}
		if !errors.Is(err, status.ErrFailedPrecondition) || !strings.Contains(err.Error(), "settled synchronously") {
			t.Fatalf("iteration %d: err = %v, want FAILED_PRECONDITION about synchronous settlement", i, err)
		}
	}
}

func TestWaitValidation(t *testing.T) {
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()
	defineGatedAgent(t, reg, "waiter", store)
	h := LookupAgent(reg, "waiter")

	// An unknown snapshot has nothing to wait on, so the wait fails the way
	// the read does rather than blocking until ctx ends.
	if _, err := h.Task("no-such-snapshot").Wait(context.Background()); !errors.Is(err, ErrSnapshotNotFound) {
		t.Fatalf("Wait on unknown snapshot error = %v, want ErrSnapshotNotFound", err)
	}
	if _, err := h.WaitForSnapshot(context.Background(), ""); !errors.Is(err, status.ErrInvalidArgument) {
		t.Fatalf("WaitForSnapshot(\"\") error = %v, want INVALID_ARGUMENT", err)
	}
}

// recordingTransport is an [agentTransport] that records what reached it and
// answers from canned values. It is how the delegation tests see the seam: a
// real transport would prove only that the call worked, not which arguments
// the handle chose to send.
type recordingTransport struct {
	runInput  *AgentInput
	runInit   *AgentInit[json.RawMessage]
	runHasCB  bool
	lookup    *GetSnapshotRequest
	waitID    string
	abortID   string
	callCount int
}

func (t *recordingTransport) Run(ctx context.Context, input *AgentInput, init *AgentInit[json.RawMessage], cb func(context.Context, json.RawMessage) error) (*AgentOutput[json.RawMessage], error) {
	t.callCount++
	t.runInput, t.runInit, t.runHasCB = input, init, cb != nil
	return &AgentOutput[json.RawMessage]{FinishReason: AgentFinishReasonStop}, nil
}

func (t *recordingTransport) GetSnapshot(ctx context.Context, lookup *GetSnapshotRequest) (*SessionSnapshot[json.RawMessage], error) {
	t.callCount++
	t.lookup = lookup
	return &SessionSnapshot[json.RawMessage]{Status: SnapshotStatusCompleted}, nil
}

func (t *recordingTransport) WaitForSnapshot(ctx context.Context, snapshotID string) (*SessionSnapshot[json.RawMessage], error) {
	t.callCount++
	t.waitID = snapshotID
	return &SessionSnapshot[json.RawMessage]{Status: SnapshotStatusCompleted}, nil
}

func (t *recordingTransport) Abort(ctx context.Context, snapshotID string) (SnapshotStatus, error) {
	t.callCount++
	t.abortID = snapshotID
	return SnapshotStatusAborted, nil
}

// TestAgentHandle_DelegatesToTransport pins the division of labour the seam
// exists to create. The handle resolves options and validates arguments; the
// transport receives a request that is already well formed and decides nothing
// about it. A remote transport can only be a drop-in if that line holds.
func TestAgentHandle_DelegatesToTransport(t *testing.T) {
	tr := &recordingTransport{}
	h := &AgentHandle{name: "researcher", transport: tr}
	ctx := context.Background()

	// One transport read serves both lookups, which is why it takes the
	// request rather than splitting into two methods.
	if _, err := h.GetSnapshot(ctx, "snap-1"); err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}
	if tr.lookup.SnapshotID != "snap-1" || tr.lookup.SessionID != "" {
		t.Errorf("GetSnapshot sent %+v, want a snapshot-ID lookup", tr.lookup)
	}
	if _, err := h.GetLatestSnapshot(ctx, "sess-1"); err != nil {
		t.Fatalf("GetLatestSnapshot: %v", err)
	}
	if tr.lookup.SessionID != "sess-1" || tr.lookup.SnapshotID != "" {
		t.Errorf("GetLatestSnapshot sent %+v, want a session-ID lookup", tr.lookup)
	}

	if _, err := h.WaitForSnapshot(ctx, "snap-2"); err != nil {
		t.Fatalf("WaitForSnapshot: %v", err)
	}
	if tr.waitID != "snap-2" {
		t.Errorf("WaitForSnapshot sent %q, want %q", tr.waitID, "snap-2")
	}

	if _, err := h.Abort(ctx, "snap-3"); err != nil {
		t.Fatalf("Abort: %v", err)
	}
	if tr.abortID != "snap-3" {
		t.Errorf("Abort sent %q, want %q", tr.abortID, "snap-3")
	}

	// Options are the handle's to resolve: a transport sees a settled init,
	// never the option values, so every transport agrees on what they mean.
	if _, err := h.Run(ctx, &AgentInput{Message: ai.NewUserTextMessage("hi")},
		WithSessionID[json.RawMessage]("sess-9")); err != nil {
		t.Fatalf("Run: %v", err)
	}
	if tr.runInit == nil || tr.runInit.SessionID != "sess-9" {
		t.Errorf("Run sent init %+v, want the resolved session ID", tr.runInit)
	}
	if tr.runHasCB {
		t.Error("Run passed a stream callback; the handle reports a turn by its final output")
	}

	// RunDetached sets the detach directive on a copy, whatever the caller
	// set, and leaves the caller's input alone. The canned output settles
	// without a snapshot, so the launch reports that; the directive is what
	// this checks.
	input := &AgentInput{Message: ai.NewUserTextMessage("go")}
	if _, err := h.RunDetached(ctx, input); !errors.Is(err, status.ErrFailedPrecondition) {
		t.Fatalf("RunDetached over a synchronously settled output: err = %v, want FAILED_PRECONDITION", err)
	}
	if tr.runInput == nil || !tr.runInput.Detach {
		t.Errorf("RunDetached sent %+v, want the detach directive set", tr.runInput)
	}
	if input.Detach {
		t.Error("RunDetached mutated the caller's input")
	}

	// Validation is the handle's too, so a bad argument costs no dispatch and
	// fails identically wherever the agent lives.
	before := tr.callCount
	for _, tc := range []struct {
		name string
		call func() error
	}{
		{"nil input", func() error { _, err := h.Run(ctx, nil); return err }},
		{"nil detached input", func() error { _, err := h.RunDetached(ctx, nil); return err }},
		{"empty snapshot ID", func() error { _, err := h.GetSnapshot(ctx, ""); return err }},
		{"empty wait ID", func() error { _, err := h.WaitForSnapshot(ctx, ""); return err }},
		{"empty session ID", func() error { _, err := h.GetLatestSnapshot(ctx, ""); return err }},
		{"empty abort ID", func() error { _, err := h.Abort(ctx, ""); return err }},
	} {
		if err := tc.call(); !errors.Is(err, status.ErrInvalidArgument) {
			t.Errorf("%s: error = %v, want INVALID_ARGUMENT", tc.name, err)
		}
	}
	if tr.callCount != before {
		t.Errorf("%d rejected calls reached the transport", tr.callCount-before)
	}
}

// TestActionTransportForwardsStreamChunks covers the one transport argument
// nothing above reaches: [AgentHandle] passes no stream callback today, so
// only a direct dispatch shows the parameter is wired rather than decorative.
// It is in the interface now because a turn streams at this level whatever
// carries it, and adding it later would reshape the seam.
func TestActionTransportForwardsStreamChunks(t *testing.T) {
	reg := newTestRegistry(t)
	agent := DefineCustomAgent(reg, "streamer",
		func(ctx context.Context, resp Responder, sess *SessionRunner[any]) (*AgentResult, error) {
			if err := sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				for _, word := range []string{"one", "two"} {
					resp.SendModelChunk(&ai.ModelResponseChunk{Content: []*ai.Part{ai.NewTextPart(word)}})
				}
				sess.AddMessages(ai.NewModelTextMessage("done"))
				return nil, nil
			}); err != nil {
				return nil, err
			}
			return sess.Result(), nil
		})

	var mu sync.Mutex
	var streamed []string
	transport := &actionTransport{name: "streamer", run: agent}
	out, err := transport.Run(context.Background(), &AgentInput{Message: ai.NewUserTextMessage("go")}, nil,
		func(ctx context.Context, raw json.RawMessage) error {
			var chunk AgentStreamChunk
			if err := json.Unmarshal(raw, &chunk); err != nil {
				return err
			}
			if chunk.ModelChunk == nil {
				return nil
			}
			mu.Lock()
			defer mu.Unlock()
			streamed = append(streamed, chunk.ModelChunk.Text())
			return nil
		})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.Message.Text() != "done" {
		t.Errorf("Message.Text() = %q, want %q", out.Message.Text(), "done")
	}
	mu.Lock()
	defer mu.Unlock()
	if want := []string{"one", "two"}; !slices.Equal(streamed, want) {
		t.Errorf("streamed chunks = %q, want %q", streamed, want)
	}
}

func TestAgentHandle_MetadataKeepsEagerValue(t *testing.T) {
	// A transport with no api.Action to derive metadata from (the shape a
	// remote handle takes) sets meta at construction and leaves metaSrc nil.
	// The lazy derivation must not run over the nil source and clobber the
	// eager value back to nil.
	meta := &AgentMetadata{StateManagement: AgentStateManagementServer, Abortable: true}
	h := &AgentHandle{name: "remote", meta: meta}
	for i := 0; i < 2; i++ {
		got := h.Metadata()
		if got == nil || got.StateManagement != AgentStateManagementServer || !got.Abortable {
			t.Fatalf("Metadata() call %d = %+v, want the eagerly set metadata kept", i+1, got)
		}
	}
}

// metadataCountingStore layers [SnapshotMetadataReader] on the test store and
// counts which read path answers, so a test can tell the capability from the
// fallback.
type metadataCountingStore struct {
	*testInMemStore[testState]
	fullReads, metadataReads, latestFullReads, latestMetadataReads int
}

func (s *metadataCountingStore) GetSnapshot(ctx context.Context, snapshotID string) (*SessionSnapshot[testState], error) {
	s.fullReads++
	return s.testInMemStore.GetSnapshot(ctx, snapshotID)
}

func (s *metadataCountingStore) GetLatestSnapshot(ctx context.Context, sessionID string) (*SessionSnapshot[testState], error) {
	s.latestFullReads++
	return s.testInMemStore.GetLatestSnapshot(ctx, sessionID)
}

func (s *metadataCountingStore) GetSnapshotMetadata(ctx context.Context, snapshotID string) (*SessionSnapshot[testState], error) {
	s.metadataReads++
	snap, err := s.testInMemStore.GetSnapshot(ctx, snapshotID)
	if snap != nil {
		snap.State = nil
	}
	return snap, err
}

func (s *metadataCountingStore) GetLatestSnapshotMetadata(ctx context.Context, sessionID string) (*SessionSnapshot[testState], error) {
	s.latestMetadataReads++
	snap, err := s.testInMemStore.GetLatestSnapshot(ctx, sessionID)
	if snap != nil {
		snap.State = nil
	}
	return snap, err
}

func TestAgentHandle_MetadataOnlyPrefersTheStoreCapability(t *testing.T) {
	// A metadata-only read takes SnapshotMetadataReader when the store offers
	// it, for both addressings, and never touches the full read. The
	// fallback, a full read with the state dropped, is what every other test
	// in this package exercises through testInMemStore, which lacks the
	// capability (see TestAgentHandle_GetSnapshotMetadataOnly).
	reg := newTestRegistry(t)
	store := &metadataCountingStore{testInMemStore: newTestInMemStore[testState]()}
	af := defineLastGoodTestAgent(reg, "metaCapability", WithSessionStore(store))
	out, err := af.RunText(context.Background(), "first")
	if err != nil {
		t.Fatalf("RunText: %v", err)
	}
	store.fullReads, store.metadataReads, store.latestFullReads, store.latestMetadataReads = 0, 0, 0, 0

	h := LookupAgent(reg, "metaCapability")
	meta, err := h.GetSnapshot(context.Background(), out.SnapshotID, WithMetadataOnly())
	if err != nil {
		t.Fatalf("GetSnapshot(WithMetadataOnly): %v", err)
	}
	if meta.State != nil || meta.Status != SnapshotStatusCompleted {
		t.Errorf("metadata read = status %q state %v, want completed with no state", meta.Status, meta.State)
	}
	latest, err := h.GetLatestSnapshot(context.Background(), out.SessionID, WithMetadataOnly())
	if err != nil {
		t.Fatalf("GetLatestSnapshot(WithMetadataOnly): %v", err)
	}
	if latest.State != nil || latest.SnapshotID != out.SnapshotID {
		t.Errorf("latest metadata read = %+v, want the run's snapshot with no state", latest)
	}
	if store.metadataReads != 1 || store.latestMetadataReads != 1 || store.fullReads != 0 || store.latestFullReads != 0 {
		t.Errorf("reads: metadata=%d latestMetadata=%d full=%d latestFull=%d, want the capability path only (1, 1, 0, 0)",
			store.metadataReads, store.latestMetadataReads, store.fullReads, store.latestFullReads)
	}
}
