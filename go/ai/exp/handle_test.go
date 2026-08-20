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
	"strings"
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

		h, err := LookupAgent(reg, "researcher")
		if err != nil {
			t.Fatalf("LookupAgent: %v", err)
		}
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

	t.Run("not found", func(t *testing.T) {
		reg := newTestRegistry(t)
		_, err := LookupAgent(reg, "ghost")
		if !errors.Is(err, status.ErrNotFound) {
			t.Fatalf("LookupAgent error = %v, want NOT_FOUND", err)
		}
	})

	t.Run("registered under the agent key but not an agent", func(t *testing.T) {
		reg := newTestRegistry(t)
		core.NewActionOf(api.ActionTypeAgent, "impostor", nil,
			func(ctx context.Context, in string) (string, error) { return in, nil },
		).Register(reg)
		_, err := LookupAgent(reg, "impostor")
		if !errors.Is(err, status.ErrInvalidArgument) {
			t.Fatalf("LookupAgent error = %v, want INVALID_ARGUMENT", err)
		}
	})
}

// fakeDescAction is an api.Action stub carrying only a descriptor, for
// AgentMetadataOf's decode paths. The embedded nil interface covers the
// methods AgentMetadataOf never calls.
type fakeDescAction struct {
	api.Action
	desc api.ActionDesc
}

func (f fakeDescAction) Desc() api.ActionDesc { return f.desc }

func TestAgentMetadataOf(t *testing.T) {
	t.Run("nil action", func(t *testing.T) {
		if got := AgentMetadataOf(nil); got != nil {
			t.Fatalf("AgentMetadataOf(nil) = %+v, want nil", got)
		}
	})

	t.Run("typed value from a live agent", func(t *testing.T) {
		reg := newTestRegistry(t)
		agent := defineEchoAgent(t, reg, "typed")
		meta := AgentMetadataOf(agent)
		if meta == nil {
			t.Fatal("AgentMetadataOf = nil, want metadata")
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
		meta := AgentMetadataOf(a)
		if meta == nil || meta.StateManagement != AgentStateManagementServer || !meta.Abortable {
			t.Fatalf("AgentMetadataOf = %+v, want server-managed abortable", meta)
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
		meta := AgentMetadataOf(fakeDescAction{desc: api.ActionDesc{Metadata: decoded}})
		if meta == nil {
			t.Fatal("AgentMetadataOf = nil, want metadata decoded from map")
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
		if meta := AgentMetadataOf(a); meta != nil {
			t.Fatalf("AgentMetadataOf = %+v, want nil for a descriptor that did not decode", meta)
		}
	})

	t.Run("returned metadata does not alias the descriptor", func(t *testing.T) {
		// The copy claim has to cover StateSchema too: it is a map, so a
		// struct copy alone shares it with every other reader of a descriptor
		// documented as immutable.
		desc := api.ActionDesc{Metadata: map[string]any{
			"agent": AgentMetadata{StateSchema: map[string]any{"type": "object"}},
		}}
		meta := AgentMetadataOf(fakeDescAction{desc: desc})
		if meta == nil {
			t.Fatal("AgentMetadataOf = nil, want typed metadata")
		}
		meta.StateSchema["type"] = "tampered"
		original := desc.Metadata["agent"].(AgentMetadata)
		if got := original.StateSchema["type"]; got != "object" {
			t.Errorf("descriptor schema type = %v, want %q untouched", got, "object")
		}
	})

	t.Run("action without agent metadata", func(t *testing.T) {
		a := fakeDescAction{desc: api.ActionDesc{Metadata: map[string]any{"other": true}}}
		if got := AgentMetadataOf(a); got != nil {
			t.Fatalf("AgentMetadataOf = %+v, want nil", got)
		}
	})
}

func TestAgentHandle_Run(t *testing.T) {
	reg := newTestRegistry(t)
	defineEchoAgent(t, reg, "echo")
	h, err := LookupAgent(reg, "echo")
	if err != nil {
		t.Fatalf("LookupAgent: %v", err)
	}

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

func TestAgentHandle_StartPollWaitTask(t *testing.T) {
	// Full background lifecycle through the handle: launch, observe pending,
	// rehydrate the task from nothing but its snapshot ID, and wait for the
	// finalized snapshot.
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()
	_, entered, release := defineGatedAgent(t, reg, "worker", store)

	h, err := LookupAgent(reg, "worker")
	if err != nil {
		t.Fatalf("LookupAgent: %v", err)
	}

	task, err := h.Start(context.Background(), &AgentInput{Message: ai.NewUserTextMessage("go")})
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	if task.SnapshotID() == "" {
		t.Fatal("Start returned a task with no snapshot ID")
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
	h2, err := LookupAgent(reg, "worker")
	if err != nil {
		t.Fatalf("LookupAgent (rehydrate): %v", err)
	}
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

func TestAgentHandle_StartRejected(t *testing.T) {
	t.Run("client-managed agent cannot detach", func(t *testing.T) {
		reg := newTestRegistry(t)
		defineEchoAgent(t, reg, "storeless")
		h, err := LookupAgent(reg, "storeless")
		if err != nil {
			t.Fatalf("LookupAgent: %v", err)
		}
		_, err = h.Start(context.Background(), &AgentInput{Message: ai.NewUserTextMessage("go")})
		if err == nil {
			t.Fatal("Start succeeded on a storeless agent, want rejection")
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
		h, err := LookupAgent(reg, "unabortable")
		if err != nil {
			t.Fatalf("LookupAgent: %v", err)
		}
		if meta := h.Metadata(); meta == nil || meta.Abortable {
			t.Errorf("Metadata() = %+v, want non-abortable", meta)
		}
		_, err = h.Abort(context.Background(), "some-id")
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
	task, err := h.Start(context.Background(), &AgentInput{Message: ai.NewUserTextMessage("go")})
	if err != nil {
		t.Fatalf("Start: %v", err)
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
	if got != SnapshotStatusAborted {
		t.Fatalf("Abort status = %q, want %q", got, SnapshotStatusAborted)
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

func TestAgentHandle_SnapshotReadErrors(t *testing.T) {
	t.Run("no session store", func(t *testing.T) {
		reg := newTestRegistry(t)
		defineEchoAgent(t, reg, "bare")
		h, err := LookupAgent(reg, "bare")
		if err != nil {
			t.Fatalf("LookupAgent: %v", err)
		}
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
		h, err := LookupAgent(reg, "ids")
		if err != nil {
			t.Fatalf("LookupAgent: %v", err)
		}
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
		h, err := LookupAgent(reg, "misses")
		if err != nil {
			t.Fatalf("LookupAgent: %v", err)
		}
		if _, err := h.GetSnapshot(context.Background(), "nope"); !errors.Is(err, ErrSnapshotNotFound) {
			t.Fatalf("GetSnapshot(missing) error = %v, want ErrSnapshotNotFound", err)
		}
		if _, err := h.Task("nope").Poll(context.Background()); !errors.Is(err, status.ErrNotFound) {
			t.Fatalf("Poll(missing) error = %v, want NOT_FOUND", err)
		}
	})
}

func TestAgentHandle_StartSettledSynchronously(t *testing.T) {
	// An agent whose fn returns without consuming its input can settle the
	// invocation before the runtime observes the detach directive. Start must
	// treat that as a first-class outcome: a task over the committed snapshot
	// when one exists, or FAILED_PRECONDITION when nothing was recorded;
	// never an INTERNAL contract-violation error. The race is scheduling-
	// dependent, so accept either outcome on each iteration and pin only the
	// contract.
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
		task, err := h.Start(context.Background(), &AgentInput{Message: ai.NewUserTextMessage("go")})
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
	h, err := LookupAgent(reg, "waiter")
	if err != nil {
		t.Fatalf("LookupAgent: %v", err)
	}

	// An unknown snapshot has nothing to wait on, so the wait fails the way
	// the read does rather than blocking until ctx ends.
	if _, err := h.Task("no-such-snapshot").Wait(context.Background()); !errors.Is(err, ErrSnapshotNotFound) {
		t.Fatalf("Wait on unknown snapshot error = %v, want ErrSnapshotNotFound", err)
	}
	if _, err := h.WaitForSnapshot(context.Background(), ""); !errors.Is(err, status.ErrInvalidArgument) {
		t.Fatalf("WaitForSnapshot(\"\") error = %v, want INVALID_ARGUMENT", err)
	}
}
