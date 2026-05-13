// Copyright 2025 Google LLC
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
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/internal/registry"
)

type testState struct {
	Counter int      `json:"counter"`
	Topics  []string `json:"topics,omitempty"`
}

type testStatus struct {
	Phase string `json:"phase"`
}

func newTestRegistry(t *testing.T) *registry.Registry {
	t.Helper()
	return registry.New()
}

func TestAgent_BasicMultiTurn(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	af := DefineCustomAgent(reg, "basicFlow",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				resp.SendStatus(testStatus{Phase: "generating"})
				// Echo back the user's message.
				if input.Message != nil {
					reply := ai.NewModelTextMessage("echo: " + input.Message.Content[0].Text)
					sess.AddMessages(reply)
				}
				sess.UpdateCustom(func(s testState) testState {
					s.Counter++
					return s
				})
				resp.SendStatus(testStatus{Phase: "complete"})
				return nil
			})
		},
	)

	conn, err := af.StreamBidi(ctx)
	if err != nil {
		t.Fatalf("StreamBidi failed: %v", err)
	}

	// Turn 1.
	if err := conn.SendText("hello"); err != nil {
		t.Fatalf("SendText failed: %v", err)
	}
	var turn1Chunks int
	for chunk, err := range conn.Receive() {
		if err != nil {
			t.Fatalf("Receive error: %v", err)
		}
		turn1Chunks++
		if chunk.TurnEnd != nil {
			break
		}
	}
	if turn1Chunks < 2 { // at least status + TurnEnd
		t.Errorf("expected at least 2 chunks in turn 1, got %d", turn1Chunks)
	}

	// Turn 2.
	if err := conn.SendText("world"); err != nil {
		t.Fatalf("SendText failed: %v", err)
	}
	for chunk, err := range conn.Receive() {
		if err != nil {
			t.Fatalf("Receive error: %v", err)
		}
		if chunk.TurnEnd != nil {
			break
		}
	}

	conn.Close()

	response, err := conn.Output()
	if err != nil {
		t.Fatalf("Output failed: %v", err)
	}

	// 2 user messages + 2 echo replies = 4.
	if got := len(response.State.Messages); got != 4 {
		t.Errorf("expected 4 messages, got %d", got)
	}
	if got := response.State.Custom.Counter; got != 2 {
		t.Errorf("expected counter=2, got %d", got)
	}
}

func TestAgent_WithSessionStore(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := NewInMemorySessionStore[testState]()

	af := DefineCustomAgent(reg, "snapshotFlow",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				if input.Message != nil {
					sess.AddMessages(ai.NewModelTextMessage("reply"))
				}
				sess.UpdateCustom(func(s testState) testState {
					s.Counter++
					return s
				})
				return nil
			})
		},
		WithSessionStore(store),
	)

	conn, err := af.StreamBidi(ctx)
	if err != nil {
		t.Fatalf("StreamBidi failed: %v", err)
	}

	conn.SendText("turn1")

	var snapshotIDs []string
	for chunk, err := range conn.Receive() {
		if err != nil {
			t.Fatalf("Receive error: %v", err)
		}
		if chunk.TurnEnd != nil {
			if chunk.TurnEnd.SnapshotID != "" {
				snapshotIDs = append(snapshotIDs, chunk.TurnEnd.SnapshotID)
			}
			break
		}
	}

	if len(snapshotIDs) != 1 {
		t.Fatalf("expected 1 snapshot from turn, got %d", len(snapshotIDs))
	}

	// Verify the snapshot was persisted.
	snap, err := store.GetSnapshot(ctx, snapshotIDs[0])
	if err != nil {
		t.Fatalf("GetSnapshot failed: %v", err)
	}
	if snap == nil {
		t.Fatal("expected snapshot, got nil")
	}
	if snap.State.Custom.Counter != 1 {
		t.Errorf("expected counter=1 in snapshot, got %d", snap.State.Custom.Counter)
	}

	conn.Close()

	response, err := conn.Output()
	if err != nil {
		t.Fatalf("Output failed: %v", err)
	}

	// Final snapshot at invocation end.
	if response.SnapshotID == "" {
		t.Error("expected final snapshot ID")
	}
}

func TestAgent_ResumeFromSnapshot(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := NewInMemorySessionStore[testState]()

	af := DefineCustomAgent(reg, "resumeFlow",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				if input.Message != nil {
					sess.AddMessages(ai.NewModelTextMessage("reply"))
				}
				sess.UpdateCustom(func(s testState) testState {
					s.Counter++
					return s
				})
				return nil
			})
		},
		WithSessionStore(store),
	)

	// First invocation: create a snapshot.
	conn1, err := af.StreamBidi(ctx)
	if err != nil {
		t.Fatalf("StreamBidi failed: %v", err)
	}
	conn1.SendText("first message")
	for chunk, err := range conn1.Receive() {
		if err != nil {
			t.Fatalf("Receive error: %v", err)
		}
		if chunk.TurnEnd != nil {
			break
		}
	}
	conn1.Close()
	resp1, err := conn1.Output()
	if err != nil {
		t.Fatalf("Output failed: %v", err)
	}
	if resp1.SnapshotID == "" {
		t.Fatal("expected snapshot ID from first invocation")
	}

	// Second invocation: resume from snapshot.
	conn2, err := af.StreamBidi(ctx, WithSnapshotID[testState](resp1.SnapshotID))
	if err != nil {
		t.Fatalf("StreamBidi with snapshot failed: %v", err)
	}
	conn2.SendText("continued message")
	for chunk, err := range conn2.Receive() {
		if err != nil {
			t.Fatalf("Receive error: %v", err)
		}
		if chunk.TurnEnd != nil {
			break
		}
	}
	conn2.Close()
	resp2, err := conn2.Output()
	if err != nil {
		t.Fatalf("Output failed: %v", err)
	}

	// The new snapshot should reference the previous as parent.
	if resp2.SnapshotID == "" {
		t.Fatal("expected snapshot ID from second invocation")
	}
	snap2, err := store.GetSnapshot(ctx, resp2.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot failed: %v", err)
	}

	// Should have messages from both invocations:
	// first: user + reply (2) + second: user + reply (2) = 4.
	if got := len(snap2.State.Messages); got != 4 {
		t.Errorf("expected 4 messages after resume, got %d", got)
	}
	// Counter should be 2 (1 from first + 1 from second).
	if got := snap2.State.Custom.Counter; got != 2 {
		t.Errorf("expected counter=2, got %d", got)
	}
	// The parent chain: snap2's parent is a turn-end snapshot from the second invocation,
	// which itself has a parent from the first invocation's final snapshot.
	// We just verify that the parent chain exists (not empty).
	if snap2.ParentID == "" {
		t.Error("expected parent ID on resumed snapshot")
	}
}

func TestAgent_ClientManagedState(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	af := DefineCustomAgent(reg, "clientStateFlow",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				if input.Message != nil {
					sess.AddMessages(ai.NewModelTextMessage("reply"))
				}
				sess.UpdateCustom(func(s testState) testState {
					s.Counter++
					return s
				})
				return nil
			})
		},
	)

	// Start with client-provided state.
	clientState := &SessionState[testState]{
		Messages: []*ai.Message{
			ai.NewUserTextMessage("previous message"),
			ai.NewModelTextMessage("previous reply"),
		},
		Custom: testState{Counter: 5},
	}

	conn, err := af.StreamBidi(ctx, WithState(clientState))
	if err != nil {
		t.Fatalf("StreamBidi failed: %v", err)
	}

	conn.SendText("new message")
	for chunk, err := range conn.Receive() {
		if err != nil {
			t.Fatalf("Receive error: %v", err)
		}
		if chunk.TurnEnd != nil {
			break
		}
	}
	conn.Close()

	response, err := conn.Output()
	if err != nil {
		t.Fatalf("Output failed: %v", err)
	}

	// 2 previous + 1 new user + 1 reply = 4.
	if got := len(response.State.Messages); got != 4 {
		t.Errorf("expected 4 messages, got %d", got)
	}
	// Counter should be 6 (started at 5, incremented once).
	if got := response.State.Custom.Counter; got != 6 {
		t.Errorf("expected counter=6, got %d", got)
	}
	// No snapshot since no store was configured.
	if response.SnapshotID != "" {
		t.Errorf("expected no snapshot ID without store, got %q", response.SnapshotID)
	}
}

func TestAgent_Artifacts(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	af := DefineCustomAgent(reg, "artifactFlow",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			err := sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {

				resp.SendArtifact(&Artifact{
					Name:  "code.go",
					Parts: []*ai.Part{ai.NewTextPart("package main")},
				})

				// Replace artifact with same name.
				resp.SendArtifact(&Artifact{
					Name:  "code.go",
					Parts: []*ai.Part{ai.NewTextPart("package main\nfunc main() {}")},
				})

				// Add another artifact.
				resp.SendArtifact(&Artifact{
					Name:  "readme.md",
					Parts: []*ai.Part{ai.NewTextPart("# README")},
				})

				sess.AddMessages(ai.NewModelTextMessage("done"))
				return nil
			})
			if err != nil {
				return nil, err
			}
			return &AgentResult{Artifacts: sess.Artifacts()}, nil
		},
	)

	conn, err := af.StreamBidi(ctx)
	if err != nil {
		t.Fatalf("StreamBidi failed: %v", err)
	}

	conn.SendText("generate code")
	var receivedArtifacts []*Artifact
	for chunk, err := range conn.Receive() {
		if err != nil {
			t.Fatalf("Receive error: %v", err)
		}
		if chunk.Artifact != nil {
			receivedArtifacts = append(receivedArtifacts, chunk.Artifact)
		}
		if chunk.TurnEnd != nil {
			break
		}
	}
	conn.Close()

	if len(receivedArtifacts) != 3 { // all 3 sends are streamed
		t.Errorf("expected 3 streamed artifacts, got %d", len(receivedArtifacts))
	}

	response, err := conn.Output()
	if err != nil {
		t.Fatalf("Output failed: %v", err)
	}

	// Output should have 2 unique artifacts (code.go was replaced).
	if got := len(response.Artifacts); got != 2 {
		t.Errorf("expected 2 artifacts, got %d", got)
	}
}

func TestAgent_SnapshotCallback(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := NewInMemorySessionStore[testState]()

	// Only snapshot on even turns.
	callbackCalls := 0
	af := DefineCustomAgent(reg, "callbackFlow",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				sess.AddMessages(ai.NewModelTextMessage("reply"))
				sess.UpdateCustom(func(s testState) testState {
					s.Counter++
					return s
				})
				return nil
			})
		},
		WithSessionStore(store),
		WithSnapshotCallback(func(ctx context.Context, sc *SnapshotContext[testState]) bool {
			callbackCalls++
			return sc.TurnIndex%2 == 0 // only snapshot on even turns
		}),
	)

	conn, err := af.StreamBidi(ctx)
	if err != nil {
		t.Fatalf("StreamBidi failed: %v", err)
	}

	var snapshotIDs []string
	for i := 0; i < 3; i++ {
		conn.SendText(fmt.Sprintf("turn %d", i))
		for chunk, err := range conn.Receive() {
			if err != nil {
				t.Fatalf("Receive error on turn %d: %v", i, err)
			}
			if chunk.TurnEnd != nil {
				if chunk.TurnEnd.SnapshotID != "" {
					snapshotIDs = append(snapshotIDs, chunk.TurnEnd.SnapshotID)
				}
				break
			}
		}
	}
	conn.Close()
	conn.Output() // drain

	// Turn 0 (even) → snapshot, Turn 1 (odd) → no, Turn 2 (even) → snapshot.
	// That's 2 turn snapshots from the callback.
	if got := len(snapshotIDs); got != 2 {
		t.Errorf("expected 2 turn snapshots, got %d", got)
	}
	// Callback should have been called 3 times (once per turn).
	if callbackCalls < 3 {
		t.Errorf("expected at least 3 callback calls, got %d", callbackCalls)
	}
}

func TestAgent_SendMessage(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	af := DefineCustomAgent(reg, "sendMsgFlow",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				return nil
			})
		},
	)

	conn, err := af.StreamBidi(ctx)
	if err != nil {
		t.Fatalf("StreamBidi failed: %v", err)
	}

	// Send a message via SendMessage.
	err = conn.SendMessage(ai.NewUserTextMessage("msg1"))
	if err != nil {
		t.Fatalf("SendMessage failed: %v", err)
	}
	for chunk, err := range conn.Receive() {
		if err != nil {
			t.Fatalf("Receive error: %v", err)
		}
		if chunk.TurnEnd != nil {
			break
		}
	}
	conn.Close()

	response, err := conn.Output()
	if err != nil {
		t.Fatalf("Output failed: %v", err)
	}

	// The message should have been added.
	if got := len(response.State.Messages); got != 1 {
		t.Errorf("expected 1 message, got %d", got)
	}
}

func TestAgent_SessionContext(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	var retrievedCounter int
	af := DefineCustomAgent(reg, "contextFlow",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				// Session should be retrievable from context.
				ctxSess := SessionFromContext[testState](ctx)
				if ctxSess == nil {
					t.Error("expected session from context")
					return nil
				}
				ctxSess.UpdateCustom(func(s testState) testState {
					s.Counter = 42
					return s
				})
				retrievedCounter = ctxSess.Custom().Counter
				return nil
			})
		},
	)

	conn, err := af.StreamBidi(ctx)
	if err != nil {
		t.Fatalf("StreamBidi failed: %v", err)
	}

	conn.SendText("test")
	for chunk, err := range conn.Receive() {
		if err != nil {
			t.Fatalf("Receive error: %v", err)
		}
		if chunk.TurnEnd != nil {
			break
		}
	}
	conn.Close()
	conn.Output()

	if retrievedCounter != 42 {
		t.Errorf("expected counter=42 from context, got %d", retrievedCounter)
	}
}

func TestAgent_ErrorInTurn(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	af := DefineCustomAgent(reg, "errorFlow",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				return fmt.Errorf("turn failed")
			})
		},
	)

	conn, err := af.StreamBidi(ctx)
	if err != nil {
		t.Fatalf("StreamBidi failed: %v", err)
	}

	conn.SendText("trigger error")
	conn.Close()

	_, err = conn.Output()
	if err == nil {
		t.Fatal("expected error from failed turn")
	}
}

func TestAgent_SetMessages(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	af := DefineCustomAgent(reg, "setMsgsFlow",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				// Replace all messages with just one.
				sess.SetMessages([]*ai.Message{ai.NewModelTextMessage("replaced")})
				return nil
			})
		},
	)

	conn, err := af.StreamBidi(ctx)
	if err != nil {
		t.Fatalf("StreamBidi failed: %v", err)
	}

	conn.SendText("original")
	for chunk, err := range conn.Receive() {
		if err != nil {
			t.Fatalf("Receive error: %v", err)
		}
		if chunk.TurnEnd != nil {
			break
		}
	}
	conn.Close()

	response, err := conn.Output()
	if err != nil {
		t.Fatalf("Output failed: %v", err)
	}

	// SetMessages replaced everything with 1 message.
	if got := len(response.State.Messages); got != 1 {
		t.Errorf("expected 1 message after SetMessages, got %d", got)
	}
}

func TestInMemorySessionStore(t *testing.T) {
	t.Run("GetMissing", func(t *testing.T) {
		store := NewInMemorySessionStore[testState]()
		snap, err := store.GetSnapshot(context.Background(), "nonexistent")
		if err != nil {
			t.Fatalf("GetSnapshot failed: %v", err)
		}
		if snap != nil {
			t.Errorf("expected nil, got %v", snap)
		}
	})

	t.Run("SaveWithFixedID", func(t *testing.T) {
		store := NewInMemorySessionStore[testState]()
		saved, err := store.SaveSnapshot(context.Background(), "snap-1",
			func(existing *SessionSnapshot[testState]) (*SessionSnapshot[testState], error) {
				if existing != nil {
					t.Errorf("expected nil existing on first save, got %+v", existing)
				}
				return &SessionSnapshot[testState]{
					Status: SnapshotStatusSucceeded,
					State:  &SessionState[testState]{Custom: testState{Counter: 1}},
				}, nil
			})
		if err != nil {
			t.Fatalf("SaveSnapshot failed: %v", err)
		}
		if saved.SnapshotID != "snap-1" {
			t.Errorf("saved SnapshotID = %q, want %q", saved.SnapshotID, "snap-1")
		}
		if saved.CreatedAt.IsZero() || saved.UpdatedAt.IsZero() {
			t.Errorf("expected CreatedAt/UpdatedAt stamped, got created=%v updated=%v",
				saved.CreatedAt, saved.UpdatedAt)
		}
	})

	t.Run("GetReturnsCopy", func(t *testing.T) {
		store := NewInMemorySessionStore[testState]()
		if _, err := store.SaveSnapshot(context.Background(), "snap-1",
			func(_ *SessionSnapshot[testState]) (*SessionSnapshot[testState], error) {
				return &SessionSnapshot[testState]{
					Status: SnapshotStatusSucceeded,
					State:  &SessionState[testState]{Custom: testState{Counter: 1}},
				}, nil
			}); err != nil {
			t.Fatalf("SaveSnapshot: %v", err)
		}
		retrieved, _ := store.GetSnapshot(context.Background(), "snap-1")
		retrieved.State.Custom.Counter = 999
		retrieved2, _ := store.GetSnapshot(context.Background(), "snap-1")
		if retrieved2.State.Custom.Counter != 1 {
			t.Errorf("expected counter=1 (isolation), got %d", retrieved2.State.Custom.Counter)
		}
	})

	t.Run("DefaultsEmptyStatusToComplete", func(t *testing.T) {
		store := NewInMemorySessionStore[testState]()
		saved, err := store.SaveSnapshot(context.Background(), "",
			func(_ *SessionSnapshot[testState]) (*SessionSnapshot[testState], error) {
				return &SessionSnapshot[testState]{}, nil
			})
		if err != nil {
			t.Fatalf("SaveSnapshot: %v", err)
		}
		if saved.SnapshotID == "" {
			t.Error("expected store to generate SnapshotID")
		}
		if saved.Status != SnapshotStatusSucceeded {
			t.Errorf("expected Status=complete by default, got %q", saved.Status)
		}
	})

	t.Run("NoopFnSkipsWrite", func(t *testing.T) {
		store := NewInMemorySessionStore[testState]()
		if _, err := store.SaveSnapshot(context.Background(), "snap-1",
			func(_ *SessionSnapshot[testState]) (*SessionSnapshot[testState], error) {
				return &SessionSnapshot[testState]{Status: SnapshotStatusSucceeded}, nil
			}); err != nil {
			t.Fatalf("seed: %v", err)
		}
		before, _ := store.GetSnapshot(context.Background(), "snap-1")
		noop, err := store.SaveSnapshot(context.Background(), "snap-1",
			func(_ *SessionSnapshot[testState]) (*SessionSnapshot[testState], error) {
				return nil, nil
			})
		if err != nil {
			t.Fatalf("noop SaveSnapshot: %v", err)
		}
		if noop != nil {
			t.Errorf("expected nil return on noop, got %+v", noop)
		}
		after, _ := store.GetSnapshot(context.Background(), "snap-1")
		if before.UpdatedAt != after.UpdatedAt {
			t.Errorf("noop should not bump UpdatedAt: before=%v after=%v", before.UpdatedAt, after.UpdatedAt)
		}
	})

	t.Run("PreservesCreatedAtOnUpdate", func(t *testing.T) {
		store := NewInMemorySessionStore[testState]()
		saved, err := store.SaveSnapshot(context.Background(), "snap-1",
			func(_ *SessionSnapshot[testState]) (*SessionSnapshot[testState], error) {
				return &SessionSnapshot[testState]{Status: SnapshotStatusSucceeded}, nil
			})
		if err != nil {
			t.Fatalf("seed: %v", err)
		}
		time.Sleep(time.Millisecond) // ensure measurable UpdatedAt delta
		updated, err := store.SaveSnapshot(context.Background(), "snap-1",
			func(existing *SessionSnapshot[testState]) (*SessionSnapshot[testState], error) {
				if existing == nil {
					t.Fatal("expected non-nil existing on update")
				}
				return &SessionSnapshot[testState]{
					Status: SnapshotStatusSucceeded,
					State:  &SessionState[testState]{Custom: testState{Counter: 2}},
				}, nil
			})
		if err != nil {
			t.Fatalf("update: %v", err)
		}
		if !updated.CreatedAt.Equal(saved.CreatedAt) {
			t.Errorf("CreatedAt not preserved: before=%v after=%v", saved.CreatedAt, updated.CreatedAt)
		}
		if !updated.UpdatedAt.After(saved.UpdatedAt) {
			t.Errorf("UpdatedAt did not advance: before=%v after=%v", saved.UpdatedAt, updated.UpdatedAt)
		}
	})
}

func TestAgent_TurnSpanOutput(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	var capturedOutputs []any

	af := DefineCustomAgent(reg, "turnOutputFlow",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			// Wrap collectTurnOutput to capture what each turn produces.
			originalCollect := sess.collectTurnOutput
			sess.collectTurnOutput = func() any {
				output := originalCollect()
				capturedOutputs = append(capturedOutputs, output)
				return output
			}

			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				resp.SendStatus(testStatus{Phase: "thinking"})
				resp.SendModelChunk(&ai.ModelResponseChunk{
					Content: []*ai.Part{ai.NewTextPart("reply")},
				})
				resp.SendArtifact(&Artifact{
					Name:  "out.txt",
					Parts: []*ai.Part{ai.NewTextPart("content")},
				})
				sess.AddMessages(ai.NewModelTextMessage("reply"))
				return nil
			})
		},
	)

	conn, err := af.StreamBidi(ctx)
	if err != nil {
		t.Fatalf("StreamBidi failed: %v", err)
	}

	// Two turns.
	for turn := range 2 {
		if err := conn.SendText(fmt.Sprintf("turn %d", turn)); err != nil {
			t.Fatalf("SendText failed on turn %d: %v", turn, err)
		}
		for chunk, err := range conn.Receive() {
			if err != nil {
				t.Fatalf("Receive error on turn %d: %v", turn, err)
			}
			if chunk.TurnEnd != nil {
				break
			}
		}
	}

	conn.Close()
	if _, err := conn.Output(); err != nil {
		t.Fatalf("Output failed: %v", err)
	}

	// Should have captured output for each turn.
	if len(capturedOutputs) != 2 {
		t.Fatalf("expected 2 captured outputs, got %d", len(capturedOutputs))
	}

	for i, output := range capturedOutputs {
		chunks, ok := output.([]*AgentStreamChunk[testStatus])
		if !ok {
			t.Fatalf("turn %d: expected []*AgentStreamChunk[testStatus], got %T", i, output)
		}
		// 3 content chunks per turn: status + model chunk + artifact.
		if len(chunks) != 3 {
			t.Errorf("turn %d: expected 3 chunks, got %d", i, len(chunks))
		}
		for j, chunk := range chunks {
			if chunk.TurnEnd != nil {
				t.Errorf("turn %d, chunk %d: TurnEnd should not be in turn output", i, j)
			}
		}
	}
}

func TestAgent_TurnSpanOutput_WithSnapshots(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := NewInMemorySessionStore[testState]()

	var capturedOutputs []any

	af := DefineCustomAgent(reg, "turnOutputSnapshotFlow",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			originalCollect := sess.collectTurnOutput
			sess.collectTurnOutput = func() any {
				output := originalCollect()
				capturedOutputs = append(capturedOutputs, output)
				return output
			}

			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				resp.SendStatus(testStatus{Phase: "working"})
				sess.AddMessages(ai.NewModelTextMessage("reply"))
				return nil
			})
		},
		WithSessionStore(store),
	)

	conn, err := af.StreamBidi(ctx)
	if err != nil {
		t.Fatalf("StreamBidi failed: %v", err)
	}

	conn.SendText("hello")
	var sawSnapshot bool
	for chunk, err := range conn.Receive() {
		if err != nil {
			t.Fatalf("Receive error: %v", err)
		}
		if chunk.TurnEnd != nil {
			if chunk.TurnEnd.SnapshotID != "" {
				sawSnapshot = true
			}
			break
		}
	}
	conn.Close()
	conn.Output()

	if !sawSnapshot {
		t.Fatal("expected a snapshot ID on the turn-end chunk")
	}

	// Turn output should contain only the status chunk, not the TurnEnd signal.
	if len(capturedOutputs) != 1 {
		t.Fatalf("expected 1 captured output, got %d", len(capturedOutputs))
	}
	chunks := capturedOutputs[0].([]*AgentStreamChunk[testStatus])
	if len(chunks) != 1 {
		t.Errorf("expected 1 content chunk, got %d", len(chunks))
	}
	if chunks[0].Status.Phase != "working" {
		t.Errorf("expected status phase 'working', got %q", chunks[0].Status.Phase)
	}
}

// setupPromptTestRegistry creates a registry with an echo model and generate action.
func setupPromptTestRegistry(t *testing.T) *registry.Registry {
	t.Helper()
	reg := registry.New()
	ctx := context.Background()

	ai.ConfigureFormats(reg)
	ai.DefineModel(reg, "test/echo", &ai.ModelOptions{Supports: &ai.ModelSupports{Multiturn: true, SystemRole: true}},
		func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			// Echo back the last user message text.
			var text string
			for i := len(req.Messages) - 1; i >= 0; i-- {
				if req.Messages[i].Role == ai.RoleUser {
					text = req.Messages[i].Text()
					break
				}
			}
			if text == "" {
				text = "no input"
			}

			resp := &ai.ModelResponse{
				Request: req,
				Message: ai.NewModelTextMessage("echo: " + text),
			}

			if cb != nil {
				if err := cb(ctx, &ai.ModelResponseChunk{
					Content: resp.Message.Content,
				}); err != nil {
					return nil, err
				}
			}

			return resp, nil
		},
	)
	ai.DefineGenerateAction(ctx, reg)
	return reg
}

func TestPromptAgent_Basic(t *testing.T) {
	ctx := context.Background()
	reg := setupPromptTestRegistry(t)

	ai.DefinePrompt(reg, "testPrompt",
		ai.WithModelName("test/echo"),
		ai.WithSystem("You are a test assistant."),
	)

	af := DefineAgent[testState](reg, "testPrompt", FromPrompt())

	conn, err := af.StreamBidi(ctx)
	if err != nil {
		t.Fatalf("StreamBidi failed: %v", err)
	}

	// Turn 1.
	if err := conn.SendText("hello"); err != nil {
		t.Fatalf("SendText failed: %v", err)
	}

	var gotChunk bool
	for chunk, err := range conn.Receive() {
		if err != nil {
			t.Fatalf("Receive error: %v", err)
		}
		if chunk.ModelChunk != nil {
			gotChunk = true
		}
		if chunk.TurnEnd != nil {
			break
		}
	}
	if !gotChunk {
		t.Error("expected at least one streaming chunk")
	}

	// Turn 2.
	if err := conn.SendText("world"); err != nil {
		t.Fatalf("SendText failed: %v", err)
	}
	for chunk, err := range conn.Receive() {
		if err != nil {
			t.Fatalf("Receive error: %v", err)
		}
		if chunk.TurnEnd != nil {
			break
		}
	}

	conn.Close()

	response, err := conn.Output()
	if err != nil {
		t.Fatalf("Output failed: %v", err)
	}

	// 2 user messages + 2 model replies = 4.
	if got := len(response.State.Messages); got != 4 {
		t.Errorf("expected 4 messages, got %d", got)
		for i, m := range response.State.Messages {
			t.Logf("  msg[%d]: role=%s text=%s", i, m.Role, m.Text())
		}
	}
}

func TestPromptAgent_MultiTurnHistory(t *testing.T) {
	ctx := context.Background()
	reg := setupPromptTestRegistry(t)

	// Use a model that echoes all message count so we can verify history grows.
	ai.DefineModel(reg, "test/history", &ai.ModelOptions{Supports: &ai.ModelSupports{Multiturn: true, SystemRole: true}},
		func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			// Count total messages received (includes prompt-rendered + history).
			var parts []string
			for _, m := range req.Messages {
				parts = append(parts, string(m.Role)+":"+m.Text())
			}
			text := strings.Join(parts, "|")

			resp := &ai.ModelResponse{
				Request: req,
				Message: ai.NewModelTextMessage(text),
			}
			if cb != nil {
				cb(ctx, &ai.ModelResponseChunk{Content: resp.Message.Content})
			}
			return resp, nil
		},
	)

	ai.DefinePrompt(reg, "historyPrompt",
		ai.WithModelName("test/history"),
		ai.WithSystem("system prompt"),
	)

	af := DefineAgent[testState](reg, "historyPrompt", FromPrompt())

	conn, err := af.StreamBidi(ctx)
	if err != nil {
		t.Fatalf("StreamBidi failed: %v", err)
	}

	// Turn 1.
	conn.SendText("turn1")
	var turn1Response string
	for chunk, err := range conn.Receive() {
		if err != nil {
			t.Fatalf("Receive error: %v", err)
		}
		if chunk.ModelChunk != nil {
			turn1Response += chunk.ModelChunk.Text()
		}
		if chunk.TurnEnd != nil {
			break
		}
	}

	// Turn 1 should have: system message + user message "turn1" (2 messages total from prompt + history).
	// The system message comes from the prompt, "turn1" from session history.
	if !strings.Contains(turn1Response, "turn1") {
		t.Errorf("turn1 response should contain 'turn1', got: %s", turn1Response)
	}

	// Turn 2.
	conn.SendText("turn2")
	var turn2Response string
	for chunk, err := range conn.Receive() {
		if err != nil {
			t.Fatalf("Receive error: %v", err)
		}
		if chunk.ModelChunk != nil {
			turn2Response += chunk.ModelChunk.Text()
		}
		if chunk.TurnEnd != nil {
			break
		}
	}

	// Turn 2 should have: system + turn1 user + turn1 model reply + turn2 user (4 messages from prompt + history).
	if !strings.Contains(turn2Response, "turn1") || !strings.Contains(turn2Response, "turn2") {
		t.Errorf("turn2 response should contain both 'turn1' and 'turn2', got: %s", turn2Response)
	}

	conn.Close()

	response, err := conn.Output()
	if err != nil {
		t.Fatalf("Output failed: %v", err)
	}

	// Session should have: turn1 user + turn1 model + turn2 user + turn2 model = 4 messages.
	if got := len(response.State.Messages); got != 4 {
		t.Errorf("expected 4 messages in session, got %d", got)
		for i, m := range response.State.Messages {
			t.Logf("  msg[%d]: role=%s text=%s", i, m.Role, m.Text())
		}
	}
}

func TestPromptAgent_SnapshotResumePreservesHistory(t *testing.T) {
	ctx := context.Background()
	reg := setupPromptTestRegistry(t)
	store := NewInMemorySessionStore[testState]()

	ai.DefinePrompt(reg, "snapPrompt",
		ai.WithModelName("test/echo"),
		ai.WithSystem("You are a test assistant."),
	)

	af := DefineAgent[testState](reg, "snapPrompt", FromPrompt(),
		WithSessionStore(store),
	)

	conn, err := af.StreamBidi(ctx)
	if err != nil {
		t.Fatalf("StreamBidi failed: %v", err)
	}

	conn.SendText("hello")
	for chunk, err := range conn.Receive() {
		if err != nil {
			t.Fatalf("Receive error: %v", err)
		}
		if chunk.TurnEnd != nil {
			break
		}
	}
	conn.Close()

	resp, err := conn.Output()
	if err != nil {
		t.Fatalf("Output failed: %v", err)
	}
	if resp.SnapshotID == "" {
		t.Fatal("expected snapshot ID")
	}

	conn2, err := af.StreamBidi(ctx, WithSnapshotID[testState](resp.SnapshotID))
	if err != nil {
		t.Fatalf("StreamBidi with snapshot failed: %v", err)
	}

	conn2.SendText("continued")
	for chunk, err := range conn2.Receive() {
		if err != nil {
			t.Fatalf("Receive error: %v", err)
		}
		if chunk.TurnEnd != nil {
			break
		}
	}
	conn2.Close()

	resp2, err := conn2.Output()
	if err != nil {
		t.Fatalf("Output failed: %v", err)
	}

	snap2, err := store.GetSnapshot(ctx, resp2.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot failed: %v", err)
	}
	if got := len(snap2.State.Messages); got != 4 {
		t.Errorf("expected 4 messages after resume, got %d", got)
	}
}

func TestPromptAgent_ToolLoopMessages(t *testing.T) {
	ctx := context.Background()
	reg := registry.New()
	ai.ConfigureFormats(reg)

	// Define two tools so the model can call them across multiple rounds.
	ai.DefineTool(reg, "greet", "returns a greeting",
		func(ctx *ai.ToolContext, input struct {
			Name string `json:"name"`
		}) (string, error) {
			return "hello " + input.Name, nil
		},
	)
	ai.DefineTool(reg, "farewell", "returns a farewell",
		func(ctx *ai.ToolContext, input struct {
			Name string `json:"name"`
		}) (string, error) {
			return "goodbye " + input.Name, nil
		},
	)

	// Model that drives a multi-round tool loop:
	//   Round 1: request "greet" tool
	//   Round 2: after seeing greet response, request "farewell" tool
	//   Round 3: after seeing farewell response, return final text
	ai.DefineModel(reg, "test/toolmodel", &ai.ModelOptions{Supports: &ai.ModelSupports{Multiturn: true, SystemRole: true, Tools: true}},
		func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			// Count tool responses to determine which round we're in.
			toolResps := 0
			for _, msg := range req.Messages {
				for _, p := range msg.Content {
					if p.IsToolResponse() {
						toolResps++
					}
				}
			}

			switch toolResps {
			case 0:
				// Round 1: request greet.
				return &ai.ModelResponse{
					Request: req,
					Message: &ai.Message{
						Role: ai.RoleModel,
						Content: []*ai.Part{ai.NewToolRequestPart(&ai.ToolRequest{
							Name:  "greet",
							Input: map[string]any{"name": "world"},
						})},
					},
				}, nil
			case 1:
				// Round 2: saw greet response, now request farewell.
				return &ai.ModelResponse{
					Request: req,
					Message: &ai.Message{
						Role: ai.RoleModel,
						Content: []*ai.Part{ai.NewToolRequestPart(&ai.ToolRequest{
							Name:  "farewell",
							Input: map[string]any{"name": "world"},
						})},
					},
				}, nil
			default:
				// Round 3: saw both tool responses, return final text.
				resp := &ai.ModelResponse{
					Request: req,
					Message: ai.NewModelTextMessage("done"),
				}
				if cb != nil {
					cb(ctx, &ai.ModelResponseChunk{Content: resp.Message.Content})
				}
				return resp, nil
			}
		},
	)
	ai.DefineGenerateAction(ctx, reg)

	ai.DefinePrompt(reg, "toolPrompt",
		ai.WithModelName("test/toolmodel"),
		ai.WithSystem("You are a test assistant."),
		ai.WithTools(ai.ToolName("greet"), ai.ToolName("farewell")),
	)

	af := DefineAgent[testState](reg, "toolPrompt", FromPrompt())

	conn, err := af.StreamBidi(ctx)
	if err != nil {
		t.Fatalf("StreamBidi failed: %v", err)
	}

	conn.SendText("go")
	for chunk, err := range conn.Receive() {
		if err != nil {
			t.Fatalf("Receive error: %v", err)
		}
		if chunk.TurnEnd != nil {
			break
		}
	}
	conn.Close()

	response, err := conn.Output()
	if err != nil {
		t.Fatalf("Output failed: %v", err)
	}

	// Session should contain all messages from the multi-round tool loop:
	// 1. user message ("go")
	// 2. model tool-call message (greet request)
	// 3. tool response message (greet result)
	// 4. model tool-call message (farewell request)
	// 5. tool response message (farewell result)
	// 6. final model text response
	msgs := response.State.Messages
	if got := len(msgs); got != 6 {
		t.Errorf("expected 6 messages, got %d", got)
		for i, m := range msgs {
			t.Logf("  msg[%d]: role=%s text=%s", i, m.Role, m.Text())
		}
		t.FailNow()
	}

	if msgs[0].Role != ai.RoleUser {
		t.Errorf("msg[0] role = %s, want user", msgs[0].Role)
	}

	// Verify the two tool request/response pairs.
	for _, pair := range []struct {
		reqIdx  int
		respIdx int
		tool    string
	}{
		{1, 2, "greet"},
		{3, 4, "farewell"},
	} {
		reqMsg := msgs[pair.reqIdx]
		if reqMsg.Role != ai.RoleModel {
			t.Errorf("msg[%d] role = %s, want model", pair.reqIdx, reqMsg.Role)
		}
		hasReq := false
		for _, p := range reqMsg.Content {
			if p.IsToolRequest() && p.ToolRequest.Name == pair.tool {
				hasReq = true
			}
		}
		if !hasReq {
			t.Errorf("msg[%d] should contain a %s tool request", pair.reqIdx, pair.tool)
		}

		respMsg := msgs[pair.respIdx]
		if respMsg.Role != ai.RoleTool {
			t.Errorf("msg[%d] role = %s, want tool", pair.respIdx, respMsg.Role)
		}
	}

	if msgs[5].Role != ai.RoleModel || msgs[5].Text() != "done" {
		t.Errorf("msg[5] should be final model response, got role=%s text=%q", msgs[5].Role, msgs[5].Text())
	}
}

func TestAgent_RunText(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	af := DefineCustomAgent(reg, "runTextFlow",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				if input.Message != nil {
					sess.AddMessages(ai.NewModelTextMessage("echo: " + input.Message.Content[0].Text))
				}
				sess.UpdateCustom(func(s testState) testState {
					s.Counter++
					return s
				})
				return nil
			})
		},
	)

	response, err := af.RunText(ctx, "hello")
	if err != nil {
		t.Fatalf("RunText failed: %v", err)
	}

	// 1 user message + 1 echo reply = 2.
	if got := len(response.State.Messages); got != 2 {
		t.Errorf("expected 2 messages, got %d", got)
	}
	if got := response.State.Custom.Counter; got != 1 {
		t.Errorf("expected counter=1, got %d", got)
	}
}

func TestAgent_Run(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	af := DefineCustomAgent(reg, "runFlow",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				if input.Message != nil {
					sess.AddMessages(ai.NewModelTextMessage("reply"))
				}
				return nil
			})
		},
	)

	input := &AgentInput{
		Message: ai.NewUserTextMessage("msg1"),
	}

	response, err := af.Run(ctx, input)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	// 1 user message + 1 reply = 2.
	if got := len(response.State.Messages); got != 2 {
		t.Errorf("expected 2 messages, got %d", got)
	}
}

func TestAgent_RunText_WithState(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	af := DefineCustomAgent(reg, "runStateFlow",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				sess.AddMessages(ai.NewModelTextMessage("reply"))
				sess.UpdateCustom(func(s testState) testState {
					s.Counter++
					return s
				})
				return nil
			})
		},
	)

	clientState := &SessionState[testState]{
		Messages: []*ai.Message{
			ai.NewUserTextMessage("previous"),
			ai.NewModelTextMessage("previous reply"),
		},
		Custom: testState{Counter: 10},
	}

	response, err := af.RunText(ctx, "new message", WithState(clientState))
	if err != nil {
		t.Fatalf("RunText with state failed: %v", err)
	}

	// 2 previous + 1 new user + 1 reply = 4.
	if got := len(response.State.Messages); got != 4 {
		t.Errorf("expected 4 messages, got %d", got)
	}
	// Counter should be 11 (started at 10, incremented once).
	if got := response.State.Custom.Counter; got != 11 {
		t.Errorf("expected counter=11, got %d", got)
	}
}

func TestAgent_RunText_WithSnapshot(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := NewInMemorySessionStore[testState]()

	af := DefineCustomAgent(reg, "runSnapshotFlow",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				sess.AddMessages(ai.NewModelTextMessage("reply"))
				sess.UpdateCustom(func(s testState) testState {
					s.Counter++
					return s
				})
				return nil
			})
		},
		WithSessionStore(store),
	)

	// First invocation via RunText.
	resp1, err := af.RunText(ctx, "first")
	if err != nil {
		t.Fatalf("first RunText failed: %v", err)
	}
	if resp1.SnapshotID == "" {
		t.Fatal("expected snapshot ID from first invocation")
	}

	// Resume from snapshot via RunText.
	resp2, err := af.RunText(ctx, "second", WithSnapshotID[testState](resp1.SnapshotID))
	if err != nil {
		t.Fatalf("second RunText failed: %v", err)
	}

	snap, err := store.GetSnapshot(ctx, resp2.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot failed: %v", err)
	}
	// 4 messages: first user + reply + second user + reply.
	if got := len(snap.State.Messages); got != 4 {
		t.Errorf("expected 4 messages after resume, got %d", got)
	}
	if got := snap.State.Custom.Counter; got != 2 {
		t.Errorf("expected counter=2, got %d", got)
	}
}

func TestPromptAgent_RunText(t *testing.T) {
	ctx := context.Background()
	reg := setupPromptTestRegistry(t)

	ai.DefinePrompt(reg, "runTextPrompt",
		ai.WithModelName("test/echo"),
		ai.WithSystem("You are a test assistant."),
	)

	af := DefineAgent[testState](reg, "runTextPrompt", FromPrompt())

	response, err := af.RunText(ctx, "hello")
	if err != nil {
		t.Fatalf("RunText failed: %v", err)
	}

	// 1 user message + 1 model reply = 2.
	if got := len(response.State.Messages); got != 2 {
		t.Errorf("expected 2 messages, got %d", got)
		for i, m := range response.State.Messages {
			t.Logf("  msg[%d]: role=%s text=%s", i, m.Role, m.Text())
		}
	}
}

func TestPromptAgent_RejectsNonUserRole(t *testing.T) {
	ctx := context.Background()
	reg := setupPromptTestRegistry(t)

	ai.DefinePrompt(reg, "rejectRolePrompt", ai.WithModelName("test/echo"))
	af := DefineAgent[testState](reg, "rejectRolePrompt", FromPrompt())

	_, err := af.Run(ctx, &AgentInput{
		Message: &ai.Message{
			Role:    ai.RoleModel,
			Content: []*ai.Part{ai.NewTextPart("hi")},
		},
	})
	if err == nil {
		t.Fatal("expected error for non-user role, got nil")
	}
	if !strings.Contains(err.Error(), "role") {
		t.Errorf("expected role-related error, got %v", err)
	}
}

func TestPromptAgent_RejectsToolRequestPart(t *testing.T) {
	ctx := context.Background()
	reg := setupPromptTestRegistry(t)

	ai.DefinePrompt(reg, "rejectToolReqPrompt", ai.WithModelName("test/echo"))
	af := DefineAgent[testState](reg, "rejectToolReqPrompt", FromPrompt())

	_, err := af.Run(ctx, &AgentInput{
		Message: &ai.Message{
			Role: ai.RoleUser,
			Content: []*ai.Part{
				ai.NewTextPart("hi"),
				ai.NewToolRequestPart(&ai.ToolRequest{Name: "doThing", Ref: "1"}),
			},
		},
	})
	if err == nil {
		t.Fatal("expected error for tool request part, got nil")
	}
	if !strings.Contains(err.Error(), "tool request") {
		t.Errorf("expected tool-request error, got %v", err)
	}
}

func TestPromptAgent_RejectsToolResponsePart(t *testing.T) {
	ctx := context.Background()
	reg := setupPromptTestRegistry(t)

	ai.DefinePrompt(reg, "rejectToolRespPrompt", ai.WithModelName("test/echo"))
	af := DefineAgent[testState](reg, "rejectToolRespPrompt", FromPrompt())

	_, err := af.Run(ctx, &AgentInput{
		Message: &ai.Message{
			Role: ai.RoleUser,
			Content: []*ai.Part{
				ai.NewToolResponsePart(&ai.ToolResponse{Name: "doThing", Ref: "1"}),
			},
		},
	})
	if err == nil {
		t.Fatal("expected error for tool response part, got nil")
	}
	if !strings.Contains(err.Error(), "tool") {
		t.Errorf("expected tool-related error, got %v", err)
	}
}

func TestAgent_SingleTurnSnapshotDedup(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := NewInMemorySessionStore[testState]()

	af := DefineCustomAgent(reg, "dedupFlow",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				sess.AddMessages(ai.NewModelTextMessage("reply"))
				sess.UpdateCustom(func(s testState) testState {
					s.Counter++
					return s
				})
				return nil
			})
		},
		WithSessionStore(store),
	)

	// Single-turn invocation: should produce exactly 1 snapshot (turn-end),
	// not 2 (turn-end + invocation-end with identical state).
	response, err := af.RunText(ctx, "hello")
	if err != nil {
		t.Fatalf("RunText failed: %v", err)
	}

	if response.SnapshotID == "" {
		t.Fatal("expected snapshot ID in response")
	}

	// Count total snapshots in the store.
	snap, err := store.GetSnapshot(ctx, response.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot failed: %v", err)
	}
	if snap.Event != SnapshotEventTurnEnd {
		t.Errorf("expected turn-end snapshot, got %s", snap.Event)
	}
	// The turn-end snapshot should have no parent (first and only snapshot).
	if snap.ParentID != "" {
		t.Errorf("expected no parent (single snapshot), got parent %q", snap.ParentID)
	}
}

func TestAgent_MultiTurnSnapshotDedup(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := NewInMemorySessionStore[testState]()

	af := DefineCustomAgent(reg, "multiDedupFlow",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				sess.AddMessages(ai.NewModelTextMessage("reply"))
				sess.UpdateCustom(func(s testState) testState {
					s.Counter++
					return s
				})
				return nil
			})
		},
		WithSessionStore(store),
	)

	// Multi-turn: last turn-end snapshot should dedup with invocation-end.
	conn, err := af.StreamBidi(ctx)
	if err != nil {
		t.Fatalf("StreamBidi failed: %v", err)
	}

	var snapshotIDs []string
	for i := 0; i < 3; i++ {
		conn.SendText(fmt.Sprintf("turn %d", i))
		for chunk, err := range conn.Receive() {
			if err != nil {
				t.Fatalf("Receive error on turn %d: %v", i, err)
			}
			if chunk.TurnEnd != nil {
				if chunk.TurnEnd.SnapshotID != "" {
					snapshotIDs = append(snapshotIDs, chunk.TurnEnd.SnapshotID)
				}
				break
			}
		}
	}
	conn.Close()

	response, err := conn.Output()
	if err != nil {
		t.Fatalf("Output failed: %v", err)
	}

	// Should have 3 turn-end snapshots (one per turn), no extra invocation-end.
	if got := len(snapshotIDs); got != 3 {
		t.Errorf("expected 3 turn-end snapshots, got %d", got)
	}

	// The output snapshot ID should reuse the last turn-end snapshot.
	if response.SnapshotID == "" {
		t.Fatal("expected snapshot ID in response")
	}
	if response.SnapshotID != snapshotIDs[len(snapshotIDs)-1] {
		t.Errorf("expected output snapshot to reuse last turn-end snapshot %q, got %q",
			snapshotIDs[len(snapshotIDs)-1], response.SnapshotID)
	}
}

func TestAgent_InvocationEndSnapshotWhenStateChangesAfterRun(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := NewInMemorySessionStore[testState]()

	af := DefineCustomAgent(reg, "postRunMutateFlow",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			if err := sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				sess.AddMessages(ai.NewModelTextMessage("reply"))
				return nil
			}); err != nil {
				return nil, err
			}
			// Mutate state AFTER sess.Run returns -- this should trigger
			// a separate invocation-end snapshot.
			sess.UpdateCustom(func(s testState) testState {
				s.Counter = 99
				return s
			})
			return sess.Result(), nil
		},
		WithSessionStore(store),
	)

	response, err := af.RunText(ctx, "hello")
	if err != nil {
		t.Fatalf("RunText failed: %v", err)
	}

	if response.SnapshotID == "" {
		t.Fatal("expected snapshot ID in response")
	}

	// The final snapshot should be an invocation-end snapshot that captured
	// the post-Run mutation.
	snap, err := store.GetSnapshot(ctx, response.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot failed: %v", err)
	}
	if snap.Event != SnapshotEventInvocationEnd {
		t.Errorf("expected invocation-end snapshot, got %s", snap.Event)
	}
	if snap.State.Custom.Counter != 99 {
		t.Errorf("expected counter=99 in final snapshot, got %d", snap.State.Custom.Counter)
	}
	// Should have a parent (the turn-end snapshot).
	if snap.ParentID == "" {
		t.Error("expected parent ID (turn-end snapshot)")
	}
}

// TestAgent_FnPanicReturnsError verifies that a panic inside the agent
// function is recovered and surfaced as an error, rather than crashing the
// process or hanging the streaming goroutine.
func TestAgent_FnPanicReturnsError(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	af := DefineCustomAgent(reg, "panicFlow",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				resp.SendStatus(testStatus{Phase: "before-panic"})
				panic("boom")
			})
		},
	)

	conn, err := af.StreamBidi(ctx)
	if err != nil {
		t.Fatalf("StreamBidi: %v", err)
	}
	if err := conn.SendText("trigger"); err != nil {
		t.Fatalf("SendText: %v", err)
	}

	done := make(chan error, 1)
	go func() {
		for chunk, err := range conn.Receive() {
			_ = chunk
			if err != nil {
				done <- err
				return
			}
		}
		_, outErr := conn.Output()
		done <- outErr
	}()

	select {
	case err := <-done:
		if err == nil {
			t.Fatal("expected error from panicking fn")
		}
		if !strings.Contains(err.Error(), "panicked") {
			t.Errorf("expected panic error, got: %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Receive/Output hung; streaming goroutine likely leaked")
	}

	conn.Close()
}

// TestAgent_CancelDuringStreamReleasesGoroutine verifies that cancelling the
// context mid-stream does not deadlock the streaming goroutine on outCh send.
func TestAgent_CancelDuringStreamReleasesGoroutine(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	reg := newTestRegistry(t)

	emitting := make(chan struct{})
	fnDone := make(chan struct{})
	af := DefineCustomAgent(reg, "cancelFlow",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			defer close(fnDone)
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				close(emitting)
				// Emit until ctx cancels. Without the goroutine's
				// ctx-aware drain, this would deadlock once the consumer
				// stops reading.
				for {
					select {
					case <-ctx.Done():
						return ctx.Err()
					default:
					}
					resp.SendStatus(testStatus{Phase: "tick"})
				}
			})
		},
	)

	conn, err := af.StreamBidi(ctx)
	if err != nil {
		t.Fatalf("StreamBidi: %v", err)
	}
	if err := conn.SendText("go"); err != nil {
		t.Fatalf("SendText: %v", err)
	}

	<-emitting
	cancel()

	select {
	case <-fnDone:
	case <-time.After(2 * time.Second):
		t.Fatal("agent fn did not return after ctx cancel; goroutine deadlock")
	}
	conn.Close()
}

// --- Detach, transform, and getSnapshot tests ---

// waitForSnapshot polls the store for a snapshot matching the predicate,
// failing the test if it doesn't show up within the timeout.
func waitForSnapshot[State any](
	t *testing.T,
	store SessionStore[State],
	id string,
	timeout time.Duration,
	predicate func(*SessionSnapshot[State]) bool,
) *SessionSnapshot[State] {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		snap, err := store.GetSnapshot(context.Background(), id)
		if err != nil {
			t.Fatalf("GetSnapshot(%q): %v", id, err)
		}
		if snap != nil && predicate(snap) {
			return snap
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("snapshot %q did not satisfy predicate within %s", id, timeout)
	return nil
}

func TestAgent_TurnEnd_CarriesSnapshotID(t *testing.T) {
	// Sanity: each TurnEnd chunk carries the snapshot ID of the turn-end
	// snapshot, and the snapshots themselves are persisted.
	reg := newTestRegistry(t)
	store := NewInMemorySessionStore[testState]()

	af := DefineCustomAgent(reg, "turnEndSnapshotFlow",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				sess.AddMessages(ai.NewModelTextMessage("ok"))
				return nil
			})
		},
		WithSessionStore(store),
	)

	conn, err := af.StreamBidi(context.Background())
	if err != nil {
		t.Fatalf("StreamBidi: %v", err)
	}

	var observed []TurnEnd
	for turn := 0; turn < 3; turn++ {
		if err := conn.SendText(fmt.Sprintf("turn %d", turn)); err != nil {
			t.Fatalf("SendText: %v", err)
		}
		for chunk, err := range conn.Receive() {
			if err != nil {
				t.Fatalf("Receive: %v", err)
			}
			if chunk.TurnEnd != nil {
				observed = append(observed, *chunk.TurnEnd)
				break
			}
		}
	}
	conn.Close()
	if _, err := conn.Output(); err != nil {
		t.Fatalf("Output: %v", err)
	}

	if got := len(observed); got != 3 {
		t.Fatalf("observed %d TurnEnd chunks, want 3", got)
	}
	for i, te := range observed {
		if te.SnapshotID == "" {
			t.Errorf("TurnEnd[%d].SnapshotID is empty", i)
			continue
		}
		snap, err := store.GetSnapshot(context.Background(), te.SnapshotID)
		if err != nil {
			t.Fatalf("GetSnapshot: %v", err)
		}
		if snap == nil {
			t.Errorf("turn %d: snapshot %q not in store", i, te.SnapshotID)
		}
	}
}

func TestAgent_Detach_SuspendsTurnSnapshotsAndProcessesQueue(t *testing.T) {
	// Detach lands while turn 0 (input A) is mid-fn and an extra turn
	// (the detach input D itself) is waiting. The pending snapshot must:
	//   - Be written with empty state and no parent (A was suspended, so
	//     no turn-end snapshot landed before pending).
	//   - NOT write a separate turn-end snapshot for A or D (suspended).
	// After release, the finalized snapshot has both A's and D's effects.
	reg := newTestRegistry(t)
	store := NewInMemorySessionStore[testState]()

	entered := make(chan struct{}, 4)
	release := make(chan struct{})

	af := DefineCustomAgent(reg, "detachInFlight",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				entered <- struct{}{}
				<-release
				sess.AddMessages(ai.NewModelTextMessage("reply-" + input.Message.Text()))
				sess.UpdateCustom(func(s testState) testState {
					s.Counter++
					return s
				})
				return nil
			})
		},
		WithSessionStore(store),
	)

	conn, err := af.StreamBidi(context.Background())
	if err != nil {
		t.Fatalf("StreamBidi: %v", err)
	}

	// Drain stream chunks in the background.
	go func() {
		for _, err := range conn.Receive() {
			if err != nil {
				return
			}
		}
	}()

	// Send A and wait for it to enter fn (so it's in-flight when detach
	// arrives).
	if err := conn.SendText("A"); err != nil {
		t.Fatalf("SendText A: %v", err)
	}
	select {
	case <-entered:
	case <-time.After(2 * time.Second):
		t.Fatal("A did not enter fn")
	}

	// Send D, then Detach. The eager intake reader sees D queued and the
	// detach signal immediately, even though the runner is blocked on A.
	if err := conn.SendText("D"); err != nil {
		t.Fatalf("SendText D: %v", err)
	}
	if err := conn.Detach(); err != nil {
		t.Fatalf("Detach: %v", err)
	}

	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	if out.SnapshotID == "" {
		t.Fatal("expected pending snapshot ID")
	}

	pending, err := store.GetSnapshot(context.Background(), out.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot pending: %v", err)
	}
	if pending.Status != SnapshotStatusPending {
		t.Errorf("pending snapshot status = %q, want pending", pending.Status)
	}
	if pending.State != nil {
		t.Errorf("pending state = %+v, want nil (live state not yet committed)", pending.State)
	}

	// No separate turn-end snapshot for A should have been written.
	// (Walk the parent chain — pending should have no parent in this
	// invocation since A was the first turn and got suspended.)
	if pending.ParentID != "" {
		t.Errorf("pending ParentID = %q, want empty (A was suspended)", pending.ParentID)
	}

	close(release)

	final := waitForSnapshot(t, store, out.SnapshotID, 2*time.Second, func(s *SessionSnapshot[testState]) bool {
		return s.Status == SnapshotStatusSucceeded
	})
	if final.State.Custom.Counter != 2 {
		t.Errorf("final counter = %d, want 2 (A + D both processed)", final.State.Custom.Counter)
	}
	if got := len(final.State.Messages); got != 4 {
		// 2 user (A, D) + 2 model replies = 4.
		t.Errorf("final messages = %d, want 4", got)
	}
}

func TestAgent_Detach_AfterPriorTurns_ChainsParent(t *testing.T) {
	// Run two normal turns first, then detach during a third (in-flight)
	// turn. The pending snapshot must chain off the second turn's snapshot.
	reg := newTestRegistry(t)
	store := NewInMemorySessionStore[testState]()

	enter := make(chan struct{}, 4)
	release := make(chan struct{}, 4)

	af := DefineCustomAgent(reg, "detachChainParent",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				enter <- struct{}{}
				<-release
				sess.AddMessages(ai.NewModelTextMessage("ok"))
				return nil
			})
		},
		WithSessionStore(store),
	)

	conn, err := af.StreamBidi(context.Background())
	if err != nil {
		t.Fatalf("StreamBidi: %v", err)
	}

	// Background drainer.
	go func() {
		for _, err := range conn.Receive() {
			if err != nil {
				return
			}
		}
	}()

	// Run two normal turns.
	for i := 0; i < 2; i++ {
		release <- struct{}{} // pre-load release so this turn's fn doesn't block
		if err := conn.SendText(fmt.Sprintf("sync-%d", i)); err != nil {
			t.Fatalf("SendText: %v", err)
		}
		<-enter
	}
	// Brief settle so the second turn-end snapshot lands before detach.
	time.Sleep(20 * time.Millisecond)
	// Drain enter signal if buffered.
	for len(enter) > 0 {
		<-enter
	}

	// Now start a third turn but DON'T release it — the third turn is
	// in-flight when detach lands.
	if err := conn.SendText("inflight"); err != nil {
		t.Fatalf("SendText inflight: %v", err)
	}
	<-enter // third turn entered fn

	// Send the queued input and detach.
	if err := conn.SendText("detach-msg"); err != nil {
		t.Fatalf("SendText detach-msg: %v", err)
	}
	if err := conn.Detach(); err != nil {
		t.Fatalf("Detach: %v", err)
	}

	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	pending, err := store.GetSnapshot(context.Background(), out.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}
	if pending.ParentID == "" {
		t.Error("pending ParentID empty; expected parent = last sync turn snapshot")
	}

	// Release remaining turns and let finalize run.
	close(release)
	waitForSnapshot(t, store, out.SnapshotID, 2*time.Second, func(s *SessionSnapshot[testState]) bool {
		return s.Status == SnapshotStatusSucceeded
	})
}

func TestAgent_Detach_RequiresStore(t *testing.T) {
	reg := newTestRegistry(t)

	af := DefineCustomAgent(reg, "detachNoStore",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				return nil
			})
		},
	)

	conn, err := af.StreamBidi(context.Background())
	if err != nil {
		t.Fatalf("StreamBidi: %v", err)
	}
	if err := conn.Detach(); err != nil {
		t.Fatalf("Detach send: %v", err)
	}
	conn.Close()

	_, err = conn.Output()
	if err == nil {
		t.Fatal("expected error when detaching without a session store")
	}
	if !strings.Contains(err.Error(), "detach requires a session store") {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestAgent_Detach_PendingThenComplete(t *testing.T) {
	// Client detaches mid-flow; flow finishes naturally; pending snapshot
	// flips to status=succeeded with the full session state.
	reg := newTestRegistry(t)
	store := NewInMemorySessionStore[testState]()

	release := make(chan struct{})
	entered := make(chan struct{})

	af := DefineCustomAgent(reg, "detachComplete",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				select {
				case entered <- struct{}{}:
				case <-ctx.Done():
				}
				<-release
				sess.AddMessages(ai.NewModelTextMessage("finished"))
				sess.UpdateCustom(func(s testState) testState {
					s.Counter = 42
					return s
				})
				return nil
			})
		},
		WithSessionStore(store),
	)

	conn, err := af.StreamBidi(context.Background())
	if err != nil {
		t.Fatalf("StreamBidi: %v", err)
	}

	// Drain chunks so the responder isn't blocked.
	go func() {
		for _, err := range conn.Receive() {
			if err != nil {
				return
			}
		}
	}()

	if err := conn.SendText("go"); err != nil {
		t.Fatalf("SendText: %v", err)
	}
	if err := conn.Detach(); err != nil {
		t.Fatalf("Detach: %v", err)
	}

	select {
	case <-entered:
	case <-time.After(2 * time.Second):
		t.Fatal("flow did not enter work phase")
	}

	// Output returns the pending snapshot ID immediately; the snapshot
	// itself should be in the store with status=pending.
	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	if out.SnapshotID == "" {
		t.Fatal("expected snapshot ID after detach")
	}

	pending, err := store.GetSnapshot(context.Background(), out.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot pending: %v", err)
	}
	if pending == nil {
		t.Fatal("pending snapshot not written")
	}
	if pending.Status != SnapshotStatusPending {
		t.Errorf("expected status=%q, got %q", SnapshotStatusPending, pending.Status)
	}
	if pending.State != nil {
		t.Errorf("pending snapshot should not carry session state, got %+v", pending.State)
	}

	// Release; finalizer rewrites the snapshot with the terminal state.
	close(release)

	finalSnap := waitForSnapshot(t, store, out.SnapshotID, 2*time.Second, func(s *SessionSnapshot[testState]) bool {
		return s.Status == SnapshotStatusSucceeded
	})
	if finalSnap.State.Custom.Counter != 42 {
		t.Errorf("expected counter=42 in final snapshot, got %d", finalSnap.State.Custom.Counter)
	}
	if got := len(finalSnap.State.Messages); got < 2 {
		t.Errorf("expected at least 2 messages in final snapshot, got %d", got)
	}
}

func TestAgent_Detach_SendArtifactPostDetachLandsInSnapshot(t *testing.T) {
	// SendArtifact must behave the same way regardless of whether detach
	// has landed: the artifact is added to the session and shows up in
	// the finalized snapshot's state. The wire forward is the only thing
	// detach suppresses, so flow authors don't need to branch on detach.
	reg := newTestRegistry(t)
	store := NewInMemorySessionStore[testState]()

	detached := make(chan struct{})
	release := make(chan struct{})

	af := DefineCustomAgent(reg, "detachArtifact",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				resp.SendArtifact(&Artifact{
					Name:  "before.txt",
					Parts: []*ai.Part{ai.NewTextPart("pre-detach")},
				})
				select {
				case <-detached:
				case <-ctx.Done():
					return ctx.Err()
				}
				resp.SendArtifact(&Artifact{
					Name:  "after.txt",
					Parts: []*ai.Part{ai.NewTextPart("post-detach")},
				})
				<-release
				return nil
			})
		},
		WithSessionStore(store),
	)

	conn, err := af.StreamBidi(context.Background())
	if err != nil {
		t.Fatalf("StreamBidi: %v", err)
	}
	go func() {
		for _, err := range conn.Receive() {
			if err != nil {
				return
			}
		}
	}()

	if err := conn.SendText("go"); err != nil {
		t.Fatalf("SendText: %v", err)
	}
	if err := conn.Detach(); err != nil {
		t.Fatalf("Detach: %v", err)
	}

	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	if out.SnapshotID == "" {
		t.Fatal("expected pending snapshot ID")
	}

	close(detached)
	close(release)

	final := waitForSnapshot(t, store, out.SnapshotID, 2*time.Second, func(s *SessionSnapshot[testState]) bool {
		return s.Status == SnapshotStatusSucceeded
	})

	names := make(map[string]bool, len(final.State.Artifacts))
	for _, a := range final.State.Artifacts {
		names[a.Name] = true
	}
	if !names["before.txt"] {
		t.Errorf("pre-detach artifact missing from final snapshot: %v", final.State.Artifacts)
	}
	if !names["after.txt"] {
		t.Errorf("post-detach artifact missing from final snapshot: %v", final.State.Artifacts)
	}
}

func TestAgent_Detach_FlowErrorsBecomesError(t *testing.T) {
	reg := newTestRegistry(t)
	store := NewInMemorySessionStore[testState]()

	release := make(chan struct{})
	entered := make(chan struct{})
	boom := errors.New("kaboom")

	af := DefineCustomAgent(reg, "detachErr",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				select {
				case entered <- struct{}{}:
				case <-time.After(time.Second):
				}
				<-release
				return boom
			})
		},
		WithSessionStore(store),
	)

	conn, err := af.StreamBidi(context.Background())
	if err != nil {
		t.Fatalf("StreamBidi: %v", err)
	}
	go func() {
		for _, err := range conn.Receive() {
			if err != nil {
				return
			}
		}
	}()

	if err := conn.SendText("go"); err != nil {
		t.Fatalf("SendText: %v", err)
	}
	if err := conn.Detach(); err != nil {
		t.Fatalf("Detach: %v", err)
	}
	<-entered

	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	if out.SnapshotID == "" {
		t.Fatal("expected pending snapshot ID")
	}

	close(release)

	snap := waitForSnapshot(t, store, out.SnapshotID, 2*time.Second, func(s *SessionSnapshot[testState]) bool {
		return s.Status == SnapshotStatusFailed
	})
	if snap.Error == nil || !strings.Contains(snap.Error.Message, "kaboom") {
		t.Errorf("expected snapshot.Error.Message to contain %q, got %+v", "kaboom", snap.Error)
	}

	// Resuming from an errored detached snapshot is rejected.
	_, err = af.RunText(context.Background(), "retry", WithSnapshotID[testState](out.SnapshotID))
	if err == nil {
		t.Fatal("expected error when resuming from errored snapshot")
	}
	if !strings.Contains(err.Error(), "kaboom") {
		t.Errorf("unexpected resume error: %v", err)
	}
}

func TestAgent_Detach_AbortSnapshotStopsFlow(t *testing.T) {
	// Client detaches, then calls AbortSnapshot. The store's status
	// subscriber notifies the runtime, which cancels the work context, and
	// the finalizer rewrites the snapshot with status=aborted.
	reg := newTestRegistry(t)
	store := NewInMemorySessionStore[testState]()

	entered := make(chan struct{})

	af := DefineCustomAgent(reg, "detachAbort",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				select {
				case entered <- struct{}{}:
				case <-time.After(time.Second):
				}
				<-ctx.Done()
				return ctx.Err()
			})
		},
		WithSessionStore(store),
	)

	conn, err := af.StreamBidi(context.Background())
	if err != nil {
		t.Fatalf("StreamBidi: %v", err)
	}
	go func() {
		for _, err := range conn.Receive() {
			if err != nil {
				return
			}
		}
	}()

	if err := conn.SendText("go"); err != nil {
		t.Fatalf("SendText: %v", err)
	}
	if err := conn.Detach(); err != nil {
		t.Fatalf("Detach: %v", err)
	}
	<-entered

	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	if out.SnapshotID == "" {
		t.Fatal("expected pending snapshot ID")
	}

	// Abort via the store. The local caller already has the store
	// reference from WithSessionStore.
	status, err := store.AbortSnapshot(context.Background(), out.SnapshotID)
	if err != nil {
		t.Fatalf("AbortSnapshot: %v", err)
	}
	if status != SnapshotStatusAborted {
		t.Errorf("AbortSnapshot status = %q, want aborted", status)
	}

	// The subscriber wakes the runtime, cancels work, and the finalizer
	// rewrites the snapshot with the aborted status.
	finalSnap := waitForSnapshot(t, store, out.SnapshotID, 2*time.Second, func(s *SessionSnapshot[testState]) bool {
		return s.Status == SnapshotStatusAborted && s.UpdatedAt.After(s.CreatedAt)
	})
	// The flow only blocked on ctx — no state mutation expected. State
	// may be nil (when AbortSnapshot landed before the finalizer's write
	// could populate it) or a populated zero-value struct.
	if finalSnap.State != nil && finalSnap.State.Custom.Counter != 0 {
		t.Errorf("unexpected counter value in aborted snapshot: %d", finalSnap.State.Custom.Counter)
	}
}

func TestAgent_Detach_NormalCompletionStillEmitsTurnEnd(t *testing.T) {
	// Sanity: a non-detached invocation against a store-backed flow still
	// behaves like a synchronous flow (turn-end snapshots, no pending row).
	reg := newTestRegistry(t)
	store := NewInMemorySessionStore[testState]()

	af := DefineCustomAgent(reg, "syncStillWorks",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				sess.AddMessages(ai.NewModelTextMessage("ok"))
				return nil
			})
		},
		WithSessionStore(store),
	)

	conn, err := af.StreamBidi(context.Background())
	if err != nil {
		t.Fatalf("StreamBidi: %v", err)
	}
	if err := conn.SendText("hi"); err != nil {
		t.Fatalf("SendText: %v", err)
	}

	var turnEndID string
	for chunk, err := range conn.Receive() {
		if err != nil {
			t.Fatalf("Receive: %v", err)
		}
		if chunk.TurnEnd != nil {
			turnEndID = chunk.TurnEnd.SnapshotID
			break
		}
	}
	if turnEndID == "" {
		t.Fatal("expected snapshot ID on TurnEnd chunk")
	}
	conn.Close()
	if _, err := conn.Output(); err != nil {
		t.Fatalf("Output: %v", err)
	}

	snap, err := store.GetSnapshot(context.Background(), turnEndID)
	if err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}
	if snap.Status != SnapshotStatusSucceeded {
		t.Errorf("turn-end snapshot status = %q, want succeeded", snap.Status)
	}
	if snap.Event != SnapshotEventTurnEnd {
		t.Errorf("turn-end snapshot event = %q, want %q", snap.Event, SnapshotEventTurnEnd)
	}
}

func TestAgent_Detach_ClientDisconnectBeforeDetachCancels(t *testing.T) {
	// Without detach, a client cancel still cancels the work — this is
	// the regression guard for "until detach=true is called, this is a
	// normal HTTP/WS connection that cancels on close."
	reg := newTestRegistry(t)
	store := NewInMemorySessionStore[testState]()

	entered := make(chan struct{})
	exited := make(chan error, 1)

	af := DefineCustomAgent(reg, "syncCancel",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			err := sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				select {
				case entered <- struct{}{}:
				case <-ctx.Done():
				}
				<-ctx.Done()
				return ctx.Err()
			})
			exited <- err
			return nil, err
		},
		WithSessionStore(store),
	)

	ctx, cancel := context.WithCancel(context.Background())
	conn, err := af.StreamBidi(ctx)
	if err != nil {
		t.Fatalf("StreamBidi: %v", err)
	}
	go func() {
		for _, err := range conn.Receive() {
			if err != nil {
				return
			}
		}
	}()

	if err := conn.SendText("go"); err != nil {
		t.Fatalf("SendText: %v", err)
	}
	<-entered
	cancel()

	select {
	case fnErr := <-exited:
		if fnErr == nil {
			t.Error("expected fn to exit with ctx error after client cancel")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("fn did not exit after client cancel")
	}
}

func TestAgent_ResumeFromErrorSnapshot_Rejected(t *testing.T) {
	reg := newTestRegistry(t)
	store := NewInMemorySessionStore[testState]()

	erroredID := "errored-456"
	if _, err := store.SaveSnapshot(context.Background(), erroredID,
		func(_ *SessionSnapshot[testState]) (*SessionSnapshot[testState], error) {
			return &SessionSnapshot[testState]{
				Event:  SnapshotEventInvocationEnd,
				Status: SnapshotStatusFailed,
				Error: &core.GenkitError{
					Status:  core.INTERNAL,
					Message: "underlying failure",
				},
				State: &SessionState[testState]{},
			}, nil
		}); err != nil {
		t.Fatalf("SaveSnapshot: %v", err)
	}

	af := DefineCustomAgent(reg, "resumeErrored",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, nil
		},
		WithSessionStore(store),
	)

	_, err := af.RunText(context.Background(), "hi", WithSnapshotID[testState](erroredID))
	if err == nil {
		t.Fatal("expected error when resuming from errored snapshot")
	}
	if !strings.Contains(err.Error(), "underlying failure") {
		t.Errorf("expected error to surface underlying failure, got: %v", err)
	}
}

func TestAgent_GetSnapshotAction_ReturnsTransformedState(t *testing.T) {
	reg := newTestRegistry(t)
	store := NewInMemorySessionStore[testState]()

	// Transform that scrubs a specific word from all messages.
	transform := func(_ context.Context, s *SessionState[testState]) *SessionState[testState] {
		for _, msg := range s.Messages {
			for _, p := range msg.Content {
				if p.Text != "" {
					p.Text = strings.ReplaceAll(p.Text, "secret", "[REDACTED]")
				}
			}
		}
		return s
	}

	af := DefineCustomAgent(reg, "transformedFlow",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				sess.AddMessages(ai.NewModelTextMessage("the secret is out"))
				return nil
			})
		},
		WithSessionStore(store),
		WithStateTransform[testState](transform),
	)

	ctx := context.Background()
	out, err := af.RunText(ctx, "tell me the secret")
	if err != nil {
		t.Fatalf("RunText: %v", err)
	}

	// Transform is action-layer behavior: invoke the registered action
	// directly the way a non-Go client would.
	action := core.ResolveActionFor[*GetSnapshotRequest, *GetSnapshotResponse[testState], struct{}, struct{}](
		reg, api.ActionTypeAgentSnapshot, "transformedFlow")
	if action == nil {
		t.Fatal("getSnapshot action not registered")
	}
	resp, err := action.Run(ctx, &GetSnapshotRequest{SnapshotID: out.SnapshotID}, nil)
	if err != nil {
		t.Fatalf("getSnapshot action: %v", err)
	}
	if resp.SnapshotID != out.SnapshotID {
		t.Errorf("SnapshotID mismatch: got %q want %q", resp.SnapshotID, out.SnapshotID)
	}
	if resp.Status != SnapshotStatusSucceeded {
		t.Errorf("expected status=succeeded, got %q", resp.Status)
	}
	if resp.State == nil {
		t.Fatal("expected state in response")
	}
	// Both messages should be redacted: user message (from input) and model reply.
	for i, msg := range resp.State.Messages {
		for _, p := range msg.Content {
			if strings.Contains(p.Text, "secret") {
				t.Errorf("message %d still contains 'secret': %q", i, p.Text)
			}
		}
	}

	// The stored snapshot must remain untransformed so the flow can be
	// resumed faithfully.
	stored, err := store.GetSnapshot(ctx, out.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot direct: %v", err)
	}
	foundRaw := false
	for _, msg := range stored.State.Messages {
		for _, p := range msg.Content {
			if strings.Contains(p.Text, "secret") {
				foundRaw = true
			}
		}
	}
	if !foundRaw {
		t.Error("expected stored snapshot to retain the original 'secret' text")
	}
}

func TestInMemorySessionStore_GetSnapshot_NotFound(t *testing.T) {
	store := NewInMemorySessionStore[testState]()

	snap, err := store.GetSnapshot(context.Background(), "nope")
	if err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}
	if snap != nil {
		t.Errorf("expected nil for missing snapshot, got %+v", snap)
	}
}

func TestAgent_GetSnapshotAction_NoStore(t *testing.T) {
	// With no SessionStore configured, neither companion action should
	// be registered: there is no server-side snapshot to fetch or abort.
	reg := newTestRegistry(t)

	DefineCustomAgent(reg, "noStoreFlow",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, nil
		},
	)

	getAction := core.ResolveActionFor[*GetSnapshotRequest, *GetSnapshotResponse[testState], struct{}, struct{}](
		reg, api.ActionTypeAgentSnapshot, "noStoreFlow")
	if getAction != nil {
		t.Error("getSnapshot action should NOT be registered without a store")
	}
	abortAction := core.ResolveActionFor[*AbortSnapshotRequest, *AbortSnapshotResponse, struct{}, struct{}](
		reg, api.ActionTypeAgentAbort, "noStoreFlow")
	if abortAction != nil {
		t.Error("abortSnapshot action should NOT be registered without a store")
	}
}

func TestLoadSession_AgentInitValidation(t *testing.T) {
	// loadSession enforces the AgentInit invariants:
	//   - snapshotId and state are mutually exclusive,
	//   - snapshotId requires a store (server-managed state),
	//   - state requires the absence of a store (client-managed state).
	ctx := context.Background()
	store := NewInMemorySessionStore[testState]()
	state := &SessionState[testState]{Custom: testState{Counter: 1}}

	cases := []struct {
		name    string
		init    *AgentInit[testState]
		store   SessionStore[testState]
		wantErr string
	}{
		{
			name:    "both snapshotId and state set",
			init:    &AgentInit[testState]{SnapshotID: "snap-1", State: state},
			store:   store,
			wantErr: "mutually exclusive",
		},
		{
			name:    "both set, no store",
			init:    &AgentInit[testState]{SnapshotID: "snap-1", State: state},
			store:   nil,
			wantErr: "mutually exclusive",
		},
		{
			name:    "state with server-managed agent",
			init:    &AgentInit[testState]{State: state},
			store:   store,
			wantErr: "server-managed state",
		},
		{
			name:    "snapshotId with client-managed agent",
			init:    &AgentInit[testState]{SnapshotID: "snap-1"},
			store:   nil,
			wantErr: "client-managed state",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, _, err := loadSession(ctx, tc.init, tc.store)
			if err == nil {
				t.Fatalf("expected error, got nil")
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("error %q does not contain %q", err.Error(), tc.wantErr)
			}
		})
	}

	t.Run("empty init with server store is allowed", func(t *testing.T) {
		sess, snap, err := loadSession(ctx, &AgentInit[testState]{}, store)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if sess == nil {
			t.Fatal("expected session, got nil")
		}
		if snap != nil {
			t.Errorf("expected no snapshot, got %+v", snap)
		}
	})

	t.Run("empty init with no store is allowed", func(t *testing.T) {
		sess, snap, err := loadSession(ctx, &AgentInit[testState]{}, nil)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if sess == nil {
			t.Fatal("expected session, got nil")
		}
		if snap != nil {
			t.Errorf("expected no snapshot, got %+v", snap)
		}
	})
}

// minimalStore is a SessionStore that does NOT implement SnapshotAborter.
// Used to verify the abort action stays unregistered for stores that
// lack the capability.
type minimalStore[State any] struct{}

func (minimalStore[State]) GetSnapshot(context.Context, string) (*SessionSnapshot[State], error) {
	return nil, nil
}
func (minimalStore[State]) SaveSnapshot(
	context.Context, string,
	func(*SessionSnapshot[State]) (*SessionSnapshot[State], error),
) (*SessionSnapshot[State], error) {
	return nil, nil
}

func TestAgent_AgentMetadata(t *testing.T) {
	// Verify the metadata["agent"] payload on the flow's action descriptor
	// correctly reports stateManagement and abortable for each combination
	// of store capabilities.
	noopFn := func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
		return nil, nil
	}

	cases := []struct {
		name        string
		define      func(reg api.Registry, flowName string)
		wantMgmt    AgentStateManagement
		wantAbortab bool
	}{
		{
			name: "no store → client-managed, not abortable",
			define: func(reg api.Registry, flowName string) {
				DefineCustomAgent(reg, flowName, noopFn)
			},
			wantMgmt:    AgentStateManagementClient,
			wantAbortab: false,
		},
		{
			name: "store missing abort capabilities → server-managed, not abortable",
			define: func(reg api.Registry, flowName string) {
				DefineCustomAgent(reg, flowName, noopFn,
					WithSessionStore[testState](minimalStore[testState]{}))
			},
			wantMgmt:    AgentStateManagementServer,
			wantAbortab: false,
		},
		{
			name: "store with full capabilities → server-managed, abortable",
			define: func(reg api.Registry, flowName string) {
				DefineCustomAgent(reg, flowName, noopFn,
					WithSessionStore(NewInMemorySessionStore[testState]()))
			},
			wantMgmt:    AgentStateManagementServer,
			wantAbortab: true,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			reg := newTestRegistry(t)
			flowName := "metaFlow"
			tc.define(reg, flowName)

			act := core.ResolveActionFor[*AgentInit[testState], *AgentOutput[testState], *AgentStreamChunk[testStatus], *AgentInput](
				reg, api.ActionTypeFlow, flowName)
			if act == nil {
				t.Fatal("flow action not registered")
			}
			desc := act.Desc()
			raw, ok := desc.Metadata["agent"]
			if !ok {
				t.Fatalf("metadata[\"agent\"] missing; got metadata = %+v", desc.Metadata)
			}
			meta, ok := raw.(AgentMetadata)
			if !ok {
				t.Fatalf("metadata[\"agent\"] type = %T, want AgentMetadata", raw)
			}
			if meta.StateManagement != tc.wantMgmt {
				t.Errorf("stateManagement = %q, want %q", meta.StateManagement, tc.wantMgmt)
			}
			if meta.Abortable != tc.wantAbortab {
				t.Errorf("abortable = %v, want %v", meta.Abortable, tc.wantAbortab)
			}
		})
	}
}

func TestAgent_AbortAction_GatedOnCapabilities(t *testing.T) {
	// Verify the abort companion action is only registered when the
	// store implements SnapshotAborter. The getSnapshot action is
	// registered regardless.
	t.Run("aborter capability → both registered", func(t *testing.T) {
		reg := newTestRegistry(t)
		store := NewInMemorySessionStore[testState]() // implements SnapshotAborter
		DefineCustomAgent(reg, "fullCaps",
			func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
				return nil, nil
			},
			WithSessionStore(store),
		)
		getAction := core.ResolveActionFor[*GetSnapshotRequest, *GetSnapshotResponse[testState], struct{}, struct{}](
			reg, api.ActionTypeAgentSnapshot, "fullCaps")
		if getAction == nil {
			t.Error("getSnapshot action should be registered")
		}
		abortAction := core.ResolveActionFor[*AbortSnapshotRequest, *AbortSnapshotResponse, struct{}, struct{}](
			reg, api.ActionTypeAgentAbort, "fullCaps")
		if abortAction == nil {
			t.Error("abortSnapshot action should be registered when store implements SnapshotAborter")
		}
	})

	t.Run("no aborter capability → abort not registered", func(t *testing.T) {
		reg := newTestRegistry(t)
		DefineCustomAgent(reg, "minCaps",
			func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
				return nil, nil
			},
			WithSessionStore[testState](minimalStore[testState]{}),
		)
		getAction := core.ResolveActionFor[*GetSnapshotRequest, *GetSnapshotResponse[testState], struct{}, struct{}](
			reg, api.ActionTypeAgentSnapshot, "minCaps")
		if getAction == nil {
			t.Error("getSnapshot action should be registered even when store lacks SnapshotAborter")
		}
		abortAction := core.ResolveActionFor[*AbortSnapshotRequest, *AbortSnapshotResponse, struct{}, struct{}](
			reg, api.ActionTypeAgentAbort, "minCaps")
		if abortAction != nil {
			t.Error("abortSnapshot action should NOT be registered when store lacks SnapshotAborter")
		}
	})
}

func TestAgent_AbortAction_NotFound(t *testing.T) {
	// The store's "not found" sentinel (empty status, nil error) must
	// surface as a core.NOT_FOUND GenkitError on the abort companion
	// action so callers (Dev UI, remote clients) see a proper status.
	reg := newTestRegistry(t)
	DefineCustomAgent(reg, "missingFlow",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, nil
		},
		WithSessionStore(NewInMemorySessionStore[testState]()),
	)

	abortAction := core.ResolveActionFor[*AbortSnapshotRequest, *AbortSnapshotResponse, struct{}, struct{}](
		reg, api.ActionTypeAgentAbort, "missingFlow")
	if abortAction == nil {
		t.Fatal("abortSnapshot action should be registered")
	}

	_, err := abortAction.Run(context.Background(), &AbortSnapshotRequest{SnapshotID: "no-such-snap"}, nil)
	if err == nil {
		t.Fatal("expected error for missing snapshot, got nil")
	}
	var ge *core.GenkitError
	if !errors.As(err, &ge) {
		t.Fatalf("expected *core.GenkitError, got %T: %v", err, err)
	}
	if ge.Status != core.NOT_FOUND {
		t.Errorf("status = %q, want %q", ge.Status, core.NOT_FOUND)
	}
}

func TestAgent_StateTransform_ClientManagedState(t *testing.T) {
	reg := newTestRegistry(t)

	// Client-managed state: transform should be applied to AgentOutput.State.
	transform := func(_ context.Context, s *SessionState[testState]) *SessionState[testState] {
		// Zero out the counter to demonstrate the transform is applied.
		s.Custom.Counter = -1
		return s
	}

	af := DefineCustomAgent(reg, "clientXformFlow",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				sess.UpdateCustom(func(s testState) testState {
					s.Counter = 7
					return s
				})
				return nil
			})
		},
		WithStateTransform[testState](transform),
	)

	out, err := af.RunText(context.Background(), "go")
	if err != nil {
		t.Fatalf("RunText: %v", err)
	}
	if out.State == nil {
		t.Fatal("expected client-managed state in output")
	}
	if out.State.Custom.Counter != -1 {
		t.Errorf("expected transformed counter=-1, got %d", out.State.Custom.Counter)
	}
}

func TestAgent_ResumeFromFinalizedDetachedSnapshot(t *testing.T) {
	// End-to-end: run a flow that the client detaches from, let it
	// finalize, then resume from its snapshot as if reconnecting later.
	reg := newTestRegistry(t)
	store := NewInMemorySessionStore[testState]()

	af := DefineCustomAgent(reg, "resumeDetachedFlow",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				sess.AddMessages(ai.NewModelTextMessage("reply"))
				sess.UpdateCustom(func(s testState) testState {
					s.Counter++
					return s
				})
				return nil
			})
		},
		WithSessionStore(store),
	)

	ctx := context.Background()

	// First invocation: detach to write a pending snapshot, then wait
	// for finalize.
	conn, err := af.StreamBidi(ctx)
	if err != nil {
		t.Fatalf("StreamBidi: %v", err)
	}
	go func() {
		for _, err := range conn.Receive() {
			if err != nil {
				return
			}
		}
	}()
	if err := conn.SendText("turn 1"); err != nil {
		t.Fatalf("SendText: %v", err)
	}
	if err := conn.Detach(); err != nil {
		t.Fatalf("Detach: %v", err)
	}
	first, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	finalSnap := waitForSnapshot(t, store, first.SnapshotID, 2*time.Second, func(s *SessionSnapshot[testState]) bool {
		return s.Status == SnapshotStatusSucceeded
	})
	if finalSnap.State.Custom.Counter != 1 {
		t.Fatalf("expected counter=1 in finalized snapshot, got %d", finalSnap.State.Custom.Counter)
	}

	// Resume from the finalized snapshot.
	second, err := af.RunText(ctx, "turn 2", WithSnapshotID[testState](first.SnapshotID))
	if err != nil {
		t.Fatalf("resume RunText: %v", err)
	}

	snap, err := store.GetSnapshot(ctx, second.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}
	if snap.State.Custom.Counter != 2 {
		t.Errorf("expected counter=2 after resume, got %d", snap.State.Custom.Counter)
	}
}

func TestInMemorySessionStore_AbortSnapshot_AtomicAndIdempotent(t *testing.T) {
	ctx := context.Background()
	store := NewInMemorySessionStore[testState]()

	// Abort on missing snapshot returns empty status, no error.
	if status, err := store.AbortSnapshot(ctx, "nope"); err != nil || status != "" {
		t.Fatalf("AbortSnapshot(missing) = %q, %v; want \"\", nil", status, err)
	}

	// Pending → aborted, UpdatedAt advances (verified via GetSnapshot).
	pending, err := store.SaveSnapshot(ctx, "snap-cas",
		func(_ *SessionSnapshot[testState]) (*SessionSnapshot[testState], error) {
			return &SessionSnapshot[testState]{
				Event:  SnapshotEventDetach,
				Status: SnapshotStatusPending,
			}, nil
		})
	if err != nil {
		t.Fatalf("SaveSnapshot: %v", err)
	}
	time.Sleep(time.Millisecond) // ensure measurable UpdatedAt delta
	status, err := store.AbortSnapshot(ctx, "snap-cas")
	if err != nil {
		t.Fatalf("AbortSnapshot: %v", err)
	}
	if status != SnapshotStatusAborted {
		t.Errorf("status after first abort = %q, want aborted", status)
	}
	afterFirst, err := store.GetSnapshot(ctx, "snap-cas")
	if err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}
	if !afterFirst.UpdatedAt.After(pending.UpdatedAt) {
		t.Errorf("UpdatedAt did not advance: %v vs %v", afterFirst.UpdatedAt, pending.UpdatedAt)
	}

	// Idempotent: second abort returns aborted, no error, no further mutation.
	firstUpdate := afterFirst.UpdatedAt
	status2, err := store.AbortSnapshot(ctx, "snap-cas")
	if err != nil {
		t.Fatalf("AbortSnapshot (second): %v", err)
	}
	if status2 != SnapshotStatusAborted {
		t.Errorf("status after second abort = %q, want aborted", status2)
	}
	afterSecond, err := store.GetSnapshot(ctx, "snap-cas")
	if err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}
	if !afterSecond.UpdatedAt.Equal(firstUpdate) {
		t.Errorf("UpdatedAt advanced on idempotent abort: %v vs %v", afterSecond.UpdatedAt, firstUpdate)
	}

	// Abort on terminal status is a no-op that returns the existing status.
	if _, err := store.SaveSnapshot(ctx, "snap-complete",
		func(_ *SessionSnapshot[testState]) (*SessionSnapshot[testState], error) {
			return &SessionSnapshot[testState]{
				Event:  SnapshotEventTurnEnd,
				Status: SnapshotStatusSucceeded,
			}, nil
		}); err != nil {
		t.Fatalf("SaveSnapshot: %v", err)
	}
	status3, err := store.AbortSnapshot(ctx, "snap-complete")
	if err != nil {
		t.Fatalf("AbortSnapshot on complete: %v", err)
	}
	if status3 != SnapshotStatusSucceeded {
		t.Errorf("abort on complete returned status=%q, want succeeded", status3)
	}
}

func TestAgent_Detach_FinalizeRespectsConcurrentAbort(t *testing.T) {
	// An abort that lands while fn is still running but does not actually
	// stop fn (because fn does not observe ctx) must still result in
	// status=aborted — the finalizer must not clobber aborted with
	// complete. The subscriber observes the status flip and the finalizer
	// reads the resulting flag.
	reg := newTestRegistry(t)
	store := NewInMemorySessionStore[testState]()

	fnRelease := make(chan struct{})
	entered := make(chan struct{})

	af := DefineCustomAgent(reg, "raceFinalize",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				select {
				case entered <- struct{}{}:
				case <-time.After(time.Second):
				}
				<-fnRelease
				// Return cleanly without observing ctx. Without the
				// subscriber/recheck, this would land status=succeeded and
				// clobber the abort.
				return nil
			})
		},
		WithSessionStore(store),
	)

	conn, err := af.StreamBidi(context.Background())
	if err != nil {
		t.Fatalf("StreamBidi: %v", err)
	}
	go func() {
		for _, err := range conn.Receive() {
			if err != nil {
				return
			}
		}
	}()

	if err := conn.SendText("go"); err != nil {
		t.Fatalf("SendText: %v", err)
	}
	if err := conn.Detach(); err != nil {
		t.Fatalf("Detach: %v", err)
	}
	<-entered

	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}

	// Externally abort before releasing fn.
	if _, err := store.AbortSnapshot(context.Background(), out.SnapshotID); err != nil {
		t.Fatalf("AbortSnapshot: %v", err)
	}

	close(fnRelease)

	finalSnap := waitForSnapshot(t, store, out.SnapshotID, 2*time.Second, func(s *SessionSnapshot[testState]) bool {
		return s.Status == SnapshotStatusAborted || s.Status == SnapshotStatusSucceeded
	})
	if finalSnap.Status != SnapshotStatusAborted {
		t.Errorf("finalize clobbered aborted with %q", finalSnap.Status)
	}
}

func TestInMemorySessionStore_OnSnapshotStatusChange(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	store := NewInMemorySessionStore[testState]()

	// Subscribe to a missing snapshot: channel returns immediately closed
	// without yielding a value.
	missing := store.OnSnapshotStatusChange(ctx, "nope")
	if _, ok := <-missing; ok {
		t.Errorf("expected channel for missing snapshot to be closed without a value")
	}

	// Persist a pending snapshot so subsequent subscribers get an initial
	// value plus updates on each status flip.
	if _, err := store.SaveSnapshot(ctx, "snap-sub",
		func(_ *SessionSnapshot[testState]) (*SessionSnapshot[testState], error) {
			return &SessionSnapshot[testState]{
				Event:  SnapshotEventDetach,
				Status: SnapshotStatusPending,
			}, nil
		}); err != nil {
		t.Fatalf("SaveSnapshot: %v", err)
	}

	subCtx, subCancel := context.WithCancel(ctx)
	defer subCancel()
	statusCh := store.OnSnapshotStatusChange(subCtx, "snap-sub")

	// Initial value reflects current status.
	select {
	case status, ok := <-statusCh:
		if !ok {
			t.Fatal("channel closed before initial status")
		}
		if status != SnapshotStatusPending {
			t.Errorf("initial status = %q, want pending", status)
		}
	case <-time.After(time.Second):
		t.Fatal("did not receive initial status")
	}

	// Abort flips status; subscriber observes aborted.
	if _, err := store.AbortSnapshot(ctx, "snap-sub"); err != nil {
		t.Fatalf("AbortSnapshot: %v", err)
	}
	select {
	case status, ok := <-statusCh:
		if !ok {
			t.Fatal("channel closed before abort notification")
		}
		if status != SnapshotStatusAborted {
			t.Errorf("status notification = %q, want aborted", status)
		}
	case <-time.After(time.Second):
		t.Fatal("did not receive abort notification")
	}

	// Cancelling the subscription closes the channel.
	subCancel()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		_, ok := <-statusCh
		if !ok {
			return
		}
	}
	t.Fatal("channel did not close after subscription ctx cancel")
}

func TestAgent_AbortSnapshot_NoOpOnTerminal(t *testing.T) {
	// Calling AbortSnapshot on an already-terminal snapshot is a no-op
	// that returns the existing status.
	reg := newTestRegistry(t)
	store := NewInMemorySessionStore[testState]()

	af := DefineCustomAgent(reg, "abortNoop",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				sess.AddMessages(ai.NewModelTextMessage("reply"))
				return nil
			})
		},
		WithSessionStore(store),
	)

	ctx := context.Background()
	out, err := af.RunText(ctx, "hi")
	if err != nil {
		t.Fatalf("RunText: %v", err)
	}

	status, err := store.AbortSnapshot(ctx, out.SnapshotID)
	if err != nil {
		t.Fatalf("AbortSnapshot: %v", err)
	}
	if status != SnapshotStatusSucceeded {
		t.Errorf("expected status=%q (existing terminal), got %q", SnapshotStatusSucceeded, status)
	}

	// Confirm the store snapshot was not flipped.
	snap, err := store.GetSnapshot(ctx, out.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}
	if snap.Status != SnapshotStatusSucceeded {
		t.Errorf("snapshot status = %q after abort-on-terminal, want succeeded", snap.Status)
	}
}

func TestAgent_ResultAndOutput_IsolatedFromSession(t *testing.T) {
	// Result() and AgentOutput must contain deep copies of session state so
	// neither the fn (after calling Result) nor the caller (after receiving
	// Output) can mutate session contents through them. Both layers
	// deep-copy: Result for fn-side ergonomics, handleFnDone for defense
	// in depth in case fn returns AgentResult built with raw session
	// pointers instead of going through Result().
	reg := newTestRegistry(t)
	store := NewInMemorySessionStore[testState]()

	var (
		sessionMsgAfterMutation string
		sessionArtAfterMutation string
		fnReturnedMessage       *ai.Message
		fnReturnedArtifact      *Artifact
	)

	af := DefineCustomAgent(reg, "isolation",
		func(ctx context.Context, resp Responder[testStatus], sess *SessionRunner[testState]) (*AgentResult, error) {
			if err := sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				sess.AddMessages(ai.NewModelTextMessage("session-msg"))
				sess.AddArtifacts(&Artifact{
					Name:  "orig",
					Parts: []*ai.Part{ai.NewTextPart("orig-part")},
				})
				return nil
			}); err != nil {
				return nil, err
			}

			result := sess.Result()
			// Mutate the returned result; must not touch session state.
			result.Message.Content[0].Text = "fn-tainted-msg"
			result.Artifacts[0].Name = "fn-tainted-art"

			// Capture the session view AFTER mutation so the outer test
			// can verify the mutation didn't bleed through.
			msgs := sess.Messages()
			sessionMsgAfterMutation = msgs[len(msgs)-1].Content[0].Text
			arts := sess.Artifacts()
			sessionArtAfterMutation = arts[0].Name

			// Capture the pointers fn is returning so the outer test
			// can verify handleFnDone copied them (i.e., out.Message
			// is not the same pointer as what fn handed back).
			fnReturnedMessage = result.Message
			fnReturnedArtifact = result.Artifacts[0]
			return result, nil
		},
		WithSessionStore(store),
	)

	out, err := af.RunText(context.Background(), "go")
	if err != nil {
		t.Fatalf("RunText: %v", err)
	}

	// Result() must have given fn an isolated copy.
	if sessionMsgAfterMutation != "session-msg" {
		t.Errorf("session message tainted by fn mutation of Result(): got %q, want %q",
			sessionMsgAfterMutation, "session-msg")
	}
	if sessionArtAfterMutation != "orig" {
		t.Errorf("session artifact tainted by fn mutation of Result(): got %q, want %q",
			sessionArtAfterMutation, "orig")
	}

	// handleFnDone must have copied fn's returned pointers at the framework
	// boundary, so caller-side mutations cannot reach what fn handed back.
	if out.Message == fnReturnedMessage {
		t.Error("AgentOutput.Message shares pointer with fn's returned message; handleFnDone defensive copy missing")
	}
	if len(out.Artifacts) > 0 && out.Artifacts[0] == fnReturnedArtifact {
		t.Error("AgentOutput.Artifacts[0] shares pointer with fn's returned artifact; handleFnDone defensive copy missing")
	}

	// The persisted snapshot must reflect the un-tainted session state.
	snap, err := store.GetSnapshot(context.Background(), out.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}
	if got := snap.State.Messages[len(snap.State.Messages)-1].Content[0].Text; got != "session-msg" {
		t.Errorf("snapshot message tainted: got %q, want %q", got, "session-msg")
	}
	if snap.State.Artifacts[0].Name != "orig" {
		t.Errorf("snapshot artifact tainted: got %q, want %q", snap.State.Artifacts[0].Name, "orig")
	}
}
