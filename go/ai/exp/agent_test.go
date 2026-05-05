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
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
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
		func(ctx context.Context, resp Responder[testStatus], sess *AgentSession[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				resp.SendStatus(testStatus{Phase: "generating"})
				// Echo back the user's message.
				if len(input.Messages) > 0 {
					reply := ai.NewModelTextMessage("echo: " + input.Messages[0].Content[0].Text)
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
		func(ctx context.Context, resp Responder[testStatus], sess *AgentSession[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				if len(input.Messages) > 0 {
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
		func(ctx context.Context, resp Responder[testStatus], sess *AgentSession[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				if len(input.Messages) > 0 {
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
		func(ctx context.Context, resp Responder[testStatus], sess *AgentSession[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				if len(input.Messages) > 0 {
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
		func(ctx context.Context, resp Responder[testStatus], sess *AgentSession[testState]) (*AgentResult, error) {
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
		func(ctx context.Context, resp Responder[testStatus], sess *AgentSession[testState]) (*AgentResult, error) {
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

func TestAgent_SendMessages(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	af := DefineCustomAgent(reg, "sendMsgsFlow",
		func(ctx context.Context, resp Responder[testStatus], sess *AgentSession[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				return nil
			})
		},
	)

	conn, err := af.StreamBidi(ctx)
	if err != nil {
		t.Fatalf("StreamBidi failed: %v", err)
	}

	// Send multiple messages at once.
	err = conn.SendMessages(
		ai.NewUserTextMessage("msg1"),
		ai.NewUserTextMessage("msg2"),
	)
	if err != nil {
		t.Fatalf("SendMessages failed: %v", err)
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

	// Both messages should have been added.
	if got := len(response.State.Messages); got != 2 {
		t.Errorf("expected 2 messages, got %d", got)
	}
}

func TestAgent_SessionContext(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	var retrievedCounter int
	af := DefineCustomAgent(reg, "contextFlow",
		func(ctx context.Context, resp Responder[testStatus], sess *AgentSession[testState]) (*AgentResult, error) {
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
		func(ctx context.Context, resp Responder[testStatus], sess *AgentSession[testState]) (*AgentResult, error) {
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
		func(ctx context.Context, resp Responder[testStatus], sess *AgentSession[testState]) (*AgentResult, error) {
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

func TestAgent_SnapshotIDInMessageMetadata(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := NewInMemorySessionStore[testState]()

	af := DefineCustomAgent(reg, "metadataFlow",
		func(ctx context.Context, resp Responder[testStatus], sess *AgentSession[testState]) (*AgentResult, error) {
			err := sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				sess.AddMessages(ai.NewModelTextMessage("reply"))
				return nil
			})
			if err != nil {
				return nil, err
			}
			msgs := sess.Messages()
			return &AgentResult{Message: msgs[len(msgs)-1]}, nil
		},
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

	response, err := conn.Output()
	if err != nil {
		t.Fatalf("Output failed: %v", err)
	}

	// The last model message should have snapshotId in its metadata.
	if response.Message == nil {
		t.Fatal("expected Message in response")
	}
	if response.Message.Metadata == nil {
		t.Fatal("expected metadata on last message")
	}
	if _, ok := response.Message.Metadata["snapshotId"]; !ok {
		t.Error("expected snapshotId in last message metadata")
	}
}

func TestInMemorySessionStore(t *testing.T) {
	ctx := context.Background()
	store := NewInMemorySessionStore[testState]()

	// Get non-existent.
	snap, err := store.GetSnapshot(ctx, "nonexistent")
	if err != nil {
		t.Fatalf("GetSnapshot failed: %v", err)
	}
	if snap != nil {
		t.Errorf("expected nil, got %v", snap)
	}

	// Save and retrieve.
	snapshot := &SessionSnapshot[testState]{
		SnapshotID: "snap-1",
		State: SessionState[testState]{
			Custom: testState{Counter: 1},
		},
	}
	if err := store.SaveSnapshot(ctx, snapshot); err != nil {
		t.Fatalf("SaveSnapshot failed: %v", err)
	}

	retrieved, err := store.GetSnapshot(ctx, "snap-1")
	if err != nil {
		t.Fatalf("GetSnapshot failed: %v", err)
	}
	if retrieved == nil {
		t.Fatal("expected snapshot")
	}
	if retrieved.State.Custom.Counter != 1 {
		t.Errorf("expected counter=1, got %d", retrieved.State.Custom.Counter)
	}

	// Verify isolation.
	snapshot.State.Custom.Counter = 999
	retrieved2, _ := store.GetSnapshot(ctx, "snap-1")
	if retrieved2.State.Custom.Counter != 1 {
		t.Errorf("expected counter=1 (isolation), got %d", retrieved2.State.Custom.Counter)
	}
}

func TestAgent_TurnSpanOutput(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	var capturedOutputs []any

	af := DefineCustomAgent(reg, "turnOutputFlow",
		func(ctx context.Context, resp Responder[testStatus], sess *AgentSession[testState]) (*AgentResult, error) {
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
		func(ctx context.Context, resp Responder[testStatus], sess *AgentSession[testState]) (*AgentResult, error) {
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

	af := DefinePromptAgent[testState, any](reg, "testPrompt", nil)

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

func TestPromptAgent_PromptInputOverride(t *testing.T) {
	ctx := context.Background()
	reg := setupPromptTestRegistry(t)

	type greetInput struct {
		Name string `json:"name"`
	}

	ai.DefineDataPrompt[greetInput, string](reg, "greetPrompt",
		ai.WithModelName("test/echo"),
		ai.WithPrompt("Hello {{name}}!"),
	)

	af := DefinePromptAgent[testState](reg, "greetPrompt", greetInput{Name: "default"})

	// Use WithPromptInput to override.
	conn, err := af.StreamBidi(ctx,
		WithInputVariables[testState](greetInput{Name: "override"}),
	)
	if err != nil {
		t.Fatalf("StreamBidi failed: %v", err)
	}

	if err := conn.SendText("hi"); err != nil {
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

	// Verify the override was stored in session state.
	if response.State.InputVariables == nil {
		t.Fatal("expected PromptInput in state")
	}

	// The model echoes the last user message, which is "hi".
	// But the prompt was rendered with "override" so "Hello override!" should appear
	// in the messages sent to the model (verified via the echo).
	// We primarily verify the state was set correctly.
	inputMap, ok := response.State.InputVariables.(map[string]any)
	if !ok {
		t.Fatalf("expected PromptInput to be map[string]any, got %T", response.State.InputVariables)
	}
	if name, _ := inputMap["name"].(string); name != "override" {
		t.Errorf("expected PromptInput name='override', got %q", name)
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

	af := DefinePromptAgent[testState, any](reg, "historyPrompt", nil)

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

func TestPromptAgent_SnapshotPersistsPromptInput(t *testing.T) {
	ctx := context.Background()
	reg := setupPromptTestRegistry(t)
	store := NewInMemorySessionStore[testState]()

	ai.DefinePrompt(reg, "snapPrompt",
		ai.WithModelName("test/echo"),
		ai.WithSystem("You are a test assistant."),
	)

	af := DefinePromptAgent[testState, any](reg, "snapPrompt", nil,
		WithSessionStore(store),
	)

	// Start with prompt input.
	conn, err := af.StreamBidi(ctx,
		WithInputVariables[testState](map[string]any{"key": "value"}),
	)
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

	// Verify the snapshot contains PromptInput.
	snap, err := store.GetSnapshot(ctx, resp.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot failed: %v", err)
	}
	if snap.State.InputVariables == nil {
		t.Error("expected InputVariables in snapshot state")
	}

	// Resume from snapshot — the PromptInput should be preserved.
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

	// Verify state via snapshot (server-managed state).
	snap2, err := store.GetSnapshot(ctx, resp2.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot failed: %v", err)
	}
	if got := len(snap2.State.Messages); got != 4 {
		t.Errorf("expected 4 messages after resume, got %d", got)
	}
	if snap2.State.InputVariables == nil {
		t.Error("expected PromptInput preserved after resume")
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

	af := DefinePromptAgent[testState, any](reg, "toolPrompt", nil)

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
		func(ctx context.Context, resp Responder[testStatus], sess *AgentSession[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				if len(input.Messages) > 0 {
					sess.AddMessages(ai.NewModelTextMessage("echo: " + input.Messages[0].Content[0].Text))
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
		func(ctx context.Context, resp Responder[testStatus], sess *AgentSession[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) error {
				if len(input.Messages) > 0 {
					sess.AddMessages(ai.NewModelTextMessage("reply"))
				}
				return nil
			})
		},
	)

	input := &AgentInput{
		Messages: []*ai.Message{
			ai.NewUserTextMessage("msg1"),
			ai.NewUserTextMessage("msg2"),
		},
	}

	response, err := af.Run(ctx, input)
	if err != nil {
		t.Fatalf("Run failed: %v", err)
	}

	// 2 user messages + 1 reply = 3.
	if got := len(response.State.Messages); got != 3 {
		t.Errorf("expected 3 messages, got %d", got)
	}
}

func TestAgent_RunText_WithState(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	af := DefineCustomAgent(reg, "runStateFlow",
		func(ctx context.Context, resp Responder[testStatus], sess *AgentSession[testState]) (*AgentResult, error) {
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
		func(ctx context.Context, resp Responder[testStatus], sess *AgentSession[testState]) (*AgentResult, error) {
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

	af := DefinePromptAgent[testState, any](reg, "runTextPrompt", nil)

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

func TestAgent_SingleTurnSnapshotDedup(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := NewInMemorySessionStore[testState]()

	af := DefineCustomAgent(reg, "dedupFlow",
		func(ctx context.Context, resp Responder[testStatus], sess *AgentSession[testState]) (*AgentResult, error) {
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
		func(ctx context.Context, resp Responder[testStatus], sess *AgentSession[testState]) (*AgentResult, error) {
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
		func(ctx context.Context, resp Responder[testStatus], sess *AgentSession[testState]) (*AgentResult, error) {
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
		func(ctx context.Context, resp Responder[testStatus], sess *AgentSession[testState]) (*AgentResult, error) {
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
		func(ctx context.Context, resp Responder[testStatus], sess *AgentSession[testState]) (*AgentResult, error) {
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

// TestAgent_DefineAgent_StateMismatchPanics verifies that passing a typed
// AgentOption (e.g., a session store) with a different State than the
// declared one on DefineAgent panics with a clear message.
func TestAgent_DefineAgent_StateMismatchPanics(t *testing.T) {
	type otherState struct{ X int }
	reg := newTestRegistry(t)
	store := NewInMemorySessionStore[testState]()

	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("expected panic on State mismatch")
		}
		msg := fmt.Sprintf("%v", r)
		if !strings.Contains(msg, "does not match agent State") {
			t.Errorf("panic message missing expected substring: %s", msg)
		}
	}()

	_ = DefineAgent[otherState](reg, "mismatch", WithSessionStore(store))
}
