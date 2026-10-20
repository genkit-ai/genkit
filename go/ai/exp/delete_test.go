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
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/status"
)

// defineQuickDeleteAgent registers a trivial agent that completes each turn at once.
func defineQuickDeleteAgent(t *testing.T, name string, store SessionStore[testState]) *Agent[testState] {
	t.Helper()
	opts := []AgentOption[testState]{}
	if store != nil {
		opts = append(opts, WithSessionStore(store))
	}
	return DefineCustomAgent(newTestRegistry(t), name,
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				sess.AddMessages(ai.NewModelTextMessage("done"))
				return nil, nil
			})
		},
		opts...,
	)
}

func TestAgent_DeleteSession_CompletedSession(t *testing.T) {
	ctx := context.Background()
	store := newTestInMemStore[testState]()
	agent := defineQuickDeleteAgent(t, "deleteCompleted", store)

	out, err := agent.Run(ctx, &AgentInput{Message: ai.NewUserTextMessage("hi")})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.SessionID == "" || out.SnapshotID == "" {
		t.Fatalf("output has no session/snapshot ID: %+v", out)
	}

	removed, err := agent.DeleteSession(ctx, out.SessionID)
	if err != nil || !removed {
		t.Fatalf("DeleteSession = %v, %v; want true, nil", removed, err)
	}
	if got, err := store.GetSnapshot(ctx, out.SnapshotID); err != nil || got != nil {
		t.Errorf("snapshot survived the delete: %v, %v", got, err)
	}
	// Idempotent.
	if removed, err := agent.DeleteSession(ctx, out.SessionID); err != nil || removed {
		t.Errorf("second DeleteSession = %v, %v; want false, nil", removed, err)
	}
	// Resuming by snapshot ID now fails; by session ID it starts fresh.
	if _, err := agent.Run(ctx, &AgentInput{Message: ai.NewUserTextMessage("again")}, WithSnapshotID[testState](out.SnapshotID)); err == nil {
		t.Error("resume by snapshot ID after delete succeeded, want an error")
	}
	fresh, err := agent.Run(ctx, &AgentInput{Message: ai.NewUserTextMessage("again")}, WithSessionID[testState](out.SessionID))
	if err != nil {
		t.Fatalf("resume by session ID: %v", err)
	}
	row, err := store.GetSnapshot(ctx, fresh.SnapshotID)
	if err != nil || row == nil || row.ParentID != "" {
		t.Errorf("resumed session row = %+v, %v; want a fresh root", row, err)
	}
}

func TestAgent_DeleteSession_AbortsPendingRunAndDoesNotResurrect(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()
	agent, entered, release := defineGatedAgent(t, reg, "deletePending", store)

	task, err := agent.RunDetached(ctx, &AgentInput{Message: ai.NewUserTextMessage("go")})
	if err != nil {
		t.Fatalf("RunDetached: %v", err)
	}
	select {
	case <-entered:
	case <-time.After(2 * time.Second):
		t.Fatal("background work did not start")
	}
	pending, err := store.GetSnapshot(ctx, task.SnapshotID())
	if err != nil || pending == nil || pending.Status != SnapshotStatusPending {
		t.Fatalf("pending row = %+v, %v", pending, err)
	}

	if removed, err := agent.DeleteSession(ctx, pending.SessionID); err != nil || !removed {
		t.Fatalf("DeleteSession = %v, %v; want true, nil", removed, err)
	}

	// Let the worker finish whichever way it will; nothing may be written back.
	close(release)
	time.Sleep(300 * time.Millisecond)
	if n := store.snapshotCount(); n != 0 {
		t.Errorf("store holds %d rows after the delete, want 0 (session came back)", n)
	}
}

func TestAgent_DeleteSession_Errors(t *testing.T) {
	ctx := context.Background()

	t.Run("client-managed agent", func(t *testing.T) {
		agent := defineQuickDeleteAgent(t, "deleteNoStore", nil)
		if _, err := agent.DeleteSession(ctx, "s"); !errors.Is(err, ErrSessionStoreNotConfigured) {
			t.Errorf("error = %v, want ErrSessionStoreNotConfigured", err)
		}
	})

	t.Run("store without the capability", func(t *testing.T) {
		// Embedding the interface hides DeleteSession.
		agent := defineQuickDeleteAgent(t, "deleteUnsupported", struct{ SessionStore[testState] }{newTestInMemStore[testState]()})
		if _, err := agent.DeleteSession(ctx, "s"); !errors.Is(err, status.ErrFailedPrecondition) {
			t.Errorf("error = %v, want FAILED_PRECONDITION", err)
		}
	})

	t.Run("empty session ID", func(t *testing.T) {
		agent := defineQuickDeleteAgent(t, "deleteEmptyID", newTestInMemStore[testState]())
		if _, err := agent.DeleteSession(ctx, ""); !errors.Is(err, status.ErrInvalidArgument) {
			t.Errorf("error = %v, want INVALID_ARGUMENT", err)
		}
	})
}
