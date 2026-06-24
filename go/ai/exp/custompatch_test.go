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
	"sync"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
)

// collectTurnPatches consumes one turn's chunks, returning the customPatch from
// each chunk that carries one (in stream order). Consuming via Receive also
// updates the connection's tracked custom state.
func collectTurnPatches(t *testing.T, conn *AgentConnection[testState]) []JSONPatch {
	t.Helper()
	var patches []JSONPatch
	for chunk, err := range conn.Receive() {
		if err != nil {
			t.Fatalf("Receive: %v", err)
		}
		if len(chunk.CustomPatch) > 0 {
			patches = append(patches, chunk.CustomPatch)
		}
		if chunk.TurnEnd != nil {
			break
		}
	}
	return patches
}

// TestCustomPatch_PerTurnRebaseAndIncremental verifies that the first patch of
// every turn is a whole-document replace and later patches in the same turn are
// minimal incremental diffs.
func TestCustomPatch_PerTurnRebaseAndIncremental(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	af := DefineCustomAgent(reg, "cp",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				// Two mutations: first emits a whole-document replace, the
				// second an incremental diff against the first.
				sess.UpdateCustom(func(s testState) testState { s.Counter++; return s })
				sess.UpdateCustom(func(s testState) testState { s.Counter++; return s })
				return nil, nil
			})
		},
	)

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer conn.Output()

	// Turn 1.
	if err := conn.SendText("hi"); err != nil {
		t.Fatalf("SendText: %v", err)
	}
	patches := collectTurnPatches(t, conn)
	if len(patches) != 2 {
		t.Fatalf("turn 1: expected 2 customPatch chunks, got %d", len(patches))
	}
	if op := patches[0][0]; len(patches[0]) != 1 || op.Op != JSONPatchOpReplace || op.Path != "" {
		t.Errorf("turn 1 first patch: want whole-document replace, got %s", patchString(patches[0]))
	}
	if op := patches[1][0]; len(patches[1]) != 1 || op.Op != JSONPatchOpReplace || op.Path != "/counter" {
		t.Errorf("turn 1 second patch: want replace /counter, got %s", patchString(patches[1]))
	}

	// Turn 2: the first patch re-bases the client with another whole-document
	// replace even though only /counter changed since the previous turn.
	if err := conn.SendText("hi"); err != nil {
		t.Fatalf("SendText: %v", err)
	}
	patches = collectTurnPatches(t, conn)
	if len(patches) != 2 {
		t.Fatalf("turn 2: expected 2 customPatch chunks, got %d", len(patches))
	}
	if op := patches[0][0]; op.Path != "" {
		t.Errorf("turn 2 first patch: want whole-document replace (path \"\"), got %s", patchString(patches[0]))
	}
}

// TestCustomPatch_ClientTracksLiveCustom verifies the connection applies the
// streamed patches so Custom reflects the server's state.
func TestCustomPatch_ClientTracksLiveCustom(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	af := DefineCustomAgent(reg, "cp",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				sess.UpdateCustom(func(s testState) testState {
					s.Counter++
					s.Topics = append(s.Topics, input.Message.Text())
					return s
				})
				return nil, nil
			})
		},
	)

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}

	// Before any patch, Custom is the zero value.
	if got, err := conn.Custom(); err != nil || got.Counter != 0 {
		t.Errorf("initial Custom = %+v (err %v), want zero value", got, err)
	}

	conn.SendText("alpha")
	collectTurnPatches(t, conn)
	conn.SendText("beta")
	collectTurnPatches(t, conn)

	got, err := conn.Custom()
	if err != nil {
		t.Fatalf("Custom: %v", err)
	}
	if got.Counter != 2 {
		t.Errorf("tracked counter = %d, want 2", got.Counter)
	}
	if want := []string{"alpha", "beta"}; len(got.Topics) != 2 || got.Topics[0] != want[0] || got.Topics[1] != want[1] {
		t.Errorf("tracked topics = %v, want %v", got.Topics, want)
	}

	conn.Close()
	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	// The live-tracked custom agrees with the authoritative final state.
	if out.State.Custom.Counter != got.Counter {
		t.Errorf("tracked counter %d != final state counter %d", got.Counter, out.State.Custom.Counter)
	}
}

// TestCustomPatch_HonorsStateTransform verifies the diff is computed on the
// client-facing custom value (after WithStateTransform), so redaction reaches
// the wire.
func TestCustomPatch_HonorsStateTransform(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	af := DefineCustomAgent(reg, "cp",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				sess.UpdateCustom(func(s testState) testState {
					s.Counter = 5
					s.Topics = []string{"secret"}
					return s
				})
				return nil, nil
			})
		},
		// Redact Topics on the way out to the client.
		WithStateTransform(func(ctx context.Context, st *SessionState[testState]) (*SessionState[testState], error) {
			st.Custom.Topics = nil
			return st, nil
		}),
	)

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer conn.Output()

	conn.SendText("hi")
	collectTurnPatches(t, conn)

	got, err := conn.Custom()
	if err != nil {
		t.Fatalf("Custom: %v", err)
	}
	if got.Counter != 5 {
		t.Errorf("counter = %d, want 5", got.Counter)
	}
	if len(got.Topics) != 0 {
		t.Errorf("topics should be redacted on the wire, got %v", got.Topics)
	}
}

// TestCustomPatch_StateTransformErrorFailsInvocationClosed verifies a state
// transform that errors while shaping a streamed custom delta fails the
// invocation closed: the patch is withheld (no delta reaches the wire) and the
// invocation resolves as a failed output carrying the transform's status.
func TestCustomPatch_StateTransformErrorFailsInvocationClosed(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	af := DefineCustomAgent(reg, "cp",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				sess.UpdateCustom(func(s testState) testState {
					s.Counter = 5
					return s
				})
				return nil, nil
			})
		},
		WithStateTransform(func(_ context.Context, st *SessionState[testState]) (*SessionState[testState], error) {
			return nil, core.NewError(core.PERMISSION_DENIED, "cannot shape custom state")
		}),
	)

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}

	conn.SendText("go")
	// The transform errors before any delta is emitted, so no customPatch
	// reaches the wire; the stream ends as the invocation fails.
	if patches := collectTurnPatches(t, conn); len(patches) != 0 {
		t.Errorf("customPatches on the wire = %d, want 0 (withheld, fail-closed)", len(patches))
	}

	out, err := outputWithin(t, conn, 2*time.Second)
	if err != nil {
		t.Fatalf("expected a graceful failed output, got error: %v", err)
	}
	if out.FinishReason != AgentFinishReasonFailed {
		t.Errorf("FinishReason = %q, want %q", out.FinishReason, AgentFinishReasonFailed)
	}
	if out.Error == nil || out.Error.Status != core.PERMISSION_DENIED {
		t.Errorf("Error = %+v, want status %q from the transform", out.Error, core.PERMISSION_DENIED)
	}
}

// TestCustomPatch_ConcurrentMutations exercises the patcher's locking when
// custom state is mutated from several goroutines at once: the streamed patches
// must converge on the final state, and there must be no data race (run with
// -race). The last patcher emit observes all completed increments, so the
// client's tracked custom equals the final counter.
func TestCustomPatch_ConcurrentMutations(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)
	const n = 30

	af := DefineCustomAgent(reg, "cp",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				var wg sync.WaitGroup
				for range n {
					wg.Add(1)
					go func() {
						defer wg.Done()
						sess.UpdateCustom(func(s testState) testState { s.Counter++; return s })
					}()
				}
				wg.Wait()
				return nil, nil
			})
		},
	)

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}

	conn.SendText("go")
	collectTurnPatches(t, conn) // drains concurrently with the producers

	got, err := conn.Custom()
	if err != nil {
		t.Fatalf("Custom: %v", err)
	}
	if got.Counter != n {
		t.Errorf("tracked counter = %d, want %d", got.Counter, n)
	}

	conn.Close()
	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	if out.State.Custom.Counter != n {
		t.Errorf("final state counter = %d, want %d", out.State.Custom.Counter, n)
	}
}

// TestCustomPatch_NoMutationNoPatch verifies a turn that does not mutate custom
// state emits no customPatch chunk.
func TestCustomPatch_NoMutationNoPatch(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	af := DefineCustomAgent(reg, "cp",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				sess.AddMessages(ai.NewModelTextMessage("reply"))
				return nil, nil
			})
		},
	)

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer conn.Output()

	conn.SendText("hi")
	if patches := collectTurnPatches(t, conn); len(patches) != 0 {
		t.Errorf("expected no customPatch chunks, got %d", len(patches))
	}
}

// TestCustomPatch_EmptyDiffEmitsNothing verifies that a no-op mutation after
// the first patch of the turn produces no chunk (an empty incremental diff).
func TestCustomPatch_EmptyDiffEmitsNothing(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	af := DefineCustomAgent(reg, "cp",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				sess.UpdateCustom(func(s testState) testState { s.Counter = 1; return s })
				// Re-applies the same value: the incremental diff is empty.
				sess.UpdateCustom(func(s testState) testState { s.Counter = 1; return s })
				return nil, nil
			})
		},
	)

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer conn.Output()

	conn.SendText("hi")
	patches := collectTurnPatches(t, conn)
	if len(patches) != 1 {
		t.Errorf("expected 1 customPatch chunk (the whole-document replace), got %d", len(patches))
	}
}
