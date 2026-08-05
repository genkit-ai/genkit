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
	"strings"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
)

// TestStreamTransform_RedactsModelChunksOnWire verifies a stream transform that
// edits a chunk in place reaches the wire: the streamed model text is redacted
// for the client.
func TestStreamTransform_RedactsModelChunksOnWire(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	af := DefineCustomAgent(reg, "st",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				resp.SendModelChunk(&ai.ModelResponseChunk{
					Content: []*ai.Part{ai.NewTextPart("the secret is 42")},
				})
				return nil, nil
			})
		},
		WithStreamTransform[testState](func(_ context.Context, c *AgentStreamChunk) (*AgentStreamChunk, error) {
			if c.ModelChunk != nil {
				for _, p := range c.ModelChunk.Content {
					p.Text = strings.ReplaceAll(p.Text, "secret", "[REDACTED]")
				}
			}
			return c, nil
		}),
	)

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	defer conn.Output()

	sendText(t, conn, "tell me")
	var got string
	for chunk, err := range conn.Receive() {
		if err != nil {
			t.Fatalf("Receive: %v", err)
		}
		if chunk.ModelChunk != nil {
			got += chunk.ModelChunk.Text()
		}
		if chunk.TurnEnd != nil {
			break
		}
	}

	if want := "the [REDACTED] is 42"; got != want {
		t.Errorf("streamed model text = %q, want %q", got, want)
	}
}

// TestStreamTransform_NilDropsFromWireKeepsSideEffects verifies returning a nil
// chunk drops it from the wire while its already-applied side effects survive:
// an artifact dropped from the stream still lands in the session and the final
// output. nil is the wire-only, successful "omit" outcome, distinct from an
// error (fail-closed).
func TestStreamTransform_NilDropsFromWireKeepsSideEffects(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	af := DefineCustomAgent(reg, "st",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			err := sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				resp.SendArtifact(&Artifact{
					Name:  "secret.txt",
					Parts: []*ai.Part{ai.NewTextPart("classified")},
				})
				return nil, nil
			})
			if err != nil {
				return nil, err
			}
			return &AgentResult{Artifacts: sess.Artifacts()}, nil
		},
		// Drop every artifact chunk from the stream (intentional omit, not an error).
		WithStreamTransform[testState](func(_ context.Context, c *AgentStreamChunk) (*AgentStreamChunk, error) {
			if c.Artifact != nil {
				return nil, nil
			}
			return c, nil
		}),
	)

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}

	sendText(t, conn, "go")
	var wireArtifacts int
	for chunk, err := range conn.Receive() {
		if err != nil {
			t.Fatalf("Receive: %v", err)
		}
		if chunk.Artifact != nil {
			wireArtifacts++
		}
		if chunk.TurnEnd != nil {
			break
		}
	}
	if wireArtifacts != 0 {
		t.Errorf("artifacts on the wire = %d, want 0 (dropped by transform)", wireArtifacts)
	}

	conn.Close()
	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	// The invocation still succeeds, and the side effect (artifact recorded on
	// the session) is untouched: dropping it from the stream is a wire-only edit.
	if out.FinishReason == AgentFinishReasonFailed {
		t.Fatalf("expected a successful invocation, got failed: %+v", out.Error)
	}
	if len(out.Artifacts) != 1 || out.Artifacts[0].Name != "secret.txt" {
		t.Errorf("output artifacts = %+v, want one named secret.txt", out.Artifacts)
	}
}

// TestStreamTransform_OwnsDeepCopy verifies the transform receives a fresh deep
// copy it owns: mutating a chunk in place changes only the wire copy, leaving
// the session/output artifact and the pointer the agent fn retained untouched.
func TestStreamTransform_OwnsDeepCopy(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	// The artifact the fn sends and keeps a pointer to. The transform renames
	// the wire copy; this pointer must not see that rename.
	var sent *Artifact

	af := DefineCustomAgent(reg, "st",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			err := sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				sent = &Artifact{Name: "real.go", Parts: []*ai.Part{ai.NewTextPart("x")}}
				resp.SendArtifact(sent)
				return nil, nil
			})
			if err != nil {
				return nil, err
			}
			return &AgentResult{Artifacts: sess.Artifacts()}, nil
		},
		WithStreamTransform[testState](func(_ context.Context, c *AgentStreamChunk) (*AgentStreamChunk, error) {
			if c.Artifact != nil {
				c.Artifact.Name = "renamed.go"
			}
			return c, nil
		}),
	)

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}

	sendText(t, conn, "go")
	var wireName string
	for chunk, err := range conn.Receive() {
		if err != nil {
			t.Fatalf("Receive: %v", err)
		}
		if chunk.Artifact != nil {
			wireName = chunk.Artifact.Name
		}
		if chunk.TurnEnd != nil {
			break
		}
	}

	conn.Close()
	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}

	// The wire copy carries the transform's edit.
	if wireName != "renamed.go" {
		t.Errorf("wire artifact name = %q, want renamed.go", wireName)
	}
	// The final output (session side effect) keeps the original name.
	if len(out.Artifacts) != 1 || out.Artifacts[0].Name != "real.go" {
		t.Errorf("output artifacts = %+v, want one named real.go", out.Artifacts)
	}
	// Output() has drained the router, so the transform has run; the fn's
	// retained pointer is still the original, proving the transform mutated a
	// copy rather than the caller's artifact.
	if sent.Name != "real.go" {
		t.Errorf("agent fn's retained artifact name = %q, want real.go", sent.Name)
	}
}

// TestStreamTransform_ReshapesTurnEnd verifies the transform also sees control
// chunks: stripping the snapshot ID from a turn-end signal hides it from the
// client while the snapshot itself is still persisted and reported on the final
// output.
func TestStreamTransform_ReshapesTurnEnd(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()

	af := DefineCustomAgent(reg, "st",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				sess.AddMessages(ai.NewModelTextMessage("done"))
				return nil, nil
			})
		},
		WithSessionStore(store),
		// Hide the server-side snapshot ID from the streamed turn-end.
		WithStreamTransform[testState](func(_ context.Context, c *AgentStreamChunk) (*AgentStreamChunk, error) {
			if c.TurnEnd != nil {
				c.TurnEnd.SnapshotID = ""
			}
			return c, nil
		}),
	)

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}

	sendText(t, conn, "go")
	te := nextTurnEnd(t, conn)
	if te.SnapshotID != "" {
		t.Errorf("streamed TurnEnd.SnapshotID = %q, want empty (stripped on the wire)", te.SnapshotID)
	}

	conn.Close()
	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	// The final output keeps the real snapshot ID (the transform is stream-only)
	// and the snapshot is genuinely persisted.
	if out.SnapshotID == "" {
		t.Fatal("output SnapshotID is empty; expected the persisted snapshot ID")
	}
	if snap, err := store.GetSnapshot(ctx, out.SnapshotID); err != nil || snap == nil {
		t.Errorf("GetSnapshot(%q) = (%v, %v), want the persisted snapshot", out.SnapshotID, snap, err)
	}
}

// TestStreamTransform_ErrorFailsInvocationClosed verifies a transform that
// returns an error fails the whole invocation closed: the offending chunk never
// reaches the wire, the invocation resolves as a failed output, and the
// transform's status code is preserved.
func TestStreamTransform_ErrorFailsInvocationClosed(t *testing.T) {
	transform := func(_ context.Context, c *AgentStreamChunk) (*AgentStreamChunk, error) {
		if c.ModelChunk != nil {
			return nil, core.NewError(core.PERMISSION_DENIED, "cannot shape chunk")
		}
		return c, nil
	}
	assertStreamTransformFailsClosed(t, transform, core.PERMISSION_DENIED, "cannot shape chunk")
}

// TestStreamTransform_PanicFailsInvocationClosed verifies a panicking transform
// is contained in the router goroutine (not a process crash) and treated as a
// fail-closed error: the invocation resolves as a failed output rather than
// leaking the chunk.
func TestStreamTransform_PanicFailsInvocationClosed(t *testing.T) {
	transform := func(_ context.Context, c *AgentStreamChunk) (*AgentStreamChunk, error) {
		if c.ModelChunk != nil {
			panic("boom")
		}
		return c, nil
	}
	assertStreamTransformFailsClosed(t, transform, core.INTERNAL, "panicked")
}

// assertStreamTransformFailsClosed runs an agent that streams one model chunk
// through transform and asserts the invocation fails closed: the chunk never
// reaches the wire and the output is a failure carrying wantStatus and a message
// containing wantMsg.
func assertStreamTransformFailsClosed(t *testing.T, transform StreamTransform, wantStatus core.StatusName, wantMsg string) {
	t.Helper()
	ctx := context.Background()
	reg := newTestRegistry(t)

	af := DefineCustomAgent(reg, "st",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				resp.SendModelChunk(&ai.ModelResponseChunk{
					Content: []*ai.Part{ai.NewTextPart("leak")},
				})
				return nil, nil
			})
		},
		WithStreamTransform[testState](transform),
	)

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}

	sendText(t, conn, "go")
	for chunk, err := range conn.Receive() {
		if err != nil {
			break // failure may surface as the stream ends; Output carries it
		}
		if chunk.ModelChunk != nil {
			t.Error("model chunk reached the wire; expected fail-closed")
		}
	}

	out, err := outputWithin(t, conn, 2*time.Second)
	if err != nil {
		t.Fatalf("expected a graceful failed output, got error: %v", err)
	}
	if out.FinishReason != AgentFinishReasonFailed {
		t.Errorf("FinishReason = %q, want %q", out.FinishReason, AgentFinishReasonFailed)
	}
	if out.Error == nil || out.Error.Status != wantStatus {
		t.Errorf("Error = %+v, want status %q", out.Error, wantStatus)
	}
	if out.Error != nil && !strings.Contains(out.Error.Message, wantMsg) {
		t.Errorf("Error.Message = %q, want it to contain %q", out.Error.Message, wantMsg)
	}
}

// TestStreamTransform_RejectsSecondOption verifies WithStreamTransform, like the
// other agent options, may be set only once.
func TestStreamTransform_RejectsSecondOption(t *testing.T) {
	reg := newTestRegistry(t)
	noop := func(_ context.Context, c *AgentStreamChunk) (*AgentStreamChunk, error) { return c, nil }
	noopFn := func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
		return nil, nil
	}
	defer func() {
		if recover() == nil {
			t.Error("expected a panic when WithStreamTransform is set twice")
		}
	}()
	DefineCustomAgent(reg, "twice", noopFn,
		WithStreamTransform[testState](noop), WithStreamTransform[testState](noop))
}
