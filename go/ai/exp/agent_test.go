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
	"sync/atomic"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/tracing"
	"github.com/firebase/genkit/go/internal/registry"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

type testState struct {
	Counter int      `json:"counter"`
	Topics  []string `json:"topics,omitempty"`
}

func newTestRegistry(t *testing.T) *registry.Registry {
	t.Helper()
	return registry.New()
}

// sendText sends a user text message, failing the test if the send is
// rejected. The few sites that expect the send to race invocation completion
// (e.g. a send after a failing turn) check the error themselves instead.
func sendText[State any](t *testing.T, conn *AgentConnection[State], text string) {
	t.Helper()
	if err := conn.SendText(text); err != nil {
		t.Fatalf("SendText(%q): %v", text, err)
	}
}

// sendTurn sends a user text message and advances the stream to that turn's
// TurnEnd, returning it. It is the send-one-turn-and-wait pattern most
// multi-turn tests share; tests that must inspect intermediate chunks should
// drive Receive directly (see nextTurnEnd).
func sendTurn[State any](t *testing.T, conn *AgentConnection[State], text string) *TurnEnd {
	t.Helper()
	sendText(t, conn, text)
	return nextTurnEnd(t, conn)
}

// drainInBackground consumes conn's stream in a goroutine so the agent's
// responder never blocks on a full stream buffer while the test orchestrates
// a detach or cancellation. The goroutine exits when the stream ends or
// errors. Used by tests that don't inspect the streamed chunks.
func drainInBackground[State any](conn *AgentConnection[State]) {
	go func() {
		for _, err := range conn.Receive() {
			if err != nil {
				return
			}
		}
	}()
}

func TestAgent_BasicMultiTurn(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	af := DefineCustomAgent(reg, "basicFlow",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				// Echo back the user's message.
				if input.Message != nil {
					reply := ai.NewModelTextMessage("echo: " + input.Message.Content[0].Text)
					sess.AddMessages(reply)
				}
				// Mutating custom state streams a customPatch chunk.
				sess.UpdateCustom(func(s testState) testState {
					s.Counter++
					return s
				})
				return nil, nil
			})
		},
	)

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect failed: %v", err)
	}

	// Turn 1.
	sendText(t, conn, "hello")
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
	if turn1Chunks < 2 { // at least customPatch + TurnEnd
		t.Errorf("expected at least 2 chunks in turn 1, got %d", turn1Chunks)
	}

	// Turn 2.
	sendTurn(t, conn, "world")

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

// TestAgentConnection_Custom_TracksStreamedPatches verifies the client-side
// live custom-state tracking: as Receive yields chunks, AgentConnection applies
// each CustomPatch to an internal copy that Custom() returns. It exercises the
// per-turn whole-document replace (the first patch of a turn re-bases the
// client) followed by an incremental diff within the same turn, and that the
// tracking carries across turns. The server-side patch emission is covered by
// TestAgent_CustomPatchWholeDocumentReplace; this is its client-side complement.
func TestAgentConnection_Custom_TracksStreamedPatches(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	af := DefineCustomAgent(reg, "liveCustomFlow",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				// Two mutations per turn: the first emits a whole-document
				// replace, the second an incremental diff.
				sess.UpdateCustom(func(s testState) testState {
					s.Counter++
					return s
				})
				sess.UpdateCustom(func(s testState) testState {
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

	// Before any patch arrives, Custom() is the zero value.
	if got, err := conn.Custom(); err != nil || got.Counter != 0 || len(got.Topics) != 0 {
		t.Errorf("Custom() before any turn = %+v (err %v), want zero value", got, err)
	}

	// Turn 1: draining the turn applies the streamed patches (replace + diff).
	sendTurn(t, conn, "alpha")
	got, err := conn.Custom()
	if err != nil {
		t.Fatalf("Custom() after turn 1: %v", err)
	}
	if got.Counter != 1 || !slices.Equal(got.Topics, []string{"alpha"}) {
		t.Errorf("tracked custom after turn 1 = %+v, want {Counter:1 Topics:[alpha]}", got)
	}

	// Turn 2: its first patch is a whole-document replace that re-bases the
	// client; the cumulative tracked state must still be correct.
	sendTurn(t, conn, "beta")
	got, err = conn.Custom()
	if err != nil {
		t.Fatalf("Custom() after turn 2: %v", err)
	}
	if got.Counter != 2 || !slices.Equal(got.Topics, []string{"alpha", "beta"}) {
		t.Errorf("tracked custom after turn 2 = %+v, want {Counter:2 Topics:[alpha beta]}", got)
	}

	conn.Close()
	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	// The authoritative final state agrees with what the streamed patches tracked.
	if out.State.Custom.Counter != 2 || !slices.Equal(out.State.Custom.Topics, []string{"alpha", "beta"}) {
		t.Errorf("final state custom = %+v, want {Counter:2 Topics:[alpha beta]}", out.State.Custom)
	}
}

func TestAgent_WithSessionStore(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()

	af := defineCounterAgent(reg, "snapshotFlow", WithSessionStore(store))

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect failed: %v", err)
	}

	sendText(t, conn, "turn1")

	var snapshotIDs []string
	if te := nextTurnEnd(t, conn); te.SnapshotID != "" {
		snapshotIDs = append(snapshotIDs, te.SnapshotID)
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

// spanCollector is a minimal in-memory sdktrace.SpanExporter that records
// finished spans so a test can assert on their attributes.
type spanCollector struct {
	mu    sync.Mutex
	spans []sdktrace.ReadOnlySpan
}

func (c *spanCollector) ExportSpans(_ context.Context, spans []sdktrace.ReadOnlySpan) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.spans = append(c.spans, spans...)
	return nil
}

func (c *spanCollector) Shutdown(context.Context) error { return nil }

// byName returns the first recorded span with the given name, or nil.
func (c *spanCollector) byName(name string) sdktrace.ReadOnlySpan {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, s := range c.spans {
		if s.Name() == name {
			return s
		}
	}
	return nil
}

// collectSpans registers an in-memory exporter on the global tracer provider
// (the one tracing.RunInNewSpan writes through) for the duration of the test.
// The SimpleSpanProcessor exports each span synchronously as it ends, so by
// the time a turn completes its span is already recorded.
func collectSpans(t *testing.T) *spanCollector {
	t.Helper()
	c := &spanCollector{}
	sp := sdktrace.NewSimpleSpanProcessor(c)
	tp := tracing.TracerProvider()
	tp.RegisterSpanProcessor(sp)
	t.Cleanup(func() { tp.UnregisterSpanProcessor(sp) })
	return c
}

// spanAttr returns the string value of the named span attribute, if present.
func spanAttr(span sdktrace.ReadOnlySpan, key string) (string, bool) {
	for _, kv := range span.Attributes() {
		if string(kv.Key) == key {
			return kv.Value.AsString(), true
		}
	}
	return "", false
}

// TestAgent_RootSpanCarriesSessionID verifies the agent's root action span is
// stamped with the invocation's session ID under
// "genkit:metadata:agent:sessionId", and that the per-turn spans are left
// untagged. This matches the JS agent, which tags the action span once via
// setCustomMetadataAttributes({'agent:sessionId': ...}) rather than each turn.
func TestAgent_RootSpanCarriesSessionID(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)
	spans := collectSpans(t)

	const agentName = "tracedCounterFlow"
	af := defineCounterAgent(reg, agentName)

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect failed: %v", err)
	}
	sendTurn(t, conn, "turn1")
	sendTurn(t, conn, "turn2")
	conn.Close()

	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output failed: %v", err)
	}
	if out.SessionID == "" {
		t.Fatal("expected a non-empty session ID on the output")
	}

	const attrKey = "genkit:metadata:agent:sessionId"

	// The root action span (named for the agent) carries the session ID.
	root := spans.byName(agentName)
	if root == nil {
		t.Fatalf("missing root action span %q", agentName)
	}
	got, ok := spanAttr(root, attrKey)
	if !ok {
		t.Fatalf("root span %q: missing attribute %q", agentName, attrKey)
	}
	if got != out.SessionID {
		t.Errorf("root span %q: %s = %q, want %q", agentName, attrKey, got, out.SessionID)
	}

	// The per-turn spans are named "runTurn-N" (1-indexed) with type flowStep
	// and no subtype, matching JS's run(); the session ID lives on the root
	// span only.
	for _, turn := range []string{"runTurn-1", "runTurn-2"} {
		span := spans.byName(turn)
		if span == nil {
			t.Fatalf("missing span %q", turn)
		}
		if v, ok := spanAttr(span, attrKey); ok {
			t.Errorf("turn span %q: unexpected attribute %q = %q (want it on the root span only)", turn, attrKey, v)
		}
		if v, _ := spanAttr(span, "genkit:type"); v != "flowStep" {
			t.Errorf("turn span %q: genkit:type = %q, want %q", turn, v, "flowStep")
		}
		if v, ok := spanAttr(span, "genkit:metadata:subtype"); ok {
			t.Errorf("turn span %q: unexpected genkit:metadata:subtype = %q (JS's run() sets no subtype)", turn, v)
		}
	}
}

func TestAgent_ResumeFromSnapshot(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()

	af := defineCounterAgent(reg, "resumeFlow", WithSessionStore(store))

	// First invocation: create a snapshot.
	conn1, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect failed: %v", err)
	}
	sendText(t, conn1, "first message")
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
	conn2, err := af.Connect(ctx, WithSnapshotID[testState](resp1.SnapshotID))
	if err != nil {
		t.Fatalf("Connect with snapshot failed: %v", err)
	}
	sendTurn(t, conn2, "continued message")
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

	af := defineCounterAgent(reg, "clientStateFlow")

	// Start with client-provided state.
	clientState := &SessionState[testState]{
		Messages: []*ai.Message{
			ai.NewUserTextMessage("previous message"),
			ai.NewModelTextMessage("previous reply"),
		},
		Custom: testState{Counter: 5},
	}

	conn, err := af.Connect(ctx, WithState(clientState))
	if err != nil {
		t.Fatalf("Connect failed: %v", err)
	}

	sendTurn(t, conn, "new message")
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
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			err := sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {

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
				return nil, nil
			})
			if err != nil {
				return nil, err
			}
			return &AgentResult{Artifacts: sess.Artifacts()}, nil
		},
	)

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect failed: %v", err)
	}

	sendText(t, conn, "generate code")
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

// TestAgent_ClientManagedState_CallerStateIsolated verifies that the
// framework deep-copies client-managed state at the entry boundary: the
// invocation must not write into the caller's state object (e.g. via
// AddArtifacts' in-place replace), and two invocations given the same
// state object must not share memory.
func TestAgent_ClientManagedState_CallerStateIsolated(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	af := DefineCustomAgent(reg, "stateIsolationFlow",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				// Replace the artifact the caller's state carried (the
				// in-place replace path) and extend history.
				resp.SendArtifact(&Artifact{
					Name:  "code.go",
					Parts: []*ai.Part{ai.NewTextPart("v2")},
				})
				sess.AddMessages(ai.NewModelTextMessage("reply"))
				return nil, nil
			})
		},
	)

	callerArtifact := &Artifact{
		Name:  "code.go",
		Parts: []*ai.Part{ai.NewTextPart("v1")},
	}
	prev := &SessionState[testState]{
		Artifacts: []*Artifact{callerArtifact},
		Messages:  []*ai.Message{ai.NewUserTextMessage("previous")},
	}

	out, err := af.RunText(ctx, "turn 1", WithState(prev))
	if err != nil {
		t.Fatalf("RunText failed: %v", err)
	}

	// The caller's state object must be untouched: same artifact pointer
	// and content, no appended messages.
	if prev.Artifacts[0] != callerArtifact {
		t.Error("invocation replaced the artifact inside the caller's state object")
	}
	if got := callerArtifact.Parts[0].Text; got != "v1" {
		t.Errorf("caller's artifact content changed to %q, want %q", got, "v1")
	}
	if got := len(prev.Messages); got != 1 {
		t.Errorf("caller's message slice grew to %d entries, want 1", got)
	}

	// The output reflects the replace: one artifact with the new content.
	if got := len(out.State.Artifacts); got != 1 {
		t.Fatalf("expected 1 artifact in output state, got %d", got)
	}
	if got := out.State.Artifacts[0].Parts[0].Text; got != "v2" {
		t.Errorf("output artifact content = %q, want %q", got, "v2")
	}
}

// TestAgent_InputMessageCloned verifies the session stores a private copy
// of the input message: a caller mutating the message it sent after the
// turn must not change conversation history.
func TestAgent_InputMessageCloned(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	af := DefineCustomAgent(reg, "inputCloneFlow",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				return nil, nil
			})
		},
	)

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect failed: %v", err)
	}

	sent := ai.NewUserTextMessage("original")
	if err := conn.SendMessage(sent); err != nil {
		t.Fatalf("SendMessage failed: %v", err)
	}
	nextTurnEnd(t, conn)

	// The turn is over, so the message is in session history. Mutating
	// the caller's copy must not reach it.
	sent.Content[0].Text = "mutated"

	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output failed: %v", err)
	}
	if got := len(out.State.Messages); got != 1 {
		t.Fatalf("expected 1 message, got %d", got)
	}
	if got := out.State.Messages[0].Content[0].Text; got != "original" {
		t.Errorf("session history reflects caller's mutation: got %q, want %q", got, "original")
	}
}

// TestAgent_SendArtifact_SynchronousAndCloned verifies SendArtifact's two
// guarantees: the artifact is visible to session reads by the time the
// call returns (Result right after SendArtifact must include it), and the
// session holds a private copy, so the sender's retained pointer cannot
// mutate session state.
func TestAgent_SendArtifact_SynchronousAndCloned(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	var (
		resultArtifacts int
		sessionContent  string
	)
	af := DefineCustomAgent(reg, "syncArtifactFlow",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				a := &Artifact{Name: "out.txt", Parts: []*ai.Part{ai.NewTextPart("original")}}
				resp.SendArtifact(a)
				// Visible as soon as SendArtifact returns.
				resultArtifacts = len(sess.Result().Artifacts)
				// The session holds its own copy.
				a.Parts[0] = ai.NewTextPart("mutated")
				if arts := sess.Artifacts(); len(arts) == 1 {
					sessionContent = arts[0].Parts[0].Text
				}
				return nil, nil
			})
		},
	)

	if _, err := af.RunText(ctx, "go"); err != nil {
		t.Fatalf("RunText failed: %v", err)
	}
	if resultArtifacts != 1 {
		t.Errorf("Result() right after SendArtifact saw %d artifacts, want 1", resultArtifacts)
	}
	if sessionContent != "original" {
		t.Errorf("session artifact content = %q, want %q (sender's mutation must not reach session state)", sessionContent, "original")
	}
}

// TestAgent_TurnEndSnapshot_IncludesSameTurnArtifact verifies that a
// turn-end snapshot captures artifacts sent during that turn: the Send
// side effect applies before the call returns, so the snapshot taken at
// turn end cannot miss it. Turn end is the agent's only snapshot point, so
// the invocation output reuses the turn-end row, which therefore must hold
// the artifact for a later resume.
func TestAgent_TurnEndSnapshot_IncludesSameTurnArtifact(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()

	af := DefineCustomAgent(reg, "turnEndArtifactFlow",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				resp.SendArtifact(&Artifact{
					Name:  "report.md",
					Parts: []*ai.Part{ai.NewTextPart("# Report")},
				})
				return nil, nil
			})
		},
		WithSessionStore[testState](store),
	)

	out, err := af.RunText(ctx, "produce the report")
	if err != nil {
		t.Fatalf("RunText failed: %v", err)
	}
	if out.SnapshotID == "" {
		t.Fatal("expected a snapshot ID on the output")
	}
	snap, err := store.GetSnapshot(ctx, out.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot failed: %v", err)
	}
	if snap == nil {
		t.Fatalf("snapshot %q not found", out.SnapshotID)
	}
	if snap.State == nil || len(snap.State.Artifacts) != 1 {
		t.Fatalf("turn-end snapshot missing the artifact sent during the turn: %+v", snap.State)
	}
	if got := snap.State.Artifacts[0].Name; got != "report.md" {
		t.Errorf("snapshot artifact name = %q, want %q", got, "report.md")
	}
}

func TestAgent_SendMessage(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	af := DefineCustomAgent(reg, "sendMsgFlow",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				return nil, nil
			})
		},
	)

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect failed: %v", err)
	}

	// Send a message via SendMessage.
	err = conn.SendMessage(ai.NewUserTextMessage("msg1"))
	if err != nil {
		t.Fatalf("SendMessage failed: %v", err)
	}
	nextTurnEnd(t, conn)
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
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				// Session should be retrievable from context.
				ctxSess := SessionFromContext[testState](ctx)
				if ctxSess == nil {
					t.Error("expected session from context")
					return nil, nil
				}
				ctxSess.UpdateCustom(func(s testState) testState {
					s.Counter = 42
					return s
				})
				retrievedCounter = ctxSess.Custom().Counter
				return nil, nil
			})
		},
	)

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect failed: %v", err)
	}

	sendTurn(t, conn, "test")
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
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				return nil, fmt.Errorf("turn failed")
			})
		},
	)

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect failed: %v", err)
	}

	sendText(t, conn, "trigger error")
	conn.Close()

	// A failed turn resolves the invocation gracefully rather than
	// failing the action: the outcome is on the output.
	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	if out.FinishReason != AgentFinishReasonFailed {
		t.Errorf("expected finish reason %q, got %q", AgentFinishReasonFailed, out.FinishReason)
	}
	if out.Error == nil || !strings.Contains(out.Error.Message, "turn failed") {
		t.Errorf("expected output error containing %q, got %+v", "turn failed", out.Error)
	}
	// Client-managed: the last-good state rides the output. No turn
	// succeeded, so it is the initial empty state — excluding the failed
	// turn's partial mutations (the user message added before fn ran).
	if out.State == nil {
		t.Fatal("expected last-good state on failed output")
	}
	if got := len(out.State.Messages); got != 0 {
		t.Errorf("expected 0 messages in last-good state, got %d", got)
	}
}

// defineCounterAgent defines a custom agent whose every turn appends a model
// "reply" message and increments the custom counter: the minimal stateful
// turn body shared by the snapshot and state-management tests. opts pass
// through to DefineCustomAgent (e.g. WithSessionStore).
func TestAgent_TurnContext(t *testing.T) {
	ctx := context.Background()

	t.Run("server-managed reserves the ID and persists under it", func(t *testing.T) {
		reg := newTestRegistry(t)
		store := newTestInMemStore[testState]()

		var mu sync.Mutex
		var seen []TurnContext
		af := DefineCustomAgent(reg, "turnCtxServer",
			func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
				return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
					tc := TurnContextFromContext(ctx)
					if tc == nil {
						return nil, fmt.Errorf("TurnContextFromContext returned nil inside the turn")
					}
					mu.Lock()
					seen = append(seen, *tc)
					mu.Unlock()
					sess.AddMessages(ai.NewModelTextMessage("reply"))
					return nil, nil
				})
			},
			WithSessionStore(store),
		)

		conn, err := af.Connect(ctx)
		if err != nil {
			t.Fatalf("Connect: %v", err)
		}
		te1 := sendTurn(t, conn, "one")
		te2 := sendTurn(t, conn, "two")
		if _, err := conn.Output(); err != nil {
			t.Fatalf("Output: %v", err)
		}

		mu.Lock()
		defer mu.Unlock()
		if len(seen) != 2 {
			t.Fatalf("got %d turn contexts, want 2", len(seen))
		}

		// Turn 0: fresh session, a reserved ID, no parent.
		if seen[0].TurnIndex != 0 {
			t.Errorf("turn 0 TurnIndex = %d, want 0", seen[0].TurnIndex)
		}
		if seen[0].SnapshotID == "" {
			t.Fatal("turn 0 reserved SnapshotID is empty; want a reserved ID")
		}
		if seen[0].ParentSnapshotID != "" {
			t.Errorf("turn 0 ParentSnapshotID = %q, want empty", seen[0].ParentSnapshotID)
		}
		// The whole point: the ID the handler saw up front is the ID the
		// snapshot persisted under (and the one the TurnEnd chunk reports).
		if seen[0].SnapshotID != te1.SnapshotID {
			t.Errorf("turn 0 reserved ID %q != persisted TurnEnd ID %q", seen[0].SnapshotID, te1.SnapshotID)
		}
		if snap, err := store.GetSnapshot(ctx, seen[0].SnapshotID); err != nil || snap == nil {
			t.Errorf("GetSnapshot(reserved %q) = (%v, %v); want a stored row", seen[0].SnapshotID, snap, err)
		}

		// Turn 1: index advances, parent is turn 0's snapshot, ID is fresh.
		if seen[1].TurnIndex != 1 {
			t.Errorf("turn 1 TurnIndex = %d, want 1", seen[1].TurnIndex)
		}
		if seen[1].SnapshotID != te2.SnapshotID {
			t.Errorf("turn 1 reserved ID %q != persisted TurnEnd ID %q", seen[1].SnapshotID, te2.SnapshotID)
		}
		if seen[1].ParentSnapshotID != seen[0].SnapshotID {
			t.Errorf("turn 1 ParentSnapshotID = %q, want turn 0's %q", seen[1].ParentSnapshotID, seen[0].SnapshotID)
		}
		if seen[1].SnapshotID == seen[0].SnapshotID {
			t.Error("turn 1 reused turn 0's reserved ID; each turn must reserve a fresh one")
		}
	})

	t.Run("client-managed reserves no ID", func(t *testing.T) {
		reg := newTestRegistry(t)

		var mu sync.Mutex
		var seen *TurnContext
		af := DefineCustomAgent(reg, "turnCtxClient",
			func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
				return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
					tc := TurnContextFromContext(ctx)
					mu.Lock()
					seen = tc
					mu.Unlock()
					return nil, nil
				})
			},
		)

		if _, err := af.RunText(ctx, "hi"); err != nil {
			t.Fatalf("RunText: %v", err)
		}

		mu.Lock()
		defer mu.Unlock()
		if seen == nil {
			t.Fatal("TurnContextFromContext returned nil for a client-managed agent")
		}
		if seen.SnapshotID != "" {
			t.Errorf("client-managed reserved SnapshotID = %q, want empty", seen.SnapshotID)
		}
		if seen.ParentSnapshotID != "" {
			t.Errorf("client-managed ParentSnapshotID = %q, want empty", seen.ParentSnapshotID)
		}
	})
}

func defineCounterAgent(reg api.Registry, name string, opts ...AgentOption[testState]) *Agent[testState] {
	return DefineCustomAgent(reg, name,
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				sess.AddMessages(ai.NewModelTextMessage("reply"))
				sess.UpdateCustom(func(s testState) testState {
					s.Counter++
					return s
				})
				return nil, nil
			})
		},
		opts...,
	)
}

// defineLastGoodTestAgent defines a client- or server-managed echo agent
// whose turn fails (with partial session mutations) when the user sends
// "boom". Successful turns report [AgentFinishReasonStop].
func defineLastGoodTestAgent(reg api.Registry, name string, opts ...AgentOption[testState]) *Agent[testState] {
	return DefineCustomAgent(reg, name,
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			if err := sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				text := input.Message.Content[0].Text
				if text == "boom" {
					// Partial mutations of the failing turn: these must not
					// leak into the recovered last-good state.
					sess.AddMessages(ai.NewModelTextMessage("partial reply"))
					sess.UpdateCustom(func(s testState) testState {
						s.Counter = 999
						return s
					})
					return nil, core.NewError(core.UNAVAILABLE, "model timeout")
				}
				sess.AddMessages(ai.NewModelTextMessage("echo: " + text))
				sess.UpdateCustom(func(s testState) testState {
					s.Counter++
					return s
				})
				return &TurnResult{FinishReason: AgentFinishReasonStop}, nil
			}); err != nil {
				return nil, err
			}
			return sess.Result(), nil
		},
		opts...,
	)
}

func TestAgent_FailedTurn_ClientManagedReturnsLastGoodState(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	af := defineLastGoodTestAgent(reg, "lastGoodClient")

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	for _, text := range []string{"one", "two", "boom"} {
		sendText(t, conn, text)
	}

	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	if out.FinishReason != AgentFinishReasonFailed {
		t.Errorf("expected finish reason %q, got %q", AgentFinishReasonFailed, out.FinishReason)
	}
	if out.Error == nil {
		t.Fatal("expected error on failed output")
	}
	if out.Error.Status != core.UNAVAILABLE {
		t.Errorf("expected original status %q preserved, got %q", core.UNAVAILABLE, out.Error.Status)
	}
	if !strings.Contains(out.Error.Message, "model timeout") {
		t.Errorf("expected error message to contain %q, got %q", "model timeout", out.Error.Message)
	}

	// The last-good state holds both successful turns (user + echo each)
	// and excludes the failed turn entirely: no "boom" user message, no
	// partial reply, no counter clobber.
	if out.State == nil {
		t.Fatal("expected last-good state on failed output")
	}
	if got := len(out.State.Messages); got != 4 {
		t.Fatalf("expected 4 messages in last-good state, got %d", got)
	}
	if got := out.State.Messages[3].Content[0].Text; got != "echo: two" {
		t.Errorf("expected last message %q, got %q", "echo: two", got)
	}
	if got := out.State.Custom.Counter; got != 2 {
		t.Errorf("expected counter=2 in last-good state, got %d", got)
	}
}

func TestAgent_FailedTurn_LastGoodStateIsResumable(t *testing.T) {
	// On failure, the client resumes a fresh invocation from the
	// last-good state in the failed output.
	ctx := context.Background()
	reg := newTestRegistry(t)

	af := defineLastGoodTestAgent(reg, "lastGoodResume")

	out, err := af.RunText(ctx, "boom", WithState(&SessionState[testState]{
		Messages: []*ai.Message{
			ai.NewUserTextMessage("one"),
			ai.NewModelTextMessage("echo: one"),
		},
		Custom: testState{Counter: 1},
	}))
	if err != nil {
		t.Fatalf("RunText: %v", err)
	}
	if out.FinishReason != AgentFinishReasonFailed {
		t.Fatalf("expected finish reason %q, got %q", AgentFinishReasonFailed, out.FinishReason)
	}
	// The failed output echoes back the state the failed turn started
	// with: exactly what the client sent.
	if got := len(out.State.Messages); got != 2 {
		t.Fatalf("expected 2 messages in last-good state, got %d", got)
	}

	retry, err := af.RunText(ctx, "two", WithState(out.State))
	if err != nil {
		t.Fatalf("RunText(retry): %v", err)
	}
	if retry.FinishReason != AgentFinishReasonStop {
		t.Errorf("expected finish reason %q, got %q", AgentFinishReasonStop, retry.FinishReason)
	}
	if retry.Error != nil {
		t.Errorf("expected no error on retry, got %+v", retry.Error)
	}
	if got := len(retry.State.Messages); got != 4 {
		t.Errorf("expected 4 messages after retry, got %d", got)
	}
	if got := retry.State.Custom.Counter; got != 2 {
		t.Errorf("expected counter=2 after retry, got %d", got)
	}
}

func TestAgent_FailedTurn_ServerManagedReturnsLastTurnSnapshot(t *testing.T) {
	// Server-managed: every successful turn snapshots, so when a later turn
	// fails the last-good state is already the newest row. The failed output
	// reuses that turn's snapshot ID and no extra row is written.
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()

	af := defineLastGoodTestAgent(reg, "recoveryDedup", WithSessionStore[testState](store))

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	turn0 := sendTurn(t, conn, "one")
	if turn0.SnapshotID == "" {
		t.Fatal("expected turn 0 snapshot")
	}

	sendText(t, conn, "boom")
	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	if out.FinishReason != AgentFinishReasonFailed {
		t.Errorf("expected finish reason %q, got %q", AgentFinishReasonFailed, out.FinishReason)
	}
	if out.SnapshotID != turn0.SnapshotID {
		t.Errorf("expected failed output to reuse the persisted last-good snapshot %q, got %q",
			turn0.SnapshotID, out.SnapshotID)
	}
	if rows := store.snapshotCount(); rows != 1 {
		t.Errorf("expected no recovery row when last-good is already persisted, got %d rows", rows)
	}
}

func TestAgent_FailedFirstTurn_AfterResume_ReturnsParentSnapshotID(t *testing.T) {
	// Resuming from a snapshot and failing before any turn completes:
	// the parent snapshot already captures the last-good state, so the
	// failed output points back at it and no recovery row is written.
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()

	parent, err := store.SaveSnapshot(ctx, "",
		func(_ *SessionSnapshot[testState]) (*SessionSnapshot[testState], error) {
			return &SessionSnapshot[testState]{
				Status: SnapshotStatusCompleted,
				State: &SessionState[testState]{
					Messages: []*ai.Message{
						ai.NewUserTextMessage("one"),
						ai.NewModelTextMessage("echo: one"),
					},
					Custom: testState{Counter: 1},
				},
			}, nil
		})
	if err != nil {
		t.Fatalf("SaveSnapshot: %v", err)
	}

	af := defineLastGoodTestAgent(reg, "resumeFailFirst", WithSessionStore[testState](store))

	out, err := af.RunText(ctx, "boom", WithSnapshotID[testState](parent.SnapshotID))
	if err != nil {
		t.Fatalf("RunText: %v", err)
	}
	if out.FinishReason != AgentFinishReasonFailed {
		t.Errorf("expected finish reason %q, got %q", AgentFinishReasonFailed, out.FinishReason)
	}
	if out.SnapshotID != parent.SnapshotID {
		t.Errorf("expected failed output to return parent snapshot %q, got %q",
			parent.SnapshotID, out.SnapshotID)
	}
	if rows := store.snapshotCount(); rows != 1 {
		t.Errorf("expected no new rows on first-turn failure after resume, got %d", rows)
	}
}

func TestAgent_FailedTurn_EmitsFailedTurnEnd(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	// Hold the agent fn open until the client has consumed the failed
	// TurnEnd, so the chunk delivery is deterministic (the runtime stops
	// forwarding chunks once fn returns with an error).
	turnEndSeen := make(chan struct{})
	af := DefineCustomAgent(reg, "failedTurnEnd",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			err := sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				return nil, fmt.Errorf("boom")
			})
			select {
			case <-turnEndSeen:
			case <-time.After(2 * time.Second):
			}
			return nil, err
		},
	)

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	sendText(t, conn, "hi")

	turnEnd := nextTurnEnd(t, conn)
	close(turnEndSeen)

	if turnEnd.FinishReason != AgentFinishReasonFailed {
		t.Errorf("expected TurnEnd finish reason %q, got %q", AgentFinishReasonFailed, turnEnd.FinishReason)
	}
	if turnEnd.SnapshotID != "" {
		t.Errorf("failed turn must not snapshot its partial state, got snapshot %q", turnEnd.SnapshotID)
	}

	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	if out.FinishReason != AgentFinishReasonFailed {
		t.Errorf("expected finish reason %q, got %q", AgentFinishReasonFailed, out.FinishReason)
	}
}

func TestAgent_CustomAgentContinuesAfterFailedTurn(t *testing.T) {
	// A custom agent may treat a turn failure as recoverable: swallow the
	// error from Run and keep processing queued inputs. The intake must
	// keep pacing inputs after a failed turn for this to work.
	ctx := context.Background()
	reg := newTestRegistry(t)

	af := DefineCustomAgent(reg, "continueAfterFail",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			for {
				err := sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
					text := input.Message.Content[0].Text
					if text == "boom" {
						return nil, fmt.Errorf("recoverable failure")
					}
					sess.AddMessages(ai.NewModelTextMessage("echo: " + text))
					return &TurnResult{FinishReason: AgentFinishReasonStop}, nil
				})
				if err == nil {
					break // input channel closed
				}
			}
			return sess.Result(), nil
		},
	)

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	for _, text := range []string{"one", "boom", "two"} {
		sendText(t, conn, text)
	}

	// A hang here means intake pacing after a failed turn is broken.
	out, err := outputWithin(t, conn, 2*time.Second)
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	// The agent recovered: the invocation succeeds with the live state
	// (including the failed turn's user message, which the agent chose
	// to keep by continuing).
	if out.FinishReason != AgentFinishReasonStop {
		t.Errorf("expected finish reason %q, got %q", AgentFinishReasonStop, out.FinishReason)
	}
	if out.Error != nil {
		t.Errorf("expected no error on recovered invocation, got %+v", out.Error)
	}
	// user one, echo one, user boom, user two, echo two.
	if got := len(out.State.Messages); got != 5 {
		t.Errorf("expected 5 messages, got %d", got)
	}
}

func TestAgent_InitFailure_FailsActionWithStatus(t *testing.T) {
	// Pre-turn precondition/validation failures fail the action outright
	// (no failed-AgentOutput conversion, no snapshot): the invocation never
	// reached the input phase, so there is no conversation state to hand
	// back. The error keeps its original status.
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()

	echo := func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
		return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
			return nil, nil
		})
	}
	serverManaged := DefineCustomAgent(reg, "initFailServer", echo, WithSessionStore[testState](store))
	clientManaged := DefineCustomAgent(reg, "initFailClient", echo)

	tests := []struct {
		name       string
		run        func() (*AgentOutput[testState], error)
		wantStatus core.StatusName
		wantMsg    string
	}{
		{
			name: "state rejected when server-managed",
			run: func() (*AgentOutput[testState], error) {
				return serverManaged.RunText(ctx, "hi", WithState(&SessionState[testState]{}))
			},
			wantStatus: core.FAILED_PRECONDITION,
			wantMsg:    "session store",
		},
		{
			name: "snapshot ID rejected when client-managed",
			run: func() (*AgentOutput[testState], error) {
				return clientManaged.RunText(ctx, "hi", WithSnapshotID[testState]("some-id"))
			},
			wantStatus: core.FAILED_PRECONDITION,
			wantMsg:    "no session store",
		},
		{
			name: "missing snapshot",
			run: func() (*AgentOutput[testState], error) {
				return serverManaged.RunText(ctx, "hi", WithSnapshotID[testState]("nope"))
			},
			wantStatus: core.NOT_FOUND,
			wantMsg:    "not found",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			out, err := tc.run()
			if err == nil {
				t.Fatalf("expected error, got output: %+v", out)
			}
			if out != nil {
				t.Errorf("expected nil output on init failure, got %+v", out)
			}
			ge := core.AsGenkitError(err)
			if ge.Status != tc.wantStatus {
				t.Errorf("expected status %q, got %q (err: %v)", tc.wantStatus, ge.Status, err)
			}
			if !strings.Contains(ge.Message, tc.wantMsg) {
				t.Errorf("expected error message to contain %q, got %q", tc.wantMsg, ge.Message)
			}
		})
	}
}

// TestAgent_JSONInitRejectsUnknownFields verifies that the agent's JSON init
// paths reject a payload carrying a field the inferred AgentInit schema does not
// declare, surfacing INVALID_ARGUMENT rather than silently dropping it and
// starting a fresh session as if nothing were wrong. This exercises the real
// inferred init schema end to end (the core mechanism is covered by
// TestBidiJSONInitRejectsUnknownFields).
func TestAgent_JSONInitRejectsUnknownFields(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	echo := func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
		return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
			return nil, nil
		})
	}
	af := DefineCustomAgent(reg, "jsonInitUnknownFields", echo)

	assertRejected := func(t *testing.T, err error) {
		t.Helper()
		if err == nil {
			t.Fatal("expected INVALID_ARGUMENT for unknown init field, got nil")
		}
		if ge := core.AsGenkitError(err); ge.Status != core.INVALID_ARGUMENT {
			t.Errorf("status = %q, want %q (err: %v)", ge.Status, core.INVALID_ARGUMENT, err)
		}
	}

	badInit := json.RawMessage(`{"bogus":true}`)

	t.Run("ConnectJSON", func(t *testing.T) {
		_, err := af.ConnectJSON(ctx, &api.BidiJSONOptions{Init: badInit})
		assertRejected(t, err)
	})

	t.Run("RunBidiJSON", func(t *testing.T) {
		input, err := json.Marshal(&AgentInput{Message: ai.NewUserTextMessage("hi")})
		if err != nil {
			t.Fatalf("marshal input: %v", err)
		}
		_, err = af.RunBidiJSON(ctx, input, nil, &api.BidiJSONOptions{Init: badInit})
		assertRejected(t, err)
	})

	// An empty init object declares no fields, so it passes validation and
	// starts a fresh session.
	t.Run("empty init accepted", func(t *testing.T) {
		conn, err := af.ConnectJSON(ctx, &api.BidiJSONOptions{Init: json.RawMessage(`{}`)})
		if err != nil {
			t.Fatalf("ConnectJSON with empty init: %v", err)
		}
		conn.Close()
		if _, err := conn.Output(); err != nil {
			t.Fatalf("Output: %v", err)
		}
	})
}

func TestAgent_SetMessages(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	af := DefineCustomAgent(reg, "setMsgsFlow",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				// Replace all messages with just one.
				sess.SetMessages([]*ai.Message{ai.NewModelTextMessage("replaced")})
				return nil, nil
			})
		},
	)

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect failed: %v", err)
	}

	sendTurn(t, conn, "original")
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

// turnSpanState parses a turn span's genkit:output attribute, which the agent
// records as {"state": <session state>} (see turnSpanOutput), and returns the
// embedded state.
func turnSpanState(t *testing.T, span sdktrace.ReadOnlySpan) *SessionState[testState] {
	t.Helper()
	raw, ok := spanAttr(span, "genkit:output")
	if !ok {
		t.Fatalf("span %q: missing genkit:output attribute", span.Name())
	}
	var out turnSpanOutput[testState]
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		t.Fatalf("span %q: genkit:output %q is not valid turn output: %v", span.Name(), raw, err)
	}
	if out.State == nil {
		t.Fatalf("span %q: genkit:output %q carries no state", span.Name(), raw)
	}
	return out.State
}

// TestAgent_TurnSpanOutput verifies each turn span's output is the committed
// session state at turn end, wrapped as {state: ...}. The agent is
// client-managed (no store), so no turn-end snapshot is written and the
// per-turn snapshot-ID attribute is absent.
func TestAgent_TurnSpanOutput(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)
	spans := collectSpans(t)

	af := DefineCustomAgent(reg, "turnOutputFlow",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				sess.UpdateCustom(func(s testState) testState {
					s.Counter++
					return s
				})
				resp.SendModelChunk(&ai.ModelResponseChunk{
					Content: []*ai.Part{ai.NewTextPart("reply")},
				})
				resp.SendArtifact(&Artifact{
					Name:  "out.txt",
					Parts: []*ai.Part{ai.NewTextPart("content")},
				})
				sess.AddMessages(ai.NewModelTextMessage("reply"))
				return nil, nil
			})
		},
	)

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect failed: %v", err)
	}

	// Two turns.
	for turn := range 2 {
		sendTurn(t, conn, fmt.Sprintf("turn %d", turn))
	}

	conn.Close()
	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output failed: %v", err)
	}

	// Each turn span's output carries the cumulative state through that turn:
	// the running counter and one user message plus one reply per turn.
	for turn := range 2 {
		name := fmt.Sprintf("runTurn-%d", turn+1)
		span := spans.byName(name)
		if span == nil {
			t.Fatalf("missing span %q", name)
		}
		state := turnSpanState(t, span)
		if state.SessionID != out.SessionID {
			t.Errorf("%s: state.sessionId = %q, want %q", name, state.SessionID, out.SessionID)
		}
		if want := turn + 1; state.Custom.Counter != want {
			t.Errorf("%s: state.custom.counter = %d, want %d", name, state.Custom.Counter, want)
		}
		if got, want := len(state.Messages), 2*(turn+1); got != want {
			t.Errorf("%s: len(state.messages) = %d, want %d", name, got, want)
		}
		// An artifact added during the turn rides the committed state too.
		if got := len(state.Artifacts); got != 1 {
			t.Errorf("%s: len(state.artifacts) = %d, want 1", name, got)
		}
		// Client-managed: no snapshot, so no snapshot-ID attribute.
		if v, ok := spanAttr(span, snapshotIDSpanAttrKey); ok {
			t.Errorf("%s: unexpected %s = %q (client-managed writes no snapshot)", name, snapshotIDSpanAttrKey, v)
		}
	}
}

// TestAgent_TurnSpanOutput_WithSnapshots verifies that for a server-managed
// agent the turn span both carries the persisted snapshot's ID under
// genkit:metadata:agent:snapshotId and records the committed state as its
// {state: ...} output, agreeing with the snapshot the ID points to.
func TestAgent_TurnSpanOutput_WithSnapshots(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()
	spans := collectSpans(t)

	af := defineCounterAgent(reg, "turnOutputSnapshotFlow", WithSessionStore(store))

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect failed: %v", err)
	}

	sendText(t, conn, "hello")
	te := nextTurnEnd(t, conn)
	if te.SnapshotID == "" {
		t.Fatal("expected a snapshot ID on the turn-end chunk")
	}
	conn.Close()
	if _, err := conn.Output(); err != nil {
		t.Fatalf("Output failed: %v", err)
	}

	span := spans.byName("runTurn-1")
	if span == nil {
		t.Fatal("missing span runTurn-1")
	}

	// The turn span is tagged with the turn-end snapshot's ID.
	got, ok := spanAttr(span, snapshotIDSpanAttrKey)
	if !ok {
		t.Fatalf("turn span: missing %s", snapshotIDSpanAttrKey)
	}
	if got != te.SnapshotID {
		t.Errorf("turn span %s = %q, want %q (the turn-end snapshot)", snapshotIDSpanAttrKey, got, te.SnapshotID)
	}

	// Its output is the committed state, matching the persisted snapshot.
	state := turnSpanState(t, span)
	if state.Custom.Counter != 1 {
		t.Errorf("turn span state.custom.counter = %d, want 1", state.Custom.Counter)
	}
	snap, err := store.GetSnapshot(ctx, te.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}
	if state.SessionID != snap.SessionID {
		t.Errorf("turn span state.sessionId = %q, want %q (snapshot's session)", state.SessionID, snap.SessionID)
	}
	if got, want := state.Custom.Counter, snap.State.Custom.Counter; got != want {
		t.Errorf("turn span state.custom.counter = %d, want %d (snapshot's counter)", got, want)
	}
}

// TestAgent_CustomPatchWholeDocumentReplace verifies the server emits the first
// custom-state mutation of a turn as a whole-document replace: a single RFC 6902
// replace at the root pointer, which re-bases a client that may not share the
// server's baseline. The client-side effect of this re-basing across turns is
// covered by TestAgentConnection_Custom_TracksStreamedPatches.
func TestAgent_CustomPatchWholeDocumentReplace(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	af := defineCounterAgent(reg, "wholeDocReplaceFlow")

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect failed: %v", err)
	}
	sendText(t, conn, "hello")

	var patch JSONPatch
	for chunk, err := range conn.Receive() {
		if err != nil {
			t.Fatalf("Receive: %v", err)
		}
		if chunk.CustomPatch != nil {
			patch = chunk.CustomPatch
			break
		}
		if chunk.TurnEnd != nil {
			break
		}
	}
	conn.Close()

	if len(patch) != 1 || patch[0].Op != JSONPatchOpReplace || patch[0].Path != "" {
		t.Errorf("first customPatch = %+v, want a single whole-document replace at root", patch)
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

// personaInput is the input schema for the shared prompt exercised by
// TestPromptAgent_NamedPromptSharedAcrossAgents.
type personaInput struct {
	Personality string `json:"personality"`
}

// TestPromptAgent_NamedPromptSharedAcrossAgents verifies that WithNamedPrompt
// backs several agents with a single prompt, rendering each with its own
// input, and that an agent's name is independent of the prompt it references.
func TestPromptAgent_NamedPromptSharedAcrossAgents(t *testing.T) {
	ctx := context.Background()
	reg := registry.New()
	ai.ConfigureFormats(reg)

	var mu sync.Mutex
	var renderedSystems []string
	ai.DefineModel(reg, "test/capture", &ai.ModelOptions{Supports: &ai.ModelSupports{Multiturn: true, SystemRole: true}},
		func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			mu.Lock()
			for _, m := range req.Messages {
				if m.Role == ai.RoleSystem {
					renderedSystems = append(renderedSystems, m.Text())
				}
			}
			mu.Unlock()
			return &ai.ModelResponse{Request: req, Message: ai.NewModelTextMessage("ok")}, nil
		},
	)
	ai.DefineGenerateAction(ctx, reg)

	// One shared prompt with a personality variable, registered under a name
	// that matches no agent.
	ai.DefinePrompt(reg, "sharedChat",
		ai.WithModelName("test/capture"),
		ai.WithInputType(personaInput{}),
		ai.WithSystem("You are {{personality}}."),
	)

	// Two agents, different names, the one prompt, different inputs.
	pirate := DefinePromptAgent[testState](reg, "pirate",
		WithNamedPrompt[testState]("sharedChat", personaInput{Personality: "a pirate"}))
	chef := DefinePromptAgent[testState](reg, "chef",
		WithNamedPrompt[testState]("sharedChat", personaInput{Personality: "a chef"}))

	// The agents register under their own names, not the prompt's.
	if pirate.Name() != "pirate" {
		t.Errorf("pirate.Name() = %q, want %q", pirate.Name(), "pirate")
	}
	if chef.Name() != "chef" {
		t.Errorf("chef.Name() = %q, want %q", chef.Name(), "chef")
	}

	if _, err := pirate.RunText(ctx, "hi"); err != nil {
		t.Fatalf("pirate RunText: %v", err)
	}
	if _, err := chef.RunText(ctx, "hi"); err != nil {
		t.Fatalf("chef RunText: %v", err)
	}

	// Each agent rendered the shared prompt with its own personality input.
	joined := strings.Join(renderedSystems, " | ")
	if !strings.Contains(joined, "You are a pirate.") {
		t.Errorf("pirate personality not rendered; system prompts = %q", joined)
	}
	if !strings.Contains(joined, "You are a chef.") {
		t.Errorf("chef personality not rendered; system prompts = %q", joined)
	}
}

// TestDefinePromptAgent_DefaultAndNamed exercises the option-configured prompt
// agent: the default same-named lookup (no source option), the WithNamedPrompt
// override sourcing a shared prompt with a code-supplied input, and that the
// shared agent options (WithDescription) compose alongside the prompt source.
func TestDefinePromptAgent_DefaultAndNamed(t *testing.T) {
	ctx := context.Background()
	reg := registry.New()
	ai.ConfigureFormats(reg)

	var mu sync.Mutex
	var renderedSystems []string
	ai.DefineModel(reg, "test/capture", &ai.ModelOptions{Supports: &ai.ModelSupports{Multiturn: true, SystemRole: true}},
		func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			mu.Lock()
			for _, m := range req.Messages {
				if m.Role == ai.RoleSystem {
					renderedSystems = append(renderedSystems, m.Text())
				}
			}
			mu.Unlock()
			return &ai.ModelResponse{Request: req, Message: ai.NewModelTextMessage("ok")}, nil
		},
	)
	ai.DefineGenerateAction(ctx, reg)

	// A same-named prompt for the default lookup, and a shared parameterized
	// prompt for the WithNamedPrompt override.
	ai.DefinePrompt(reg, "chef",
		ai.WithModelName("test/capture"),
		ai.WithSystem("You are a chef."),
	)
	ai.DefinePrompt(reg, "sharedChat",
		ai.WithModelName("test/capture"),
		ai.WithInputType(personaInput{}),
		ai.WithSystem("You are {{personality}}."),
	)

	// Default: no source option resolves the prompt named after the agent.
	chef := DefinePromptAgent[testState](reg, "chef",
		WithDescription[testState]("a chef agent"))
	if chef.Name() != "chef" {
		t.Fatalf("chef.Name() = %q, want %q", chef.Name(), "chef")
	}
	if got := chef.Desc().Description; got != "a chef agent" {
		t.Errorf("chef description = %q, want %q", got, "a chef agent")
	}

	// Override: source the shared prompt with a code-supplied input. The agent
	// name ("pirate") is independent of the referenced prompt ("sharedChat").
	pirate := DefinePromptAgent[testState](reg, "pirate",
		WithNamedPrompt[testState]("sharedChat", personaInput{Personality: "a pirate"}))

	if _, err := chef.RunText(ctx, "hi"); err != nil {
		t.Fatalf("chef RunText: %v", err)
	}
	if _, err := pirate.RunText(ctx, "hi"); err != nil {
		t.Fatalf("pirate RunText: %v", err)
	}

	joined := strings.Join(renderedSystems, " | ")
	if !strings.Contains(joined, "You are a chef.") {
		t.Errorf("default same-named prompt not rendered; systems = %q", joined)
	}
	if !strings.Contains(joined, "You are a pirate.") {
		t.Errorf("WithNamedPrompt input not rendered; systems = %q", joined)
	}
}

// TestDefinePromptAgent_Panics covers the definition-time failures: a prompt
// that is not registered, and setting the prompt source more than once.
func TestDefinePromptAgent_Panics(t *testing.T) {
	reg := setupPromptTestRegistry(t)
	ai.DefinePrompt(reg, "present", ai.WithModelName("test/echo"))

	t.Run("missing prompt", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("expected panic for missing prompt")
			}
		}()
		DefinePromptAgent[testState](reg, "absent")
	})

	t.Run("duplicate prompt source", func(t *testing.T) {
		defer func() {
			if r := recover(); r == nil {
				t.Fatal("expected panic for duplicate WithNamedPrompt")
			}
		}()
		DefinePromptAgent[testState](reg, "present",
			WithNamedPrompt[testState]("present", nil),
			WithNamedPrompt[testState]("present", nil))
	})
}

func TestPromptAgent_Basic(t *testing.T) {
	ctx := context.Background()
	reg := setupPromptTestRegistry(t)

	ai.DefinePrompt(reg, "testPrompt",
		ai.WithModelName("test/echo"),
		ai.WithSystem("You are a test assistant."),
	)

	af := DefinePromptAgent[testState](reg, "testPrompt")

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect failed: %v", err)
	}

	// Turn 1.
	sendText(t, conn, "hello")

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
	sendTurn(t, conn, "world")

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

	af := DefinePromptAgent[testState](reg, "historyPrompt")

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect failed: %v", err)
	}

	// Turn 1.
	sendText(t, conn, "turn1")
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
	sendText(t, conn, "turn2")
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
	store := newTestInMemStore[testState]()

	ai.DefinePrompt(reg, "snapPrompt",
		ai.WithModelName("test/echo"),
		ai.WithSystem("You are a test assistant."),
	)

	af := DefinePromptAgent[testState](reg, "snapPrompt",
		WithSessionStore(store),
	)

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect failed: %v", err)
	}

	sendTurn(t, conn, "hello")
	conn.Close()

	resp, err := conn.Output()
	if err != nil {
		t.Fatalf("Output failed: %v", err)
	}
	if resp.SnapshotID == "" {
		t.Fatal("expected snapshot ID")
	}

	conn2, err := af.Connect(ctx, WithSnapshotID[testState](resp.SnapshotID))
	if err != nil {
		t.Fatalf("Connect with snapshot failed: %v", err)
	}

	sendTurn(t, conn2, "continued")
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

	af := DefinePromptAgent[testState](reg, "toolPrompt")

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect failed: %v", err)
	}

	sendTurn(t, conn, "go")
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
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				if input.Message != nil {
					sess.AddMessages(ai.NewModelTextMessage("echo: " + input.Message.Content[0].Text))
				}
				sess.UpdateCustom(func(s testState) testState {
					s.Counter++
					return s
				})
				return nil, nil
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
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				if input.Message != nil {
					sess.AddMessages(ai.NewModelTextMessage("reply"))
				}
				return nil, nil
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

	af := defineCounterAgent(reg, "runStateFlow")

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
	store := newTestInMemStore[testState]()

	af := defineCounterAgent(reg, "runSnapshotFlow", WithSessionStore(store))

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

	af := DefinePromptAgent[testState](reg, "runTextPrompt")

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

// TestPromptAgent_RejectsInvalidInputMessage verifies the prompt-backed loop
// rejects turn messages it cannot safely consume: a non-user role, or tool
// request/response parts (which belong on AgentInput.Resume, not a turn
// message). Each resolves as a failed output carrying an INVALID_ARGUMENT
// error rather than reaching the model.
func TestPromptAgent_RejectsInvalidInputMessage(t *testing.T) {
	ctx := context.Background()
	reg := setupPromptTestRegistry(t)
	var modelCalls atomic.Int64
	ai.DefineModel(reg, "test/reject", &ai.ModelOptions{Supports: &ai.ModelSupports{Multiturn: true}},
		func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			modelCalls.Add(1)
			return &ai.ModelResponse{Message: ai.NewModelTextMessage("unexpected")}, nil
		},
	)
	ai.DefinePrompt(reg, "rejectPrompt", ai.WithModelName("test/reject"))
	af := DefinePromptAgent[testState](reg, "rejectPrompt")

	tests := []struct {
		name    string
		message *ai.Message
		resume  *ToolResume
		wantMsg string
	}{
		{
			name:    "missing message",
			message: nil,
			wantMsg: "message",
		},
		{
			// A Resume struct that is non-nil but carries no Respond/Restart
			// entries must be treated the same as no Resume at all; only
			// checking for a nil Resume would let this slip through to the
			// model with an effectively empty turn.
			name:    "empty resume payload",
			message: nil,
			resume:  &ToolResume{},
			wantMsg: "message",
		},
		{
			name:    "non-user role",
			message: &ai.Message{Role: ai.RoleModel, Content: []*ai.Part{ai.NewTextPart("hi")}},
			wantMsg: "role",
		},
		{
			name: "tool request part",
			message: &ai.Message{Role: ai.RoleUser, Content: []*ai.Part{
				ai.NewTextPart("hi"),
				ai.NewToolRequestPart(&ai.ToolRequest{Name: "doThing", Ref: "1"}),
			}},
			wantMsg: "tool request",
		},
		{
			name: "tool response part",
			message: &ai.Message{Role: ai.RoleUser, Content: []*ai.Part{
				ai.NewToolResponsePart(&ai.ToolResponse{Name: "doThing", Ref: "1"}),
			}},
			wantMsg: "tool",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			out, err := af.Run(ctx, &AgentInput{Message: tc.message, Resume: tc.resume})
			if err != nil {
				t.Fatalf("Run: %v", err)
			}
			if out.FinishReason != AgentFinishReasonFailed {
				t.Errorf("FinishReason = %q, want %q", out.FinishReason, AgentFinishReasonFailed)
			}
			if out.Error == nil {
				t.Fatal("expected output error, got nil")
			}
			if out.Error.Status != core.INVALID_ARGUMENT {
				t.Errorf("Error.Status = %q, want %q", out.Error.Status, core.INVALID_ARGUMENT)
			}
			if !strings.Contains(out.Error.Message, tc.wantMsg) {
				t.Errorf("Error.Message = %q, want substring %q", out.Error.Message, tc.wantMsg)
			}
		})
	}
	if got := modelCalls.Load(); got != 0 {
		t.Fatalf("model calls = %d, want 0", got)
	}
}

func TestValidateResumeAgainstHistory(t *testing.T) {
	// History spans two model messages (each carrying a tool request) plus a
	// user message that must be ignored. The whole history is searched, not
	// just the last turn.
	history := []*ai.Message{
		{Role: ai.RoleUser, Content: []*ai.Part{ai.NewTextPart("hi")}},
		{Role: ai.RoleModel, Content: []*ai.Part{
			ai.NewToolRequestPart(&ai.ToolRequest{Name: "first", Ref: "r1", Input: map[string]any{"a": float64(1)}}),
		}},
		{Role: ai.RoleModel, Content: []*ai.Part{
			ai.NewTextPart("thinking"),
			ai.NewToolRequestPart(&ai.ToolRequest{Name: "second", Ref: "r2", Input: map[string]any{"b": "x"}}),
		}},
	}

	respond := func(name, ref string) []*ai.Part {
		return []*ai.Part{ai.NewToolResponsePart(&ai.ToolResponse{Name: name, Ref: ref, Output: "ok"})}
	}
	restart := func(name, ref string, input any) []*ai.Part {
		return []*ai.Part{ai.NewToolRequestPart(&ai.ToolRequest{Name: name, Ref: ref, Input: input})}
	}

	tests := []struct {
		name    string
		resume  *ToolResume
		wantErr string // substring the error must contain; "" means it must succeed
	}{
		{name: "nil resume", resume: nil},
		{name: "empty resume", resume: &ToolResume{}},
		{name: "respond matches first model message", resume: &ToolResume{Respond: respond("first", "r1")}},
		{name: "respond matches a later model message", resume: &ToolResume{Respond: respond("second", "r2")}},
		{name: "restart matches input exactly", resume: &ToolResume{Restart: restart("first", "r1", map[string]any{"a": float64(1)})}},
		{
			name:    "respond references unknown tool",
			resume:  &ToolResume{Respond: respond("ghost", "r1")},
			wantErr: "not found in session history",
		},
		{
			name:    "respond references known tool with wrong ref",
			resume:  &ToolResume{Respond: respond("first", "wrong")},
			wantErr: "not found in session history",
		},
		{
			name:    "restart references unknown tool",
			resume:  &ToolResume{Restart: restart("ghost", "r1", nil)},
			wantErr: "not found in session history",
		},
		{
			name:    "restart forges modified input",
			resume:  &ToolResume{Restart: restart("first", "r1", map[string]any{"a": float64(2)})},
			wantErr: "modified inputs",
		},
		{
			// An int 1 normalizes to the same JSON shape as the stored float64
			// 1, so a faithful restart is not mistaken for a forgery.
			name:   "restart input matches across json number types",
			resume: &ToolResume{Restart: restart("first", "r1", map[string]any{"a": 1})},
		},
		{
			// A kind-PartToolRequest part with a nil ToolRequest pointer (e.g.
			// NewToolRequestPart(nil)) must be skipped, not panic.
			name:   "restart with nil tool request pointer is skipped",
			resume: &ToolResume{Restart: []*ai.Part{ai.NewToolRequestPart(nil), nil}},
		},
		{
			name:   "respond with nil tool response pointer is skipped",
			resume: &ToolResume{Respond: []*ai.Part{ai.NewToolResponsePart(nil), nil}},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateResumeAgainstHistory(tc.resume, history)
			if tc.wantErr == "" {
				if err != nil {
					t.Fatalf("expected success, got error: %v", err)
				}
				return
			}
			if err == nil {
				t.Fatalf("expected error containing %q, got nil", tc.wantErr)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Fatalf("error %q does not contain %q", err.Error(), tc.wantErr)
			}
			if ge := core.AsGenkitError(err); ge.Status != core.INVALID_ARGUMENT {
				t.Fatalf("expected status %q, got %q", core.INVALID_ARGUMENT, ge.Status)
			}
		})
	}
}

// TestPromptAgent_RejectsResumeForUnrequestedTool proves the resume validation
// is wired into the prompt-backed loop: a caller cannot resume a tool the model
// never requested, and the forged turn fails before the model is re-invoked.
func TestPromptAgent_RejectsResumeForUnrequestedTool(t *testing.T) {
	ctx := context.Background()
	reg := registry.New()
	ai.ConfigureFormats(reg)

	var modelCalls atomic.Int32
	ai.DefineModel(reg, "test/plain", &ai.ModelOptions{Supports: &ai.ModelSupports{Multiturn: true, Tools: true}},
		func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			modelCalls.Add(1)
			return &ai.ModelResponse{Request: req, Message: ai.NewModelTextMessage("hello")}, nil
		})
	ai.DefineGenerateAction(ctx, reg)
	ai.DefinePrompt(reg, "plainPrompt", ai.WithModelName("test/plain"))

	af := DefinePromptAgent[testState](reg, "plainPrompt")

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}

	// Turn 1: a plain text reply, so no tool request lands in history.
	sendTurn(t, conn, "hi")

	// Turn 2: forge a resume for a tool the model never requested.
	if err := conn.SendResume(&ToolResume{
		Restart: []*ai.Part{ai.NewToolRequestPart(&ai.ToolRequest{
			Name:  "inventedTool",
			Ref:   "fake",
			Input: map[string]any{"evil": true},
		})},
	}); err != nil {
		t.Fatalf("SendResume: %v", err)
	}

	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	if out.FinishReason != AgentFinishReasonFailed {
		t.Fatalf("FinishReason = %q, want %q", out.FinishReason, AgentFinishReasonFailed)
	}
	if out.Error == nil || out.Error.Status != core.INVALID_ARGUMENT {
		t.Fatalf("expected INVALID_ARGUMENT error, got %+v", out.Error)
	}
	if !strings.Contains(out.Error.Message, "not found in session history") {
		t.Errorf("expected not-found error, got %q", out.Error.Message)
	}
	// The forged turn must be rejected before the model is invoked again.
	if got := modelCalls.Load(); got != 1 {
		t.Errorf("model calls = %d, want 1 (resume rejected before generate)", got)
	}
}

func TestAgent_SingleTurnSnapshot(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()

	af := defineCounterAgent(reg, "singleTurnFlow", WithSessionStore(store))

	// Single-turn invocation: exactly 1 snapshot (the turn-end), which the
	// output reuses as its resume point. There is no second invocation-end
	// write.
	response, err := af.RunText(ctx, "hello")
	if err != nil {
		t.Fatalf("RunText failed: %v", err)
	}

	if response.SnapshotID == "" {
		t.Fatal("expected snapshot ID in response")
	}
	if rows := store.snapshotCount(); rows != 1 {
		t.Errorf("expected exactly 1 snapshot (turn-end only), got %d", rows)
	}

	snap, err := store.GetSnapshot(ctx, response.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot failed: %v", err)
	}
	// The turn-end snapshot should have no parent (first and only snapshot).
	if snap.ParentID != "" {
		t.Errorf("expected no parent (single snapshot), got parent %q", snap.ParentID)
	}
}

func TestAgent_MultiTurnSnapshot(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()

	af := defineCounterAgent(reg, "multiDedupFlow", WithSessionStore(store))

	// Multi-turn: one snapshot per turn; the output reuses the last one.
	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect failed: %v", err)
	}

	var snapshotIDs []string
	for i := 0; i < 3; i++ {
		sendText(t, conn, fmt.Sprintf("turn %d", i))
		if te := nextTurnEnd(t, conn); te.SnapshotID != "" {
			snapshotIDs = append(snapshotIDs, te.SnapshotID)
		}
	}
	conn.Close()

	response, err := conn.Output()
	if err != nil {
		t.Fatalf("Output failed: %v", err)
	}

	// Should have 3 turn-end snapshots, one per turn.
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

func TestAgent_PostRunMutationNotSnapshotted(t *testing.T) {
	// Snapshots happen at turn end only. State a custom agent mutates after
	// its turn loop rides on the returned output but is not persisted: the
	// output's snapshot ID is the last turn-end row, which predates the
	// mutation, and no extra row is written.
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()

	af := DefineCustomAgent(reg, "postRunMutateFlow",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			if err := sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				sess.AddMessages(ai.NewModelTextMessage("reply"))
				sess.UpdateCustom(func(s testState) testState {
					s.Counter = 1
					return s
				})
				return nil, nil
			}); err != nil {
				return nil, err
			}
			// Mutate state after sess.Run returns: rides on the output but is
			// not snapshotted.
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
	// Exactly one snapshot (the turn-end); the post-loop mutation wrote none.
	if rows := store.snapshotCount(); rows != 1 {
		t.Errorf("expected exactly 1 snapshot (no post-loop write), got %d", rows)
	}

	snap, err := store.GetSnapshot(ctx, response.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot failed: %v", err)
	}
	// The snapshot holds the turn-end state (counter=1), not the post-loop
	// mutation (counter=99).
	if snap.State.Custom.Counter != 1 {
		t.Errorf("expected turn-end counter=1 in snapshot, got %d", snap.State.Custom.Counter)
	}
}

// TestAgent_FnPanicReturnsError verifies that a panic inside the agent
// function is recovered and surfaced as an error, rather than crashing the
// process or hanging the streaming goroutine.
func TestAgent_FnPanicResolvesAsFailedOutput(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	af := DefineCustomAgent(reg, "panicFlow",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				resp.SendModelChunk(&ai.ModelResponseChunk{
					Content: []*ai.Part{ai.NewTextPart("before-panic")},
				})
				panic("boom")
			})
		},
	)

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	sendText(t, conn, "trigger")

	// A hang here means the streaming goroutine leaked.
	out, err := outputWithin(t, conn, 2*time.Second)
	if err != nil {
		t.Fatalf("expected panic to resolve as failed output, got error: %v", err)
	}
	if out.FinishReason != AgentFinishReasonFailed {
		t.Errorf("expected finish reason %q, got %q", AgentFinishReasonFailed, out.FinishReason)
	}
	if out.Error == nil || !strings.Contains(out.Error.Message, "panicked") {
		t.Errorf("expected panic error on output, got: %+v", out.Error)
	}
	if out.Error != nil && out.Error.Status != core.INTERNAL {
		t.Errorf("expected status %q, got %q", core.INTERNAL, out.Error.Status)
	}
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
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			defer close(fnDone)
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				close(emitting)
				// Emit until ctx cancels. Without the goroutine's
				// ctx-aware drain, this would deadlock once the consumer
				// stops reading.
				for {
					select {
					case <-ctx.Done():
						return nil, ctx.Err()
					default:
					}
					resp.SendModelChunk(&ai.ModelResponseChunk{
						Content: []*ai.Part{ai.NewTextPart("tick")},
					})
				}
			})
		},
	)

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	sendText(t, conn, "go")

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

// nextTurnEnd consumes conn's stream until the next TurnEnd chunk and returns
// a copy of it, failing the test if the stream errors or ends first. Use it
// in tests that only need to advance to a turn boundary; tests that must
// inspect intermediate chunks should range over Receive directly.
func nextTurnEnd[State any](t *testing.T, conn *AgentConnection[State]) *TurnEnd {
	t.Helper()
	for chunk, err := range conn.Receive() {
		if err != nil {
			t.Fatalf("Receive: %v", err)
		}
		if chunk.TurnEnd != nil {
			te := *chunk.TurnEnd
			return &te
		}
	}
	t.Fatal("no TurnEnd chunk observed")
	return nil
}

// outputWithin finalizes conn and returns its output, failing the test if
// finalization does not complete within d. Use it in tests where a
// regression would make Output hang rather than fail.
func outputWithin[State any](t *testing.T, conn *AgentConnection[State], d time.Duration) (*AgentOutput[State], error) {
	t.Helper()
	type outcome struct {
		out *AgentOutput[State]
		err error
	}
	done := make(chan outcome, 1)
	go func() {
		out, err := conn.Output()
		done <- outcome{out, err}
	}()
	select {
	case oc := <-done:
		return oc.out, oc.err
	case <-time.After(d):
		t.Fatal("Output did not complete in time; the runtime likely hung")
		return nil, nil
	}
}

func TestAgent_TurnEnd_CarriesSnapshotID(t *testing.T) {
	// Sanity: each TurnEnd chunk carries the snapshot ID of the turn-end
	// snapshot, and the snapshots themselves are persisted.
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()

	af := DefineCustomAgent(reg, "turnEndSnapshotFlow",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				sess.AddMessages(ai.NewModelTextMessage("ok"))
				return nil, nil
			})
		},
		WithSessionStore(store),
	)

	conn, err := af.Connect(context.Background())
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}

	var observed []TurnEnd
	for turn := 0; turn < 3; turn++ {
		sendText(t, conn, fmt.Sprintf("turn %d", turn))
		observed = append(observed, *nextTurnEnd(t, conn))
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
	store := newTestInMemStore[testState]()

	entered := make(chan struct{}, 4)
	release := make(chan struct{})

	af := DefineCustomAgent(reg, "detachInFlight",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				entered <- struct{}{}
				<-release
				sess.AddMessages(ai.NewModelTextMessage("reply-" + input.Message.Text()))
				sess.UpdateCustom(func(s testState) testState {
					s.Counter++
					return s
				})
				return nil, nil
			})
		},
		WithSessionStore(store),
	)

	conn, err := af.Connect(context.Background())
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}

	// Drain stream chunks in the background.
	drainInBackground(conn)

	// Send A and wait for it to enter fn (so it's in-flight when detach
	// arrives).
	sendText(t, conn, "A")
	select {
	case <-entered:
	case <-time.After(2 * time.Second):
		t.Fatal("A did not enter fn")
	}

	// Send D, then Detach. The eager intake reader sees D queued and the
	// detach signal immediately, even though the runner is blocked on A.
	sendText(t, conn, "D")
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
		return s.Status == SnapshotStatusCompleted
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
	store := newTestInMemStore[testState]()

	enter := make(chan struct{}, 4)
	release := make(chan struct{}, 4)

	af := DefineCustomAgent(reg, "detachChainParent",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				enter <- struct{}{}
				<-release
				sess.AddMessages(ai.NewModelTextMessage("ok"))
				return nil, nil
			})
		},
		WithSessionStore(store),
	)

	conn, err := af.Connect(context.Background())
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}

	// Background drainer.
	drainInBackground(conn)

	// Run two normal turns.
	for i := 0; i < 2; i++ {
		release <- struct{}{} // pre-load release so this turn's fn doesn't block
		sendText(t, conn, fmt.Sprintf("sync-%d", i))
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
	sendText(t, conn, "inflight")
	<-enter // third turn entered fn

	// Send the queued input and detach.
	sendText(t, conn, "detach-msg")
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
		return s.Status == SnapshotStatusCompleted
	})
}

func TestAgent_Detach_RequiresStore(t *testing.T) {
	reg := newTestRegistry(t)

	af := DefineCustomAgent(reg, "detachNoStore",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				return nil, nil
			})
		},
	)

	conn, err := af.Connect(context.Background())
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	if err := conn.Detach(); err != nil {
		t.Fatalf("Detach send: %v", err)
	}
	conn.Close()

	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	if out.FinishReason != AgentFinishReasonFailed {
		t.Errorf("expected finish reason %q, got %q", AgentFinishReasonFailed, out.FinishReason)
	}
	if out.Error == nil || !strings.Contains(out.Error.Message, "detach requires a session store") {
		t.Errorf("unexpected output error: %+v", out.Error)
	}
	if out.Error != nil && out.Error.Status != core.FAILED_PRECONDITION {
		t.Errorf("expected status %q, got %q", core.FAILED_PRECONDITION, out.Error.Status)
	}
}

func TestAgent_Detach_PendingThenComplete(t *testing.T) {
	// Client detaches mid-flow; flow finishes naturally; pending snapshot
	// flips to status=completed with the full session state.
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()

	release := make(chan struct{})
	entered := make(chan struct{})

	af := DefineCustomAgent(reg, "detachComplete",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
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
				return nil, nil
			})
		},
		WithSessionStore(store),
	)

	conn, err := af.Connect(context.Background())
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}

	// Drain chunks so the responder isn't blocked.
	drainInBackground(conn)

	sendText(t, conn, "go")
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
		return s.Status == SnapshotStatusCompleted
	})
	if finalSnap.State.Custom.Counter != 42 {
		t.Errorf("expected counter=42 in final snapshot, got %d", finalSnap.State.Custom.Counter)
	}
	if got := len(finalSnap.State.Messages); got < 2 {
		t.Errorf("expected at least 2 messages in final snapshot, got %d", got)
	}
}

// getSnapshotViaAction invokes the agent's getSnapshot companion action the
// way a remote client (or the Dev UI) would. Unlike a direct store read it
// resolves compute-on-read status such as expired.
func getSnapshotViaAction[State any](t *testing.T, reg *registry.Registry, agentName, snapshotID string) *SessionSnapshot[State] {
	t.Helper()
	action := core.ResolveActionFor[*GetSnapshotRequest, *SessionSnapshot[State], struct{}](
		reg, api.ActionTypeAgentSnapshot, agentName)
	if action == nil {
		t.Fatalf("getSnapshot action not registered for %q", agentName)
	}
	resp, err := action.Run(context.Background(), &GetSnapshotRequest{SnapshotID: snapshotID}, nil)
	if err != nil {
		t.Fatalf("getSnapshot action: %v", err)
	}
	return resp
}

// savePendingWithHeartbeat writes a pending snapshot carrying the given
// heartbeat (nil for none) directly into the store, returning its ID. It
// stands in for an orphaned detached invocation whose worker died without
// finalizing the row.
func savePendingWithHeartbeat(t *testing.T, store *testInMemStore[testState], sessionID string, heartbeat *time.Time) string {
	t.Helper()
	now := time.Now()
	saved, err := store.SaveSnapshot(context.Background(), "",
		func(_ *SessionSnapshot[testState]) (*SessionSnapshot[testState], error) {
			return &SessionSnapshot[testState]{
				Status:      SnapshotStatusPending,
				HeartbeatAt: heartbeat,
				CreatedAt:   now,
				UpdatedAt:   now,
				State:       &SessionState[testState]{SessionID: sessionID, Custom: testState{Counter: 7}},
			}, nil
		})
	if err != nil {
		t.Fatalf("SaveSnapshot pending: %v", err)
	}
	return saved.SnapshotID
}

// defineNoopHeartbeatAgent registers a trivial server-managed agent so its
// getSnapshot companion action exists; the heartbeat compute-on-read tests
// drive that action against snapshots they write into the store directly.
func defineNoopHeartbeatAgent(t *testing.T, reg *registry.Registry, name string, store *testInMemStore[testState]) {
	t.Helper()
	DefineCustomAgent(reg, name,
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				return nil, nil
			})
		},
		WithSessionStore(store),
	)
}

func TestAgent_Detach_StampsHeartbeatOnPendingSnapshot(t *testing.T) {
	// A detached invocation's pending snapshot carries an initial heartbeat,
	// which the background refresh loop later advances and a reader uses to
	// tell a live worker from a dead one.
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()

	release := make(chan struct{})
	entered := make(chan struct{})

	af := DefineCustomAgent(reg, "heartbeatStamp",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				select {
				case entered <- struct{}{}:
				case <-ctx.Done():
				}
				<-release
				sess.AddMessages(ai.NewModelTextMessage("done"))
				return nil, nil
			})
		},
		WithSessionStore(store),
	)

	conn, err := af.Connect(context.Background())
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	drainInBackground(conn)

	sendText(t, conn, "go")
	if err := conn.Detach(); err != nil {
		t.Fatalf("Detach: %v", err)
	}

	select {
	case <-entered:
	case <-time.After(2 * time.Second):
		t.Fatal("flow did not enter work phase")
	}

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
		t.Errorf("status = %q, want %q", pending.Status, SnapshotStatusPending)
	}
	if pending.HeartbeatAt == nil {
		t.Error("pending detached snapshot should carry a heartbeatAt")
	}

	// Release; the finalizer stops the heartbeat and rewrites the row.
	close(release)
	waitForSnapshot(t, store, out.SnapshotID, 2*time.Second, func(s *SessionSnapshot[testState]) bool {
		return s.Status == SnapshotStatusCompleted
	})
}

func TestAgent_GetSnapshotAction_StaleHeartbeatReportsExpired(t *testing.T) {
	// An orphaned pending snapshot (heartbeat older than the expiry timeout) is
	// surfaced as expired on read, while the raw store row stays pending:
	// compute-on-read never writes the expired status back.
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()
	defineNoopHeartbeatAgent(t, reg, "heartbeatExpired", store)

	stale := time.Now().Add(-2 * defaultHeartbeatTimeout)
	id := savePendingWithHeartbeat(t, store, "sess-expired", &stale)

	raw, err := store.GetSnapshot(context.Background(), id)
	if err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}
	if raw.Status != SnapshotStatusPending {
		t.Errorf("raw store status = %q, want %q (compute-on-read must not write back)", raw.Status, SnapshotStatusPending)
	}

	viaAction := getSnapshotViaAction[testState](t, reg, "heartbeatExpired", id)
	if viaAction.Status != SnapshotStatusExpired {
		t.Errorf("getSnapshot status = %q, want %q", viaAction.Status, SnapshotStatusExpired)
	}
}

func TestAgent_GetSnapshotAction_FreshHeartbeatStaysPending(t *testing.T) {
	// A pending snapshot whose heartbeat is fresh is reported as pending: its
	// worker is presumed alive.
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()
	defineNoopHeartbeatAgent(t, reg, "heartbeatFresh", store)

	fresh := time.Now()
	id := savePendingWithHeartbeat(t, store, "sess-fresh", &fresh)

	viaAction := getSnapshotViaAction[testState](t, reg, "heartbeatFresh", id)
	if viaAction.Status != SnapshotStatusPending {
		t.Errorf("getSnapshot status = %q, want %q", viaAction.Status, SnapshotStatusPending)
	}
}

func TestAgent_GetSnapshotAction_NoHeartbeatStaysPending(t *testing.T) {
	// A pending snapshot that has not written a first heartbeat yet is not
	// expired: the beat may simply not have fired.
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()
	defineNoopHeartbeatAgent(t, reg, "heartbeatNone", store)

	id := savePendingWithHeartbeat(t, store, "sess-noheartbeat", nil)

	viaAction := getSnapshotViaAction[testState](t, reg, "heartbeatNone", id)
	if viaAction.Status != SnapshotStatusPending {
		t.Errorf("getSnapshot status = %q, want %q", viaAction.Status, SnapshotStatusPending)
	}
}

func TestAgent_Heartbeat_BeatAdvancesHeartbeatNotUpdatedAt(t *testing.T) {
	// A beat advances the liveness timestamp but must NOT advance UpdatedAt: a
	// heartbeat is not a state change, which is the whole reason HeartbeatAt is
	// a field of its own rather than reusing UpdatedAt.
	store := newTestInMemStore[testState]()
	old := time.Now().Add(-time.Hour)
	id := savePendingWithHeartbeat(t, store, "sess", &old)
	before, err := store.GetSnapshot(context.Background(), id)
	if err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}

	if err := beatHeartbeat(context.Background(), store, id); err != nil {
		t.Fatalf("beatHeartbeat: %v", err)
	}

	after, err := store.GetSnapshot(context.Background(), id)
	if err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}
	if after.HeartbeatAt == nil || !after.HeartbeatAt.After(*before.HeartbeatAt) {
		t.Errorf("HeartbeatAt did not advance: before=%v after=%v", before.HeartbeatAt, after.HeartbeatAt)
	}
	if !after.UpdatedAt.Equal(before.UpdatedAt) {
		t.Errorf("UpdatedAt changed on heartbeat: before=%v after=%v", before.UpdatedAt, after.UpdatedAt)
	}
	if after.Status != SnapshotStatusPending {
		t.Errorf("status = %q, want %q", after.Status, SnapshotStatusPending)
	}
}

func TestAgent_Heartbeat_BeatIsNoopOnTerminalSnapshot(t *testing.T) {
	// A stray beat against a settled snapshot must never resurrect or mutate
	// it: the mutator's pending-guard returns nil, so no write happens.
	store := newTestInMemStore[testState]()
	saved, err := store.SaveSnapshot(context.Background(), "",
		func(_ *SessionSnapshot[testState]) (*SessionSnapshot[testState], error) {
			return &SessionSnapshot[testState]{
				Status: SnapshotStatusCompleted,
				State:  &SessionState[testState]{SessionID: "sess"},
			}, nil
		})
	if err != nil {
		t.Fatalf("SaveSnapshot: %v", err)
	}
	before, err := store.GetSnapshot(context.Background(), saved.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}

	if err := beatHeartbeat(context.Background(), store, saved.SnapshotID); err != nil {
		t.Fatalf("beatHeartbeat: %v", err)
	}

	after, err := store.GetSnapshot(context.Background(), saved.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}
	if after.Status != SnapshotStatusCompleted {
		t.Errorf("status = %q, want %q (beat must not change it)", after.Status, SnapshotStatusCompleted)
	}
	if after.HeartbeatAt != nil {
		t.Errorf("beat stamped a heartbeat on a terminal snapshot: %v", after.HeartbeatAt)
	}
	if !after.UpdatedAt.Equal(before.UpdatedAt) {
		t.Errorf("beat bumped UpdatedAt on a terminal snapshot: before=%v after=%v", before.UpdatedAt, after.UpdatedAt)
	}
}

func TestAgent_Detach_SendArtifactPostDetachLandsInSnapshot(t *testing.T) {
	// SendArtifact must behave the same way regardless of whether detach
	// has landed: the artifact is added to the session and shows up in
	// the finalized snapshot's state. The wire forward is the only thing
	// detach suppresses, so flow authors don't need to branch on detach.
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()

	detached := make(chan struct{})
	release := make(chan struct{})

	af := DefineCustomAgent(reg, "detachArtifact",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				resp.SendArtifact(&Artifact{
					Name:  "before.txt",
					Parts: []*ai.Part{ai.NewTextPart("pre-detach")},
				})
				select {
				case <-detached:
				case <-ctx.Done():
					return nil, ctx.Err()
				}
				resp.SendArtifact(&Artifact{
					Name:  "after.txt",
					Parts: []*ai.Part{ai.NewTextPart("post-detach")},
				})
				<-release
				return nil, nil
			})
		},
		WithSessionStore(store),
	)

	conn, err := af.Connect(context.Background())
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	drainInBackground(conn)

	sendText(t, conn, "go")
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
		return s.Status == SnapshotStatusCompleted
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
	store := newTestInMemStore[testState]()

	release := make(chan struct{})
	entered := make(chan struct{})
	boom := errors.New("kaboom")

	af := DefineCustomAgent(reg, "detachErr",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				select {
				case entered <- struct{}{}:
				case <-time.After(time.Second):
				}
				<-release
				return nil, boom
			})
		},
		WithSessionStore(store),
	)

	conn, err := af.Connect(context.Background())
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	drainInBackground(conn)

	sendText(t, conn, "go")
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

	// Resuming from an errored detached snapshot is rejected before the
	// invocation starts, so the action fails with the original error.
	resumeOut, err := af.RunText(context.Background(), "retry", WithSnapshotID[testState](out.SnapshotID))
	if err == nil {
		t.Fatalf("expected error resuming errored snapshot, got output: %+v", resumeOut)
	}
	if !strings.Contains(err.Error(), "kaboom") {
		t.Errorf("expected resume error to surface the original failure, got: %v", err)
	}
}

func TestAgent_Detach_AbortStopsFlow(t *testing.T) {
	// Client detaches, then calls abortPendingSnapshot. The store's status
	// subscriber notifies the runtime, which cancels the work context, and
	// the finalizer rewrites the snapshot with status=aborted.
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()

	entered := make(chan struct{})

	af := DefineCustomAgent(reg, "detachAbort",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				select {
				case entered <- struct{}{}:
				case <-time.After(time.Second):
				}
				<-ctx.Done()
				return nil, ctx.Err()
			})
		},
		WithSessionStore(store),
	)

	conn, err := af.Connect(context.Background())
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	drainInBackground(conn)

	sendText(t, conn, "go")
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
	status, err := abortPendingSnapshot(context.Background(), store, out.SnapshotID)
	if err != nil {
		t.Fatalf("abortPendingSnapshot: %v", err)
	}
	if status != SnapshotStatusAborted {
		t.Errorf("abortPendingSnapshot status = %q, want aborted", status)
	}

	// The subscriber wakes the runtime, cancels work, and the finalizer
	// rewrites the snapshot with the aborted status.
	finalSnap := waitForSnapshot(t, store, out.SnapshotID, 2*time.Second, func(s *SessionSnapshot[testState]) bool {
		return s.Status == SnapshotStatusAborted && s.UpdatedAt.After(s.CreatedAt)
	})
	// The flow only blocked on ctx — no state mutation expected. State
	// may be nil (when abortPendingSnapshot landed before the finalizer's write
	// could populate it) or a populated zero-value struct.
	if finalSnap.State != nil && finalSnap.State.Custom.Counter != 0 {
		t.Errorf("unexpected counter value in aborted snapshot: %d", finalSnap.State.Custom.Counter)
	}
}

func TestAgent_Detach_NormalCompletionStillEmitsTurnEnd(t *testing.T) {
	// Sanity: a non-detached invocation against a store-backed flow still
	// behaves like a synchronous flow (turn-end snapshots, no pending row).
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()

	af := DefineCustomAgent(reg, "syncStillWorks",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				sess.AddMessages(ai.NewModelTextMessage("ok"))
				return nil, nil
			})
		},
		WithSessionStore(store),
	)

	conn, err := af.Connect(context.Background())
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	sendText(t, conn, "hi")

	var turnEndID string
	turnEndID = nextTurnEnd(t, conn).SnapshotID
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
	if snap.Status != SnapshotStatusCompleted {
		t.Errorf("turn-end snapshot status = %q, want completed", snap.Status)
	}
}

func TestAgent_Detach_ClientDisconnectBeforeDetachCancels(t *testing.T) {
	// Without detach, a client cancel still cancels the work — this is
	// the regression guard for "until detach=true is called, this is a
	// normal HTTP/WS connection that cancels on close."
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()

	entered := make(chan struct{})
	exited := make(chan error, 1)

	af := DefineCustomAgent(reg, "syncCancel",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			err := sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				select {
				case entered <- struct{}{}:
				case <-ctx.Done():
				}
				<-ctx.Done()
				return nil, ctx.Err()
			})
			exited <- err
			return nil, err
		},
		WithSessionStore(store),
	)

	ctx, cancel := context.WithCancel(context.Background())
	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	drainInBackground(conn)

	sendText(t, conn, "go")
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
	store := newTestInMemStore[testState]()

	erroredID := "errored-456"
	if _, err := store.SaveSnapshot(context.Background(), erroredID,
		func(_ *SessionSnapshot[testState]) (*SessionSnapshot[testState], error) {
			return &SessionSnapshot[testState]{
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
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, nil
		},
		WithSessionStore(store),
	)

	out, err := af.RunText(context.Background(), "hi", WithSnapshotID[testState](erroredID))
	if err == nil {
		t.Fatalf("expected error when resuming from errored snapshot, got output: %+v", out)
	}
	ge := core.AsGenkitError(err)
	if ge.Status != core.FAILED_PRECONDITION {
		t.Errorf("expected status %q, got %q", core.FAILED_PRECONDITION, ge.Status)
	}
	if !strings.Contains(ge.Message, "underlying failure") {
		t.Errorf("expected error to surface underlying failure, got: %v", err)
	}
}

func TestAgent_GetSnapshotAction_ReturnsTransformedState(t *testing.T) {
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()

	// Transform that scrubs a specific word from all messages. It also
	// (incorrectly) drops the framework-owned session ID, which the
	// action must re-stamp on the way out.
	transform := func(_ context.Context, s *SessionState[testState]) (*SessionState[testState], error) {
		for _, msg := range s.Messages {
			for _, p := range msg.Content {
				if p.Text != "" {
					p.Text = strings.ReplaceAll(p.Text, "secret", "[REDACTED]")
				}
			}
		}
		s.SessionID = ""
		return s, nil
	}

	af := DefineCustomAgent(reg, "transformedFlow",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				sess.AddMessages(ai.NewModelTextMessage("the secret is out"))
				return nil, nil
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
	action := core.ResolveActionFor[*GetSnapshotRequest, *SessionSnapshot[testState], struct{}](
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
	if resp.Status != SnapshotStatusCompleted {
		t.Errorf("expected status=completed, got %q", resp.Status)
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
	// The transform dropped the state-carried session ID; the action
	// re-stamps it from the row so outbound state stays self-describing.
	if resp.State.SessionID != resp.SessionID {
		t.Errorf("state-carried session ID = %q, want re-stamped %q", resp.State.SessionID, resp.SessionID)
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

// assertGenkitStatus fails the test unless err is a *core.GenkitError with the
// given status.
func assertGenkitStatus(t *testing.T, err error, want core.StatusName, label string) {
	t.Helper()
	var ge *core.GenkitError
	if !errors.As(err, &ge) {
		t.Errorf("%s: expected *core.GenkitError, got %v", label, err)
		return
	}
	if ge.Status != want {
		t.Errorf("%s: status = %q, want %q", label, ge.Status, want)
	}
}

// TestAgent_GetSnapshot_FacadeTransformsRawStoreDoesNot is the crux of the
// agent-as-facade design: reads through the agent apply the configured
// [WithStateTransform], while a direct store read returns raw state. A caller
// that reaches past the agent into the store therefore sees unredacted data,
// which is exactly why GetSnapshot/GetLatestSnapshot live on the agent.
func TestAgent_GetSnapshot_FacadeTransformsRawStoreDoesNot(t *testing.T) {
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()

	transform := func(_ context.Context, s *SessionState[testState]) (*SessionState[testState], error) {
		for _, msg := range s.Messages {
			for _, p := range msg.Content {
				if p.Text != "" {
					p.Text = strings.ReplaceAll(p.Text, "secret", "[REDACTED]")
				}
			}
		}
		return s, nil
	}

	af := DefineCustomAgent(reg, "facadeTransform",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				sess.AddMessages(ai.NewModelTextMessage("the secret is out"))
				return nil, nil
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

	hasSecret := func(snap *SessionSnapshot[testState]) bool {
		for _, msg := range snap.State.Messages {
			for _, p := range msg.Content {
				if strings.Contains(p.Text, "secret") {
					return true
				}
			}
		}
		return false
	}

	// Through the agent: transformed (redacted).
	viaAgent, err := af.GetSnapshot(ctx, out.SnapshotID)
	if err != nil {
		t.Fatalf("agent.GetSnapshot: %v", err)
	}
	if hasSecret(viaAgent) {
		t.Error("agent.GetSnapshot returned untransformed state (found 'secret')")
	}

	// GetLatestSnapshot resolves the same row by session and also transforms.
	viaLatest, err := af.GetLatestSnapshot(ctx, out.SessionID)
	if err != nil {
		t.Fatalf("agent.GetLatestSnapshot: %v", err)
	}
	if viaLatest.SnapshotID != out.SnapshotID {
		t.Errorf("GetLatestSnapshot id = %q, want %q", viaLatest.SnapshotID, out.SnapshotID)
	}
	if hasSecret(viaLatest) {
		t.Error("agent.GetLatestSnapshot returned untransformed state (found 'secret')")
	}

	// Directly via the store: raw, untransformed. This contrast is the point.
	raw, err := store.GetSnapshot(ctx, out.SnapshotID)
	if err != nil {
		t.Fatalf("store.GetSnapshot: %v", err)
	}
	if !hasSecret(raw) {
		t.Error("raw store.GetSnapshot should retain the original 'secret' text")
	}
}

// TestAgent_SnapshotFacade_Errors covers the guard rails on the agent facade:
// a client-managed agent has no store, and empty IDs / missing rows are
// rejected before touching the store.
func TestAgent_SnapshotFacade_Errors(t *testing.T) {
	reg := newTestRegistry(t)
	ctx := context.Background()

	// Client-managed agent (no store): every facade method is FAILED_PRECONDITION.
	client := defineCounterAgent(reg, "facadeClient")
	_, e1 := client.GetSnapshot(ctx, "x")
	assertGenkitStatus(t, e1, core.FAILED_PRECONDITION, "client GetSnapshot")
	_, e2 := client.GetLatestSnapshot(ctx, "x")
	assertGenkitStatus(t, e2, core.FAILED_PRECONDITION, "client GetLatestSnapshot")
	_, e3 := client.Abort(ctx, "x")
	assertGenkitStatus(t, e3, core.FAILED_PRECONDITION, "client abortPendingSnapshot")

	// Server-managed agent: empty IDs are INVALID_ARGUMENT, missing rows NOT_FOUND.
	store := newTestInMemStore[testState]()
	server := defineCounterAgent(reg, "facadeServer", WithSessionStore(store))
	_, e4 := server.GetSnapshot(ctx, "")
	assertGenkitStatus(t, e4, core.INVALID_ARGUMENT, "empty GetSnapshot")
	_, e5 := server.GetLatestSnapshot(ctx, "")
	assertGenkitStatus(t, e5, core.INVALID_ARGUMENT, "empty GetLatestSnapshot")
	_, e6 := server.Abort(ctx, "")
	assertGenkitStatus(t, e6, core.INVALID_ARGUMENT, "empty abortPendingSnapshot")
	_, e7 := server.GetSnapshot(ctx, "missing")
	assertGenkitStatus(t, e7, core.NOT_FOUND, "missing GetSnapshot")
}

// TestAgent_Abort_Method verifies the in-process convenience flips a
// pending row to aborted through the store, mirroring the package-level
// [abortPendingSnapshot] and the abort companion action.
func TestAgent_Abort_Method(t *testing.T) {
	reg := newTestRegistry(t)
	ctx := context.Background()
	store := newTestInMemStore[testState]()
	af := defineCounterAgent(reg, "facadeAbort", WithSessionStore(store))

	// Seed a pending row the way a detached turn would.
	pending, err := store.SaveSnapshot(ctx, "", func(*SessionSnapshot[testState]) (*SessionSnapshot[testState], error) {
		return &SessionSnapshot[testState]{SessionID: "s1", Status: SnapshotStatusPending}, nil
	})
	if err != nil {
		t.Fatalf("seed pending: %v", err)
	}

	status, err := af.Abort(ctx, pending.SnapshotID)
	if err != nil {
		t.Fatalf("agent.Abort: %v", err)
	}
	if status != SnapshotStatusAborted {
		t.Errorf("returned status = %q, want aborted", status)
	}
	got, err := store.GetSnapshot(ctx, pending.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}
	if got.Status != SnapshotStatusAborted {
		t.Errorf("stored status = %q, want aborted", got.Status)
	}
}

// TestAgent_GetSnapshotAction_ReturnsFinishReason verifies the remote
// getSnapshot companion action surfaces the persisted finish reason, so a
// non-Go client or the Dev UI polling a detached/background invocation can
// report how it ended without re-deriving it from the messages.
func TestAgent_GetSnapshotAction_ReturnsFinishReason(t *testing.T) {
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()

	af := DefineCustomAgent(reg, "finishReasonActionFlow",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				sess.AddMessages(ai.NewModelTextMessage("done"))
				return &TurnResult{FinishReason: AgentFinishReasonStop}, nil
			})
		},
		WithSessionStore(store),
	)

	ctx := context.Background()
	out, err := af.RunText(ctx, "hi")
	if err != nil {
		t.Fatalf("RunText: %v", err)
	}

	action := core.ResolveActionFor[*GetSnapshotRequest, *SessionSnapshot[testState], struct{}](
		reg, api.ActionTypeAgentSnapshot, "finishReasonActionFlow")
	if action == nil {
		t.Fatal("getSnapshot action not registered")
	}
	resp, err := action.Run(ctx, &GetSnapshotRequest{SnapshotID: out.SnapshotID}, nil)
	if err != nil {
		t.Fatalf("getSnapshot action: %v", err)
	}
	if resp.FinishReason != AgentFinishReasonStop {
		t.Errorf("getSnapshot FinishReason = %q, want %q", resp.FinishReason, AgentFinishReasonStop)
	}
}

// TestAgent_GetSnapshotAction_BySessionID verifies the getSnapshot companion
// action's session-ID modes: fetching the session's latest snapshot (whatever
// its status, the way a reconnecting client observes a session), composing
// with a snapshot ID as an integrity assertion, and the argument-validation
// edges.
func TestAgent_GetSnapshotAction_BySessionID(t *testing.T) {
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()
	af := defineLastGoodTestAgent(reg, "getBySessionFlow", WithSessionStore(store))

	ctx := context.Background()
	out1, err := af.RunText(ctx, "first")
	if err != nil {
		t.Fatalf("RunText: %v", err)
	}
	out2, err := af.RunText(ctx, "second", WithSessionID[testState](out1.SessionID))
	if err != nil {
		t.Fatalf("RunText resume: %v", err)
	}

	action := core.ResolveActionFor[*GetSnapshotRequest, *SessionSnapshot[testState], struct{}](
		reg, api.ActionTypeAgentSnapshot, "getBySessionFlow")
	if action == nil {
		t.Fatal("getSnapshot action not registered")
	}

	// sessionId alone resolves the session's latest snapshot (the second turn).
	resp, err := action.Run(ctx, &GetSnapshotRequest{SessionID: out1.SessionID}, nil)
	if err != nil {
		t.Fatalf("getSnapshot by sessionId: %v", err)
	}
	if resp.SnapshotID != out2.SnapshotID {
		t.Errorf("latest snapshot = %q, want most recent %q", resp.SnapshotID, out2.SnapshotID)
	}

	// sessionId + matching snapshotId returns that exact snapshot.
	resp, err = action.Run(ctx, &GetSnapshotRequest{SessionID: out1.SessionID, SnapshotID: out1.SnapshotID}, nil)
	if err != nil {
		t.Fatalf("getSnapshot by snapshotId+sessionId: %v", err)
	}
	if resp.SnapshotID != out1.SnapshotID {
		t.Errorf("snapshot = %q, want %q", resp.SnapshotID, out1.SnapshotID)
	}

	// A snapshotId whose session does not match the asserted sessionId is rejected.
	if _, err := action.Run(ctx, &GetSnapshotRequest{SessionID: "other-session", SnapshotID: out1.SnapshotID}, nil); err == nil {
		t.Fatal("expected snapshot/session mismatch to be rejected")
	} else if ge := core.AsGenkitError(err); ge.Status != core.INVALID_ARGUMENT {
		t.Errorf("mismatch status = %q, want INVALID_ARGUMENT (err: %v)", ge.Status, err)
	}

	// Neither field set is rejected.
	if _, err := action.Run(ctx, &GetSnapshotRequest{}, nil); err == nil {
		t.Fatal("expected empty request to be rejected")
	} else if ge := core.AsGenkitError(err); ge.Status != core.INVALID_ARGUMENT {
		t.Errorf("empty-request status = %q, want INVALID_ARGUMENT (err: %v)", ge.Status, err)
	}

	// An unknown session resolves to no snapshot (NOT_FOUND).
	if _, err := action.Run(ctx, &GetSnapshotRequest{SessionID: "no-such-session"}, nil); err == nil {
		t.Fatal("expected NOT_FOUND for unknown session")
	} else if ge := core.AsGenkitError(err); ge.Status != core.NOT_FOUND {
		t.Errorf("unknown-session status = %q, want NOT_FOUND (err: %v)", ge.Status, err)
	}

	// The session-ID lookup returns the latest row whatever its status, so a
	// reconnecting client can observe a failed/pending tip (unlike resume,
	// which rejects it).
	failedAt := time.Now()
	failed, err := store.SaveSnapshot(ctx, "", func(_ *SessionSnapshot[testState]) (*SessionSnapshot[testState], error) {
		return &SessionSnapshot[testState]{
			SessionID:    out1.SessionID,
			ParentID:     out2.SnapshotID,
			Status:       SnapshotStatusFailed,
			FinishReason: AgentFinishReasonFailed,
			CreatedAt:    failedAt,
			UpdatedAt:    failedAt,
		}, nil
	})
	if err != nil {
		t.Fatalf("SaveSnapshot failed tip: %v", err)
	}
	resp, err = action.Run(ctx, &GetSnapshotRequest{SessionID: out1.SessionID}, nil)
	if err != nil {
		t.Fatalf("getSnapshot by sessionId (failed tip): %v", err)
	}
	if resp.SnapshotID != failed.SnapshotID || resp.Status != SnapshotStatusFailed {
		t.Errorf("latest = %q/%q, want failed tip %q/failed", resp.SnapshotID, resp.Status, failed.SnapshotID)
	}
}

func TestInMemorySessionStore_GetSnapshot_NotFound(t *testing.T) {
	store := newTestInMemStore[testState]()

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
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, nil
		},
	)

	getAction := core.ResolveActionFor[*GetSnapshotRequest, *SessionSnapshot[testState], struct{}](
		reg, api.ActionTypeAgentSnapshot, "noStoreFlow")
	if getAction != nil {
		t.Error("getSnapshot action should NOT be registered without a store")
	}
	abortAction := core.ResolveActionFor[*AgentAbortRequest, *AgentAbortResponse, struct{}](
		reg, api.ActionTypeAgentAbort, "noStoreFlow")
	if abortAction != nil {
		t.Error("abort action should NOT be registered without a store")
	}
}

func TestLoadSession_AgentInitValidation(t *testing.T) {
	// loadSession enforces the AgentInit invariants:
	//   - state is mutually exclusive with sessionId and snapshotId (a
	//     client-managed conversation's identity rides inside the state),
	//   - sessionId and snapshotId require a store (server-managed state),
	//   - state requires the absence of a store (client-managed state),
	//   - sessionId composes with snapshotId as an integrity assertion on
	//     the loaded snapshot.
	ctx := context.Background()
	store := newTestInMemStore[testState]()
	state := &SessionState[testState]{Custom: testState{Counter: 1}}

	// A persisted snapshot belonging to session "sess-1", for the
	// sessionId+snapshotId match and mismatch cases.
	saved, err := store.SaveSnapshot(ctx, "", func(_ *SessionSnapshot[testState]) (*SessionSnapshot[testState], error) {
		return &SessionSnapshot[testState]{
			SessionID: "sess-1",
			State:     state,
		}, nil
	})
	if err != nil {
		t.Fatalf("SaveSnapshot: %v", err)
	}

	errCases := []struct {
		name    string
		init    *AgentInit[testState]
		store   SessionStore[testState]
		wantErr string
	}{
		{
			name:    "both snapshotId and state set",
			init:    &AgentInit[testState]{SnapshotID: saved.SnapshotID, State: state},
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
			name:    "sessionId and state set",
			init:    &AgentInit[testState]{SessionID: "sess-1", State: state},
			store:   nil,
			wantErr: "mutually exclusive",
		},
		{
			name:    "all three set",
			init:    &AgentInit[testState]{SessionID: "sess-1", SnapshotID: saved.SnapshotID, State: state},
			store:   store,
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
		{
			name:    "sessionId with client-managed agent",
			init:    &AgentInit[testState]{SessionID: "sess-1"},
			store:   nil,
			wantErr: "client-managed state",
		},
		{
			name:    "sessionId mismatching the loaded snapshot",
			init:    &AgentInit[testState]{SessionID: "sess-other", SnapshotID: saved.SnapshotID},
			store:   store,
			wantErr: "does not belong to session",
		},
	}

	for _, tc := range errCases {
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

	okCases := []struct {
		name     string
		init     *AgentInit[testState]
		store    SessionStore[testState]
		wantSnap bool
	}{
		{
			name:  "empty init with server store",
			init:  &AgentInit[testState]{},
			store: store,
		},
		{
			name: "empty init with no store",
			init: &AgentInit[testState]{},
		},
		{
			name: "state carrying its session ID with client-managed agent",
			init: &AgentInit[testState]{State: &SessionState[testState]{SessionID: "client-sess", Custom: testState{Counter: 1}}},
		},
		{
			name:     "sessionId matching the loaded snapshot",
			init:     &AgentInit[testState]{SessionID: "sess-1", SnapshotID: saved.SnapshotID},
			store:    store,
			wantSnap: true,
		},
		{
			// A session ID with no persisted snapshots is not an error: the
			// caller is starting a brand-new conversation under an ID of its
			// own choosing. loadSession returns a fresh session and no parent
			// snapshot; the runtime stamps the chosen ID.
			name:     "sessionId with no matching snapshots starts fresh",
			init:     &AgentInit[testState]{SessionID: "sess-unknown"},
			store:    store,
			wantSnap: false,
		},
	}

	for _, tc := range okCases {
		t.Run(tc.name, func(t *testing.T) {
			sess, snap, err := loadSession(ctx, tc.init, tc.store)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if sess == nil {
				t.Fatal("expected session, got nil")
			}
			if tc.wantSnap != (snap != nil) {
				t.Errorf("snapshot presence = %v, want %v (snap: %+v)", snap != nil, tc.wantSnap, snap)
			}
			if tc.init.State != nil && sess.State().Custom.Counter != tc.init.State.Custom.Counter {
				t.Errorf("state not loaded: got %+v", sess.State())
			}
		})
	}

	t.Run("latest-snapshot lookup validates the store's answer", func(t *testing.T) {
		// A non-conforming store that resolves a session to a snapshot from
		// a different session must be caught rather than silently resuming
		// the wrong conversation.
		_, _, err := loadSession(ctx, &AgentInit[testState]{SessionID: "sess-lied-about"},
			wrongSessionStore[testState]{SessionStore: store, snapshotID: saved.SnapshotID})
		if err == nil {
			t.Fatal("expected error, got nil")
		}
		if !strings.Contains(err.Error(), "violates the GetLatestSnapshot contract") {
			t.Errorf("error %q does not mention the contract violation", err.Error())
		}
	})
}

// wrongSessionStore wraps a SessionStore but resolves every
// GetLatestSnapshot call to a fixed snapshot, regardless of the requested
// session: a deliberately non-conforming implementation.
type wrongSessionStore[State any] struct {
	SessionStore[State]
	snapshotID string
}

func (s wrongSessionStore[State]) GetLatestSnapshot(ctx context.Context, sessionID string) (*SessionSnapshot[State], error) {
	return s.SessionStore.GetSnapshot(ctx, s.snapshotID)
}

// minimalStore is a SessionStore that does NOT implement SnapshotSubscriber.
// Used to verify the abort action stays unregistered for stores that
// lack the capability.
type minimalStore[State any] struct{}

func (minimalStore[State]) GetSnapshot(context.Context, string) (*SessionSnapshot[State], error) {
	return nil, nil
}
func (minimalStore[State]) GetLatestSnapshot(context.Context, string) (*SessionSnapshot[State], error) {
	return nil, nil
}
func (minimalStore[State]) SaveSnapshot(
	context.Context, string,
	func(*SessionSnapshot[State]) (*SessionSnapshot[State], error),
) (*SessionSnapshot[State], error) {
	return nil, nil
}

func TestAgent_AgentMetadata(t *testing.T) {
	// Verify the metadata["agent"] payload on the agent's action descriptor
	// correctly reports stateManagement and abortable for each combination
	// of store capabilities.
	noopFn := func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
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
					WithSessionStore(newTestInMemStore[testState]()))
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

			// The registry holds the agent's bidi action under the agent key
			// (see Agent.Register); resolve it as a plain api.Action and read
			// its descriptor.
			act := reg.LookupAction(api.NewKey(api.ActionTypeAgent, "", flowName))
			if act == nil {
				t.Fatal("agent action not registered")
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

func TestAgent_AgentMetadata_StateSchema(t *testing.T) {
	// metadata["agent"].stateSchema advertises a JSON schema for the agent's
	// custom state type, inferred the same way action input/output schemas
	// are: a struct state yields an object schema with its fields; an
	// unstructured `any` state yields none (omitted).
	readMeta := func(t *testing.T, reg api.Registry, flowName string) AgentMetadata {
		t.Helper()
		act := reg.LookupAction(api.NewKey(api.ActionTypeAgent, "", flowName))
		if act == nil {
			t.Fatal("agent action not registered")
		}
		meta, ok := act.Desc().Metadata["agent"].(AgentMetadata)
		if !ok {
			t.Fatalf("metadata[\"agent\"] missing or wrong type: %+v", act.Desc().Metadata["agent"])
		}
		return meta
	}

	t.Run("struct state yields an object schema with its fields", func(t *testing.T) {
		reg := newTestRegistry(t)
		DefineCustomAgent(reg, "structStateFlow",
			func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
				return nil, nil
			})
		meta := readMeta(t, reg, "structStateFlow")
		if meta.StateSchema == nil {
			t.Fatal("expected a state schema for a struct state type")
		}
		if got := meta.StateSchema["type"]; got != "object" {
			t.Errorf("state schema type = %v, want object", got)
		}
		props, ok := meta.StateSchema["properties"].(map[string]any)
		if !ok {
			t.Fatalf("state schema properties = %T, want map", meta.StateSchema["properties"])
		}
		if _, ok := props["counter"]; !ok {
			t.Errorf("state schema missing the 'counter' field: %+v", props)
		}
	})

	t.Run("any state yields no schema", func(t *testing.T) {
		reg := newTestRegistry(t)
		DefineCustomAgent[any](reg, "anyStateFlow",
			func(ctx context.Context, resp Responder, sess *SessionRunner[any]) (*AgentResult, error) {
				return nil, nil
			})
		meta := readMeta(t, reg, "anyStateFlow")
		if meta.StateSchema != nil {
			t.Errorf("expected no state schema for an any state type, got %+v", meta.StateSchema)
		}
	})
}

func TestAgent_AbortAction_GatedOnCapabilities(t *testing.T) {
	// Verify the abort companion action is only registered when the
	// store implements SnapshotSubscriber. The getSnapshot action is
	// registered regardless.
	t.Run("aborter capability → both registered", func(t *testing.T) {
		reg := newTestRegistry(t)
		store := newTestInMemStore[testState]() // implements SnapshotSubscriber
		DefineCustomAgent(reg, "fullCaps",
			func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
				return nil, nil
			},
			WithSessionStore(store),
		)
		getAction := core.ResolveActionFor[*GetSnapshotRequest, *SessionSnapshot[testState], struct{}](
			reg, api.ActionTypeAgentSnapshot, "fullCaps")
		if getAction == nil {
			t.Error("getSnapshot action should be registered")
		}
		abortAction := core.ResolveActionFor[*AgentAbortRequest, *AgentAbortResponse, struct{}](
			reg, api.ActionTypeAgentAbort, "fullCaps")
		if abortAction == nil {
			t.Error("abort action should be registered when store implements SnapshotSubscriber")
		}
	})

	t.Run("no aborter capability → abort not registered", func(t *testing.T) {
		reg := newTestRegistry(t)
		DefineCustomAgent(reg, "minCaps",
			func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
				return nil, nil
			},
			WithSessionStore[testState](minimalStore[testState]{}),
		)
		getAction := core.ResolveActionFor[*GetSnapshotRequest, *SessionSnapshot[testState], struct{}](
			reg, api.ActionTypeAgentSnapshot, "minCaps")
		if getAction == nil {
			t.Error("getSnapshot action should be registered even when store lacks SnapshotSubscriber")
		}
		abortAction := core.ResolveActionFor[*AgentAbortRequest, *AgentAbortResponse, struct{}](
			reg, api.ActionTypeAgentAbort, "minCaps")
		if abortAction != nil {
			t.Error("abort action should NOT be registered when store lacks SnapshotSubscriber")
		}
	})
}

// TestAgent_CompanionActionAccessors verifies the agent ref hands out the
// same companion actions the registry holds, so transports can mount them
// on custom routes without registry lookups, and that the accessors mirror
// the store's capabilities by returning nil for actions that were not
// registered.
func TestAgent_CompanionActionAccessors(t *testing.T) {
	noopFn := func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
		return nil, nil
	}

	t.Run("no store → no companions", func(t *testing.T) {
		reg := newTestRegistry(t)
		af := DefineCustomAgent(reg, "noCompanions", noopFn)
		if got := af.GetSnapshotAction(); got != nil {
			t.Errorf("GetSnapshotAction() = %v, want nil", got)
		}
		if got := af.AbortAction(); got != nil {
			t.Errorf("AbortAction() = %v, want nil", got)
		}
	})

	t.Run("store without aborter → getSnapshot only", func(t *testing.T) {
		reg := newTestRegistry(t)
		af := DefineCustomAgent(reg, "getOnly", noopFn,
			WithSessionStore[testState](minimalStore[testState]{}))
		if af.GetSnapshotAction() == nil {
			t.Error("GetSnapshotAction() = nil, want action")
		}
		if got := af.AbortAction(); got != nil {
			t.Errorf("AbortAction() = %v, want nil", got)
		}
	})

	t.Run("aborter store → both, identical to the registered actions", func(t *testing.T) {
		reg := newTestRegistry(t)
		af := DefineCustomAgent(reg, "bothCompanions", noopFn,
			WithSessionStore(newTestInMemStore[testState]()))
		if got, want := af.GetSnapshotAction(), reg.LookupAction("/agent-snapshot/bothCompanions"); got == nil || got != want {
			t.Errorf("GetSnapshotAction() = %v, want registered action %v", got, want)
		}
		if got, want := af.AbortAction(), reg.LookupAction("/agent-abort/bothCompanions"); got == nil || got != want {
			t.Errorf("AbortAction() = %v, want registered action %v", got, want)
		}
	})
}

// TestAgent_Store verifies the agent hands back the store it was
// configured with (so local Go code need not thread a separate reference),
// and nil when the agent is client-managed.
func TestAgent_Store(t *testing.T) {
	noopFn := func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
		return nil, nil
	}

	t.Run("returns the configured store", func(t *testing.T) {
		reg := newTestRegistry(t)
		store := newTestInMemStore[testState]()
		af := DefineCustomAgent(reg, "withStore", noopFn, WithSessionStore[testState](store))
		if got := af.Store(); got != SessionStore[testState](store) {
			t.Errorf("Store() = %v, want the configured store %v", got, store)
		}
		// The returned store is usable directly, and store-specific
		// capabilities are reachable by type assertion.
		if _, ok := af.Store().(SnapshotSubscriber); !ok {
			t.Error("expected the configured store to satisfy SnapshotSubscriber")
		}
	})

	t.Run("nil for a client-managed agent", func(t *testing.T) {
		reg := newTestRegistry(t)
		af := DefineCustomAgent(reg, "noStore", noopFn)
		if got := af.Store(); got != nil {
			t.Errorf("Store() = %v, want nil for a client-managed agent", got)
		}
	})
}

// TestAgent_Description verifies that WithDescription lands on the agent
// action's descriptor as the standard top-level Description (lifted from
// metadata["description"] by core), so reflective tooling and local
// callers read it the same way they read any other action's description.
func TestAgent_Description(t *testing.T) {
	noopFn := func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
		return nil, nil
	}

	t.Run("set via WithDescription", func(t *testing.T) {
		reg := newTestRegistry(t)
		const want = "A concise test agent."
		af := DefineCustomAgent(reg, "described", noopFn, WithDescription[testState](want))
		if got := af.Desc().Description; got != want {
			t.Errorf("Desc().Description = %q, want %q", got, want)
		}
		// It must also be in the metadata map, the place core lifts it
		// from and where the wire descriptor carries it.
		if got, _ := af.Desc().Metadata["description"].(string); got != want {
			t.Errorf("Desc().Metadata[\"description\"] = %q, want %q", got, want)
		}
	})

	t.Run("empty when unset", func(t *testing.T) {
		reg := newTestRegistry(t)
		af := DefineCustomAgent(reg, "undescribed", noopFn)
		if got := af.Desc().Description; got != "" {
			t.Errorf("Desc().Description = %q, want empty", got)
		}
		if _, ok := af.Desc().Metadata["description"]; ok {
			t.Error("expected no description key in metadata when unset")
		}
	})

	t.Run("rejects a second WithDescription", func(t *testing.T) {
		reg := newTestRegistry(t)
		defer func() {
			if recover() == nil {
				t.Error("expected a panic when WithDescription is set twice")
			}
		}()
		DefineCustomAgent(reg, "twice", noopFn,
			WithDescription[testState]("first"), WithDescription[testState]("second"))
	})
}

// TestAgent_RegisterCarriesCompanions verifies that registering an agent
// ref into another registry brings the companion actions along, so the
// agent travels as a unit (see Agent.Register).
func TestAgent_RegisterCarriesCompanions(t *testing.T) {
	reg := newTestRegistry(t)
	af := DefineCustomAgent(reg, "mover",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, nil
		},
		WithSessionStore(newTestInMemStore[testState]()),
	)

	reg2 := newTestRegistry(t)
	af.Register(reg2)
	for _, key := range []string{"/agent/mover", "/agent-snapshot/mover", "/agent-abort/mover"} {
		if reg2.LookupAction(key) == nil {
			t.Errorf("action %q missing from the registry after Register", key)
		}
	}

	// The agent key resolves to the bidi run action, not a companion-bearing
	// facade: consumers recover the companions by their own keys (as the loop
	// above and the route builders do), not by asserting an interface on the
	// agent action.
	runAction := reg2.LookupAction("/agent/mover")
	if _, ok := runAction.(api.BidiAction); !ok {
		t.Errorf("registered agent action = %T, want an api.BidiAction", runAction)
	}
	if _, ok := runAction.(interface{ GetSnapshotAction() api.Action }); ok {
		t.Error("registered agent action should be the bidi run action, not the Agent facade")
	}
}

// TestNewCustomAgent_UnregisteredUntilRegister verifies the non-registering
// constructor: the agent is fully usable before it touches a registry, and
// registering it later (the genkit.RegisterAction path) surfaces the run
// action and its companions together.
func TestNewCustomAgent_UnregisteredUntilRegister(t *testing.T) {
	ctx := context.Background()
	store := newTestInMemStore[testState]()

	af := NewCustomAgent("standalone",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				sess.AddMessages(ai.NewModelTextMessage("hi"))
				return &TurnResult{FinishReason: AgentFinishReasonStop}, nil
			})
		},
		WithSessionStore(store),
	)

	// Companion refs are wired at construction, before any registry.
	if af.GetSnapshotAction() == nil || af.AbortAction() == nil {
		t.Fatal("companion actions should be built by NewCustomAgent before registration")
	}

	// The agent runs without ever being registered: the runtime never
	// consults a registry for a custom agent.
	out, err := af.RunText(ctx, "hello")
	if err != nil {
		t.Fatalf("RunText on unregistered agent: %v", err)
	}
	if out.FinishReason != AgentFinishReasonStop {
		t.Errorf("finishReason = %q, want %q", out.FinishReason, AgentFinishReasonStop)
	}

	// Registering later brings the whole unit into the registry.
	reg := newTestRegistry(t)
	af.Register(reg)
	for _, key := range []string{"/agent/standalone", "/agent-snapshot/standalone", "/agent-abort/standalone"} {
		if reg.LookupAction(key) == nil {
			t.Errorf("action %q missing after Register", key)
		}
	}
}

func TestAgent_AbortAction_NotFound(t *testing.T) {
	// The store's "not found" sentinel (empty status, nil error) must
	// surface as a core.NOT_FOUND GenkitError on the abort companion
	// action so callers (Dev UI, remote clients) see a proper status.
	reg := newTestRegistry(t)
	DefineCustomAgent(reg, "missingFlow",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, nil
		},
		WithSessionStore(newTestInMemStore[testState]()),
	)

	abortAction := core.ResolveActionFor[*AgentAbortRequest, *AgentAbortResponse, struct{}](
		reg, api.ActionTypeAgentAbort, "missingFlow")
	if abortAction == nil {
		t.Fatal("abort action should be registered")
	}

	_, err := abortAction.Run(context.Background(), &AgentAbortRequest{SnapshotID: "no-such-snap"}, nil)
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
	transform := func(_ context.Context, s *SessionState[testState]) (*SessionState[testState], error) {
		// Zero out the counter to demonstrate the transform is applied.
		s.Custom.Counter = -1
		return s, nil
	}

	af := DefineCustomAgent(reg, "clientXformFlow",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				sess.UpdateCustom(func(s testState) testState {
					s.Counter = 7
					return s
				})
				return nil, nil
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

// TestAgent_StateTransform_ErrorFailsClientManagedOutputClosed verifies a state
// transform that returns an error fails the invocation closed: rather than
// handing back unshaped state, the otherwise-successful client-managed
// invocation resolves as a failed output, with the transform's status preserved
// and no state attached.
func TestAgent_StateTransform_ErrorFailsClientManagedOutputClosed(t *testing.T) {
	reg := newTestRegistry(t)

	transform := func(_ context.Context, s *SessionState[testState]) (*SessionState[testState], error) {
		return nil, core.NewError(core.PERMISSION_DENIED, "cannot shape state")
	}

	af := DefineCustomAgent(reg, "clientXformErr",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				sess.AddMessages(ai.NewModelTextMessage("done"))
				return nil, nil
			})
		},
		WithStateTransform[testState](transform),
	)

	out, err := af.RunText(context.Background(), "go")
	if err != nil {
		t.Fatalf("expected a graceful failed output, got error: %v", err)
	}
	if out.FinishReason != AgentFinishReasonFailed {
		t.Errorf("FinishReason = %q, want %q", out.FinishReason, AgentFinishReasonFailed)
	}
	if out.Error == nil || out.Error.Status != core.PERMISSION_DENIED {
		t.Errorf("Error = %+v, want status %q from the transform", out.Error, core.PERMISSION_DENIED)
	}
	// failedOutput shapes the last-good state through the same transform, which
	// errors again here; the runtime omits state rather than leaking it.
	if out.State != nil {
		t.Errorf("expected no state on the failed output (transform failed closed), got %+v", out.State)
	}
}

// TestAgent_StateTransform_ErrorFailsSnapshotReadClosed verifies a state
// transform that errors fails a snapshot read closed: both the typed
// Agent.GetSnapshot facade and the getSnapshot companion action surface the
// transform's error (status preserved) instead of returning unshaped state.
func TestAgent_StateTransform_ErrorFailsSnapshotReadClosed(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()

	transform := func(_ context.Context, s *SessionState[testState]) (*SessionState[testState], error) {
		return nil, core.NewError(core.PERMISSION_DENIED, "cannot shape state")
	}

	af := DefineCustomAgent(reg, "snapXformErr",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				sess.AddMessages(ai.NewModelTextMessage("done"))
				return nil, nil
			})
		},
		WithSessionStore(store),
		WithStateTransform[testState](transform),
	)

	// A successful run persists a snapshot (the run itself does not read state
	// back through the transform, so it is unaffected).
	out, err := af.RunText(ctx, "go")
	if err != nil {
		t.Fatalf("RunText: %v", err)
	}
	if out.SnapshotID == "" {
		t.Fatal("expected a persisted snapshot ID")
	}

	// The typed facade fails closed with the transform's status.
	if _, err := af.GetSnapshot(ctx, out.SnapshotID); err == nil {
		t.Error("Agent.GetSnapshot: expected the transform error, got nil")
	} else if core.AsGenkitError(err).Status != core.PERMISSION_DENIED {
		t.Errorf("Agent.GetSnapshot error status = %q, want %q", core.AsGenkitError(err).Status, core.PERMISSION_DENIED)
	}

	// The getSnapshot companion action fails the same way for non-Go clients.
	action := core.ResolveActionFor[*GetSnapshotRequest, *SessionSnapshot[testState], struct{}](
		reg, api.ActionTypeAgentSnapshot, "snapXformErr")
	if action == nil {
		t.Fatal("getSnapshot action not registered")
	}
	if _, err := action.Run(ctx, &GetSnapshotRequest{SnapshotID: out.SnapshotID}, nil); err == nil {
		t.Error("getSnapshot action: expected the transform error, got nil")
	} else if core.AsGenkitError(err).Status != core.PERMISSION_DENIED {
		t.Errorf("getSnapshot action error status = %q, want %q", core.AsGenkitError(err).Status, core.PERMISSION_DENIED)
	}

	// The stored snapshot itself is untouched: the failure is read-time shaping,
	// not corruption, so the row is still resumable.
	if snap, err := store.GetSnapshot(ctx, out.SnapshotID); err != nil || snap == nil {
		t.Errorf("stored GetSnapshot = (%v, %v), want the intact snapshot", snap, err)
	}
}

func TestAgent_ResumeFromFinalizedDetachedSnapshot(t *testing.T) {
	// End-to-end: run a flow that the client detaches from, let it
	// finalize, then resume from its snapshot as if reconnecting later.
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()

	af := defineCounterAgent(reg, "resumeDetachedFlow", WithSessionStore(store))

	ctx := context.Background()

	// First invocation: detach to write a pending snapshot, then wait
	// for finalize.
	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	drainInBackground(conn)
	sendText(t, conn, "turn 1")
	if err := conn.Detach(); err != nil {
		t.Fatalf("Detach: %v", err)
	}
	first, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	finalSnap := waitForSnapshot(t, store, first.SnapshotID, 2*time.Second, func(s *SessionSnapshot[testState]) bool {
		return s.Status == SnapshotStatusCompleted
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

func TestAbortPendingSnapshot_AtomicAndIdempotent(t *testing.T) {
	// Aborting is an ordinary SaveSnapshot flip (abortPendingSnapshot); this
	// exercises its CAS semantics against the store's atomic read-mutate-write.
	ctx := context.Background()
	store := newTestInMemStore[testState]()

	// Abort on missing snapshot returns empty status, no error.
	if status, err := abortPendingSnapshot(ctx, store, "nope"); err != nil || status != "" {
		t.Fatalf("abort(missing) = %q, %v; want \"\", nil", status, err)
	}

	// Pending → aborted, UpdatedAt advances (verified via GetSnapshot).
	seedNow := time.Now()
	pending, err := store.SaveSnapshot(ctx, "snap-cas",
		func(_ *SessionSnapshot[testState]) (*SessionSnapshot[testState], error) {
			return &SessionSnapshot[testState]{
				Status:    SnapshotStatusPending,
				CreatedAt: seedNow,
				UpdatedAt: seedNow,
			}, nil
		})
	if err != nil {
		t.Fatalf("SaveSnapshot: %v", err)
	}
	time.Sleep(time.Millisecond) // ensure measurable UpdatedAt delta
	status, err := abortPendingSnapshot(ctx, store, "snap-cas")
	if err != nil {
		t.Fatalf("abort: %v", err)
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
	status2, err := abortPendingSnapshot(ctx, store, "snap-cas")
	if err != nil {
		t.Fatalf("abort (second): %v", err)
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
				Status: SnapshotStatusCompleted,
			}, nil
		}); err != nil {
		t.Fatalf("SaveSnapshot: %v", err)
	}
	status3, err := abortPendingSnapshot(ctx, store, "snap-complete")
	if err != nil {
		t.Fatalf("abort on complete: %v", err)
	}
	if status3 != SnapshotStatusCompleted {
		t.Errorf("abort on complete returned status=%q, want completed", status3)
	}
}

func TestAgent_Detach_FinalizeRespectsConcurrentAbort(t *testing.T) {
	// An abort that lands while fn is still running but does not actually
	// stop fn (because fn does not observe ctx) must still result in
	// status=aborted — the finalizer must not clobber aborted with
	// complete. The subscriber observes the status flip and the finalizer
	// reads the resulting flag.
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()

	fnRelease := make(chan struct{})
	entered := make(chan struct{})

	af := DefineCustomAgent(reg, "raceFinalize",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				select {
				case entered <- struct{}{}:
				case <-time.After(time.Second):
				}
				<-fnRelease
				// Return cleanly without observing ctx. Without the
				// subscriber/recheck, this would land status=completed and
				// clobber the abort.
				return nil, nil
			})
		},
		WithSessionStore(store),
	)

	conn, err := af.Connect(context.Background())
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	drainInBackground(conn)

	sendText(t, conn, "go")
	if err := conn.Detach(); err != nil {
		t.Fatalf("Detach: %v", err)
	}
	<-entered

	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}

	// Externally abort before releasing fn.
	if _, err := abortPendingSnapshot(context.Background(), store, out.SnapshotID); err != nil {
		t.Fatalf("abortPendingSnapshot: %v", err)
	}

	close(fnRelease)

	finalSnap := waitForSnapshot(t, store, out.SnapshotID, 2*time.Second, func(s *SessionSnapshot[testState]) bool {
		return s.Status == SnapshotStatusAborted || s.Status == SnapshotStatusCompleted
	})
	if finalSnap.Status != SnapshotStatusAborted {
		t.Errorf("finalize clobbered aborted with %q", finalSnap.Status)
	}
}

func TestInMemorySessionStore_OnSnapshotStatusChange(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	store := newTestInMemStore[testState]()

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
	if _, err := abortPendingSnapshot(ctx, store, "snap-sub"); err != nil {
		t.Fatalf("abortPendingSnapshot: %v", err)
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

func TestAgent_Abort_NoOpOnTerminal(t *testing.T) {
	// Calling abortPendingSnapshot on an already-terminal snapshot is a no-op
	// that returns the existing status.
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()

	af := DefineCustomAgent(reg, "abortNoop",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				sess.AddMessages(ai.NewModelTextMessage("reply"))
				return nil, nil
			})
		},
		WithSessionStore(store),
	)

	ctx := context.Background()
	out, err := af.RunText(ctx, "hi")
	if err != nil {
		t.Fatalf("RunText: %v", err)
	}

	status, err := abortPendingSnapshot(ctx, store, out.SnapshotID)
	if err != nil {
		t.Fatalf("abortPendingSnapshot: %v", err)
	}
	if status != SnapshotStatusCompleted {
		t.Errorf("expected status=%q (existing terminal), got %q", SnapshotStatusCompleted, status)
	}

	// Confirm the store snapshot was not flipped.
	snap, err := store.GetSnapshot(ctx, out.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}
	if snap.Status != SnapshotStatusCompleted {
		t.Errorf("snapshot status = %q after abort-on-terminal, want completed", snap.Status)
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
	store := newTestInMemStore[testState]()

	var (
		sessionMsgAfterMutation string
		sessionArtAfterMutation string
		fnReturnedMessage       *ai.Message
		fnReturnedArtifact      *Artifact
	)

	af := DefineCustomAgent(reg, "isolation",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			if err := sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				sess.AddMessages(ai.NewModelTextMessage("session-msg"))
				sess.AddArtifacts(&Artifact{
					Name:  "orig",
					Parts: []*ai.Part{ai.NewTextPart("orig-part")},
				})
				return nil, nil
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

func TestAgent_Name(t *testing.T) {
	reg := newTestRegistry(t)
	a := DefineCustomAgent(reg, "name-accessor",
		func(ctx context.Context, _ Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return sess.Result(), nil
		})
	if got := a.Name(); got != "name-accessor" {
		t.Errorf("Name() = %q, want %q", got, "name-accessor")
	}
}

// --- Finish reasons ---------------------------------------------------------

// TestAgent_FinishReason_TurnAndInvocation verifies that the reason a custom
// agent reports per turn (via TurnResult) rides the TurnEnd chunk, is
// persisted on the turn-end snapshot, and defaults the invocation's reason on
// AgentOutput.
func TestAgent_FinishReason_TurnAndInvocation(t *testing.T) {
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()

	af := DefineCustomAgent(reg, "finishReasonFlow",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				sess.AddMessages(ai.NewModelTextMessage("ok"))
				return &TurnResult{FinishReason: AgentFinishReasonStop}, nil
			})
		},
		WithSessionStore(store),
	)

	conn, err := af.Connect(context.Background())
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	sendText(t, conn, "hi")

	turnEnd := nextTurnEnd(t, conn)
	if turnEnd.FinishReason != AgentFinishReasonStop {
		t.Errorf("TurnEnd.FinishReason = %q, want %q", turnEnd.FinishReason, AgentFinishReasonStop)
	}

	// The turn-end snapshot records the reason.
	snap, err := store.GetSnapshot(context.Background(), turnEnd.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}
	if snap == nil {
		t.Fatalf("turn-end snapshot %q missing", turnEnd.SnapshotID)
	}
	if snap.FinishReason != AgentFinishReasonStop {
		t.Errorf("snapshot.FinishReason = %q, want %q", snap.FinishReason, AgentFinishReasonStop)
	}

	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	if out.FinishReason != AgentFinishReasonStop {
		t.Errorf("AgentOutput.FinishReason = %q, want %q (defaulted from last turn)", out.FinishReason, AgentFinishReasonStop)
	}
}

// TestAgent_FinishReason_OmittedWhenNil verifies that returning a nil
// TurnResult performs no implicit inference: the reason is omitted on both the
// turn-end signal and the invocation output.
func TestAgent_FinishReason_OmittedWhenNil(t *testing.T) {
	reg := newTestRegistry(t)

	af := DefineCustomAgent(reg, "noReasonFlow",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				sess.AddMessages(ai.NewModelTextMessage("ok"))
				return nil, nil
			})
		},
	)

	conn, err := af.Connect(context.Background())
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	turnEnd := sendTurn(t, conn, "hi")
	if turnEnd.FinishReason != "" {
		t.Errorf("TurnEnd.FinishReason = %q, want empty", turnEnd.FinishReason)
	}

	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	if out.FinishReason != "" {
		t.Errorf("AgentOutput.FinishReason = %q, want empty", out.FinishReason)
	}
}

// TestAgent_FinishReason_InvocationOverride verifies that a custom agent can
// override the invocation's finish reason via AgentResult.FinishReason without
// affecting the per-turn reason on TurnEnd.
func TestAgent_FinishReason_InvocationOverride(t *testing.T) {
	reg := newTestRegistry(t)

	af := DefineCustomAgent(reg, "overrideReasonFlow",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			if err := sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				sess.AddMessages(ai.NewModelTextMessage("ok"))
				return &TurnResult{FinishReason: AgentFinishReasonStop}, nil
			}); err != nil {
				return nil, err
			}
			res := sess.Result()
			res.FinishReason = AgentFinishReasonOther
			return res, nil
		},
	)

	conn, err := af.Connect(context.Background())
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	turnEnd := sendTurn(t, conn, "hi")
	if turnEnd.FinishReason != AgentFinishReasonStop {
		t.Errorf("TurnEnd.FinishReason = %q, want %q (per-turn, unaffected by override)", turnEnd.FinishReason, AgentFinishReasonStop)
	}

	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	if out.FinishReason != AgentFinishReasonOther {
		t.Errorf("AgentOutput.FinishReason = %q, want %q (override)", out.FinishReason, AgentFinishReasonOther)
	}
}

// TestAgent_FinishReason_MultiTurnDistinct verifies that each turn's TurnEnd
// carries that turn's own reason, and the invocation defaults to the last
// turn's reason.
func TestAgent_FinishReason_MultiTurnDistinct(t *testing.T) {
	reg := newTestRegistry(t)

	// Turn 0 reports "stop"; turn 1 reports "interrupted".
	reasons := []AgentFinishReason{AgentFinishReasonStop, AgentFinishReasonInterrupted}

	turn := 0
	af := DefineCustomAgent(reg, "multiReasonFlow",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				sess.AddMessages(ai.NewModelTextMessage("ok"))
				r := reasons[turn]
				turn++
				return &TurnResult{FinishReason: r}, nil
			})
		},
	)

	conn, err := af.Connect(context.Background())
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}

	var got []AgentFinishReason
	for i := 0; i < len(reasons); i++ {
		sendText(t, conn, "turn")
		got = append(got, nextTurnEnd(t, conn).FinishReason)
	}
	for i, want := range reasons {
		if got[i] != want {
			t.Errorf("turn %d TurnEnd.FinishReason = %q, want %q", i, got[i], want)
		}
	}

	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	if out.FinishReason != AgentFinishReasonInterrupted {
		t.Errorf("AgentOutput.FinishReason = %q, want %q (last turn)", out.FinishReason, AgentFinishReasonInterrupted)
	}
}

// TestPromptAgent_ForwardsFinishReason verifies that a prompt-backed agent
// forwards the underlying generate response's finish reason automatically.
func TestPromptAgent_ForwardsFinishReason(t *testing.T) {
	ctx := context.Background()
	reg := registry.New()
	ai.ConfigureFormats(reg)
	ai.DefineModel(reg, "test/length", &ai.ModelOptions{Supports: &ai.ModelSupports{Multiturn: true, SystemRole: true}},
		func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			return &ai.ModelResponse{
				Request:      req,
				Message:      ai.NewModelTextMessage("partial"),
				FinishReason: ai.FinishReasonLength,
			}, nil
		},
	)
	ai.DefineGenerateAction(ctx, reg)
	ai.DefinePrompt(reg, "lengthPrompt", ai.WithModelName("test/length"))

	af := DefinePromptAgent[testState](reg, "lengthPrompt")

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	turnEnd := sendTurn(t, conn, "hi")
	if turnEnd.FinishReason != AgentFinishReasonLength {
		t.Errorf("TurnEnd.FinishReason = %q, want %q", turnEnd.FinishReason, AgentFinishReasonLength)
	}

	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	if out.FinishReason != AgentFinishReasonLength {
		t.Errorf("AgentOutput.FinishReason = %q, want %q", out.FinishReason, AgentFinishReasonLength)
	}
}

// TestAgent_Detach_BackgroundWorkSurvivesActionReturn verifies that the
// detached background work's context stays alive after the invocation
// returns the detached output and the framework releases the action
// context. The regression this guards: the pre-detach watcher that
// mirrors the client context could observe both its wake conditions
// (client context released, detach landed) at once and randomly pick the
// cancel arm, killing the background work. The race is scheduler-driven,
// so the test stacks iterations with a settle window; under
// single-threaded scheduling (GOMAXPROCS=1) the regression trips within
// a few iterations.
func TestAgent_Detach_BackgroundWorkSurvivesActionReturn(t *testing.T) {
	ctx := context.Background()
	for i := 0; i < 20; i++ {
		reg := newTestRegistry(t)
		store := newTestInMemStore[testState]()
		release := make(chan struct{})
		fnSaw := make(chan string, 1)

		af := DefineCustomAgent(reg, "detachSurviveFlow",
			func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
				return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
					select {
					case <-release:
						fnSaw <- "release"
					case <-ctx.Done():
						fnSaw <- "ctx"
					}
					return nil, nil
				})
			},
			WithSessionStore(store),
		)

		conn, err := af.Connect(ctx)
		if err != nil {
			t.Fatalf("iteration %d: Connect: %v", i, err)
		}
		drainInBackground(conn)
		if err := conn.SendText("go"); err != nil {
			t.Fatalf("iteration %d: SendText: %v", i, err)
		}
		if err := conn.Detach(); err != nil {
			t.Fatalf("iteration %d: Detach: %v", i, err)
		}
		out, err := conn.Output()
		if err != nil {
			t.Fatalf("iteration %d: Output: %v", i, err)
		}

		// The action has returned, so the framework's release of the
		// action context is imminent or done. Give a wrongly-cancelling
		// watcher time to land before letting the turn proceed.
		time.Sleep(20 * time.Millisecond)
		close(release)
		if saw := <-fnSaw; saw != "release" {
			t.Fatalf("iteration %d: detached background work saw its context cancelled", i)
		}
		// Wait out the finalizer so the iteration's goroutines wind down.
		waitForSnapshot(t, store, out.SnapshotID, 2*time.Second, func(s *SessionSnapshot[testState]) bool {
			return s.Status == SnapshotStatusCompleted
		})
	}
}

// TestAgent_Detach_FinishReasons covers the three detach outcomes: the output
// returned to the detaching client always reports "detached", while the
// persisted snapshot records how the background work actually ended
// (completed -> last turn's reason, failed, or aborted).
func TestAgent_Detach_FinishReasons(t *testing.T) {
	t.Run("complete", func(t *testing.T) {
		reg := newTestRegistry(t)
		store := newTestInMemStore[testState]()
		release := make(chan struct{})
		entered := make(chan struct{})

		af := DefineCustomAgent(reg, "detachReasonComplete",
			func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
				return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
					select {
					case entered <- struct{}{}:
					case <-ctx.Done():
					}
					<-release
					sess.AddMessages(ai.NewModelTextMessage("done"))
					return &TurnResult{FinishReason: AgentFinishReasonStop}, nil
				})
			},
			WithSessionStore(store),
		)

		conn, err := af.Connect(context.Background())
		if err != nil {
			t.Fatalf("Connect: %v", err)
		}
		drainInBackground(conn)
		sendText(t, conn, "go")
		if err := conn.Detach(); err != nil {
			t.Fatalf("Detach: %v", err)
		}
		<-entered

		out, err := conn.Output()
		if err != nil {
			t.Fatalf("Output: %v", err)
		}
		if out.FinishReason != AgentFinishReasonDetached {
			t.Errorf("AgentOutput.FinishReason = %q, want %q", out.FinishReason, AgentFinishReasonDetached)
		}

		close(release)
		snap := waitForSnapshot(t, store, out.SnapshotID, 2*time.Second, func(s *SessionSnapshot[testState]) bool {
			return s.Status == SnapshotStatusCompleted
		})
		if snap.FinishReason != AgentFinishReasonStop {
			t.Errorf("finalized snapshot.FinishReason = %q, want %q", snap.FinishReason, AgentFinishReasonStop)
		}
	})

	t.Run("failed", func(t *testing.T) {
		reg := newTestRegistry(t)
		store := newTestInMemStore[testState]()
		release := make(chan struct{})
		entered := make(chan struct{})

		af := DefineCustomAgent(reg, "detachReasonFailed",
			func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
				return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
					select {
					case entered <- struct{}{}:
					case <-time.After(time.Second):
					}
					<-release
					return nil, errors.New("kaboom")
				})
			},
			WithSessionStore(store),
		)

		conn, err := af.Connect(context.Background())
		if err != nil {
			t.Fatalf("Connect: %v", err)
		}
		drainInBackground(conn)
		sendText(t, conn, "go")
		if err := conn.Detach(); err != nil {
			t.Fatalf("Detach: %v", err)
		}
		<-entered

		out, err := conn.Output()
		if err != nil {
			t.Fatalf("Output: %v", err)
		}
		if out.FinishReason != AgentFinishReasonDetached {
			t.Errorf("AgentOutput.FinishReason = %q, want %q", out.FinishReason, AgentFinishReasonDetached)
		}

		close(release)
		snap := waitForSnapshot(t, store, out.SnapshotID, 2*time.Second, func(s *SessionSnapshot[testState]) bool {
			return s.Status == SnapshotStatusFailed
		})
		if snap.FinishReason != AgentFinishReasonFailed {
			t.Errorf("finalized snapshot.FinishReason = %q, want %q", snap.FinishReason, AgentFinishReasonFailed)
		}
	})

	t.Run("aborted", func(t *testing.T) {
		reg := newTestRegistry(t)
		store := newTestInMemStore[testState]()
		entered := make(chan struct{})

		af := DefineCustomAgent(reg, "detachReasonAborted",
			func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
				return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
					select {
					case entered <- struct{}{}:
					case <-time.After(time.Second):
					}
					<-ctx.Done()
					return nil, ctx.Err()
				})
			},
			WithSessionStore(store),
		)

		conn, err := af.Connect(context.Background())
		if err != nil {
			t.Fatalf("Connect: %v", err)
		}
		drainInBackground(conn)
		sendText(t, conn, "go")
		if err := conn.Detach(); err != nil {
			t.Fatalf("Detach: %v", err)
		}
		<-entered

		out, err := conn.Output()
		if err != nil {
			t.Fatalf("Output: %v", err)
		}
		if _, err := abortPendingSnapshot(context.Background(), store, out.SnapshotID); err != nil {
			t.Fatalf("abortPendingSnapshot: %v", err)
		}
		// abortPendingSnapshot flips status=aborted (finishReason still empty); the
		// finalizer then annotates the row with finishReason=aborted. Wait
		// for that second write rather than the bare status flip.
		snap := waitForSnapshot(t, store, out.SnapshotID, 2*time.Second, func(s *SessionSnapshot[testState]) bool {
			return s.Status == SnapshotStatusAborted && s.FinishReason == AgentFinishReasonAborted
		})
		if snap.Status != SnapshotStatusAborted {
			t.Errorf("finalized snapshot.Status = %q, want %q", snap.Status, SnapshotStatusAborted)
		}
	})
}

// TestAgent_FinishReason_InvocationOverride_OutputOnly verifies that when a
// custom agent overrides the invocation reason (differing from the last
// turn's), the override rides on AgentOutput.FinishReason but is not
// persisted: with no invocation-end snapshot, the output points at the last
// turn-end row, which keeps that turn's own reason.
func TestAgent_FinishReason_InvocationOverride_OutputOnly(t *testing.T) {
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()

	af := DefineCustomAgent(reg, "overrideOutputFlow",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			if err := sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				sess.AddMessages(ai.NewModelTextMessage("ok"))
				return &TurnResult{FinishReason: AgentFinishReasonStop}, nil
			}); err != nil {
				return nil, err
			}
			res := sess.Result()
			res.FinishReason = AgentFinishReasonOther
			return res, nil
		},
		WithSessionStore(store),
	)

	ctx := context.Background()
	out, err := af.RunText(ctx, "hi")
	if err != nil {
		t.Fatalf("RunText: %v", err)
	}
	// The override is reported on the output...
	if out.FinishReason != AgentFinishReasonOther {
		t.Fatalf("AgentOutput.FinishReason = %q, want %q", out.FinishReason, AgentFinishReasonOther)
	}
	if out.SnapshotID == "" {
		t.Fatal("expected a snapshot ID")
	}
	// ...but only the turn-end snapshot exists, and no extra row was written
	// for the override.
	if rows := store.snapshotCount(); rows != 1 {
		t.Errorf("expected exactly 1 snapshot (turn-end only), got %d", rows)
	}
	snap, err := store.GetSnapshot(ctx, out.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}
	if snap.FinishReason != AgentFinishReasonStop {
		t.Errorf("snapshot.FinishReason = %q, want %q (the turn's own reason, not the invocation override)", snap.FinishReason, AgentFinishReasonStop)
	}
}

// TestAgent_FinishReason_MultiTurnDistinct_Persisted verifies each turn-end
// snapshot persists that turn's own reason and chains to its parent.
func TestAgent_FinishReason_MultiTurnDistinct_Persisted(t *testing.T) {
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()
	reasons := []AgentFinishReason{AgentFinishReasonStop, AgentFinishReasonInterrupted}

	turn := 0
	af := DefineCustomAgent(reg, "multiReasonPersistedFlow",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				sess.AddMessages(ai.NewModelTextMessage("ok"))
				r := reasons[turn]
				turn++
				return &TurnResult{FinishReason: r}, nil
			})
		},
		WithSessionStore(store),
	)

	conn, err := af.Connect(context.Background())
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	var ids []string
	for i := 0; i < len(reasons); i++ {
		te := sendTurn(t, conn, "turn")
		if te.FinishReason != reasons[i] {
			t.Errorf("turn %d TurnEnd.FinishReason = %q, want %q", i, te.FinishReason, reasons[i])
		}
		ids = append(ids, te.SnapshotID)
	}
	if _, err := conn.Output(); err != nil {
		t.Fatalf("Output: %v", err)
	}

	for i, id := range ids {
		snap, err := store.GetSnapshot(context.Background(), id)
		if err != nil {
			t.Fatalf("GetSnapshot[%d]: %v", i, err)
		}
		if snap.FinishReason != reasons[i] {
			t.Errorf("snapshot[%d].FinishReason = %q, want %q", i, snap.FinishReason, reasons[i])
		}
	}
	// The second turn's snapshot chains to the first.
	snap1, _ := store.GetSnapshot(context.Background(), ids[1])
	if snap1.ParentID != ids[0] {
		t.Errorf("snapshot[1].ParentID = %q, want %q", snap1.ParentID, ids[0])
	}
}

// TestAgent_FinishReason_OmittedPersisted verifies a turn that reports no
// reason persists an empty finishReason (no implicit inference at rest).
func TestAgent_FinishReason_OmittedPersisted(t *testing.T) {
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()

	af := DefineCustomAgent(reg, "noReasonPersistedFlow",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				sess.AddMessages(ai.NewModelTextMessage("ok"))
				return nil, nil
			})
		},
		WithSessionStore(store),
	)

	conn, err := af.Connect(context.Background())
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	sendText(t, conn, "hi")
	snapID := nextTurnEnd(t, conn).SnapshotID
	if _, err := conn.Output(); err != nil {
		t.Fatalf("Output: %v", err)
	}
	snap, err := store.GetSnapshot(context.Background(), snapID)
	if err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}
	if snap.FinishReason != "" {
		t.Errorf("snapshot.FinishReason = %q, want empty", snap.FinishReason)
	}
}

// TestPromptAgent_ForwardsInterruptedFinishReason drives a real interrupted
// generate response through the prompt-backed loop: the turn must stream the
// interrupt parts AND report finishReason=interrupted (the proposal's
// motivating case), without the client scanning message content.
func TestPromptAgent_ForwardsInterruptedFinishReason(t *testing.T) {
	ctx := context.Background()
	reg := registry.New()
	ai.ConfigureFormats(reg)

	interruptTool := ai.DefineTool(reg, "interruptor", "always interrupts",
		func(tc *ai.ToolContext, input any) (any, error) {
			return nil, tc.Interrupt(&ai.InterruptOptions{
				Metadata: map[string]any{"reason": "needs approval"},
			})
		},
	)
	ai.DefineModel(reg, "test/interrupt", &ai.ModelOptions{Supports: &ai.ModelSupports{Multiturn: true, Tools: true}},
		func(ctx context.Context, req *ai.ModelRequest, cb ai.ModelStreamCallback) (*ai.ModelResponse, error) {
			return &ai.ModelResponse{
				Request: req,
				Message: &ai.Message{
					Role:    ai.RoleModel,
					Content: []*ai.Part{ai.NewToolRequestPart(&ai.ToolRequest{Name: "interruptor"})},
				},
			}, nil
		})
	ai.DefineGenerateAction(ctx, reg)
	ai.DefinePrompt(reg, "interruptPrompt",
		ai.WithModelName("test/interrupt"),
		ai.WithTools(interruptTool),
	)

	af := DefinePromptAgent[testState](reg, "interruptPrompt")

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	sendText(t, conn, "do it")
	var (
		turnEnd      *TurnEnd
		gotToolChunk bool
	)
	for chunk, err := range conn.Receive() {
		if err != nil {
			t.Fatalf("Receive: %v", err)
		}
		if chunk.ModelChunk != nil && chunk.ModelChunk.Role == ai.RoleTool {
			gotToolChunk = true
		}
		if chunk.TurnEnd != nil {
			te := *chunk.TurnEnd
			turnEnd = &te
			break
		}
	}
	if !gotToolChunk {
		t.Error("expected a tool-role chunk carrying the interrupt parts")
	}
	if turnEnd == nil || turnEnd.FinishReason != AgentFinishReasonInterrupted {
		t.Fatalf("TurnEnd.FinishReason = %v, want %q", turnEnd, AgentFinishReasonInterrupted)
	}

	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	if out.FinishReason != AgentFinishReasonInterrupted {
		t.Errorf("AgentOutput.FinishReason = %q, want %q", out.FinishReason, AgentFinishReasonInterrupted)
	}
}

// TestAgent_Detach_CompletedHonorsResultOverride verifies the detach finalizer
// applies an AgentResult.FinishReason override on clean success, matching the
// synchronous path (the override does not leak into the failed/aborted cases,
// which are covered by TestAgent_Detach_FinishReasons).
func TestAgent_Detach_CompletedHonorsResultOverride(t *testing.T) {
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()
	release := make(chan struct{})
	entered := make(chan struct{})

	af := DefineCustomAgent(reg, "detachOverride",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			if err := sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				select {
				case entered <- struct{}{}:
				case <-ctx.Done():
				}
				<-release
				sess.AddMessages(ai.NewModelTextMessage("done"))
				return &TurnResult{FinishReason: AgentFinishReasonStop}, nil
			}); err != nil {
				return nil, err
			}
			res := sess.Result()
			res.FinishReason = AgentFinishReasonOther
			return res, nil
		},
		WithSessionStore(store),
	)

	conn, err := af.Connect(context.Background())
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	drainInBackground(conn)
	sendText(t, conn, "go")
	if err := conn.Detach(); err != nil {
		t.Fatalf("Detach: %v", err)
	}
	<-entered

	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	if out.FinishReason != AgentFinishReasonDetached {
		t.Errorf("AgentOutput.FinishReason = %q, want %q", out.FinishReason, AgentFinishReasonDetached)
	}

	close(release)
	snap := waitForSnapshot(t, store, out.SnapshotID, 2*time.Second, func(s *SessionSnapshot[testState]) bool {
		return s.Status == SnapshotStatusCompleted
	})
	if snap.FinishReason != AgentFinishReasonOther {
		t.Errorf("finalized snapshot.FinishReason = %q, want %q (AgentResult override)", snap.FinishReason, AgentFinishReasonOther)
	}
}

// --- Session ID tests ---

func TestAgent_SessionID_AssignedAndStable(t *testing.T) {
	// The runtime assigns the session ID when the invocation starts; every
	// snapshot the invocation persists carries it and the output reports it.
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()
	af := defineLastGoodTestAgent(reg, "sessionAssignFlow", WithSessionStore(store))

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}

	var snapshotIDs []string
	for _, text := range []string{"turn one", "turn two"} {
		te := sendTurn(t, conn, text)
		if te.SnapshotID != "" {
			snapshotIDs = append(snapshotIDs, te.SnapshotID)
		}
	}

	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	if out.SessionID == "" {
		t.Fatal("expected session ID on output")
	}
	if len(snapshotIDs) != 2 {
		t.Fatalf("expected 2 turn-end snapshots, got %d", len(snapshotIDs))
	}

	for _, id := range append(snapshotIDs, out.SnapshotID) {
		snap, err := store.GetSnapshot(ctx, id)
		if err != nil {
			t.Fatalf("GetSnapshot(%q): %v", id, err)
		}
		if snap.SessionID != out.SessionID {
			t.Errorf("snapshot %q SessionID = %q, want %q", id, snap.SessionID, out.SessionID)
		}
		// The persisted state blob is self-describing: it mirrors the
		// row's session ID.
		if snap.State == nil || snap.State.SessionID != out.SessionID {
			t.Errorf("snapshot %q state-carried session ID = %v, want %q", id, snap.State, out.SessionID)
		}
	}
}

func TestAgent_SessionID_AssignedBeforeFirstSnapshot(t *testing.T) {
	// The session ID exists from invocation start, not from the first
	// snapshot write: an invocation that commits no turn (and so writes no
	// snapshot) still reports the session it belongs to and exposes it to
	// the agent fn, with no snapshot to show for it yet.
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()

	var fnSawSessionID, ctxSawSessionID string
	af := DefineCustomAgent(reg, "sessionAlwaysAssigned",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			fnSawSessionID = sess.SessionID()
			// The ID lives on the session itself, so code holding only the
			// context-carried session (e.g. a tool) can read it too.
			if s := SessionFromContext[testState](ctx); s != nil {
				ctxSawSessionID = s.SessionID()
			}
			// Return without running a turn: nothing is committed, so no
			// snapshot is written, yet the session ID is already settled.
			return nil, nil
		},
		WithSessionStore(store),
	)

	out, err := af.RunText(ctx, "hi")
	if err != nil {
		t.Fatalf("RunText: %v", err)
	}
	if out.FinishReason == AgentFinishReasonFailed {
		t.Fatalf("invocation failed: %+v", out.Error)
	}
	if out.SessionID == "" {
		t.Fatal("expected session ID assigned at invocation start")
	}
	if fnSawSessionID != out.SessionID {
		t.Errorf("fn saw session ID %q, output reports %q", fnSawSessionID, out.SessionID)
	}
	if ctxSawSessionID != out.SessionID {
		t.Errorf("context-carried session saw ID %q, output reports %q", ctxSawSessionID, out.SessionID)
	}
	if out.SnapshotID != "" {
		t.Errorf("expected no snapshot (no turn committed), got %q", out.SnapshotID)
	}

	// A session with no persisted snapshots is not resumable, but supplying
	// its ID is not an error: it starts a brand-new conversation under that
	// caller-chosen ID rather than failing. The supplied ID carries through
	// to the output (see also TestAgent_ResumeFromSessionID_NewConversation).
	out2, err := af.RunText(ctx, "again", WithSessionID[testState](out.SessionID))
	if err != nil {
		t.Fatalf("RunText with session ID for a snapshot-less session: %v", err)
	}
	if out2.FinishReason == AgentFinishReasonFailed {
		t.Fatalf("second invocation failed: %+v", out2.Error)
	}
	if out2.SessionID != out.SessionID {
		t.Errorf("second invocation session ID = %q, want the caller-supplied %q", out2.SessionID, out.SessionID)
	}
}

func TestAgent_SessionID_StableAcrossSnapshotResume(t *testing.T) {
	// Resuming from a snapshot keeps extending the same session: rows
	// written by the resumed invocation inherit the chain's session ID.
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()
	af := defineLastGoodTestAgent(reg, "sessionStableFlow", WithSessionStore(store))

	out1, err := af.RunText(ctx, "first")
	if err != nil {
		t.Fatalf("RunText: %v", err)
	}
	if out1.SessionID == "" {
		t.Fatal("expected session ID from first invocation")
	}

	out2, err := af.RunText(ctx, "second", WithSnapshotID[testState](out1.SnapshotID))
	if err != nil {
		t.Fatalf("RunText resume: %v", err)
	}
	if out2.SessionID != out1.SessionID {
		t.Errorf("resumed invocation SessionID = %q, want %q", out2.SessionID, out1.SessionID)
	}
	snap2, err := store.GetSnapshot(ctx, out2.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}
	if snap2.SessionID != out1.SessionID {
		t.Errorf("resumed snapshot SessionID = %q, want %q", snap2.SessionID, out1.SessionID)
	}
}

func TestAgent_ResumeFromSessionID(t *testing.T) {
	// WithSessionID resolves the session's latest snapshot and continues
	// the conversation from it.
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()
	af := defineLastGoodTestAgent(reg, "sessionResumeFlow", WithSessionStore(store))

	out1, err := af.RunText(ctx, "first")
	if err != nil {
		t.Fatalf("RunText: %v", err)
	}

	out2, err := af.RunText(ctx, "second", WithSessionID[testState](out1.SessionID))
	if err != nil {
		t.Fatalf("RunText session resume: %v", err)
	}
	if out2.FinishReason == AgentFinishReasonFailed {
		t.Fatalf("session resume failed: %+v", out2.Error)
	}
	if out2.SessionID != out1.SessionID {
		t.Errorf("SessionID = %q, want %q", out2.SessionID, out1.SessionID)
	}

	snap2, err := store.GetSnapshot(ctx, out2.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}
	// Continued the conversation: both invocations' messages and both
	// counter increments, chained off the first invocation's snapshot.
	if got := len(snap2.State.Messages); got != 4 {
		t.Errorf("expected 4 messages after session resume, got %d", got)
	}
	if got := snap2.State.Custom.Counter; got != 2 {
		t.Errorf("expected counter=2 after session resume, got %d", got)
	}
	if snap2.ParentID != out1.SnapshotID {
		t.Errorf("resumed snapshot ParentID = %q, want %q", snap2.ParentID, out1.SnapshotID)
	}
}

func TestAgent_ResumeFromSessionID_ForkContinuesLatestBranch(t *testing.T) {
	// Re-invoking the agent from a non-tip snapshot forks the session.
	// Session-ID init does not care: the most recently updated branch
	// wins, so the conversation continues where activity last happened.
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()
	af := defineLastGoodTestAgent(reg, "sessionForkFlow", WithSessionStore(store))

	out1, err := af.RunText(ctx, "first")
	if err != nil {
		t.Fatalf("RunText: %v", err)
	}
	// Two sibling branches off the same parent: a fork.
	var branches []*AgentOutput[testState]
	for _, text := range []string{"branch b", "branch c"} {
		time.Sleep(2 * time.Millisecond) // order branches unambiguously by UpdatedAt
		out, err := af.RunText(ctx, text, WithSnapshotID[testState](out1.SnapshotID))
		if err != nil {
			t.Fatalf("RunText branch: %v", err)
		}
		if out.SessionID != out1.SessionID {
			t.Fatalf("branch SessionID = %q, want %q", out.SessionID, out1.SessionID)
		}
		branches = append(branches, out)
	}

	out, err := af.RunText(ctx, "which branch?", WithSessionID[testState](out1.SessionID))
	if err != nil {
		t.Fatalf("RunText session resume: %v", err)
	}
	if out.FinishReason == AgentFinishReasonFailed {
		t.Fatalf("session resume failed: %+v", out.Error)
	}
	snap, err := store.GetSnapshot(ctx, out.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}
	if want := branches[1].SnapshotID; snap.ParentID != want {
		t.Errorf("resumed snapshot ParentID = %q, want most recent branch %q", snap.ParentID, want)
	}
}

func TestAgent_ResumeFromSessionID_FailedTipRejected(t *testing.T) {
	// GetLatestSnapshot returns the session's literal latest row, so a failed
	// (or aborted) tip is no longer skipped: resuming the session by ID hits
	// the dead end and is rejected. To continue past it the caller names an
	// earlier good snapshot via WithSnapshotID.
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()
	af := defineLastGoodTestAgent(reg, "sessionDeadEndFlow", WithSessionStore(store))

	out1, err := af.RunText(ctx, "first")
	if err != nil {
		t.Fatalf("RunText: %v", err)
	}
	// A failed detach-style row chained off the tip, as a background
	// invocation that failed would leave behind. Created after out1, so it is
	// the session's latest by CreatedAt.
	failedAt := time.Now()
	if _, err := store.SaveSnapshot(ctx, "", func(_ *SessionSnapshot[testState]) (*SessionSnapshot[testState], error) {
		return &SessionSnapshot[testState]{
			SessionID:    out1.SessionID,
			ParentID:     out1.SnapshotID,
			Status:       SnapshotStatusFailed,
			FinishReason: AgentFinishReasonFailed,
			CreatedAt:    failedAt,
			UpdatedAt:    failedAt,
		}, nil
	}); err != nil {
		t.Fatalf("SaveSnapshot failed row: %v", err)
	}

	// Resuming by session ID hits the failed tip and is rejected.
	if _, err := af.RunText(ctx, "second", WithSessionID[testState](out1.SessionID)); err == nil {
		t.Fatal("expected resume to be rejected for a failed tip, got nil")
	} else if ge := core.AsGenkitError(err); ge.Status != core.FAILED_PRECONDITION {
		t.Fatalf("expected FAILED_PRECONDITION, got %q (err: %v)", ge.Status, err)
	}

	// Naming the last good snapshot explicitly still resumes past the dead end.
	out3, err := af.RunText(ctx, "third", WithSnapshotID[testState](out1.SnapshotID))
	if err != nil {
		t.Fatalf("RunText resume from good snapshot: %v", err)
	}
	if out3.FinishReason == AgentFinishReasonFailed {
		t.Fatalf("snapshot resume failed: %+v", out3.Error)
	}
	snap3, err := store.GetSnapshot(ctx, out3.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}
	if got := snap3.State.Custom.Counter; got != 2 {
		t.Errorf("expected counter=2 (resumed from last good state), got %d", got)
	}
}

func TestAgent_ResumeFromSessionID_NewConversation(t *testing.T) {
	// A caller may name a brand-new session: with no snapshot yet under that
	// ID, the invocation starts a fresh conversation and adopts the ID for
	// the whole session lifecycle (every snapshot carries it) rather than
	// failing with NOT_FOUND or minting a server ID.
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()
	af := defineLastGoodTestAgent(reg, "sessionNewConvoFlow", WithSessionStore(store))

	const sessID = "client-chosen-session"
	out, err := af.RunText(ctx, "hello", WithSessionID[testState](sessID))
	if err != nil {
		t.Fatalf("RunText: %v", err)
	}
	if out.FinishReason == AgentFinishReasonFailed {
		t.Fatalf("invocation failed: %+v", out.Error)
	}
	if out.SessionID != sessID {
		t.Errorf("output session ID = %q, want caller-supplied %q", out.SessionID, sessID)
	}
	// The persisted snapshot carries the caller-chosen session ID.
	snap, err := store.GetSnapshot(ctx, out.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}
	if snap.SessionID != sessID {
		t.Errorf("snapshot session ID = %q, want %q", snap.SessionID, sessID)
	}

	// A second invocation under the same ID now resumes the conversation it
	// just created (that snapshot is the session's resumable tip).
	out2, err := af.RunText(ctx, "again", WithSessionID[testState](sessID))
	if err != nil {
		t.Fatalf("RunText resume: %v", err)
	}
	if out2.SessionID != sessID {
		t.Errorf("resumed session ID = %q, want %q", out2.SessionID, sessID)
	}
	snap2, err := store.GetSnapshot(ctx, out2.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}
	if snap2.ParentID != out.SnapshotID {
		t.Errorf("resumed snapshot ParentID = %q, want first snapshot %q", snap2.ParentID, out.SnapshotID)
	}
	if got := snap2.State.Custom.Counter; got != 2 {
		t.Errorf("expected counter=2 after resuming the new conversation, got %d", got)
	}
}

func TestAgent_ResumeFromSessionID_AfterFailureResumesLastTurn(t *testing.T) {
	// After an invocation fails, the session's newest row is the last
	// successful turn's snapshot (a failed turn writes none), holding the
	// last-good state. Resuming by session ID continues from it like any
	// other snapshot.
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()
	af := defineLastGoodTestAgent(reg, "sessionRecoveryFlow", WithSessionStore[testState](store))

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	for _, text := range []string{"one", "two", "boom"} {
		sendText(t, conn, text)
	}
	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	if out.FinishReason != AgentFinishReasonFailed {
		t.Fatalf("expected failed invocation, got %q", out.FinishReason)
	}
	// The failed output points at the last successful turn's snapshot (turn
	// "two", counter=2), not the failed "boom" turn.
	lastGood, err := store.GetSnapshot(ctx, out.SnapshotID)
	if err != nil || lastGood == nil {
		t.Fatalf("GetSnapshot(%q): %v, %v", out.SnapshotID, lastGood, err)
	}
	if got := lastGood.State.Custom.Counter; got != 2 {
		t.Fatalf("expected last-good snapshot counter=2, got %d", got)
	}

	out2, err := af.RunText(ctx, "three", WithSessionID[testState](out.SessionID))
	if err != nil {
		t.Fatalf("RunText session resume: %v", err)
	}
	if out2.FinishReason == AgentFinishReasonFailed {
		t.Fatalf("session resume failed: %+v", out2.Error)
	}
	snap2, err := store.GetSnapshot(ctx, out2.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}
	if snap2.ParentID != lastGood.SnapshotID {
		t.Errorf("resumed snapshot ParentID = %q, want last-good row %q", snap2.ParentID, lastGood.SnapshotID)
	}
	// Last-good state (two successful turns, counter=2) plus the resumed
	// turn: the failed turn's partial mutations never made it in.
	if got := snap2.State.Custom.Counter; got != 3 {
		t.Errorf("expected counter=3 after resuming last-good state, got %d", got)
	}
	if got := len(snap2.State.Messages); got != 6 {
		t.Errorf("expected 6 messages after resuming last-good state, got %d", got)
	}
}

func TestAgent_ResumeFromSessionID_PendingTipRejected(t *testing.T) {
	// A pending tip means a detached invocation is still running; resuming
	// the session would race the background writer, so it is rejected
	// until the row finalizes.
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()
	af := defineLastGoodTestAgent(reg, "sessionPendingFlow", WithSessionStore(store))

	out1, err := af.RunText(ctx, "first")
	if err != nil {
		t.Fatalf("RunText: %v", err)
	}
	// Chained off the tip and created later, so it is the session's latest.
	pendingAt := time.Now()
	if _, err := store.SaveSnapshot(ctx, "", func(_ *SessionSnapshot[testState]) (*SessionSnapshot[testState], error) {
		return &SessionSnapshot[testState]{
			SessionID: out1.SessionID,
			ParentID:  out1.SnapshotID,
			Status:    SnapshotStatusPending,
			CreatedAt: pendingAt,
			UpdatedAt: pendingAt,
		}, nil
	}); err != nil {
		t.Fatalf("SaveSnapshot pending row: %v", err)
	}

	out, err := af.RunText(ctx, "second", WithSessionID[testState](out1.SessionID))
	if err == nil {
		t.Fatalf("expected error for pending tip, got output: %+v", out)
	}
	ge := core.AsGenkitError(err)
	if ge.Status != core.FAILED_PRECONDITION {
		t.Fatalf("expected FAILED_PRECONDITION, got %q (err: %v)", ge.Status, err)
	}
	if !strings.Contains(ge.Message, "still pending") {
		t.Errorf("expected error message to mention pending, got %q", ge.Message)
	}
}

func TestAgent_ClientManagedState_MintsSessionID(t *testing.T) {
	// With no store configured and a state object that carries no session
	// ID, the framework mints one and stamps it inside the output state,
	// so the client's opaque round-trip picks up a stable conversation
	// identity without tracking a separate field.
	ctx := context.Background()
	reg := newTestRegistry(t)
	af := defineLastGoodTestAgent(reg, "clientSessionFlow")

	out, err := af.RunText(ctx, "hi", WithState(&SessionState[testState]{Custom: testState{Counter: 1}}))
	if err != nil {
		t.Fatalf("RunText: %v", err)
	}
	if out.SessionID == "" {
		t.Fatal("expected minted SessionID for client-managed agent")
	}
	if out.State == nil || out.State.Custom.Counter != 2 {
		t.Fatalf("expected state passthrough with counter=2, got %+v", out.State)
	}
	if out.State.SessionID != out.SessionID {
		t.Errorf("state-carried session ID %q, want output's %q", out.State.SessionID, out.SessionID)
	}
}

func TestAgent_ClientManagedState_SessionIDRoundTrip(t *testing.T) {
	// With no store, the conversation's identity rides inside the state
	// object: an ID carried on SessionState is kept, the fn can read it,
	// and the output echoes it both top-level and inside the state, so
	// resending the state object preserves it across invocations.
	ctx := context.Background()
	reg := newTestRegistry(t)

	var fnSawSessionID string
	af := DefineCustomAgent(reg, "clientPassthroughFlow",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			fnSawSessionID = sess.SessionID()
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				sess.UpdateCustom(func(s testState) testState {
					s.Counter++
					return s
				})
				return nil, nil
			})
		},
	)

	out, err := af.RunText(ctx, "hi",
		WithState(&SessionState[testState]{SessionID: "client-chosen-id", Custom: testState{Counter: 1}}))
	if err != nil {
		t.Fatalf("RunText: %v", err)
	}
	if out.SessionID != "client-chosen-id" {
		t.Errorf("output SessionID = %q, want %q", out.SessionID, "client-chosen-id")
	}
	if fnSawSessionID != "client-chosen-id" {
		t.Errorf("fn saw session ID %q, want %q", fnSawSessionID, "client-chosen-id")
	}
	if out.State == nil || out.State.Custom.Counter != 2 {
		t.Fatalf("expected state passthrough with counter=2, got %+v", out.State)
	}
	if out.State.SessionID != "client-chosen-id" {
		t.Errorf("state-carried session ID = %q, want %q", out.State.SessionID, "client-chosen-id")
	}

	// Resending the output state opaquely keeps the identity.
	out2, err := af.RunText(ctx, "again", WithState(out.State))
	if err != nil {
		t.Fatalf("RunText: %v", err)
	}
	if out2.SessionID != "client-chosen-id" {
		t.Errorf("round-tripped SessionID = %q, want %q", out2.SessionID, "client-chosen-id")
	}
	if out2.State == nil || out2.State.Custom.Counter != 3 {
		t.Errorf("expected continued state with counter=3, got %+v", out2.State)
	}
}

func TestAgent_ClientManagedState_WithSessionIDRejected(t *testing.T) {
	// WithSessionID is a store lookup; a client-managed agent has no store
	// and carries the conversation's identity inside the state object, so
	// the option is rejected up front.
	ctx := context.Background()
	reg := newTestRegistry(t)
	af := defineLastGoodTestAgent(reg, "clientNoSessionLookupFlow")

	_, err := af.RunText(ctx, "hi", WithSessionID[testState]("some-id"))
	if err == nil {
		t.Fatal("expected error for WithSessionID on a client-managed agent")
	}
	if !strings.Contains(err.Error(), "SessionState.SessionID") {
		t.Errorf("expected error to point at SessionState.SessionID, got %q", err.Error())
	}
}

func TestAgent_ResumeFromSnapshotID_WithSessionID(t *testing.T) {
	// A session ID sent alongside a snapshot ID asserts which conversation
	// the snapshot belongs to: a match resumes normally, a mismatch is
	// rejected before the invocation starts.
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()
	af := defineLastGoodTestAgent(reg, "snapshotWithSessionFlow", WithSessionStore(store))

	out1, err := af.RunText(ctx, "first")
	if err != nil {
		t.Fatalf("RunText: %v", err)
	}

	out2, err := af.RunText(ctx, "second",
		WithSnapshotID[testState](out1.SnapshotID),
		WithSessionID[testState](out1.SessionID))
	if err != nil {
		t.Fatalf("RunText resume with matching session ID: %v", err)
	}
	if out2.SessionID != out1.SessionID {
		t.Errorf("SessionID = %q, want %q", out2.SessionID, out1.SessionID)
	}

	out3, err := af.RunText(ctx, "third",
		WithSnapshotID[testState](out2.SnapshotID),
		WithSessionID[testState]("not-the-right-session"))
	if err == nil {
		t.Fatalf("expected mismatch error, got output: %+v", out3)
	}
	ge := core.AsGenkitError(err)
	if ge.Status != core.INVALID_ARGUMENT {
		t.Errorf("expected status %q, got %q (err: %v)", core.INVALID_ARGUMENT, ge.Status, err)
	}
	if !strings.Contains(ge.Message, "does not belong to session") {
		t.Errorf("expected mismatch message, got %q", ge.Message)
	}
}

func TestAgent_Detach_AssignsSessionID(t *testing.T) {
	// A detach on a fresh conversation carries the runtime-assigned session
	// ID on the pending row; the detached output reports it and the
	// finalized row keeps it.
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()

	release := make(chan struct{})
	entered := make(chan struct{})
	af := DefineCustomAgent(reg, "detachSessionFlow",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				select {
				case entered <- struct{}{}:
				case <-ctx.Done():
				}
				<-release
				sess.AddMessages(ai.NewModelTextMessage("finished"))
				return nil, nil
			})
		},
		WithSessionStore(store),
	)

	conn, err := af.Connect(context.Background())
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	drainInBackground(conn)
	sendText(t, conn, "go")
	if err := conn.Detach(); err != nil {
		t.Fatalf("Detach: %v", err)
	}
	<-entered

	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	if out.SessionID == "" {
		t.Fatal("expected session ID on detached output")
	}
	pending, err := store.GetSnapshot(context.Background(), out.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}
	if pending.SessionID != out.SessionID {
		t.Errorf("pending row SessionID = %q, want %q", pending.SessionID, out.SessionID)
	}

	close(release)
	final := waitForSnapshot(t, store, out.SnapshotID, 2*time.Second, func(s *SessionSnapshot[testState]) bool {
		return s.Status == SnapshotStatusCompleted
	})
	if final.SessionID != out.SessionID {
		t.Errorf("finalized row SessionID = %q, want %q", final.SessionID, out.SessionID)
	}
}

// blockingSaveStore wraps testInMemStore so a test can hold the first
// SaveSnapshot call open: the agent's turn-end write blocks until release
// is closed, modeling a slow store at the exact moment a detach lands.
type blockingSaveStore[State any] struct {
	*testInMemStore[State]
	entered chan struct{}
	release chan struct{}
	once    sync.Once
}

func (s *blockingSaveStore[State]) SaveSnapshot(
	ctx context.Context,
	id string,
	fn func(existing *SessionSnapshot[State]) (*SessionSnapshot[State], error),
) (*SessionSnapshot[State], error) {
	blocked := false
	s.once.Do(func() { blocked = true })
	if blocked {
		close(s.entered)
		<-s.release
	}
	return s.testInMemStore.SaveSnapshot(ctx, id, fn)
}

func TestAgent_Detach_WaitsForInFlightTurnSnapshot(t *testing.T) {
	// A detach that lands while a turn-end snapshot write is still in
	// flight must wait for it: the pending row chains off the just-written
	// row instead of becoming its sibling (which would fork the session and
	// permanently break resume-by-session-ID), and the conversation stays
	// in one session.
	reg := newTestRegistry(t)
	store := &blockingSaveStore[testState]{
		testInMemStore: newTestInMemStore[testState](),
		entered:        make(chan struct{}),
		release:        make(chan struct{}),
	}
	af := defineLastGoodTestAgent(reg, "detachMidWrite", WithSessionStore[testState](store))

	conn, err := af.Connect(context.Background())
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	drainInBackground(conn)
	sendText(t, conn, "one")
	// Wait until the turn-end snapshot write is in flight, then detach
	// while it is still blocked inside the store.
	<-store.entered
	if err := conn.Detach(); err != nil {
		t.Fatalf("Detach: %v", err)
	}
	close(store.release)

	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	if out.FinishReason != AgentFinishReasonDetached {
		t.Fatalf("FinishReason = %q, want %q", out.FinishReason, AgentFinishReasonDetached)
	}

	final := waitForSnapshot[testState](t, store, out.SnapshotID, 2*time.Second, func(s *SessionSnapshot[testState]) bool {
		return s.Status == SnapshotStatusCompleted
	})

	// Find the turn-end row (the only row besides the detach row).
	var turnRowID, turnRowSession string
	others := 0
	store.testInMemStore.mu.RLock()
	for _, r := range store.testInMemStore.snapshots {
		if r.SnapshotID == out.SnapshotID {
			continue
		}
		others++
		turnRowID, turnRowSession = r.SnapshotID, r.SessionID
	}
	store.testInMemStore.mu.RUnlock()
	if others != 1 {
		t.Fatalf("expected exactly one turn-end row besides the detach row, got %d", others)
	}
	if final.ParentID != turnRowID {
		t.Errorf("detach row ParentID = %q, want turn-end row %q (a sibling row forks the session)", final.ParentID, turnRowID)
	}
	if final.SessionID != out.SessionID || turnRowSession != out.SessionID {
		t.Errorf("conversation split across sessions: turn=%q detach=%q output=%q", turnRowSession, final.SessionID, out.SessionID)
	}
	// The session stays linear and resolves to the finalized detach row.
	tip, err := store.GetLatestSnapshot(context.Background(), out.SessionID)
	if err != nil {
		t.Fatalf("GetLatestSnapshot: %v", err)
	}
	if tip == nil || tip.SnapshotID != out.SnapshotID {
		t.Errorf("session tip = %+v, want %q", tip, out.SnapshotID)
	}
}

func TestAgent_FailedTurn_OutputCarriesSessionID(t *testing.T) {
	// A failed invocation still reports the session it belongs to, next to
	// the last-good snapshot ID.
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()
	af := defineLastGoodTestAgent(reg, "failedSessionFlow", WithSessionStore(store))

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect: %v", err)
	}
	sendTurn(t, conn, "turn one")
	if err := conn.SendText("boom"); err != nil && !errors.Is(err, core.ErrActionCompleted) {
		t.Fatalf("SendText: %v", err)
	}
	out, err := conn.Output()
	if err != nil {
		t.Fatalf("Output: %v", err)
	}
	if out.FinishReason != AgentFinishReasonFailed {
		t.Fatalf("expected failed output, got %q", out.FinishReason)
	}
	if out.SessionID == "" {
		t.Fatal("expected session ID on failed output")
	}
	snap, err := store.GetSnapshot(ctx, out.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}
	if snap.SessionID != out.SessionID {
		t.Errorf("last-good snapshot SessionID = %q, want %q", snap.SessionID, out.SessionID)
	}
}

func TestAgent_ResumeFromLegacySnapshot_MintsFreshSessionID(t *testing.T) {
	// Snapshots written before session IDs existed have none; resuming one
	// starts a fresh session, assigned at invocation start and stamped on
	// every new row.
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()
	af := defineLastGoodTestAgent(reg, "legacySessionFlow", WithSessionStore(store))

	legacy := &SessionSnapshot[testState]{
		SnapshotID: "legacy-1",
		Status:     SnapshotStatusCompleted,
		CreatedAt:  time.Now(),
		UpdatedAt:  time.Now(),
		State:      &SessionState[testState]{Custom: testState{Counter: 5}},
	}
	store.mu.Lock()
	store.snapshots[legacy.SnapshotID] = legacy
	store.mu.Unlock()

	out, err := af.RunText(ctx, "continue", WithSnapshotID[testState]("legacy-1"))
	if err != nil {
		t.Fatalf("RunText: %v", err)
	}
	if out.FinishReason == AgentFinishReasonFailed {
		t.Fatalf("resume failed: %+v", out.Error)
	}
	if out.SessionID == "" {
		t.Fatal("expected freshly minted session ID after legacy resume")
	}
	snap, err := store.GetSnapshot(ctx, out.SnapshotID)
	if err != nil {
		t.Fatalf("GetSnapshot: %v", err)
	}
	if snap.SessionID != out.SessionID {
		t.Errorf("snapshot SessionID = %q, want %q", snap.SessionID, out.SessionID)
	}
	if got := snap.State.Custom.Counter; got != 6 {
		t.Errorf("expected counter=6 (legacy state + 1), got %d", got)
	}
}

func TestAgent_WithSessionID_OptionValidation(t *testing.T) {
	// The invocation-option layer rejects empty session IDs, duplicate
	// options, and conflicting state sources before the action is ever
	// invoked. WithSessionID composes with either state source, so those
	// combinations pass the option layer and the init-level checks take
	// over from there.
	ctx := context.Background()
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()
	af := defineLastGoodTestAgent(reg, "sessionOptFlow", WithSessionStore(store))

	if _, err := af.Connect(ctx, WithState(&SessionState[testState]{}), WithSnapshotID[testState]("x")); err == nil ||
		!strings.Contains(err.Error(), "mutually exclusive") {
		t.Errorf("WithState+WithSnapshotID: expected mutual-exclusion error, got %v", err)
	}
	if _, err := af.Connect(ctx, WithSessionID[testState]("s"), WithSessionID[testState]("s2")); err == nil ||
		!strings.Contains(err.Error(), "more than once") {
		t.Errorf("WithSessionID twice: expected duplicate-option error, got %v", err)
	}
	// An empty session ID is an explicit error, not a silent no-op: a
	// pipelined AgentOutput.SessionID from a storeless invocation must not
	// quietly start a fresh conversation.
	if _, err := af.Connect(ctx, WithSessionID[testState]("")); err == nil ||
		!strings.Contains(err.Error(), "session ID is empty") {
		t.Errorf("WithSessionID(\"\"): expected empty-ID error, got %v", err)
	}
	// WithSessionID composes with WithSnapshotID: the option layer accepts
	// the pair and the init-level checks (here: unknown snapshot) decide.
	conn, err := af.Connect(ctx, WithSessionID[testState]("s"), WithSnapshotID[testState]("x"))
	if err != nil {
		t.Fatalf("WithSessionID+WithSnapshotID: expected option layer to accept, got %v", err)
	}
	if _, err := conn.Output(); err == nil {
		t.Error("expected init-level error for unknown snapshot, got nil")
	}
}

func TestAgent_GetSnapshotAction_ReturnsSessionID(t *testing.T) {
	reg := newTestRegistry(t)
	store := newTestInMemStore[testState]()
	af := defineLastGoodTestAgent(reg, "sessionActionFlow", WithSessionStore(store))

	ctx := context.Background()
	out, err := af.RunText(ctx, "hi")
	if err != nil {
		t.Fatalf("RunText: %v", err)
	}
	if out.SessionID == "" {
		t.Fatal("expected session ID on output")
	}

	action := core.ResolveActionFor[*GetSnapshotRequest, *SessionSnapshot[testState], struct{}](
		reg, api.ActionTypeAgentSnapshot, "sessionActionFlow")
	if action == nil {
		t.Fatal("getSnapshot action not registered")
	}
	resp, err := action.Run(ctx, &GetSnapshotRequest{SnapshotID: out.SnapshotID}, nil)
	if err != nil {
		t.Fatalf("getSnapshot action: %v", err)
	}
	if resp.SessionID != out.SessionID {
		t.Errorf("getSnapshot SessionID = %q, want %q", resp.SessionID, out.SessionID)
	}
}

func TestPromptAgent_InlineMessages_DoesNotMutateSharedMetadata(t *testing.T) {
	ctx := context.Background()
	reg := setupPromptTestRegistry(t)

	// Render aliases the metadata map of messages registered via
	// WithMessages, so the agent loop must not tag the rendered
	// messages in place.
	shared := ai.NewModelTextMessage("inline context message")
	shared.Metadata = map[string]any{"origin": "config"}

	af := DefineAgent[testState](reg, "inlineMetaPrompt", InlinePrompt{
		ai.WithModelName("test/echo"),
		ai.WithMessages(shared),
	})

	response, err := af.RunText(ctx, "hello")
	if err != nil {
		t.Fatalf("RunText failed: %v", err)
	}

	if _, ok := shared.Metadata[promptMessageKey]; ok {
		t.Errorf("prompt message tag leaked into shared config message metadata: %v", shared.Metadata)
	}
	// The base message must still be filtered out of session history:
	// 1 user message + 1 model reply = 2.
	if got := len(response.State.Messages); got != 2 {
		t.Errorf("expected 2 messages, got %d", got)
		for i, m := range response.State.Messages {
			t.Logf("  msg[%d]: role=%s text=%s", i, m.Role, m.Text())
		}
	}
}

func TestPromptAgent_InlineMessages_ConcurrentInvocations(t *testing.T) {
	ctx := context.Background()
	reg := setupPromptTestRegistry(t)

	// All invocations render the same inline message whose metadata map
	// is shared with the registered prompt's config; tagging it in
	// place is a concurrent map write under the race detector.
	shared := ai.NewModelTextMessage("inline context message")
	shared.Metadata = map[string]any{"origin": "config"}

	af := DefineAgent[testState](reg, "inlineConcurrentPrompt", InlinePrompt{
		ai.WithModelName("test/echo"),
		ai.WithMessages(shared),
	})

	var wg sync.WaitGroup
	errs := make(chan error, 8)
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := af.RunText(ctx, "hello"); err != nil {
				errs <- err
			}
		}()
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Errorf("RunText failed: %v", err)
	}
}

func TestAgent_SendNilInput_Rejected(t *testing.T) {
	ctx := context.Background()
	reg := newTestRegistry(t)

	af := DefineCustomAgent(reg, "nilInputFlow",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
				if input.Message != nil {
					sess.AddMessages(ai.NewModelTextMessage("echo: " + input.Message.Text()))
				}
				return nil, nil
			})
		},
	)

	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect failed: %v", err)
	}

	if err := conn.Send(nil); err == nil {
		t.Error("expected Send(nil) to fail")
	}

	// The connection must remain usable after the rejected input.
	sendTurn(t, conn, "hello")
	conn.Close()

	response, err := conn.Output()
	if err != nil {
		t.Fatalf("Output failed: %v", err)
	}
	if got := len(response.State.Messages); got != 2 {
		t.Errorf("expected 2 messages, got %d", got)
	}
}

// TestDetachIntake_SkipsNilInputs covers nil inputs that bypass the typed
// Send API (e.g. a JSON null decoded by a transport): the intake must drop
// them rather than crash its reader goroutine or end the stream early.
func TestDetachIntake_SkipsNilInputs(t *testing.T) {
	t.Run("read path", func(t *testing.T) {
		src := make(chan *AgentInput, 4)
		src <- nil
		src <- &AgentInput{Message: ai.NewUserTextMessage("one")}
		src <- nil
		close(src)

		intake := startDetachIntake(src)
		defer intake.stopAndWait()

		var got []string
		for in := range intake.out() {
			got = append(got, in.Message.Text())
			intake.releaseForward()
		}
		if !slices.Equal(got, []string{"one"}) {
			t.Errorf("expected [one], got %v", got)
		}
	})

	t.Run("detach drain path", func(t *testing.T) {
		src := make(chan *AgentInput, 4)
		src <- &AgentInput{Detach: true, Message: ai.NewUserTextMessage("final")}
		src <- nil
		src <- &AgentInput{Message: ai.NewUserTextMessage("two")}
		close(src)

		intake := startDetachIntake(src)
		defer intake.stopAndWait()
		go func() { <-intake.detachSignal() }()

		var got []string
		for in := range intake.out() {
			got = append(got, in.Message.Text())
			intake.releaseForward()
		}
		if !slices.Equal(got, []string{"final", "two"}) {
			t.Errorf("expected [final two], got %v", got)
		}
	})
}

// TestAgent_ClientCancelMidStream reproduces an invocation hang: fn
// returns nil (the closed input stream ended sess.Run cleanly) while its
// last accepted chunk is still in the router's hands, parked on the full
// stream buffer, and the client then cancels instead of draining. The
// fn-done success path skips stopAndWait, so the router's forward must
// observe the cancelled action context itself (nothing will drain the
// stream again) for the invocation to resolve, and Output must unblock
// rather than waiting on completion unconditionally.
func TestAgent_ClientCancelMidStream(t *testing.T) {
	for i := range 10 {
		t.Run(fmt.Sprintf("iteration%d", i), func(t *testing.T) {
			reg := newTestRegistry(t)

			af := DefineCustomAgent(reg, "cancelFlow",
				func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
					return nil, sess.Run(ctx, func(ctx context.Context, input *AgentInput) (*TurnResult, error) {
						resp.SendModelChunk(&ai.ModelResponseChunk{
							Content: []*ai.Part{ai.NewTextPart("step0")},
						})
						resp.SendModelChunk(&ai.ModelResponseChunk{
							Content: []*ai.Part{ai.NewTextPart("step1")},
						})
						return nil, nil
					})
				},
			)

			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			conn, err := af.Connect(ctx)
			if err != nil {
				t.Fatalf("Connect failed: %v", err)
			}
			sendText(t, conn, "hello")
			// Close the input side so sess.Run ends cleanly and fn returns
			// nil once its sends are accepted.
			conn.Close()

			// Consume a single chunk: that frees a buffer slot so the fn's
			// remaining sends are accepted and fn returns, leaving the
			// router parked mid-forward on the full stream buffer.
			for range conn.Receive() {
				break
			}
			// Give the agent time to reach that parked state, then cancel
			// instead of draining.
			time.Sleep(50 * time.Millisecond)
			cancel()

			outputDone := make(chan struct{})
			go func() {
				defer close(outputDone)
				conn.Output()
			}()
			select {
			case <-outputDone:
			case <-time.After(10 * time.Second):
				t.Fatal("Output did not return after client cancellation")
			}

			// The invocation itself must also resolve: a wedged router
			// would leave the action goroutine (and its trace span) open
			// forever.
			select {
			case <-conn.Done():
			case <-time.After(10 * time.Second):
				t.Fatal("invocation did not complete after client cancellation")
			}
		})
	}
}

// TestAgent_OutputUnblocksOnCancel covers the caller's escape hatch when
// the agent fn does not observe cancellation: Output must return once the
// connection's context is cancelled instead of blocking on completion
// that may never come.
func TestAgent_OutputUnblocksOnCancel(t *testing.T) {
	reg := newTestRegistry(t)

	block := make(chan struct{})
	t.Cleanup(func() { close(block) }) // let the stubborn fn unwind

	af := DefineCustomAgent(reg, "stubbornFlow",
		func(ctx context.Context, resp Responder, sess *SessionRunner[testState]) (*AgentResult, error) {
			<-block // ignores ctx
			return nil, nil
		},
	)

	ctx, cancel := context.WithCancel(context.Background())
	conn, err := af.Connect(ctx)
	if err != nil {
		t.Fatalf("Connect failed: %v", err)
	}
	cancel()

	type result struct {
		out *AgentOutput[testState]
		err error
	}
	resultCh := make(chan result, 1)
	go func() {
		out, err := conn.Output()
		resultCh <- result{out, err}
	}()

	select {
	case res := <-resultCh:
		if res.err == nil {
			t.Errorf("expected an error from Output after cancellation, got output %+v", res.out)
		}
	case <-time.After(10 * time.Second):
		t.Fatal("Output did not return after cancellation; no context escape")
	}
}
