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

package sessionstoretest

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/core/status"
)

// row returns a row with status st and both timestamps set to at.
func (s *suite[State]) row(sessionID, parentID string, st exp.SnapshotStatus, state *exp.SessionState[State], at time.Time) *exp.SessionSnapshot[State] {
	return &exp.SessionSnapshot[State]{
		SessionID: sessionID,
		ParentID:  parentID,
		Status:    st,
		State:     state,
		CreatedAt: at,
		UpdatedAt: at,
	}
}

// pendingRow returns the row a detach writes: pending, no state, and a first
// heartbeat.
func (s *suite[State]) pendingRow(sessionID, parentID string, at time.Time) *exp.SessionSnapshot[State] {
	snap := s.row(sessionID, parentID, exp.SnapshotStatusPending, nil, at)
	snap.HeartbeatAt = &at
	return snap
}

// failedRow returns a failed turn's row, with a finish reason and an error
// that carries details.
func (s *suite[State]) failedRow(sessionID, parentID string, state *exp.SessionState[State], at time.Time) *exp.SessionSnapshot[State] {
	snap := s.row(sessionID, parentID, exp.SnapshotStatusFailed, state, at)
	snap.FinishReason = exp.AgentFinishReasonFailed
	snap.Error = status.Errorf(status.ErrUnavailable, "model unavailable").WithDetails(map[string]any{"step": "generate", "attempt": 3})
	return snap
}

// talk returns a state whose messages carry texts, alternating user and model.
func (s *suite[State]) talk(texts ...string) *exp.SessionState[State] {
	msgs := make([]*ai.Message, len(texts))
	for i, text := range texts {
		if i%2 == 0 {
			msgs[i] = ai.NewUserTextMessage(text)
		} else {
			msgs[i] = ai.NewModelTextMessage(text)
		}
	}
	return &exp.SessionState[State]{Messages: msgs}
}

// richState returns a state that uses every kind of content a conversation can
// hold: each part type, nested metadata, artifacts, usage, and text a store
// can mangle (escapes, NUL, characters outside the Basic Multilingual Plane).
func (s *suite[State]) richState() *exp.SessionState[State] {
	tricky := "quotes \" and \\ backslash, <tag> & ampersand, tab\t, newline\n, NUL \x00, accents é, CJK 世界, emoji 🌍, astral 𝄞, RTL \u202e mark, BOM \ufeff"
	return &exp.SessionState[State]{
		Messages: []*ai.Message{
			ai.NewSystemTextMessage("Answer briefly."),
			{
				Role:    ai.RoleUser,
				Content: []*ai.Part{ai.NewTextPart(tricky)},
				Metadata: map[string]any{
					"source": "web",
					"tags":   []any{"a", "b"},
					"nested": map[string]any{"ok": true, "ratio": 1.5, "none": nil, "list": []any{1.0, "two", false}},
				},
			},
			{
				Role: ai.RoleModel,
				Content: []*ai.Part{
					ai.NewReasoningPart("thinking it over", []byte{0x00, 0x01, 0xfe, 0xff}),
					ai.NewTextPart("Let me look that up."),
					ai.NewToolRequestPart(&ai.ToolRequest{Name: "lookup", Ref: "call-1", Input: map[string]any{"query": "weather", "days": 3.0, "units": []any{"c", "f"}, "strict": true, "extra": nil}}),
				},
			},
			{
				Role:    ai.RoleTool,
				Content: []*ai.Part{ai.NewToolResponsePart(&ai.ToolResponse{Name: "lookup", Ref: "call-1", Output: map[string]any{"temp": 21.5, "conditions": "sunny"}})},
			},
			{
				Role: ai.RoleModel,
				Content: []*ai.Part{
					ai.NewMediaPart("image/png", "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABpfZFQAAAAABJRU5ErkJggg=="),
					ai.NewDataPart(map[string]any{"chart": []any{1.0, 2.0, 3.0}}),
					ai.NewCustomPart(map[string]any{"kind": "widget", "id": "w-1"}),
					ai.NewTextPart(""),
				},
			},
		},
		Artifacts: []*exp.Artifact{
			{Name: "notes.md", Parts: []*ai.Part{ai.NewTextPart("# Notes\n- one\n- two")}, Metadata: map[string]any{"version": 2.0}},
			{Name: "chart.png", Parts: []*ai.Part{ai.NewMediaPart("image/png", "https://example.com/chart.png")}},
		},
		// Counts above 2^53 catch a store that rounds numbers through
		// float64, as a JSON decode into an any does.
		Usage: &ai.GenerationUsage{
			InputTokens:         1<<53 + 1,
			OutputTokens:        340,
			ThoughtsTokens:      56,
			TotalTokens:         1<<53 + 397,
			CachedContentTokens: 800,
			Custom:              map[string]float64{"cost": 0.0123},
		},
	}
}

// largeState returns a state of about size bytes of JSON, in messages of 64 KiB
// of varied text.
func (s *suite[State]) largeState(size int) *exp.SessionState[State] {
	const chunk = 64 << 10
	const alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,;:!?-"
	seed := uint32(1)
	var msgs []*ai.Message
	for total := 0; total < size; total += chunk {
		b := make([]byte, chunk)
		for i := range b {
			seed = seed*1664525 + 1013904223
			b[i] = alphabet[int(seed>>24)%len(alphabet)]
		}
		if len(msgs)%2 == 0 {
			msgs = append(msgs, ai.NewUserTextMessage(string(b)))
		} else {
			msgs = append(msgs, ai.NewModelTextMessage(string(b)))
		}
	}
	return &exp.SessionState[State]{Messages: msgs}
}

// clock hands out increasing timestamps a millisecond apart, so the checks do
// not depend on the wall clock's resolution and fit any store that keeps at
// least millisecond precision. The zone is not UTC, so a store that drops the
// offset instead of converting the instant fails.
type clock struct{ next time.Time }

func newClock() *clock {
	return &clock{next: time.Date(2026, 3, 4, 5, 6, 7, 0, time.FixedZone("UTC+05:30", 5*3600+30*60))}
}

func (c *clock) now() time.Time {
	at := c.next
	c.next = c.next.Add(time.Millisecond)
	return at
}

// --- Store operations ---

// put saves snap at id and returns the row the store reports. fn hands the
// store a fresh copy on every call, since a store may call it more than once
// and may write into the value it returns.
func put[State any](t *testing.T, ctx context.Context, store exp.SessionStore[State], id string, snap *exp.SessionSnapshot[State]) *exp.SessionSnapshot[State] {
	t.Helper()
	saved, err := store.SaveSnapshot(ctx, id, func(*exp.SessionSnapshot[State]) (*exp.SessionSnapshot[State], error) {
		return cloneJSON(snap), nil
	})
	if err != nil {
		t.Fatalf("SaveSnapshot(%q): %v", id, err)
	}
	if saved == nil {
		t.Fatalf("SaveSnapshot(%q) returned nil for a write", id)
	}
	return saved
}

// rewrite replaces the state of an existing row the way a finalize does: it
// keeps CreatedAt and advances UpdatedAt.
func rewrite[State any](t *testing.T, ctx context.Context, store exp.SessionStore[State], row *exp.SessionSnapshot[State], state *exp.SessionState[State], at time.Time) {
	t.Helper()
	next := cloneJSON(row)
	next.State = state
	next.UpdatedAt = at
	put(t, ctx, store, row.SnapshotID, next)
}

// get reads a row by ID, failing t on an error. It returns nil for a miss.
func get[State any](t *testing.T, ctx context.Context, store exp.SessionStore[State], id string) *exp.SessionSnapshot[State] {
	t.Helper()
	snap, err := store.GetSnapshot(ctx, id)
	if err != nil {
		t.Fatalf("GetSnapshot(%q): %v", id, err)
	}
	return snap
}

// mustGet reads a row by ID, failing t on an error or a miss.
func mustGet[State any](t *testing.T, ctx context.Context, store exp.SessionStore[State], id string) *exp.SessionSnapshot[State] {
	t.Helper()
	snap := get(t, ctx, store, id)
	if snap == nil {
		t.Fatalf("GetSnapshot(%q) = nil, want the row", id)
	}
	return snap
}

// getLatest reads a session's latest row, failing t on an error. It returns
// nil for a miss.
func getLatest[State any](t *testing.T, ctx context.Context, store exp.SessionStore[State], sessionID string) *exp.SessionSnapshot[State] {
	t.Helper()
	snap, err := store.GetLatestSnapshot(ctx, sessionID)
	if err != nil {
		t.Fatalf("GetLatestSnapshot(%q): %v", sessionID, err)
	}
	return snap
}

// beat refreshes a row's heartbeat with the runtime's mutator: a pending or
// aborting row is carried through unchanged but for HeartbeatAt, and a settled
// row is left alone.
func beat[State any](t *testing.T, ctx context.Context, store exp.SessionStore[State], id string, at time.Time) {
	t.Helper()
	if _, err := store.SaveSnapshot(ctx, id, func(existing *exp.SessionSnapshot[State]) (*exp.SessionSnapshot[State], error) {
		if existing == nil || existing.Status.Terminal() {
			return nil, nil
		}
		updated := *existing
		updated.HeartbeatAt = &at
		return &updated, nil
	}); err != nil {
		t.Fatalf("heartbeat of %q: %v", id, err)
	}
}

// abort flips a pending row to aborting with the runtime's mutator, and
// returns the row's status afterward, or "" when the row does not exist. A
// row that is not pending is rewritten verbatim.
func abort[State any](t *testing.T, ctx context.Context, store exp.SessionStore[State], id string, at time.Time) exp.SnapshotStatus {
	t.Helper()
	saved, err := store.SaveSnapshot(ctx, id, func(existing *exp.SessionSnapshot[State]) (*exp.SessionSnapshot[State], error) {
		if existing == nil {
			return nil, nil
		}
		if existing.Status != exp.SnapshotStatusPending {
			return existing, nil
		}
		updated := *existing
		updated.Status = exp.SnapshotStatusAborting
		updated.UpdatedAt = at
		return &updated, nil
	})
	if err != nil {
		t.Fatalf("abort of %q: %v", id, err)
	}
	if saved == nil {
		return ""
	}
	return saved.Status
}

// finalize lands a detached run's outcome on its pending row with the
// runtime's mutator: a settled row is left alone, and anything else is
// replaced by a row built from the pending one, with no heartbeat.
func finalize[State any](t *testing.T, ctx context.Context, store exp.SessionStore[State], pending *exp.SessionSnapshot[State], st exp.SnapshotStatus, reason exp.AgentFinishReason, state *exp.SessionState[State], at time.Time) {
	t.Helper()
	if _, err := store.SaveSnapshot(ctx, pending.SnapshotID, func(existing *exp.SessionSnapshot[State]) (*exp.SessionSnapshot[State], error) {
		if existing != nil && existing.Status.Terminal() {
			return nil, nil
		}
		return &exp.SessionSnapshot[State]{
			SessionID:    pending.SessionID,
			ParentID:     pending.ParentID,
			Status:       st,
			FinishReason: reason,
			State:        cloneJSON(state),
			CreatedAt:    pending.CreatedAt,
			UpdatedAt:    at,
		}, nil
	}); err != nil {
		t.Fatalf("finalize of %q: %v", pending.SnapshotID, err)
	}
}

// retry calls save until it succeeds, up to writeAttempts times, and returns
// the last error.
func retry(save func() error) error {
	var err error
	for range writeAttempts {
		if err = save(); err == nil {
			return nil
		}
		time.Sleep(10 * time.Millisecond)
	}
	return err
}

// --- Subscriptions ---

// nextStatus returns the next status ch yields, failing t if ch closes or
// stays silent past eventTimeout.
func nextStatus(t *testing.T, ch <-chan exp.SnapshotStatus) exp.SnapshotStatus {
	t.Helper()
	select {
	case st, ok := <-ch:
		if !ok {
			t.Fatal("the subscription closed, want a status")
		}
		return st
	case <-time.After(eventTimeout):
		t.Fatalf("no status within %v", eventTimeout)
		return ""
	}
}

// awaitStatus reads ch until it yields want. Subscriptions are level-triggered:
// a store may skip a status that was overwritten before it was read, or repeat
// one, so values before want are passed over.
func awaitStatus(t *testing.T, ch <-chan exp.SnapshotStatus, want exp.SnapshotStatus) {
	t.Helper()
	var seen []exp.SnapshotStatus
	timeout := time.After(eventTimeout)
	for {
		select {
		case st, ok := <-ch:
			if !ok {
				t.Fatalf("the subscription closed before %q (saw %v)", want, seen)
			}
			if st == want {
				return
			}
			seen = append(seen, st)
		case <-timeout:
			t.Fatalf("no %q within %v (saw %v)", want, eventTimeout, seen)
		}
	}
}

// awaitClose reads ch until it closes, failing t past eventTimeout. Unless
// allowValues is set, a value before the close fails t too.
func awaitClose(t *testing.T, ch <-chan exp.SnapshotStatus, allowValues bool) {
	t.Helper()
	timeout := time.After(eventTimeout)
	for {
		select {
		case st, ok := <-ch:
			if !ok {
				return
			}
			if !allowValues {
				t.Fatalf("the subscription yielded %q, want it closed without a value", st)
			}
		case <-timeout:
			t.Fatalf("the subscription did not close within %v", eventTimeout)
		}
	}
}

// --- Comparisons ---

// checkRow fails t unless got carries want's fields: identity, lineage, and
// lifecycle exactly, timestamps as instants, the error by its wire form, and
// the state as JSON.
func checkRow[State any](t *testing.T, label string, got, want *exp.SessionSnapshot[State]) {
	t.Helper()
	if got == nil {
		t.Errorf("%s: row is nil, want %s", label, describe(want))
		return
	}
	if got.SnapshotID != want.SnapshotID || got.SessionID != want.SessionID || got.ParentID != want.ParentID ||
		got.Status != want.Status || got.FinishReason != want.FinishReason {
		t.Errorf("%s: got %s, want %s", label, describe(got), describe(want))
	}
	if !got.CreatedAt.Equal(want.CreatedAt) {
		t.Errorf("%s: CreatedAt = %v, want %v", label, got.CreatedAt, want.CreatedAt)
	}
	if !got.UpdatedAt.Equal(want.UpdatedAt) {
		t.Errorf("%s: UpdatedAt = %v, want %v", label, got.UpdatedAt, want.UpdatedAt)
	}
	switch {
	case got.HeartbeatAt == nil && want.HeartbeatAt == nil:
	case got.HeartbeatAt == nil || want.HeartbeatAt == nil || !got.HeartbeatAt.Equal(*want.HeartbeatAt):
		t.Errorf("%s: HeartbeatAt = %v, want %v", label, timeOrNil(got.HeartbeatAt), timeOrNil(want.HeartbeatAt))
	}
	if ok, diff := sameJSON(want.Error, got.Error); !ok {
		t.Errorf("%s: Error differs from the one written (JSON Patch from want to got): %s", label, diff)
	}
	checkState(t, label, got.State, want.State)
}

// checkState fails t unless got and want encode to the same JSON, nil included.
func checkState[State any](t *testing.T, label string, got, want *exp.SessionState[State]) {
	t.Helper()
	if (got == nil) != (want == nil) {
		t.Errorf("%s: state is %v, want %v", label, stateOrNil(got), stateOrNil(want))
		return
	}
	if ok, diff := sameJSON(want, got); !ok {
		t.Errorf("%s: state differs from the one written (JSON Patch from want to got): %s", label, diff)
	}
}

// sameJSON reports whether want and got encode to the same JSON. When they do
// not, it describes the difference as a JSON Patch from want to got.
func sameJSON(want, got any) (bool, string) {
	wantJSON, err := json.Marshal(want)
	if err != nil {
		return false, fmt.Sprintf("marshal want: %v", err)
	}
	gotJSON, err := json.Marshal(got)
	if err != nil {
		return false, fmt.Sprintf("marshal got: %v", err)
	}
	if bytes.Equal(wantJSON, gotJSON) {
		return true, ""
	}
	patch, err := json.Marshal(exp.Diff(json.RawMessage(wantJSON), json.RawMessage(gotJSON)))
	if err != nil {
		return false, fmt.Sprintf("marshal diff: %v", err)
	}
	return false, truncate(string(patch), 2000)
}

// describe renders a row's identity and lifecycle for a failure message.
func describe[State any](snap *exp.SessionSnapshot[State]) string {
	if snap == nil {
		return "<nil>"
	}
	return fmt.Sprintf("{id=%q session=%q parent=%q status=%q finish=%q}", snap.SnapshotID, snap.SessionID, snap.ParentID, snap.Status, snap.FinishReason)
}

func timeOrNil(at *time.Time) any {
	if at == nil {
		return "<nil>"
	}
	return *at
}

func stateOrNil[State any](st *exp.SessionState[State]) string {
	if st == nil {
		return "nil"
	}
	return "non-nil"
}

// truncate cuts s to at most n bytes, dropping a character the cut splits.
func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return strings.ToValidUTF8(s[:n], "") + "... (" + fmt.Sprint(len(s)-n) + " more bytes)"
}

// cloneJSON deep-copies v through JSON, the way stores copy rows. It returns
// nil for nil.
func cloneJSON[T any](v *T) *T {
	if v == nil {
		return nil
	}
	b, err := json.Marshal(v)
	if err != nil {
		panic(fmt.Sprintf("sessionstoretest: clone: %v", err))
	}
	var out T
	if err := json.Unmarshal(b, &out); err != nil {
		panic(fmt.Sprintf("sessionstoretest: clone: %v", err))
	}
	return &out
}
