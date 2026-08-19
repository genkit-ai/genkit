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
	"reflect"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/internal/base"
)

// TestNewSessionContextPublishesPromptState verifies that attaching a session to
// a context also exposes its custom state through internal/base, which is how
// ai.prompt injects {{@state}} into templates without importing this package.
func TestNewSessionContextPublishesPromptState(t *testing.T) {
	s := &Session[map[string]any]{
		state: SessionState[map[string]any]{
			Custom: map[string]any{
				"name":        "Alice",
				"preferences": map[string]any{"theme": "dark"},
			},
		},
	}

	ctx := NewSessionContext(context.Background(), s)

	got := base.PromptStateFromContext(ctx)
	want := map[string]any{
		"name":        "Alice",
		"preferences": map[string]any{"theme": "dark"},
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("PromptStateFromContext() = %#v, want %#v", got, want)
	}
}

// TestPromptStateReflectsLatestCustom verifies the published state getter is
// evaluated lazily, so a template rendered later sees custom-state mutations
// made after the context was built.
func TestPromptStateReflectsLatestCustom(t *testing.T) {
	s := &Session[map[string]any]{
		state: SessionState[map[string]any]{Custom: map[string]any{"n": float64(1)}},
	}
	ctx := NewSessionContext(context.Background(), s)

	s.UpdateCustom(func(map[string]any) map[string]any {
		return map[string]any{"n": float64(2)}
	})

	got := base.PromptStateFromContext(ctx)
	want := map[string]any{"n": float64(2)}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("PromptStateFromContext() = %#v, want %#v", got, want)
	}
}

// TestPromptStateNilWithoutSession verifies that no state is published when no
// session is attached to the context.
func TestPromptStateNilWithoutSession(t *testing.T) {
	if got := base.PromptStateFromContext(context.Background()); got != nil {
		t.Errorf("PromptStateFromContext() = %#v, want nil", got)
	}
}

func TestSnapshotStatus_Terminal(t *testing.T) {
	cases := []struct {
		status SnapshotStatus
		want   bool
	}{
		{SnapshotStatusPending, false},
		{SnapshotStatusCompleted, true},
		{SnapshotStatusAborted, true},
		{SnapshotStatusFailed, true},
		{SnapshotStatusExpired, true},
		// Empty counts as completed, matching the documented default.
		{SnapshotStatus(""), true},
	}
	for _, tc := range cases {
		if got := tc.status.Terminal(); got != tc.want {
			t.Errorf("SnapshotStatus(%q).Terminal() = %v, want %v", tc.status, got, tc.want)
		}
	}
}

func TestSessionState_LastModelMessage(t *testing.T) {
	toolOnly := &ai.Message{Role: ai.RoleModel, Content: []*ai.Part{
		ai.NewToolRequestPart(&ai.ToolRequest{Name: "search"}),
	}}

	cases := []struct {
		name     string
		messages []*ai.Message
		want     string // "" means nil expected
	}{
		{name: "empty history"},
		{name: "no model messages", messages: []*ai.Message{ai.NewUserTextMessage("hi")}},
		{
			name: "latest text-bearing model message wins",
			messages: []*ai.Message{
				ai.NewUserTextMessage("q1"),
				ai.NewModelTextMessage("a1"),
				ai.NewUserTextMessage("q2"),
				ai.NewModelTextMessage("a2"),
			},
			want: "a2",
		},
		{
			name: "tool-request-only tip is skipped",
			messages: []*ai.Message{
				ai.NewModelTextMessage("spoken answer"),
				ai.NewUserTextMessage("follow-up"),
				toolOnly,
			},
			want: "spoken answer",
		},
		{
			name:     "only tool requests",
			messages: []*ai.Message{toolOnly},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			state := &SessionState[any]{Messages: tc.messages}
			got := state.LastModelMessage()
			if tc.want == "" {
				if got != nil {
					t.Fatalf("LastModelMessage() = %+v, want nil", got)
				}
				return
			}
			if got == nil || got.Text() != tc.want {
				t.Fatalf("LastModelMessage().Text() = %v, want %q", got, tc.want)
			}
		})
	}

	// A nil receiver (e.g. a pending snapshot's nil state) is tolerated.
	var nilState *SessionState[any]
	if got := nilState.LastModelMessage(); got != nil {
		t.Fatalf("nil receiver LastModelMessage() = %+v, want nil", got)
	}
}
