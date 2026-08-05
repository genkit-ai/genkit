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
