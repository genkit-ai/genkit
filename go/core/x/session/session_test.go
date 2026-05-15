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

package session

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
)

// UserState is a test state type with various field types.
type UserState struct {
	Name        string            `json:"name"`
	Count       int               `json:"count"`
	Preferences map[string]string `json:"preferences,omitempty"`
}

func TestNew_DefaultID(t *testing.T) {
	ctx := context.Background()
	sess, err := New[UserState](ctx)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	if sess.ID() == "" {
		t.Error("Expected session to have a generated ID")
	}
}

func TestNew_WithID(t *testing.T) {
	ctx := context.Background()
	customID := "my-custom-id"
	sess, err := New(ctx, WithID[UserState](customID))
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	if sess.ID() != customID {
		t.Errorf("Expected ID %q, got %q", customID, sess.ID())
	}
}

func TestNew_WithInitialState(t *testing.T) {
	ctx := context.Background()
	initial := UserState{Name: "Alice", Count: 42}
	sess, err := New(ctx, WithInitialState(initial))
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	got := sess.State()
	if got.Name != initial.Name {
		t.Errorf("Expected Name %q, got %q", initial.Name, got.Name)
	}
	if got.Count != initial.Count {
		t.Errorf("Expected Count %d, got %d", initial.Count, got.Count)
	}
}

func TestNew_WithStore(t *testing.T) {
	ctx := context.Background()
	store := NewInMemoryStore[UserState]()
	sess, err := New(ctx, WithStore(store))
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	if sess.store != store {
		t.Error("Expected store to be set")
	}
}

func TestNew_MultipleOptions(t *testing.T) {
	ctx := context.Background()
	store := NewInMemoryStore[UserState]()
	customID := "multi-option-id"
	initial := UserState{Name: "Bob", Count: 100}

	sess, err := New(ctx,
		WithID[UserState](customID),
		WithInitialState(initial),
		WithStore(store),
	)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	if sess.ID() != customID {
		t.Errorf("Expected ID %q, got %q", customID, sess.ID())
	}
	if sess.State().Name != initial.Name {
		t.Errorf("Expected Name %q, got %q", initial.Name, sess.State().Name)
	}
	if sess.store != store {
		t.Error("Expected store to be set")
	}
}

func TestNew_DuplicateID(t *testing.T) {
	ctx := context.Background()
	_, err := New(ctx,
		WithID[UserState]("first"),
		WithID[UserState]("second"),
	)
	if err == nil {
		t.Fatal("Expected error for duplicate WithID")
	}
	if !strings.Contains(err.Error(), "cannot set ID more than once") {
		t.Errorf("Expected duplicate ID error, got: %v", err)
	}
}

func TestNew_DuplicateInitialState(t *testing.T) {
	ctx := context.Background()
	_, err := New(ctx,
		WithInitialState(UserState{Name: "First"}),
		WithInitialState(UserState{Name: "Second"}),
	)
	if err == nil {
		t.Fatal("Expected error for duplicate WithInitialState")
	}
	if !strings.Contains(err.Error(), "cannot set initial state more than once") {
		t.Errorf("Expected duplicate state error, got: %v", err)
	}
}

func TestNew_DuplicateStore(t *testing.T) {
	ctx := context.Background()
	store1 := NewInMemoryStore[UserState]()
	store2 := NewInMemoryStore[UserState]()
	_, err := New(ctx,
		WithStore(store1),
		WithStore(store2),
	)
	if err == nil {
		t.Fatal("Expected error for duplicate WithStore")
	}
	if !strings.Contains(err.Error(), "cannot set store more than once") {
		t.Errorf("Expected duplicate store error, got: %v", err)
	}
}

func TestSession_State(t *testing.T) {
	ctx := context.Background()
	initial := UserState{
		Name:        "Alice",
		Count:       10,
		Preferences: map[string]string{"theme": "dark"},
	}
	sess, err := New(ctx, WithInitialState(initial))
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	t.Run("returns correct values", func(t *testing.T) {
		got := sess.State()
		if got.Name != initial.Name {
			t.Errorf("Expected Name %q, got %q", initial.Name, got.Name)
		}
		if got.Count != initial.Count {
			t.Errorf("Expected Count %d, got %d", initial.Count, got.Count)
		}
		if got.Preferences["theme"] != "dark" {
			t.Errorf("Expected theme %q, got %q", "dark", got.Preferences["theme"])
		}
	})

	t.Run("modifications to returned copy do not affect session", func(t *testing.T) {
		// Get a copy of the state
		copy1 := sess.State()

		// Modify the map in the returned copy
		copy1.Preferences["theme"] = "light"
		copy1.Preferences["newKey"] = "newValue"
		copy1.Name = "Modified"
		copy1.Count = 999

		// Get another copy and verify the session's internal state is unchanged
		copy2 := sess.State()

		if copy2.Name != "Alice" {
			t.Errorf("Session state was mutated: expected Name %q, got %q", "Alice", copy2.Name)
		}
		if copy2.Count != 10 {
			t.Errorf("Session state was mutated: expected Count %d, got %d", 10, copy2.Count)
		}
		if copy2.Preferences["theme"] != "dark" {
			t.Errorf("Session state was mutated: expected theme %q, got %q", "dark", copy2.Preferences["theme"])
		}
		if _, exists := copy2.Preferences["newKey"]; exists {
			t.Errorf("Session state was mutated: unexpected key 'newKey' in Preferences")
		}
	})
}

func TestSession_UpdateState_NoStore(t *testing.T) {
	ctx := context.Background()
	sess, err := New(ctx, WithInitialState(UserState{Name: "Alice"}))
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	// Verify no store is set when not provided
	if sess.store != nil {
		t.Fatal("Expected no store when not provided")
	}

	newState := UserState{Name: "Bob", Count: 5}
	if err := sess.UpdateState(ctx, newState); err != nil {
		t.Fatalf("UpdateState failed: %v", err)
	}

	// State should still be updated in memory
	got := sess.State()
	if got.Name != newState.Name {
		t.Errorf("Expected Name %q, got %q", newState.Name, got.Name)
	}
	if got.Count != newState.Count {
		t.Errorf("Expected Count %d, got %d", newState.Count, got.Count)
	}
}

func TestSession_UpdateState_WithStore(t *testing.T) {
	ctx := context.Background()
	store := NewInMemoryStore[UserState]()
	sess, err := New(ctx,
		WithID[UserState]("test-session"),
		WithInitialState(UserState{Name: "Alice"}),
		WithStore(store),
	)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	newState := UserState{Name: "Bob", Count: 5}
	if err := sess.UpdateState(ctx, newState); err != nil {
		t.Fatalf("UpdateState failed: %v", err)
	}

	// Verify state is updated in session
	got := sess.State()
	if got.Name != newState.Name {
		t.Errorf("Expected Name %q, got %q", newState.Name, got.Name)
	}

	// Verify state is persisted in store
	data, err := store.Get(ctx, "test-session")
	if err != nil {
		t.Fatalf("Store.Get failed: %v", err)
	}
	if data == nil {
		t.Fatal("Expected data in store, got nil")
	}
	if data.State.Name != newState.Name {
		t.Errorf("Store: expected Name %q, got %q", newState.Name, data.State.Name)
	}
}

func TestLoad_Success(t *testing.T) {
	store := NewInMemoryStore[UserState]()
	ctx := context.Background()

	// Save some data
	data := &Data[UserState]{
		ID:    "existing-session",
		State: UserState{Name: "Charlie", Count: 99},
	}
	if err := store.Save(ctx, data.ID, data); err != nil {
		t.Fatalf("Store.Save failed: %v", err)
	}

	// Load the session
	loaded, err := Load(ctx, store, "existing-session")
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	if loaded.ID() != "existing-session" {
		t.Errorf("Expected ID %q, got %q", "existing-session", loaded.ID())
	}
	if loaded.State().Name != "Charlie" {
		t.Errorf("Expected Name %q, got %q", "Charlie", loaded.State().Name)
	}
	if loaded.State().Count != 99 {
		t.Errorf("Expected Count %d, got %d", 99, loaded.State().Count)
	}
}

func TestLoad_NotFound(t *testing.T) {
	store := NewInMemoryStore[UserState]()
	ctx := context.Background()

	_, err := Load(ctx, store, "non-existent")
	if err == nil {
		t.Fatal("Expected error for non-existent session")
	}

	var notFoundErr *NotFoundError
	if !errors.As(err, &notFoundErr) {
		t.Errorf("Expected NotFoundError, got %T: %v", err, err)
	}
	if notFoundErr.SessionID != "non-existent" {
		t.Errorf("Expected SessionID %q, got %q", "non-existent", notFoundErr.SessionID)
	}
}

func TestNewContext_FromContext(t *testing.T) {
	ctx := context.Background()
	sess, err := New(ctx,
		WithID[UserState]("ctx-test"),
		WithInitialState(UserState{Name: "Diana"}),
	)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	// Attach session to context
	ctx = NewContext(ctx, sess)

	// Retrieve from context
	retrieved := FromContext[UserState](ctx)
	if retrieved == nil {
		t.Fatal("Expected session from context, got nil")
	}
	if retrieved.ID() != "ctx-test" {
		t.Errorf("Expected ID %q, got %q", "ctx-test", retrieved.ID())
	}
	if retrieved.State().Name != "Diana" {
		t.Errorf("Expected Name %q, got %q", "Diana", retrieved.State().Name)
	}
}

func TestStateFromContext(t *testing.T) {
	t.Run("returns state when session exists", func(t *testing.T) {
		ctx := context.Background()
		initial := UserState{
			Name:        "Alice",
			Count:       42,
			Preferences: map[string]string{"theme": "dark"},
		}
		sess, err := New(ctx, WithInitialState(initial))
		if err != nil {
			t.Fatalf("New failed: %v", err)
		}

		ctx = NewContext(ctx, sess)

		state := StateFromContext(ctx)
		if state == nil {
			t.Fatal("Expected state from context, got nil")
		}

		// StateFromContext returns the state as any, so we need to type assert
		userState, ok := state.(UserState)
		if !ok {
			t.Fatalf("Expected UserState, got %T", state)
		}

		if userState.Name != "Alice" {
			t.Errorf("Expected Name %q, got %q", "Alice", userState.Name)
		}
		if userState.Count != 42 {
			t.Errorf("Expected Count %d, got %d", 42, userState.Count)
		}
		if userState.Preferences["theme"] != "dark" {
			t.Errorf("Expected theme %q, got %q", "dark", userState.Preferences["theme"])
		}
	})

	t.Run("returns nil when no session in context", func(t *testing.T) {
		ctx := context.Background()
		state := StateFromContext(ctx)
		if state != nil {
			t.Errorf("Expected nil for empty context, got %v", state)
		}
	})

	t.Run("returns deep copy that cannot mutate session", func(t *testing.T) {
		ctx := context.Background()
		initial := UserState{
			Name:        "Bob",
			Preferences: map[string]string{"lang": "en"},
		}
		sess, err := New(ctx, WithInitialState(initial))
		if err != nil {
			t.Fatalf("New failed: %v", err)
		}

		ctx = NewContext(ctx, sess)

		// Get state via StateFromContext
		state := StateFromContext(ctx)
		userState := state.(UserState)

		// Modify the returned state
		userState.Name = "Modified"
		userState.Preferences["lang"] = "fr"

		// Verify the session's internal state is unchanged
		originalState := sess.State()
		if originalState.Name != "Bob" {
			t.Errorf("Session state was mutated: expected Name %q, got %q", "Bob", originalState.Name)
		}
		if originalState.Preferences["lang"] != "en" {
			t.Errorf("Session state was mutated: expected lang %q, got %q", "en", originalState.Preferences["lang"])
		}
	})
}

func TestFromContext_NoSession(t *testing.T) {
	ctx := context.Background()

	retrieved := FromContext[UserState](ctx)
	if retrieved != nil {
		t.Errorf("Expected nil for empty context, got %v", retrieved)
	}
}

func TestFromContext_WrongType(t *testing.T) {
	ctx := context.Background()
	// Create session with one type
	type OtherState struct {
		Value string
	}
	sess, err := New(ctx, WithInitialState(OtherState{Value: "test"}))
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	ctx = NewContext(ctx, sess)

	// Try to retrieve with different type
	retrieved := FromContext[UserState](ctx)
	if retrieved != nil {
		t.Errorf("Expected nil for wrong type, got %v", retrieved)
	}
}

func TestInMemoryStore_GetSave(t *testing.T) {
	store := NewInMemoryStore[UserState]()
	ctx := context.Background()

	// Initially empty
	data, err := store.Get(ctx, "test-id")
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if data != nil {
		t.Errorf("Expected nil for non-existent key, got %v", data)
	}

	// Save data
	original := &Data[UserState]{
		ID:    "test-id",
		State: UserState{Name: "Eve", Count: 7},
	}
	if err := store.Save(ctx, "test-id", original); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Retrieve data
	retrieved, err := store.Get(ctx, "test-id")
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if retrieved == nil {
		t.Fatal("Expected data, got nil")
	}
	if retrieved.ID != original.ID {
		t.Errorf("Expected ID %q, got %q", original.ID, retrieved.ID)
	}
	if retrieved.State.Name != original.State.Name {
		t.Errorf("Expected Name %q, got %q", original.State.Name, retrieved.State.Name)
	}
}

func TestInMemoryStore_Isolation(t *testing.T) {
	store := NewInMemoryStore[UserState]()
	ctx := context.Background()

	// Save data
	original := &Data[UserState]{
		ID:    "isolation-test",
		State: UserState{Name: "Frank", Count: 1},
	}
	if err := store.Save(ctx, "isolation-test", original); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Modify original after save
	original.State.Name = "Modified"

	// Retrieved data should not be affected
	retrieved, err := store.Get(ctx, "isolation-test")
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if retrieved.State.Name != "Frank" {
		t.Errorf("Expected Name %q (isolation), got %q", "Frank", retrieved.State.Name)
	}

	// Modify retrieved data
	retrieved.State.Name = "Also Modified"

	// Get again - should still be original
	retrieved2, err := store.Get(ctx, "isolation-test")
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if retrieved2.State.Name != "Frank" {
		t.Errorf("Expected Name %q (isolation), got %q", "Frank", retrieved2.State.Name)
	}
}

func TestInMemoryStore_Overwrite(t *testing.T) {
	store := NewInMemoryStore[UserState]()
	ctx := context.Background()

	// Save initial data
	initial := &Data[UserState]{
		ID:    "overwrite-test",
		State: UserState{Name: "Grace", Count: 1},
	}
	if err := store.Save(ctx, "overwrite-test", initial); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Overwrite with new data
	updated := &Data[UserState]{
		ID:    "overwrite-test",
		State: UserState{Name: "Grace Updated", Count: 2},
	}
	if err := store.Save(ctx, "overwrite-test", updated); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Retrieve and verify
	retrieved, err := store.Get(ctx, "overwrite-test")
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if retrieved.State.Name != "Grace Updated" {
		t.Errorf("Expected Name %q, got %q", "Grace Updated", retrieved.State.Name)
	}
	if retrieved.State.Count != 2 {
		t.Errorf("Expected Count %d, got %d", 2, retrieved.State.Count)
	}
}

func TestSession_ConcurrentAccess(t *testing.T) {
	ctx := context.Background()
	store := NewInMemoryStore[UserState]()
	sess, err := New(ctx,
		WithID[UserState]("concurrent-test"),
		WithInitialState(UserState{Name: "Initial", Count: 0}),
		WithStore(store),
	)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	const numGoroutines = 10
	const numUpdates = 100

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			defer wg.Done()
			for j := 0; j < numUpdates; j++ {
				// Read state
				_ = sess.State()

				// Update state
				_ = sess.UpdateState(ctx, UserState{
					Name:  "Goroutine",
					Count: id*numUpdates + j,
				})
			}
		}(i)
	}

	wg.Wait()

	// Verify no data corruption
	state := sess.State()
	if state.Name != "Goroutine" {
		t.Errorf("Expected Name %q, got %q", "Goroutine", state.Name)
	}
}

func TestInMemoryStore_ConcurrentAccess(t *testing.T) {
	store := NewInMemoryStore[UserState]()
	ctx := context.Background()

	const numGoroutines = 10
	const numOperations = 100

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for i := 0; i < numGoroutines; i++ {
		go func(id int) {
			defer wg.Done()
			key := "shared-key"
			for j := 0; j < numOperations; j++ {
				// Save
				data := &Data[UserState]{
					ID:    key,
					State: UserState{Name: "Concurrent", Count: id*numOperations + j},
				}
				_ = store.Save(ctx, key, data)

				// Get
				_, _ = store.Get(ctx, key)
			}
		}(i)
	}

	wg.Wait()

	// Verify we can still read
	data, err := store.Get(ctx, "shared-key")
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if data == nil {
		t.Fatal("Expected data, got nil")
	}
}

func TestNotFoundError(t *testing.T) {
	err := &NotFoundError{SessionID: "test-123"}

	expected := "session not found: test-123"
	if err.Error() != expected {
		t.Errorf("Expected error message %q, got %q", expected, err.Error())
	}
}

func TestSession_ZeroState(t *testing.T) {
	ctx := context.Background()
	// Create session without initial state
	sess, err := New[UserState](ctx)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	state := sess.State()
	if state.Name != "" {
		t.Errorf("Expected empty Name, got %q", state.Name)
	}
	if state.Count != 0 {
		t.Errorf("Expected zero Count, got %d", state.Count)
	}
	if state.Preferences != nil {
		t.Errorf("Expected nil Preferences, got %v", state.Preferences)
	}
}

func TestSession_ComplexState(t *testing.T) {
	ctx := context.Background()
	type NestedState struct {
		Inner struct {
			Value string `json:"value"`
		} `json:"inner"`
		List []int `json:"list"`
	}

	store := NewInMemoryStore[NestedState]()
	initial := NestedState{
		List: []int{1, 2, 3},
	}
	initial.Inner.Value = "nested"

	sess, err := New(ctx,
		WithID[NestedState]("complex-state"),
		WithInitialState(initial),
		WithStore(store),
	)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	// Update with nested modifications
	newState := NestedState{
		List: []int{4, 5, 6, 7},
	}
	newState.Inner.Value = "updated nested"

	if err := sess.UpdateState(ctx, newState); err != nil {
		t.Fatalf("UpdateState failed: %v", err)
	}

	// Verify nested state is correct
	got := sess.State()
	if got.Inner.Value != "updated nested" {
		t.Errorf("Expected Inner.Value %q, got %q", "updated nested", got.Inner.Value)
	}
	if len(got.List) != 4 {
		t.Errorf("Expected List length %d, got %d", 4, len(got.List))
	}

	// Verify persistence
	data, err := store.Get(ctx, "complex-state")
	if err != nil {
		t.Fatalf("Store.Get failed: %v", err)
	}
	if data.State.Inner.Value != "updated nested" {
		t.Errorf("Store: expected Inner.Value %q, got %q", "updated nested", data.State.Inner.Value)
	}
}

// mockFailingStore is a store that fails on Save for testing error handling.
type mockFailingStore[S any] struct {
	saveErr error
}

func (s *mockFailingStore[S]) Get(_ context.Context, _ string) (*Data[S], error) {
	return nil, nil
}

func (s *mockFailingStore[S]) Save(_ context.Context, _ string, _ *Data[S]) error {
	return s.saveErr
}
func TestNew_StoreError(t *testing.T) {
	ctx := context.Background()
	expectedErr := errors.New("store failure")
	store := &mockFailingStore[UserState]{saveErr: expectedErr}
	_, err := New(ctx,
		WithID[UserState]("error-test"),
		WithStore(store),
	)
	if err == nil {
		t.Fatal("Expected error from failing store")
	}
	if !strings.Contains(err.Error(), "failed to persist initial state") {
		t.Errorf("Expected persist error, got: %v", err)
	}
	if !errors.Is(err, expectedErr) {
		t.Errorf("Expected wrapped error %v, got %v", expectedErr, err)
	}
}

func TestSession_UpdateState_StoreError(t *testing.T) {
	ctx := context.Background()
	store := NewInMemoryStore[UserState]()
	sess, err := New(ctx,
		WithID[UserState]("error-test"),
		WithStore(store),
	)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	expectedErr := errors.New("store failure")
	sess.store = &mockFailingStore[UserState]{saveErr: expectedErr}

	err = sess.UpdateState(ctx, UserState{Name: "Test"})
	if err == nil {
		t.Fatal("Expected error from failing store")
	}
	if err != expectedErr {
		t.Errorf("Expected error %v, got %v", expectedErr, err)
	}
}
