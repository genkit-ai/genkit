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

package x

import (
	"context"
	"flag"
	"testing"
	"time"

	"cloud.google.com/go/firestore"
	"github.com/firebase/genkit/go/core/x/session"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/firebase"
	"google.golang.org/api/iterator"
)

var (
	testSessionProjectID  = flag.String("test-session-project-id", "", "GCP Project ID to use for session store tests")
	testSessionCollection = flag.String("test-session-collection", "genkit-sessions", "Firestore collection to use for session store tests")
)

/*
 * Pre-requisites to run this test:
 *
 * 1. **Option A - Use Firestore Emulator (Recommended for local development):**
 *    Start the Firestore emulator:
 *    ```bash
 *    export FIRESTORE_EMULATOR_HOST=127.0.0.1:8080
 *    gcloud emulators firestore start --host-port=127.0.0.1:8080
 *    ```
 *
 * 2. **Option B - Use a Real Firestore Database:**
 *    - Set up a Firebase project with Firestore enabled
 *    - Authenticate using:
 *      ```bash
 *      gcloud auth application-default login
 *      ```
 *
 * 3. **Running the Test:**
 *    ```bash
 *    go test -test-session-project-id=<YOUR_PROJECT_ID> -test-session-collection=genkit-sessions
 *    ```
 */

// TestState is a test state type with various field types.
type TestState struct {
	Name        string            `json:"name"`
	Count       int               `json:"count"`
	Preferences map[string]string `json:"preferences,omitempty"`
}

func skipIfNoFirestoreSession(t *testing.T) {
	if *testSessionProjectID == "" {
		t.Skip("Skipping test: -test-session-project-id flag not provided")
	}
}

func setupTestSessionStore(t *testing.T) (*FirestoreSessionStore[TestState], *firestore.Client, func()) {
	skipIfNoFirestoreSession(t)

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&firebase.Firebase{ProjectId: *testSessionProjectID}))

	f := genkit.LookupPlugin(g, "firebase").(*firebase.Firebase)
	client, err := f.Firestore(ctx)
	if err != nil {
		t.Fatalf("Failed to get Firestore client: %v", err)
	}

	store, err := NewFirestoreSessionStore[TestState](ctx, g,
		WithCollection(*testSessionCollection),
	)
	if err != nil {
		t.Fatalf("Failed to create session store: %v", err)
	}

	cleanup := func() {
		deleteSessionCollection(ctx, client, *testSessionCollection, t)
	}

	return store, client, cleanup
}

func deleteSessionCollection(ctx context.Context, client *firestore.Client, collectionName string, t *testing.T) {
	iter := client.Collection(collectionName).Documents(ctx)
	for {
		doc, err := iter.Next()
		if err == iterator.Done {
			break
		}
		if err != nil {
			t.Logf("Failed to iterate documents for deletion: %v", err)
			return
		}
		_, err = doc.Ref.Delete(ctx)
		if err != nil {
			t.Logf("Failed to delete document %s: %v", doc.Ref.ID, err)
		}
	}
}

func TestNewFirestoreSessionStore_MissingCollection(t *testing.T) {
	skipIfNoFirestoreSession(t)

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&firebase.Firebase{ProjectId: *testSessionProjectID}))

	_, err := NewFirestoreSessionStore[TestState](ctx, g)
	if err == nil {
		t.Fatal("Expected error when collection is missing")
	}
}

func TestFirestoreSessionStore_SaveAndGet(t *testing.T) {
	store, _, cleanup := setupTestSessionStore(t)
	defer cleanup()

	ctx := context.Background()
	sessionID := "test-session-save-get"

	// Initially empty
	data, err := store.Get(ctx, sessionID)
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if data != nil {
		t.Errorf("Expected nil for non-existent session, got %v", data)
	}

	// Save data
	original := &session.Data[TestState]{
		ID: sessionID,
		State: TestState{
			Name:        "Alice",
			Count:       42,
			Preferences: map[string]string{"theme": "dark"},
		},
	}
	if err := store.Save(ctx, sessionID, original); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Retrieve data
	retrieved, err := store.Get(ctx, sessionID)
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if retrieved == nil {
		t.Fatal("Expected data, got nil")
	}
	if retrieved.ID != sessionID {
		t.Errorf("Expected ID %q, got %q", sessionID, retrieved.ID)
	}
	if retrieved.State.Name != original.State.Name {
		t.Errorf("Expected Name %q, got %q", original.State.Name, retrieved.State.Name)
	}
	if retrieved.State.Count != original.State.Count {
		t.Errorf("Expected Count %d, got %d", original.State.Count, retrieved.State.Count)
	}
	if retrieved.State.Preferences["theme"] != "dark" {
		t.Errorf("Expected theme %q, got %q", "dark", retrieved.State.Preferences["theme"])
	}
}

func TestFirestoreSessionStore_Overwrite(t *testing.T) {
	store, client, cleanup := setupTestSessionStore(t)
	defer cleanup()

	ctx := context.Background()
	sessionID := "test-session-overwrite"

	// Save initial data
	initial := &session.Data[TestState]{
		ID:    sessionID,
		State: TestState{Name: "Alice", Count: 1},
	}
	if err := store.Save(ctx, sessionID, initial); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Get the initial document to capture CreatedAt and UpdatedAt
	snapshot1, err := client.Collection(*testSessionCollection).Doc(sessionID).Get(ctx)
	if err != nil {
		t.Fatalf("Failed to get initial document: %v", err)
	}
	initialData := snapshot1.Data()
	initialCreatedAt, ok := initialData["createdAt"].(time.Time)
	if !ok {
		t.Fatal("Expected createdAt to be a timestamp")
	}
	initialUpdatedAt, ok := initialData["updatedAt"].(time.Time)
	if !ok {
		t.Fatal("Expected updatedAt to be a timestamp")
	}

	// Wait a moment to ensure timestamp difference is detectable
	time.Sleep(10 * time.Millisecond)

	// Overwrite with new data
	updated := &session.Data[TestState]{
		ID:    sessionID,
		State: TestState{Name: "Alice Updated", Count: 2},
	}
	if err := store.Save(ctx, sessionID, updated); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Get the updated document to verify timestamps
	snapshot2, err := client.Collection(*testSessionCollection).Doc(sessionID).Get(ctx)
	if err != nil {
		t.Fatalf("Failed to get updated document: %v", err)
	}
	updatedData := snapshot2.Data()
	updatedCreatedAt, ok := updatedData["createdAt"].(time.Time)
	if !ok {
		t.Fatal("Expected createdAt to be a timestamp after update")
	}
	updatedUpdatedAt, ok := updatedData["updatedAt"].(time.Time)
	if !ok {
		t.Fatal("Expected updatedAt to be a timestamp after update")
	}

	// Verify CreatedAt is preserved (not modified during overwrite)
	if !updatedCreatedAt.Equal(initialCreatedAt) {
		t.Errorf("CreatedAt was modified during overwrite: initial=%v, after=%v", initialCreatedAt, updatedCreatedAt)
	}

	// Verify UpdatedAt is modified (should be later than initial)
	if !updatedUpdatedAt.After(initialUpdatedAt) {
		t.Errorf("UpdatedAt should be later after overwrite: initial=%v, after=%v", initialUpdatedAt, updatedUpdatedAt)
	}

	// Retrieve and verify state data
	retrieved, err := store.Get(ctx, sessionID)
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if retrieved.State.Name != "Alice Updated" {
		t.Errorf("Expected Name %q, got %q", "Alice Updated", retrieved.State.Name)
	}
	if retrieved.State.Count != 2 {
		t.Errorf("Expected Count %d, got %d", 2, retrieved.State.Count)
	}
}

func TestFirestoreSessionStore_ExpiresAt(t *testing.T) {
	store, client, cleanup := setupTestSessionStore(t)
	defer cleanup()

	ctx := context.Background()
	sessionID := "test-session-expires"

	data := &session.Data[TestState]{
		ID:    sessionID,
		State: TestState{Name: "ExpiresTest"},
	}
	if err := store.Save(ctx, sessionID, data); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Verify expiresAt is set in Firestore
	snapshot, err := client.Collection(*testSessionCollection).Doc(sessionID).Get(ctx)
	if err != nil {
		t.Fatalf("Failed to get document: %v", err)
	}

	docData := snapshot.Data()
	if docData["expiresAt"] == nil {
		t.Error("Expected expiresAt to be set")
	}
}

func TestFirestoreSessionStore_WithTTL(t *testing.T) {
	skipIfNoFirestoreSession(t)

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&firebase.Firebase{ProjectId: *testSessionProjectID}))

	f := genkit.LookupPlugin(g, "firebase").(*firebase.Firebase)
	client, err := f.Firestore(ctx)
	if err != nil {
		t.Fatalf("Failed to get Firestore client: %v", err)
	}
	defer deleteSessionCollection(ctx, client, *testSessionCollection, t)

	customTTL := 1 * time.Hour
	store, err := NewFirestoreSessionStore[TestState](ctx, g,
		WithCollection(*testSessionCollection),
		WithTTL(customTTL),
	)
	if err != nil {
		t.Fatalf("Failed to create session store: %v", err)
	}

	if store.ttl != customTTL {
		t.Errorf("Expected TTL %v, got %v", customTTL, store.ttl)
	}
}

func TestFirestoreSessionStore_EmptyState(t *testing.T) {
	store, _, cleanup := setupTestSessionStore(t)
	defer cleanup()

	ctx := context.Background()
	sessionID := "test-session-empty"

	// Save session with zero-value state
	data := &session.Data[TestState]{
		ID:    sessionID,
		State: TestState{},
	}
	if err := store.Save(ctx, sessionID, data); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Retrieve and verify
	retrieved, err := store.Get(ctx, sessionID)
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if retrieved == nil {
		t.Fatal("Expected data, got nil")
	}
	if retrieved.State.Name != "" {
		t.Errorf("Expected empty Name, got %q", retrieved.State.Name)
	}
	if retrieved.State.Count != 0 {
		t.Errorf("Expected zero Count, got %d", retrieved.State.Count)
	}
}

func TestFirestoreSessionStore_ComplexState(t *testing.T) {
	skipIfNoFirestoreSession(t)

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&firebase.Firebase{ProjectId: *testSessionProjectID}))

	f := genkit.LookupPlugin(g, "firebase").(*firebase.Firebase)
	client, err := f.Firestore(ctx)
	if err != nil {
		t.Fatalf("Failed to get Firestore client: %v", err)
	}
	defer deleteSessionCollection(ctx, client, *testSessionCollection, t)

	type NestedState struct {
		Inner struct {
			Value string `json:"value"`
		} `json:"inner"`
		List []int `json:"list"`
	}

	store, err := NewFirestoreSessionStore[NestedState](ctx, g,
		WithCollection(*testSessionCollection),
	)
	if err != nil {
		t.Fatalf("Failed to create session store: %v", err)
	}

	sessionID := "test-session-complex"

	// Save complex state
	state := NestedState{
		List: []int{1, 2, 3, 4, 5},
	}
	state.Inner.Value = "nested value"

	data := &session.Data[NestedState]{
		ID:    sessionID,
		State: state,
	}
	if err := store.Save(ctx, sessionID, data); err != nil {
		t.Fatalf("Save failed: %v", err)
	}

	// Retrieve and verify
	retrieved, err := store.Get(ctx, sessionID)
	if err != nil {
		t.Fatalf("Get failed: %v", err)
	}
	if retrieved == nil {
		t.Fatal("Expected data, got nil")
	}
	if retrieved.State.Inner.Value != "nested value" {
		t.Errorf("Expected Inner.Value %q, got %q", "nested value", retrieved.State.Inner.Value)
	}
	if len(retrieved.State.List) != 5 {
		t.Errorf("Expected List length %d, got %d", 5, len(retrieved.State.List))
	}
}

func TestFirestoreSessionStore_IntegrationWithSession(t *testing.T) {
	store, _, cleanup := setupTestSessionStore(t)
	defer cleanup()

	ctx := context.Background()

	// Create a session with the Firestore store
	sess, err := session.New(ctx,
		session.WithID[TestState]("integration-test"),
		session.WithInitialState(TestState{Name: "Integration", Count: 0}),
		session.WithStore(store),
	)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}

	// Update state (should persist to Firestore)
	if err := sess.UpdateState(ctx, TestState{Name: "Updated", Count: 10}); err != nil {
		t.Fatalf("UpdateState failed: %v", err)
	}

	// Load session from store
	loaded, err := session.Load(ctx, store, "integration-test")
	if err != nil {
		t.Fatalf("Load failed: %v", err)
	}

	if loaded.State().Name != "Updated" {
		t.Errorf("Expected Name %q, got %q", "Updated", loaded.State().Name)
	}
	if loaded.State().Count != 10 {
		t.Errorf("Expected Count %d, got %d", 10, loaded.State().Count)
	}
}
