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

package x

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"testing"
	"time"

	"cloud.google.com/go/firestore"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/x/streaming"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/firebase"
	"google.golang.org/api/iterator"
)

var (
	testStreamProjectID  = flag.String("test-stream-project-id", "", "GCP Project ID to use for stream manager tests")
	testStreamCollection = flag.String("test-stream-collection", "genkit-streams", "Firestore collection to use for stream manager tests")
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
 *    go test -test-stream-project-id=<YOUR_PROJECT_ID> -test-stream-collection=genkit-streams
 *    ```
 */

func skipIfNoFirestore(t *testing.T) {
	if *testStreamProjectID == "" {
		t.Skip("Skipping test: -test-stream-project-id flag not provided")
	}
}

func setupTestStreamManager(t *testing.T) (*FirestoreStreamManager, *firestore.Client, func()) {
	skipIfNoFirestore(t)

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&firebase.Firebase{ProjectId: *testStreamProjectID}))

	f := genkit.LookupPlugin(g, "firebase").(*firebase.Firebase)
	client, err := f.Firestore(ctx)
	if err != nil {
		t.Fatalf("Failed to get Firestore client: %v", err)
	}

	manager, err := NewFirestoreStreamManager(ctx, g,
		WithCollection(*testStreamCollection),
	)
	if err != nil {
		t.Fatalf("Failed to create stream manager: %v", err)
	}

	cleanup := func() {
		deleteStreamCollection(ctx, client, *testStreamCollection, t)
	}

	return manager, client, cleanup
}

func deleteStreamCollection(ctx context.Context, client *firestore.Client, collectionName string, t *testing.T) {
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

func TestFirestoreStreamManager_OpenDuplicateFails(t *testing.T) {
	manager, _, cleanup := setupTestStreamManager(t)
	defer cleanup()

	ctx := context.Background()
	streamID := "test-stream-dup"

	_, err := manager.Open(ctx, streamID)
	if err != nil {
		t.Fatalf("First Open failed: %v", err)
	}

	_, err = manager.Open(ctx, streamID)
	if err == nil {
		t.Fatal("Expected error when opening duplicate stream")
	}

	publicErr, ok := err.(*core.UserFacingError)
	if !ok {
		t.Fatalf("Expected UserFacingError, got %T", err)
	}
	if publicErr.Status != core.ALREADY_EXISTS {
		t.Errorf("Expected ALREADY_EXISTS error, got %v", publicErr.Status)
	}
}

func TestFirestoreStreamManager_OpenAndWrite(t *testing.T) {
	manager, client, cleanup := setupTestStreamManager(t)
	defer cleanup()

	ctx := context.Background()
	streamID := "test-stream-open-write"

	stream, err := manager.Open(ctx, streamID)
	if err != nil {
		t.Fatalf("Failed to open stream: %v", err)
	}

	chunk1, _ := json.Marshal(map[string]string{"foo": "bar"})
	chunk2, _ := json.Marshal(map[string]string{"bar": "baz"})

	if err := stream.Write(ctx, chunk1); err != nil {
		t.Fatalf("Failed to write chunk 1: %v", err)
	}
	if err := stream.Write(ctx, chunk2); err != nil {
		t.Fatalf("Failed to write chunk 2: %v", err)
	}

	snapshot, err := client.Collection(*testStreamCollection).Doc(streamID).Get(ctx)
	if err != nil {
		t.Fatalf("Failed to get document: %v", err)
	}

	data := snapshot.Data()
	streamArr, ok := data["stream"].([]interface{})
	if !ok {
		t.Fatalf("Expected stream array, got %T", data["stream"])
	}

	if len(streamArr) != 2 {
		t.Errorf("Expected 2 stream entries, got %d", len(streamArr))
	}

	entry0, _ := streamArr[0].(map[string]interface{})
	if entry0["type"] != streamEventChunk {
		t.Errorf("Expected type 'chunk', got %v", entry0["type"])
	}
	if entry0["uuid"] == nil || entry0["uuid"] == "" {
		t.Error("Expected uuid to be set for chunk")
	}

	if data["expiresAt"] == nil {
		t.Error("Expected expiresAt to be set on open (for abandoned stream cleanup)")
	}
}

func TestFirestoreStreamManager_PreserveDuplicateChunks(t *testing.T) {
	manager, client, cleanup := setupTestStreamManager(t)
	defer cleanup()

	ctx := context.Background()
	streamID := "test-stream-dupes"

	stream, err := manager.Open(ctx, streamID)
	if err != nil {
		t.Fatalf("Failed to open stream: %v", err)
	}

	chunk, _ := json.Marshal(map[string]string{"foo": "bar"})

	if err := stream.Write(ctx, chunk); err != nil {
		t.Fatalf("Failed to write chunk 1: %v", err)
	}
	if err := stream.Write(ctx, chunk); err != nil {
		t.Fatalf("Failed to write chunk 2: %v", err)
	}

	snapshot, err := client.Collection(*testStreamCollection).Doc(streamID).Get(ctx)
	if err != nil {
		t.Fatalf("Failed to get document: %v", err)
	}

	data := snapshot.Data()
	streamArr, _ := data["stream"].([]interface{})

	if len(streamArr) != 2 {
		t.Errorf("Expected 2 stream entries (duplicates should be preserved), got %d", len(streamArr))
	}

	entry0, _ := streamArr[0].(map[string]interface{})
	entry1, _ := streamArr[1].(map[string]interface{})
	if entry0["uuid"] == entry1["uuid"] {
		t.Error("UUIDs should be different for duplicate chunks")
	}
}

func TestFirestoreStreamManager_Done(t *testing.T) {
	manager, client, cleanup := setupTestStreamManager(t)
	defer cleanup()

	ctx := context.Background()
	streamID := "test-stream-done"

	stream, err := manager.Open(ctx, streamID)
	if err != nil {
		t.Fatalf("Failed to open stream: %v", err)
	}

	output, _ := json.Marshal(map[string]string{"result": "success"})
	if err := stream.Done(ctx, output); err != nil {
		t.Fatalf("Failed to mark stream done: %v", err)
	}

	snapshot, err := client.Collection(*testStreamCollection).Doc(streamID).Get(ctx)
	if err != nil {
		t.Fatalf("Failed to get document: %v", err)
	}

	data := snapshot.Data()
	streamArr, _ := data["stream"].([]interface{})

	if len(streamArr) != 1 {
		t.Errorf("Expected 1 stream entry, got %d", len(streamArr))
	}

	entry, _ := streamArr[0].(map[string]interface{})
	if entry["type"] != streamEventDone {
		t.Errorf("Expected type 'done', got %v", entry["type"])
	}

	if data["expiresAt"] == nil {
		t.Error("Expected expiresAt to be set after done")
	}
}

func TestFirestoreStreamManager_Error(t *testing.T) {
	manager, client, cleanup := setupTestStreamManager(t)
	defer cleanup()

	ctx := context.Background()
	streamID := "test-stream-error"

	stream, err := manager.Open(ctx, streamID)
	if err != nil {
		t.Fatalf("Failed to open stream: %v", err)
	}

	testError := errors.New("test error message")
	if err := stream.Error(ctx, testError); err != nil {
		t.Fatalf("Failed to mark stream error: %v", err)
	}

	snapshot, err := client.Collection(*testStreamCollection).Doc(streamID).Get(ctx)
	if err != nil {
		t.Fatalf("Failed to get document: %v", err)
	}

	data := snapshot.Data()
	streamArr, _ := data["stream"].([]interface{})

	if len(streamArr) != 1 {
		t.Errorf("Expected 1 stream entry, got %d", len(streamArr))
	}

	entry, _ := streamArr[0].(map[string]interface{})
	if entry["type"] != streamEventError {
		t.Errorf("Expected type 'error', got %v", entry["type"])
	}

	errData, _ := entry["err"].(map[string]interface{})
	if errData["message"] != "test error message" {
		t.Errorf("Expected error message 'test error message', got %v", errData["message"])
	}
	if errData["status"] != string(core.UNKNOWN) {
		t.Errorf("Expected status UNKNOWN for plain error, got %v", errData["status"])
	}

	if data["expiresAt"] == nil {
		t.Error("Expected expiresAt to be set after error")
	}
}

func TestFirestoreStreamManager_ErrorStatusPreserved(t *testing.T) {
	manager, client, cleanup := setupTestStreamManager(t)
	defer cleanup()

	ctx := context.Background()
	streamID := "test-stream-error-status"

	stream, err := manager.Open(ctx, streamID)
	if err != nil {
		t.Fatalf("Failed to open stream: %v", err)
	}

	testError := core.NewPublicError(core.INVALID_ARGUMENT, "invalid input", nil)
	if err := stream.Error(ctx, testError); err != nil {
		t.Fatalf("Failed to mark stream error: %v", err)
	}

	snapshot, err := client.Collection(*testStreamCollection).Doc(streamID).Get(ctx)
	if err != nil {
		t.Fatalf("Failed to get document: %v", err)
	}

	data := snapshot.Data()
	streamArr, _ := data["stream"].([]interface{})
	entry, _ := streamArr[0].(map[string]interface{})
	errData, _ := entry["err"].(map[string]interface{})

	if errData["status"] != string(core.INVALID_ARGUMENT) {
		t.Errorf("Expected status INVALID_ARGUMENT, got %v", errData["status"])
	}
	if errData["message"] != "invalid input" {
		t.Errorf("Expected message 'invalid input', got %v", errData["message"])
	}
}

func TestFirestoreStreamManager_Subscribe(t *testing.T) {
	manager, client, cleanup := setupTestStreamManager(t)
	defer cleanup()

	ctx := context.Background()
	streamID := "test-stream-subscribe"

	chunk1, _ := json.Marshal(map[string]string{"foo": "bar"})
	chunk2, _ := json.Marshal(map[string]string{"bar": "baz"})
	output, _ := json.Marshal(map[string]string{"result": "success"})

	_, err := client.Collection(*testStreamCollection).Doc(streamID).Set(ctx, map[string]interface{}{
		"stream": []map[string]interface{}{
			{"type": "chunk", "chunk": chunk1, "uuid": "uuid1"},
			{"type": "chunk", "chunk": chunk2, "uuid": "uuid2"},
			{"type": "done", "output": output},
		},
		"createdAt": time.Now(),
		"updatedAt": time.Now(),
	})
	if err != nil {
		t.Fatalf("Failed to create test document: %v", err)
	}

	ch, unsubscribe, err := manager.Subscribe(ctx, streamID)
	if err != nil {
		t.Fatalf("Failed to subscribe: %v", err)
	}
	defer unsubscribe()

	var chunks []json.RawMessage
	var finalOutput json.RawMessage
	timeout := time.After(5 * time.Second)

	for {
		select {
		case event, ok := <-ch:
			if !ok {
				goto verify
			}
			switch event.Type {
			case streaming.StreamEventChunk:
				chunks = append(chunks, event.Chunk)
			case streaming.StreamEventDone:
				finalOutput = event.Output
				goto verify
			case streaming.StreamEventError:
				t.Fatalf("Unexpected error: %v", event.Err)
			}
		case <-timeout:
			t.Fatal("Timeout waiting for stream events")
		}
	}

verify:
	if len(chunks) != 2 {
		t.Errorf("Expected 2 chunks, got %d", len(chunks))
	}
	if finalOutput == nil {
		t.Error("Expected final output")
	}
}

func TestFirestoreStreamManager_SubscribeErrorStatusPreserved(t *testing.T) {
	manager, client, cleanup := setupTestStreamManager(t)
	defer cleanup()

	ctx := context.Background()
	streamID := "test-stream-sub-error-status"

	_, err := client.Collection(*testStreamCollection).Doc(streamID).Set(ctx, map[string]interface{}{
		"stream": []map[string]interface{}{
			{"type": "error", "err": map[string]interface{}{
				"status":  string(core.INVALID_ARGUMENT),
				"message": "bad input",
			}},
		},
		"createdAt": time.Now(),
		"updatedAt": time.Now(),
	})
	if err != nil {
		t.Fatalf("Failed to create test document: %v", err)
	}

	ch, unsubscribe, err := manager.Subscribe(ctx, streamID)
	if err != nil {
		t.Fatalf("Failed to subscribe: %v", err)
	}
	defer unsubscribe()

	timeout := time.After(5 * time.Second)
	select {
	case event, ok := <-ch:
		if !ok {
			t.Fatal("Channel closed unexpectedly")
		}
		if event.Type != streaming.StreamEventError {
			t.Fatalf("Expected error event, got %v", event.Type)
		}
		publicErr, ok := event.Err.(*core.UserFacingError)
		if !ok {
			t.Fatalf("Expected UserFacingError, got %T", event.Err)
		}
		if publicErr.Status != core.INVALID_ARGUMENT {
			t.Errorf("Expected INVALID_ARGUMENT status, got %v", publicErr.Status)
		}
	case <-timeout:
		t.Fatal("Timeout waiting for error event")
	}
}

func TestFirestoreStreamManager_SubscribeNotFound(t *testing.T) {
	manager, _, cleanup := setupTestStreamManager(t)
	defer cleanup()

	ctx := context.Background()
	_, _, err := manager.Subscribe(ctx, "non-existent-stream")
	if err == nil {
		t.Fatal("Expected error for non-existent stream")
	}

	publicErr, ok := err.(*core.UserFacingError)
	if !ok {
		t.Fatalf("Expected UserFacingError, got %T", err)
	}
	if publicErr.Status != core.NOT_FOUND {
		t.Errorf("Expected NOT_FOUND error, got %v", publicErr.Status)
	}
}

func TestFirestoreStreamManager_Timeout(t *testing.T) {
	skipIfNoFirestore(t)

	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&firebase.Firebase{ProjectId: *testStreamProjectID}))

	f := genkit.LookupPlugin(g, "firebase").(*firebase.Firebase)
	client, err := f.Firestore(ctx)
	if err != nil {
		t.Fatalf("Failed to get Firestore client: %v", err)
	}
	defer deleteStreamCollection(ctx, client, *testStreamCollection, t)

	manager, err := NewFirestoreStreamManager(ctx, g,
		WithCollection(*testStreamCollection),
		WithTimeout(100*time.Millisecond),
	)
	if err != nil {
		t.Fatalf("Failed to create stream manager: %v", err)
	}

	streamID := "test-stream-timeout"

	_, err = manager.Open(ctx, streamID)
	if err != nil {
		t.Fatalf("Failed to open stream: %v", err)
	}

	ch, _, err := manager.Subscribe(ctx, streamID)
	if err != nil {
		t.Fatalf("Failed to subscribe: %v", err)
	}

	timeout := time.After(2 * time.Second)
	for {
		select {
		case event, ok := <-ch:
			if !ok {
				t.Fatal("Channel closed without timeout error")
				return
			}
			if event.Type == streaming.StreamEventError {
				publicErr, ok := event.Err.(*core.UserFacingError)
				if !ok {
					t.Fatalf("Expected UserFacingError, got %T", event.Err)
				}
				if publicErr.Status != core.DEADLINE_EXCEEDED {
					t.Errorf("Expected DEADLINE_EXCEEDED, got %v", publicErr.Status)
				}
				return
			}
		case <-timeout:
			t.Fatal("Test timeout - stream timeout didn't trigger")
		}
	}
}

func TestFirestoreStreamManager_WriteAfterClose(t *testing.T) {
	manager, _, cleanup := setupTestStreamManager(t)
	defer cleanup()

	ctx := context.Background()
	streamID := "test-stream-write-after-close"

	stream, err := manager.Open(ctx, streamID)
	if err != nil {
		t.Fatalf("Failed to open stream: %v", err)
	}

	if err := stream.Close(); err != nil {
		t.Fatalf("Failed to close stream: %v", err)
	}

	chunk, _ := json.Marshal(map[string]string{"foo": "bar"})
	err = stream.Write(ctx, chunk)
	if err == nil {
		t.Fatal("Expected error when writing after close")
	}

	publicErr, ok := err.(*core.UserFacingError)
	if !ok {
		t.Fatalf("Expected UserFacingError, got %T", err)
	}
	if publicErr.Status != core.FAILED_PRECONDITION {
		t.Errorf("Expected FAILED_PRECONDITION, got %v", publicErr.Status)
	}
}
