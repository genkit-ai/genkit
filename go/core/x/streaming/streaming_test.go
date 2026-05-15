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

package streaming

import (
	"context"
	"encoding/json"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/firebase/genkit/go/core"
)

func TestInMemoryStreamManager_OpenAndSubscribe(t *testing.T) {
	m := NewInMemoryStreamManager()
	defer m.Close()

	ctx := context.Background()
	streamID := "test-stream-1"

	// Open a new stream
	writer, err := m.Open(ctx, streamID)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	if writer == nil {
		t.Fatal("Open returned nil writer")
	}

	// Subscribe to the stream
	events, unsubscribe, err := m.Subscribe(ctx, streamID)
	if err != nil {
		t.Fatalf("Subscribe failed: %v", err)
	}
	defer unsubscribe()

	if events == nil {
		t.Fatal("Subscribe returned nil channel")
	}
}

func TestInMemoryStreamManager_OpenDuplicateFails(t *testing.T) {
	m := NewInMemoryStreamManager()
	defer m.Close()

	ctx := context.Background()
	streamID := "test-stream-dup"

	// Open first stream
	_, err := m.Open(ctx, streamID)
	if err != nil {
		t.Fatalf("First Open failed: %v", err)
	}

	// Try to open duplicate
	_, err = m.Open(ctx, streamID)
	if err == nil {
		t.Fatal("Expected error when opening duplicate stream")
	}

	var ufErr *core.UserFacingError
	if !errors.As(err, &ufErr) {
		t.Fatalf("Expected UserFacingError, got %T", err)
	}
	if ufErr.Status != core.ALREADY_EXISTS {
		t.Errorf("Expected ALREADY_EXISTS status, got %v", ufErr.Status)
	}
}

func TestInMemoryStreamManager_SubscribeNonExistent(t *testing.T) {
	m := NewInMemoryStreamManager()
	defer m.Close()

	ctx := context.Background()

	_, _, err := m.Subscribe(ctx, "non-existent")
	if err == nil {
		t.Fatal("Expected error when subscribing to non-existent stream")
	}

	var ufErr *core.UserFacingError
	if !errors.As(err, &ufErr) {
		t.Fatalf("Expected UserFacingError, got %T", err)
	}
	if ufErr.Status != core.NOT_FOUND {
		t.Errorf("Expected NOT_FOUND status, got %v", ufErr.Status)
	}
}

func TestInMemoryStreamManager_WriteAndReceiveChunks(t *testing.T) {
	m := NewInMemoryStreamManager()
	defer m.Close()

	ctx := context.Background()
	streamID := "test-stream-chunks"

	writer, err := m.Open(ctx, streamID)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	events, unsubscribe, err := m.Subscribe(ctx, streamID)
	if err != nil {
		t.Fatalf("Subscribe failed: %v", err)
	}
	defer unsubscribe()

	// Write chunks
	chunks := []string{"chunk1", "chunk2", "chunk3"}
	for _, chunk := range chunks {
		if err := writer.Write(ctx, json.RawMessage(`"`+chunk+`"`)); err != nil {
			t.Fatalf("Write failed: %v", err)
		}
	}

	// Read chunks
	for i, expected := range chunks {
		select {
		case event := <-events:
			if event.Type != StreamEventChunk {
				t.Errorf("Expected chunk event, got %v", event.Type)
			}
			var got string
			if err := json.Unmarshal(event.Chunk, &got); err != nil {
				t.Fatalf("Failed to unmarshal chunk: %v", err)
			}
			if got != expected {
				t.Errorf("Chunk %d: expected %q, got %q", i, expected, got)
			}
		case <-time.After(time.Second):
			t.Fatalf("Timeout waiting for chunk %d", i)
		}
	}
}

func TestInMemoryStreamManager_Done(t *testing.T) {
	m := NewInMemoryStreamManager()
	defer m.Close()

	ctx := context.Background()
	streamID := "test-stream-done"

	writer, err := m.Open(ctx, streamID)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	events, unsubscribe, err := m.Subscribe(ctx, streamID)
	if err != nil {
		t.Fatalf("Subscribe failed: %v", err)
	}
	defer unsubscribe()

	// Write a chunk
	if err := writer.Write(ctx, json.RawMessage(`"test-chunk"`)); err != nil {
		t.Fatalf("Write failed: %v", err)
	}

	// Mark as done
	output := json.RawMessage(`{"result": "success"}`)
	if err := writer.Done(ctx, output); err != nil {
		t.Fatalf("Done failed: %v", err)
	}

	// Should receive chunk then done
	select {
	case event := <-events:
		if event.Type != StreamEventChunk {
			t.Errorf("Expected chunk event first, got %v", event.Type)
		}
	case <-time.After(time.Second):
		t.Fatal("Timeout waiting for chunk")
	}

	select {
	case event := <-events:
		if event.Type != StreamEventDone {
			t.Errorf("Expected done event, got %v", event.Type)
		}
		if string(event.Output) != string(output) {
			t.Errorf("Expected output %s, got %s", output, event.Output)
		}
	case <-time.After(time.Second):
		t.Fatal("Timeout waiting for done event")
	}
}

func TestInMemoryStreamManager_Error(t *testing.T) {
	m := NewInMemoryStreamManager()
	defer m.Close()

	ctx := context.Background()
	streamID := "test-stream-error"

	writer, err := m.Open(ctx, streamID)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	events, unsubscribe, err := m.Subscribe(ctx, streamID)
	if err != nil {
		t.Fatalf("Subscribe failed: %v", err)
	}
	defer unsubscribe()

	// Mark as error
	streamErr := core.NewPublicError(core.INTERNAL, "test error", nil)
	if err := writer.Error(ctx, streamErr); err != nil {
		t.Fatalf("Error failed: %v", err)
	}

	select {
	case event := <-events:
		if event.Type != StreamEventError {
			t.Errorf("Expected error event, got %v", event.Type)
		}
		if event.Err == nil {
			t.Error("Expected error to be set")
		}
	case <-time.After(time.Second):
		t.Fatal("Timeout waiting for error event")
	}
}

func TestInMemoryStreamManager_WriteAfterDone(t *testing.T) {
	m := NewInMemoryStreamManager()
	defer m.Close()

	ctx := context.Background()
	streamID := "test-stream-write-after-done"

	writer, err := m.Open(ctx, streamID)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	if err := writer.Done(ctx, json.RawMessage(`"done"`)); err != nil {
		t.Fatalf("Done failed: %v", err)
	}

	// Try to write after done
	err = writer.Write(ctx, json.RawMessage(`"chunk"`))
	if err == nil {
		t.Fatal("Expected error when writing after done")
	}

	var ufErr *core.UserFacingError
	if !errors.As(err, &ufErr) {
		t.Fatalf("Expected UserFacingError, got %T", err)
	}
	if ufErr.Status != core.FAILED_PRECONDITION {
		t.Errorf("Expected FAILED_PRECONDITION status, got %v", ufErr.Status)
	}
}

func TestInMemoryStreamManager_WriteAfterClose(t *testing.T) {
	m := NewInMemoryStreamManager()
	defer m.Close()

	ctx := context.Background()
	streamID := "test-stream-write-after-close"

	writer, err := m.Open(ctx, streamID)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	if err := writer.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	// Try to write after close
	err = writer.Write(ctx, json.RawMessage(`"chunk"`))
	if err == nil {
		t.Fatal("Expected error when writing after close")
	}

	var ufErr *core.UserFacingError
	if !errors.As(err, &ufErr) {
		t.Fatalf("Expected UserFacingError, got %T", err)
	}
	if ufErr.Status != core.FAILED_PRECONDITION {
		t.Errorf("Expected FAILED_PRECONDITION status, got %v", ufErr.Status)
	}
}

func TestInMemoryStreamManager_DoneAfterError(t *testing.T) {
	m := NewInMemoryStreamManager()
	defer m.Close()

	ctx := context.Background()
	streamID := "test-stream-done-after-error"

	writer, err := m.Open(ctx, streamID)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	if err := writer.Error(ctx, core.NewPublicError(core.INTERNAL, "test", nil)); err != nil {
		t.Fatalf("Error failed: %v", err)
	}

	// Try to mark done after error
	err = writer.Done(ctx, json.RawMessage(`"done"`))
	if err == nil {
		t.Fatal("Expected error when calling Done after Error")
	}
}

func TestInMemoryStreamManager_MultipleSubscribers(t *testing.T) {
	m := NewInMemoryStreamManager()
	defer m.Close()

	ctx := context.Background()
	streamID := "test-stream-multi-sub"

	writer, err := m.Open(ctx, streamID)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	// Create multiple subscribers
	events1, unsub1, err := m.Subscribe(ctx, streamID)
	if err != nil {
		t.Fatalf("Subscribe 1 failed: %v", err)
	}
	defer unsub1()

	events2, unsub2, err := m.Subscribe(ctx, streamID)
	if err != nil {
		t.Fatalf("Subscribe 2 failed: %v", err)
	}
	defer unsub2()

	// Write a chunk
	chunk := json.RawMessage(`"shared-chunk"`)
	if err := writer.Write(ctx, chunk); err != nil {
		t.Fatalf("Write failed: %v", err)
	}

	// Both subscribers should receive the chunk
	for i, events := range []<-chan StreamEvent{events1, events2} {
		select {
		case event := <-events:
			if event.Type != StreamEventChunk {
				t.Errorf("Subscriber %d: expected chunk event, got %v", i+1, event.Type)
			}
			if string(event.Chunk) != string(chunk) {
				t.Errorf("Subscriber %d: expected chunk %s, got %s", i+1, chunk, event.Chunk)
			}
		case <-time.After(time.Second):
			t.Fatalf("Subscriber %d: timeout waiting for chunk", i+1)
		}
	}
}

func TestInMemoryStreamManager_LateSubscriberGetsBufferedChunks(t *testing.T) {
	m := NewInMemoryStreamManager()
	defer m.Close()

	ctx := context.Background()
	streamID := "test-stream-late-sub"

	writer, err := m.Open(ctx, streamID)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	// Write chunks before any subscriber
	chunks := []string{"early1", "early2"}
	for _, chunk := range chunks {
		if err := writer.Write(ctx, json.RawMessage(`"`+chunk+`"`)); err != nil {
			t.Fatalf("Write failed: %v", err)
		}
	}

	// Late subscriber joins
	events, unsubscribe, err := m.Subscribe(ctx, streamID)
	if err != nil {
		t.Fatalf("Subscribe failed: %v", err)
	}
	defer unsubscribe()

	// Should receive buffered chunks
	for i, expected := range chunks {
		select {
		case event := <-events:
			if event.Type != StreamEventChunk {
				t.Errorf("Expected chunk event, got %v", event.Type)
			}
			var got string
			if err := json.Unmarshal(event.Chunk, &got); err != nil {
				t.Fatalf("Failed to unmarshal chunk: %v", err)
			}
			if got != expected {
				t.Errorf("Chunk %d: expected %q, got %q", i, expected, got)
			}
		case <-time.After(time.Second):
			t.Fatalf("Timeout waiting for buffered chunk %d", i)
		}
	}
}

func TestInMemoryStreamManager_SubscribeToCompletedStream(t *testing.T) {
	m := NewInMemoryStreamManager()
	defer m.Close()

	ctx := context.Background()
	streamID := "test-stream-completed"

	writer, err := m.Open(ctx, streamID)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	// Write and complete before subscribing
	if err := writer.Write(ctx, json.RawMessage(`"chunk1"`)); err != nil {
		t.Fatalf("Write failed: %v", err)
	}
	if err := writer.Write(ctx, json.RawMessage(`"chunk2"`)); err != nil {
		t.Fatalf("Write failed: %v", err)
	}
	output := json.RawMessage(`{"final": true}`)
	if err := writer.Done(ctx, output); err != nil {
		t.Fatalf("Done failed: %v", err)
	}

	// Subscribe after completion
	events, unsubscribe, err := m.Subscribe(ctx, streamID)
	if err != nil {
		t.Fatalf("Subscribe failed: %v", err)
	}
	defer unsubscribe()

	// Should receive all buffered chunks
	for i := 0; i < 2; i++ {
		select {
		case event := <-events:
			if event.Type != StreamEventChunk {
				t.Errorf("Expected chunk event %d, got %v", i, event.Type)
			}
		case <-time.After(time.Second):
			t.Fatalf("Timeout waiting for chunk %d", i)
		}
	}

	// Should receive done event
	select {
	case event := <-events:
		if event.Type != StreamEventDone {
			t.Errorf("Expected done event, got %v", event.Type)
		}
		if string(event.Output) != string(output) {
			t.Errorf("Expected output %s, got %s", output, event.Output)
		}
	case <-time.After(time.Second):
		t.Fatal("Timeout waiting for done event")
	}

	// Channel should be closed
	select {
	case _, ok := <-events:
		if ok {
			t.Error("Expected channel to be closed")
		}
	case <-time.After(100 * time.Millisecond):
		t.Error("Channel not closed after done")
	}
}

func TestInMemoryStreamManager_SubscribeToErroredStream(t *testing.T) {
	m := NewInMemoryStreamManager()
	defer m.Close()

	ctx := context.Background()
	streamID := "test-stream-errored"

	writer, err := m.Open(ctx, streamID)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	// Write and error before subscribing
	if err := writer.Write(ctx, json.RawMessage(`"chunk1"`)); err != nil {
		t.Fatalf("Write failed: %v", err)
	}
	streamErr := core.NewPublicError(core.INTERNAL, "test error", nil)
	if err := writer.Error(ctx, streamErr); err != nil {
		t.Fatalf("Error failed: %v", err)
	}

	// Subscribe after error
	events, unsubscribe, err := m.Subscribe(ctx, streamID)
	if err != nil {
		t.Fatalf("Subscribe failed: %v", err)
	}
	defer unsubscribe()

	// Should receive buffered chunk
	select {
	case event := <-events:
		if event.Type != StreamEventChunk {
			t.Errorf("Expected chunk event, got %v", event.Type)
		}
	case <-time.After(time.Second):
		t.Fatal("Timeout waiting for chunk")
	}

	// Should receive error event
	select {
	case event := <-events:
		if event.Type != StreamEventError {
			t.Errorf("Expected error event, got %v", event.Type)
		}
	case <-time.After(time.Second):
		t.Fatal("Timeout waiting for error event")
	}
}

func TestInMemoryStreamManager_Unsubscribe(t *testing.T) {
	m := NewInMemoryStreamManager()
	defer m.Close()

	ctx := context.Background()
	streamID := "test-stream-unsub"

	writer, err := m.Open(ctx, streamID)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	events, unsubscribe, err := m.Subscribe(ctx, streamID)
	if err != nil {
		t.Fatalf("Subscribe failed: %v", err)
	}

	// Unsubscribe
	unsubscribe()

	// Write a chunk - should not panic
	if err := writer.Write(ctx, json.RawMessage(`"chunk"`)); err != nil {
		t.Fatalf("Write failed: %v", err)
	}

	// Events channel should be closed
	select {
	case _, ok := <-events:
		if ok {
			t.Error("Expected channel to be closed after unsubscribe")
		}
	case <-time.After(100 * time.Millisecond):
		t.Error("Channel not closed after unsubscribe")
	}
}

func TestInMemoryStreamManager_WithTTL(t *testing.T) {
	m := NewInMemoryStreamManager(WithTTL(10 * time.Millisecond))
	defer m.Close()

	if m.ttl != 10*time.Millisecond {
		t.Errorf("Expected TTL 10ms, got %v", m.ttl)
	}
}

func TestInMemoryStreamManager_ConcurrentOperations(t *testing.T) {
	m := NewInMemoryStreamManager()
	defer m.Close()

	ctx := context.Background()
	streamID := "test-stream-concurrent"

	writer, err := m.Open(ctx, streamID)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	const numSubscribers = 5
	const numChunks = 10

	var wg sync.WaitGroup
	errors := make(chan error, numSubscribers*numChunks)

	// Start subscribers
	for i := 0; i < numSubscribers; i++ {
		wg.Add(1)
		go func(subID int) {
			defer wg.Done()

			events, unsubscribe, err := m.Subscribe(ctx, streamID)
			if err != nil {
				errors <- err
				return
			}
			defer unsubscribe()

			received := 0
			for event := range events {
				if event.Type == StreamEventChunk {
					received++
				} else if event.Type == StreamEventDone {
					break
				}
			}

			if received != numChunks {
				errors <- core.NewPublicError(core.INTERNAL, "subscriber %d received %d chunks, expected %d", nil)
			}
		}(i)
	}

	// Give subscribers time to set up
	time.Sleep(50 * time.Millisecond)

	// Write chunks concurrently
	for i := 0; i < numChunks; i++ {
		if err := writer.Write(ctx, json.RawMessage(`"chunk"`)); err != nil {
			t.Fatalf("Write failed: %v", err)
		}
	}

	// Complete the stream
	if err := writer.Done(ctx, json.RawMessage(`"done"`)); err != nil {
		t.Fatalf("Done failed: %v", err)
	}

	wg.Wait()
	close(errors)

	for err := range errors {
		t.Errorf("Subscriber error: %v", err)
	}
}

func TestInMemoryStreamManager_Close(t *testing.T) {
	m := NewInMemoryStreamManager()

	// Close should not block
	done := make(chan struct{})
	go func() {
		m.Close()
		close(done)
	}()

	select {
	case <-done:
		// Success
	case <-time.After(time.Second):
		t.Fatal("Close blocked")
	}
}

func TestInMemoryStreamManager_CleanupExpiredStreams(t *testing.T) {
	m := NewInMemoryStreamManager(WithTTL(10 * time.Millisecond))
	defer m.Close()

	ctx := context.Background()

	// Create and complete a stream
	writer, err := m.Open(ctx, "expired-stream")
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}
	if err := writer.Done(ctx, json.RawMessage(`"done"`)); err != nil {
		t.Fatalf("Done failed: %v", err)
	}

	// Wait for TTL to expire
	time.Sleep(20 * time.Millisecond)

	// Trigger cleanup
	m.cleanupExpiredStreams()

	// Stream should be gone
	_, _, err = m.Subscribe(ctx, "expired-stream")
	if err == nil {
		t.Fatal("Expected error subscribing to expired stream")
	}

	var ufErr *core.UserFacingError
	if !errors.As(err, &ufErr) {
		t.Fatalf("Expected UserFacingError, got %T", err)
	}
	if ufErr.Status != core.NOT_FOUND {
		t.Errorf("Expected NOT_FOUND status, got %v", ufErr.Status)
	}
}

func TestInMemoryStreamManager_OpenStreamsNotCleanedUp(t *testing.T) {
	m := NewInMemoryStreamManager(WithTTL(10 * time.Millisecond))
	defer m.Close()

	ctx := context.Background()

	// Create an open stream (not completed)
	_, err := m.Open(ctx, "open-stream")
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	// Wait longer than TTL
	time.Sleep(20 * time.Millisecond)

	// Trigger cleanup
	m.cleanupExpiredStreams()

	// Stream should still exist
	_, _, err = m.Subscribe(ctx, "open-stream")
	if err != nil {
		t.Fatalf("Subscribe failed: %v", err)
	}
}

func TestInMemoryStreamManager_ErrorAfterClose(t *testing.T) {
	m := NewInMemoryStreamManager()
	defer m.Close()

	ctx := context.Background()
	streamID := "test-stream-error-after-close"

	writer, err := m.Open(ctx, streamID)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	if err := writer.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	// Try to error after close
	err = writer.Error(ctx, core.NewPublicError(core.INTERNAL, "test", nil))
	if err == nil {
		t.Fatal("Expected error when calling Error after Close")
	}

	var ufErr *core.UserFacingError
	if !errors.As(err, &ufErr) {
		t.Fatalf("Expected UserFacingError, got %T", err)
	}
	if ufErr.Status != core.FAILED_PRECONDITION {
		t.Errorf("Expected FAILED_PRECONDITION status, got %v", ufErr.Status)
	}
}

func TestInMemoryStreamManager_DoneAfterClose(t *testing.T) {
	m := NewInMemoryStreamManager()
	defer m.Close()

	ctx := context.Background()
	streamID := "test-stream-done-after-close"

	writer, err := m.Open(ctx, streamID)
	if err != nil {
		t.Fatalf("Open failed: %v", err)
	}

	if err := writer.Close(); err != nil {
		t.Fatalf("Close failed: %v", err)
	}

	// Try to done after close
	err = writer.Done(ctx, json.RawMessage(`"done"`))
	if err == nil {
		t.Fatal("Expected error when calling Done after Close")
	}

	var ufErr *core.UserFacingError
	if !errors.As(err, &ufErr) {
		t.Fatalf("Expected UserFacingError, got %T", err)
	}
	if ufErr.Status != core.FAILED_PRECONDITION {
		t.Errorf("Expected FAILED_PRECONDITION status, got %v", ufErr.Status)
	}
}
