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

// Package streaming provides experimental durable streaming APIs for Genkit.
//
// APIs in this package are under active development and may change in any
// minor version release. Use with caution in production environments.
//
// When these APIs stabilize, they will be moved to their parent packages
// (e.g., core and genkit) and these exports will be deprecated.
package streaming

import (
	"context"
	"encoding/json"
	"sync"
	"time"

	"github.com/firebase/genkit/go/core"
)

// StreamEventType indicates the type of stream event.
type StreamEventType int

const (
	StreamEventChunk StreamEventType = iota
	StreamEventDone
	StreamEventError
)

// StreamEvent represents an event in a durable stream.
type StreamEvent struct {
	Type   StreamEventType
	Chunk  json.RawMessage // set when Type == StreamEventChunk
	Output json.RawMessage // set when Type == StreamEventDone
	Err    error           // set when Type == StreamEventError
}

// StreamInput provides methods for writing to a durable stream.
type StreamInput interface {
	// Write sends a chunk to the stream and notifies all subscribers.
	Write(ctx context.Context, chunk json.RawMessage) error
	// Done marks the stream as successfully completed with the given output.
	Done(ctx context.Context, output json.RawMessage) error
	// Error marks the stream as failed with the given error.
	Error(ctx context.Context, err error) error
	// Close releases resources without marking the stream as done or errored.
	Close() error
}

// StreamManager manages durable streams, allowing creation and subscription.
// Implementations can provide different storage backends (e.g., in-memory, database, cache).
type StreamManager interface {
	// Open creates a new stream for writing.
	// Returns an error if a stream with the given ID already exists.
	Open(ctx context.Context, streamID string) (StreamInput, error)
	// Subscribe subscribes to an existing stream.
	// Returns a channel that receives stream events, an unsubscribe function, and an error.
	// If the stream has already completed, all buffered events are sent before the done/error event.
	// Returns NOT_FOUND error if the stream doesn't exist.
	Subscribe(ctx context.Context, streamID string) (<-chan StreamEvent, func(), error)
}

// inMemoryStreamBufferSize is the buffer size for subscriber event channels.
const inMemoryStreamBufferSize = 100

// streamStatus represents the current state of a stream.
type streamStatus int

const (
	streamStatusOpen streamStatus = iota
	streamStatusDone
	streamStatusError
)

// streamState holds the internal state of a single stream.
type streamState struct {
	status      streamStatus
	chunks      []json.RawMessage
	output      json.RawMessage
	err         error
	subscribers []chan StreamEvent
	lastTouched time.Time
	mu          sync.RWMutex
}

// InMemoryStreamManager is an in-memory implementation of StreamManager.
// Useful for testing or single-instance deployments where persistence is not required.
// Call Close to stop the background cleanup goroutine when the manager is no longer needed.
type InMemoryStreamManager struct {
	streams map[string]*streamState
	mu      sync.RWMutex
	ttl     time.Duration
	stopCh  chan struct{}
	doneCh  chan struct{}
}

// StreamManagerOption configures an InMemoryStreamManager.
type StreamManagerOption interface {
	applyInMemoryStreamManager(*streamManagerOptions)
}

// streamManagerOptions holds configuration for InMemoryStreamManager.
type streamManagerOptions struct {
	TTL time.Duration // Time-to-live for completed streams.
}

func (o *streamManagerOptions) applyInMemoryStreamManager(opts *streamManagerOptions) {
	if o.TTL > 0 {
		opts.TTL = o.TTL
	}
}

// WithTTL sets the time-to-live for completed streams.
// Streams that have completed (done or error) will be cleaned up after this duration.
// Default is 5 minutes.
func WithTTL(ttl time.Duration) StreamManagerOption {
	return &streamManagerOptions{TTL: ttl}
}

// NewInMemoryStreamManager creates a new InMemoryStreamManager.
// A background goroutine is started to periodically clean up expired streams.
// Call Close to stop the goroutine when the manager is no longer needed.
func NewInMemoryStreamManager(opts ...StreamManagerOption) *InMemoryStreamManager {
	options := &streamManagerOptions{
		TTL: 5 * time.Minute,
	}
	for _, opt := range opts {
		opt.applyInMemoryStreamManager(options)
	}
	m := &InMemoryStreamManager{
		streams: make(map[string]*streamState),
		ttl:     options.TTL,
		stopCh:  make(chan struct{}),
		doneCh:  make(chan struct{}),
	}
	go m.cleanupLoop()
	return m
}

// cleanupLoop runs periodically to remove expired streams.
func (m *InMemoryStreamManager) cleanupLoop() {
	ticker := time.NewTicker(time.Minute)
	defer ticker.Stop()
	defer close(m.doneCh)

	for {
		select {
		case <-m.stopCh:
			return
		case <-ticker.C:
			m.cleanupExpiredStreams()
		}
	}
}

// cleanupExpiredStreams removes streams that have completed and exceeded the TTL.
func (m *InMemoryStreamManager) cleanupExpiredStreams() {
	now := time.Now()
	m.mu.Lock()
	defer m.mu.Unlock()

	for id, state := range m.streams {
		state.mu.RLock()
		shouldDelete := state.status != streamStatusOpen && now.Sub(state.lastTouched) > m.ttl
		state.mu.RUnlock()
		if shouldDelete {
			delete(m.streams, id)
		}
	}
}

// Close stops the background cleanup goroutine and releases resources.
// This method blocks until the cleanup goroutine has stopped.
func (m *InMemoryStreamManager) Close() {
	close(m.stopCh)
	<-m.doneCh
}

// Open creates a new stream for writing.
func (m *InMemoryStreamManager) Open(ctx context.Context, streamID string) (StreamInput, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.streams[streamID]; exists {
		return nil, core.NewPublicError(core.ALREADY_EXISTS, "stream already exists", nil)
	}

	state := &streamState{
		status:      streamStatusOpen,
		chunks:      make([]json.RawMessage, 0),
		subscribers: make([]chan StreamEvent, 0),
		lastTouched: time.Now(),
	}
	m.streams[streamID] = state

	return &inMemoryStreamInput{
		manager:  m,
		streamID: streamID,
		state:    state,
	}, nil
}

// Subscribe subscribes to an existing stream.
func (m *InMemoryStreamManager) Subscribe(ctx context.Context, streamID string) (<-chan StreamEvent, func(), error) {
	m.mu.RLock()
	state, exists := m.streams[streamID]
	m.mu.RUnlock()

	if !exists {
		return nil, nil, core.NewPublicError(core.NOT_FOUND, "stream not found", nil)
	}

	ch := make(chan StreamEvent, inMemoryStreamBufferSize)

	state.mu.Lock()
	defer state.mu.Unlock()

	// Send all buffered chunks
	for _, chunk := range state.chunks {
		select {
		case ch <- StreamEvent{Type: StreamEventChunk, Chunk: chunk}:
		case <-ctx.Done():
			close(ch)
			return nil, nil, ctx.Err()
		}
	}

	// Handle completed streams
	switch state.status {
	case streamStatusDone:
		ch <- StreamEvent{Type: StreamEventDone, Output: state.output}
		close(ch)
		return ch, func() {}, nil
	case streamStatusError:
		ch <- StreamEvent{Type: StreamEventError, Err: state.err}
		close(ch)
		return ch, func() {}, nil
	}

	// Stream is still open, add subscriber
	state.subscribers = append(state.subscribers, ch)

	unsubscribe := func() {
		state.mu.Lock()
		defer state.mu.Unlock()
		for i, sub := range state.subscribers {
			if sub == ch {
				state.subscribers = append(state.subscribers[:i], state.subscribers[i+1:]...)
				close(ch)
				break
			}
		}
	}

	return ch, unsubscribe, nil
}

// inMemoryStreamInput implements ActionStreamInput for the in-memory manager.
type inMemoryStreamInput struct {
	manager  *InMemoryStreamManager
	streamID string
	state    *streamState
	closed   bool
	mu       sync.Mutex
}

func (s *inMemoryStreamInput) Write(_ context.Context, chunk json.RawMessage) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return core.NewPublicError(core.FAILED_PRECONDITION, "stream writer is closed", nil)
	}

	s.state.mu.Lock()
	defer s.state.mu.Unlock()

	if s.state.status != streamStatusOpen {
		return core.NewPublicError(core.FAILED_PRECONDITION, "stream has already completed", nil)
	}

	s.state.chunks = append(s.state.chunks, chunk)
	s.state.lastTouched = time.Now()

	event := StreamEvent{Type: StreamEventChunk, Chunk: chunk}
	for _, ch := range s.state.subscribers {
		select {
		case ch <- event:
		default:
			// Channel full, skip (subscriber is slow)
		}
	}

	return nil
}

func (s *inMemoryStreamInput) Done(_ context.Context, output json.RawMessage) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return core.NewPublicError(core.FAILED_PRECONDITION, "stream writer is closed", nil)
	}
	s.closed = true

	s.state.mu.Lock()
	defer s.state.mu.Unlock()

	if s.state.status != streamStatusOpen {
		return core.NewPublicError(core.FAILED_PRECONDITION, "stream has already completed", nil)
	}

	s.state.status = streamStatusDone
	s.state.output = output
	s.state.lastTouched = time.Now()

	event := StreamEvent{Type: StreamEventDone, Output: output}
	for _, ch := range s.state.subscribers {
		select {
		case ch <- event:
		default:
		}
		close(ch)
	}
	s.state.subscribers = nil

	return nil
}

func (s *inMemoryStreamInput) Error(_ context.Context, err error) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return core.NewPublicError(core.FAILED_PRECONDITION, "stream writer is closed", nil)
	}
	s.closed = true

	s.state.mu.Lock()
	defer s.state.mu.Unlock()

	if s.state.status != streamStatusOpen {
		return core.NewPublicError(core.FAILED_PRECONDITION, "stream has already completed", nil)
	}

	s.state.status = streamStatusError
	s.state.err = err
	s.state.lastTouched = time.Now()

	event := StreamEvent{Type: StreamEventError, Err: err}
	for _, ch := range s.state.subscribers {
		select {
		case ch <- event:
		default:
		}
		close(ch)
	}
	s.state.subscribers = nil

	return nil
}

func (s *inMemoryStreamInput) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closed = true
	return nil
}
