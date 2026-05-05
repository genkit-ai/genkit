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

package exp

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/internal/base"
)

// --- Snapshot ---

// SessionSnapshot is a persisted point-in-time capture of session state.
type SessionSnapshot[State any] struct {
	// SnapshotID is the unique identifier for this snapshot (UUID).
	SnapshotID string `json:"snapshotId"`
	// ParentID is the ID of the previous snapshot in this timeline.
	ParentID string `json:"parentId,omitempty"`
	// CreatedAt is when the snapshot was created.
	CreatedAt time.Time `json:"createdAt"`
	// Event is what triggered this snapshot.
	Event SnapshotEvent `json:"event"`
	// State is the actual conversation state.
	State SessionState[State] `json:"state"`
}

// SnapshotContext provides context for snapshot decision callbacks.
type SnapshotContext[State any] struct {
	// State is the current state that will be snapshotted if the callback returns true.
	State *SessionState[State]
	// PrevState is the state at the last snapshot, or nil if none exists.
	PrevState *SessionState[State]
	// TurnIndex is the turn number in the current invocation.
	TurnIndex int
	// Event is what triggered this snapshot check.
	Event SnapshotEvent
}

// SnapshotCallback decides whether to create a snapshot.
// If not provided and a store is configured, snapshots are always created.
type SnapshotCallback[State any] = func(ctx context.Context, sc *SnapshotContext[State]) bool

// --- Session store ---

// SessionStore persists and retrieves snapshots.
type SessionStore[State any] interface {
	// GetSnapshot retrieves a snapshot by ID. Returns nil if not found.
	GetSnapshot(ctx context.Context, snapshotID string) (*SessionSnapshot[State], error)
	// SaveSnapshot persists a snapshot.
	SaveSnapshot(ctx context.Context, snapshot *SessionSnapshot[State]) error
}

// InMemorySessionStore provides a thread-safe in-memory snapshot store.
type InMemorySessionStore[State any] struct {
	snapshots map[string]*SessionSnapshot[State]
	mu        sync.RWMutex
}

// NewInMemorySessionStore creates a new in-memory snapshot store.
func NewInMemorySessionStore[State any]() *InMemorySessionStore[State] {
	return &InMemorySessionStore[State]{
		snapshots: make(map[string]*SessionSnapshot[State]),
	}
}

// GetSnapshot retrieves a snapshot by ID. Returns nil if not found.
func (s *InMemorySessionStore[State]) GetSnapshot(_ context.Context, snapshotID string) (*SessionSnapshot[State], error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	snap, exists := s.snapshots[snapshotID]
	if !exists {
		return nil, nil
	}

	copied, err := copySnapshot(snap)
	if err != nil {
		return nil, err
	}
	return copied, nil
}

// SaveSnapshot persists a snapshot.
func (s *InMemorySessionStore[State]) SaveSnapshot(_ context.Context, snapshot *SessionSnapshot[State]) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	copied, err := copySnapshot(snapshot)
	if err != nil {
		return err
	}
	s.snapshots[copied.SnapshotID] = copied
	return nil
}

// copySnapshot creates a deep copy of a snapshot using JSON marshaling.
func copySnapshot[State any](snap *SessionSnapshot[State]) (*SessionSnapshot[State], error) {
	if snap == nil {
		return nil, nil
	}
	bytes, err := json.Marshal(snap)
	if err != nil {
		return nil, fmt.Errorf("copy snapshot: marshal: %w", err)
	}
	var copied SessionSnapshot[State]
	if err := json.Unmarshal(bytes, &copied); err != nil {
		return nil, fmt.Errorf("copy snapshot: unmarshal: %w", err)
	}
	return &copied, nil
}

// --- Session ---

// Session holds conversation state and provides thread-safe read/write access to messages,
// input variables, custom state, and artifacts.
type Session[State any] struct {
	mu      sync.RWMutex
	state   SessionState[State]
	store   SessionStore[State]
	version uint64 // incremented on every mutation; used to skip redundant snapshots
}

// State returns a copy of the current state.
func (s *Session[State]) State() *SessionState[State] {
	s.mu.RLock()
	defer s.mu.RUnlock()
	copied := s.copyStateLocked()
	return &copied
}

// Messages returns the current conversation history.
func (s *Session[State]) Messages() []*ai.Message {
	s.mu.RLock()
	defer s.mu.RUnlock()
	msgs := make([]*ai.Message, len(s.state.Messages))
	copy(msgs, s.state.Messages)
	return msgs
}

// AddMessages appends messages to the conversation history.
func (s *Session[State]) AddMessages(messages ...*ai.Message) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state.Messages = append(s.state.Messages, messages...)
	s.version++
}

// SetMessages replaces the conversation history with the given messages.
func (s *Session[State]) SetMessages(messages []*ai.Message) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state.Messages = messages
	s.version++
}

// UpdateMessages atomically reads the current messages, applies the given
// function, and writes the result back.
func (s *Session[State]) UpdateMessages(fn func([]*ai.Message) []*ai.Message) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state.Messages = fn(s.state.Messages)
	s.version++
}

// Custom returns the current user-defined custom state.
func (s *Session[State]) Custom() State {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.state.Custom
}

// UpdateCustom atomically reads the current custom state, applies the given
// function, and writes the result back.
func (s *Session[State]) UpdateCustom(fn func(State) State) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state.Custom = fn(s.state.Custom)
	s.version++
}

// InputVariables returns the prompt input stored in the session state.
func (s *Session[State]) InputVariables() any {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.state.InputVariables
}

// Artifacts returns the current artifacts.
func (s *Session[State]) Artifacts() []*Artifact {
	s.mu.RLock()
	defer s.mu.RUnlock()
	arts := make([]*Artifact, len(s.state.Artifacts))
	copy(arts, s.state.Artifacts)
	return arts
}

// AddArtifacts adds artifacts to the session. If an artifact with the same
// name already exists, it is replaced.
func (s *Session[State]) AddArtifacts(artifacts ...*Artifact) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, a := range artifacts {
		replaced := false
		if a.Name != "" {
			for i, existing := range s.state.Artifacts {
				if existing.Name == a.Name {
					s.state.Artifacts[i] = a
					replaced = true
					break
				}
			}
		}
		if !replaced {
			s.state.Artifacts = append(s.state.Artifacts, a)
		}
	}
	s.version++
}

// UpdateArtifacts atomically reads the current artifacts, applies the given
// function, and writes the result back.
func (s *Session[State]) UpdateArtifacts(fn func([]*Artifact) []*Artifact) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.state.Artifacts = fn(s.state.Artifacts)
	s.version++
}

// copyStateLocked returns a deep copy of the state. Caller must hold mu (read or write).
func (s *Session[State]) copyStateLocked() SessionState[State] {
	bytes, err := json.Marshal(s.state)
	if err != nil {
		panic(fmt.Sprintf("agent: failed to marshal state: %v", err))
	}
	var copied SessionState[State]
	if err := json.Unmarshal(bytes, &copied); err != nil {
		panic(fmt.Sprintf("agent: failed to unmarshal state: %v", err))
	}
	return copied
}

// --- Session context ---

var sessionCtxKey = base.NewContextKey[any]()

// NewSessionContext returns a new context with the session attached.
func NewSessionContext[State any](ctx context.Context, s *Session[State]) context.Context {
	return sessionCtxKey.NewContext(ctx, s)
}

// SessionFromContext retrieves the current session from context.
// Returns nil if no session is in context or if the type doesn't match.
func SessionFromContext[State any](ctx context.Context) *Session[State] {
	session, _ := sessionCtxKey.FromContext(ctx).(*Session[State])
	return session
}
