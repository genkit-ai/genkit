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

// Package session provides experimental session management APIs for Genkit.
//
// A session encapsulates a stateful execution environment with strongly-typed
// state that can be persisted across requests. Sessions are useful for maintaining
// user preferences, conversation context, or any application state that needs
// to survive between interactions.
//
// APIs in this package are under active development and may change in any
// minor version release. Use with caution in production environments.
//
// When these APIs stabilize, they will be moved to the core package
// and these exports will be deprecated.
package session

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"

	"github.com/google/uuid"
)

// Session represents a stateful environment with typed state.
// The type parameter S defines the shape of the session state and must be
// JSON-serializable for persistence.
type Session[S any] struct {
	id    string
	state S
	store Store[S]
	mu    sync.RWMutex
}

// Data is the serializable session state persisted by Store.
type Data[S any] struct {
	ID    string `json:"id"`
	State S      `json:"state,omitempty"`
}

// Store persists session data to a backend (database, file, memory, etc).
// Implementations must be safe for concurrent use.
type Store[S any] interface {
	// Get retrieves session data by ID. Returns nil if not found.
	Get(ctx context.Context, sessionID string) (*Data[S], error)
	// Save persists session data, creating or updating as needed.
	Save(ctx context.Context, sessionID string, data *Data[S]) error
}

// options holds configuration for creating a Session.
type options[S any] struct {
	ID           string
	InitialState S
	Store        Store[S]
	hasID        bool
	hasState     bool
	hasStore     bool
}

// Option configures a Session during creation.
type Option[S any] interface {
	apply(*options[S]) error
}

// apply implements Option for options, enabling composition.
func (o *options[S]) apply(opts *options[S]) error {
	if o.hasID {
		if opts.hasID {
			return errors.New("cannot set ID more than once (WithID)")
		}
		opts.ID = o.ID
		opts.hasID = true
	}

	if o.hasState {
		if opts.hasState {
			return errors.New("cannot set initial state more than once (WithInitialState)")
		}
		opts.InitialState = o.InitialState
		opts.hasState = true
	}

	if o.hasStore {
		if opts.hasStore {
			return errors.New("cannot set store more than once (WithStore)")
		}
		opts.Store = o.Store
		opts.hasStore = true
	}

	return nil
}

// WithID sets a custom session ID. If not provided, a UUID is generated.
func WithID[S any](id string) Option[S] {
	return &options[S]{ID: id, hasID: true}
}

// WithInitialState sets the initial state for a new session.
func WithInitialState[S any](state S) Option[S] {
	return &options[S]{InitialState: state, hasState: true}
}

// WithStore sets the persistence backend for the session.
// If not provided, the session is not persisted and exists only in memory.
func WithStore[S any](store Store[S]) Option[S] {
	return &options[S]{Store: store, hasStore: true}
}

// New creates a new session with the provided options.
// If a store is provided via [WithStore], the session is persisted immediately.
// If no store is provided, the session exists only in memory for the current
// request and can be propagated via context using [NewContext].
// If no ID is provided, a new UUID is generated.
// If no initial state is provided, the session is created with an empty state.
func New[S any](ctx context.Context, opts ...Option[S]) (*Session[S], error) {
	o := &options[S]{}
	for _, opt := range opts {
		if err := opt.apply(o); err != nil {
			return nil, fmt.Errorf("session.New: %w", err)
		}
	}

	id := o.ID
	if !o.hasID {
		id = uuid.New().String()
	}

	// Only persist if a store was explicitly provided
	if o.hasStore {
		data := &Data[S]{
			ID:    id,
			State: o.InitialState,
		}
		if err := o.Store.Save(ctx, id, data); err != nil {
			return nil, fmt.Errorf("session.New: failed to persist initial state: %w", err)
		}
	}

	return &Session[S]{
		id:    id,
		state: o.InitialState,
		store: o.Store, // nil if no store provided
	}, nil
}

// Load loads an existing session from the store.
// Returns an error if the session is not found or if loading fails.
func Load[S any](ctx context.Context, store Store[S], sessionID string) (*Session[S], error) {
	data, err := store.Get(ctx, sessionID)
	if err != nil {
		return nil, err
	}
	if data == nil {
		return nil, &NotFoundError{SessionID: sessionID}
	}

	return &Session[S]{
		id:    data.ID,
		state: data.State,
		store: store,
	}, nil
}

// ID returns the session's unique identifier.
func (s *Session[S]) ID() string {
	return s.id
}

// State returns the current session state.
// The returned value is a copy; modifications do not affect the session.
func (s *Session[S]) State() S {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return deepCopyState(s.state)
}

// deepCopyState creates a deep copy of the state using JSON marshaling.
// Panics if serialization fails, as this indicates a programming error
// (the state type S must be JSON-serializable per the Session contract).
func deepCopyState[S any](state S) S {
	bytes, err := json.Marshal(state)
	if err != nil {
		panic(fmt.Sprintf("session.State: failed to marshal state: %v", err))
	}

	var copied S
	if err := json.Unmarshal(bytes, &copied); err != nil {
		panic(fmt.Sprintf("session.State: failed to unmarshal state: %v", err))
	}

	return copied
}

// UpdateState updates the session state and persists it to the store (if configured).
func (s *Session[S]) UpdateState(ctx context.Context, state S) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.state = state

	if s.store != nil {
		data := &Data[S]{
			ID:    s.id,
			State: state,
		}
		if err := s.store.Save(ctx, s.id, data); err != nil {
			return err
		}
	}

	return nil
}

// contextKey is a private type for context keys to avoid collisions.
type contextKey struct{}

// sessionContextKey is the key used to store sessions in context.
var sessionContextKey = contextKey{}

// sessionHolder wraps a session with its type erased for context storage.
type sessionHolder struct {
	session any
}

// NewContext returns a new context with the session attached.
func NewContext[S any](ctx context.Context, s *Session[S]) context.Context {
	return context.WithValue(ctx, sessionContextKey, &sessionHolder{session: s})
}

// FromContext retrieves the current session from context.
// Returns nil if no session is in context or if the type doesn't match.
func FromContext[S any](ctx context.Context) *Session[S] {
	holder, ok := ctx.Value(sessionContextKey).(*sessionHolder)
	if !ok || holder == nil {
		return nil
	}
	session, ok := holder.session.(*Session[S])
	if !ok {
		return nil
	}
	return session
}

// stateGetter is an internal interface for retrieving state without type parameters.
type stateGetter interface {
	getState() any
}

// getState implements stateGetter, returning the session state as any.
func (s *Session[S]) getState() any {
	return s.State()
}

// StateFromContext retrieves the current session state from context without
// requiring knowledge of the state type. This is useful for template rendering
// where the state type is not known at compile time.
// Returns nil if no session is in context.
func StateFromContext(ctx context.Context) any {
	holder, ok := ctx.Value(sessionContextKey).(*sessionHolder)
	if !ok || holder == nil {
		return nil
	}
	if getter, ok := holder.session.(stateGetter); ok {
		return getter.getState()
	}
	return nil
}

// NotFoundError is returned when a session cannot be found in the store.
type NotFoundError struct {
	SessionID string
}

func (e *NotFoundError) Error() string {
	return "session not found: " + e.SessionID
}

// InMemoryStore is a thread-safe in-memory implementation of Store.
// Useful for testing or single-instance deployments where persistence is not required.
type InMemoryStore[S any] struct {
	data map[string]*Data[S]
	mu   sync.RWMutex
}

// NewInMemoryStore creates a new in-memory session store.
func NewInMemoryStore[S any]() *InMemoryStore[S] {
	return &InMemoryStore[S]{
		data: make(map[string]*Data[S]),
	}
}

// Get retrieves session data by ID.
func (s *InMemoryStore[S]) Get(_ context.Context, sessionID string) (*Data[S], error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	data, exists := s.data[sessionID]
	if !exists {
		return nil, nil
	}

	// Return a copy to prevent external modifications
	copied, err := copyData(data)
	if err != nil {
		return nil, err
	}
	return copied, nil
}

// Save persists session data.
func (s *InMemoryStore[S]) Save(_ context.Context, sessionID string, data *Data[S]) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Store a copy to prevent external modifications
	copied, err := copyData(data)
	if err != nil {
		return err
	}
	s.data[sessionID] = copied
	return nil
}

// copyData creates a deep copy of Data using JSON marshaling.
func copyData[S any](data *Data[S]) (*Data[S], error) {
	if data == nil {
		return nil, nil
	}

	bytes, err := json.Marshal(data)
	if err != nil {
		return nil, err
	}

	var copied Data[S]
	if err := json.Unmarshal(bytes, &copied); err != nil {
		return nil, err
	}

	return &copied, nil
}
