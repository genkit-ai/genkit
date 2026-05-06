// Copyright 2024 Google LLC
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

package session

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/internal/base"
	"github.com/google/uuid"
	"github.com/invopop/jsonschema"
)

// Sessions are by default stored in memory only.
// Implement the Store interface to persist state.
type Store interface {
	Get(sessionId string) (data Data, err error)
	Save(sessionId string, data Data) error
}

type Data struct {
	State       map[string]any           `json:"state,omitempty"`       // Any state that should be stored
	StateSchema *jsonschema.Schema       `json:"stateschema,omitempty"` // Schema for state variables
	Threads     map[string][]*ai.Message `json:"threads,omitempty"`     // Messages by thread name
}

type Session struct {
	id    string // The session id
	data  Data   // The data for the session
	store Store  // The store for the session, defaults to in-memory storage
}

type SessionOption func(s *Session) error       // SessionOption configures params for the session
var sessionKey = base.NewContextKey[*Session]() // A session key

// NewSession creates a new session with the provided options.
// If no store is provided, it defaults to in-memory storage.
func New(ctx context.Context, opts ...SessionOption) (session *Session, err error) {
	s := &Session{}

	for _, with := range opts {
		err := with(s)
		if err != nil {
			return nil, err
		}
	}

	if s.store == nil {
		s.store = &InMemorySessionStore{}
	}

	if s.id == "" {
		s.id = uuid.New().String()
	}

	if s.data.Threads == nil {
		s.data.Threads = make(map[string][]*ai.Message)
	}

	// Initialize session in store
	s.store.Save(s.id, s.data)

	return s, nil
}

// Load loads sessiondata from store, and returns a session.
func Load(ctx context.Context, sessionId string, store Store) (session *Session, err error) {
	sessionData, err := store.Get(sessionId)
	if err != nil {
		return nil, err
	}

	session = &Session{
		id:    sessionId,
		data:  sessionData,
		store: store,
	}

	return session, nil
}

// GetID returns the session id.
func (s *Session) GetID() string {
	return s.id
}

// GetData returns the sessions data
func (s *Session) GetData() (Data, error) {
	return s.store.Get(s.id)
}

// UpdateState takes any data and sets it as state in the store.
func (s *Session) UpdateState(state any) error {
	var err error

	// Ensure sessionData is up to date
	sessionData, err := s.store.Get(s.id)
	if err != nil {
		return err
	}
	s.data = sessionData

	// Allow setting state schema on the fly, if not already set
	if s.data.StateSchema == nil {
		s.data.StateSchema, s.data.State, err = getSchemaAndData(state)
		if err != nil {
			return err
		}
	} else {
		// Use existing schema to unmarshal data
		data, err := json.Marshal(state)
		if err != nil {
			return err
		}

		err = json.Unmarshal(data, &s.data.State)
		if err != nil {
			return fmt.Errorf("genkit.UpdateState: %w, state doesn't match schema", err)
		}
	}

	return s.store.Save(s.id, s.data)
}

// UpdateMessages takes a threadName and a slice of messages.
func (s *Session) UpdateMessages(threadName string, messages []*ai.Message) error {
	// Ensure sessionData is up to date
	sessionData, err := s.store.Get(s.id)
	if err != nil {
		return err
	}
	s.data = sessionData

	s.data.Threads[threadName] = messages
	return s.store.Save(s.id, s.data)
}

// Set current session in context.
func (s *Session) SetContext(ctx context.Context) context.Context {
	return sessionKey.NewContext(ctx, s)
}

// Find current session in context, returns session if found.
func FromContext(ctx context.Context) (session *Session, err error) {
	if s := sessionKey.FromContext(ctx); s != nil {
		return s, nil
	}

	return nil, errors.New("genkit.SessionFromContext: session not found")
}

// WithID sets the session id.
func WithID(id string) SessionOption {
	return func(s *Session) error {
		if s.id != "" {
			return errors.New("genkit.WithID: cannot set session id more than once")
		}
		s.id = id
		return nil
	}
}

// WithData sets the session data.
func WithData(data Data) SessionOption {
	return func(s *Session) error {
		if s.data.Threads != nil {
			return errors.New("genkit.WithData: cannot set session data more than once")
		}
		s.data = data
		return nil
	}
}

// WithStore sets a session store for the session.
func WithStore(store Store) SessionOption {
	return func(s *Session) error {
		if s.store != nil {
			return errors.New("genkit.WithStore: cannot set session store more than once")
		}
		s.store = store
		return nil
	}
}

// WithStateType uses the struct provided to derive the state schema.
// If passing a struct with values, the struct definition will serve as the schema, the values will serve as the data.
func WithStateType(state any) SessionOption {
	return func(s *Session) error {
		if s.data.StateSchema != nil {
			return errors.New("genkit.WithStateType: cannot set state type more than once")
		}

		var err error
		s.data.StateSchema, s.data.State, err = getSchemaAndData(state)
		if err != nil {
			return err
		}

		return nil
	}
}

// Helper function to derive schema and defaults from state.
func getSchemaAndData(state any) (*jsonschema.Schema, map[string]any, error) {
	schema := base.InferJSONSchemaNonReferencing(state)

	// Set values as data
	structMap := base.SchemaAsMap(schema)
	data, err := json.Marshal(state)
	if err != nil {
		return nil, nil, err
	}

	err = json.Unmarshal(data, &structMap)
	if err != nil {
		return nil, nil, fmt.Errorf("genkit.getSchemaAndData: %w, only structs are allowed as types", err)
	}

	return schema, structMap, nil
}

// Default in-memory session store.
type InMemorySessionStore struct {
	data map[string]Data // The session data keyed by session id
	mu   sync.RWMutex
}

func (s *InMemorySessionStore) Get(sessionId string) (data Data, err error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if _, ok := s.data[sessionId]; !ok {
		return data, errors.New("genkit.InMemorySessionStore.Get: session not found")
	}
	return s.data[sessionId], nil
}

func (s *InMemorySessionStore) Save(sessionId string, data Data) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.data[sessionId]; !ok {
		s.data = make(map[string]Data)
	}
	s.data[sessionId] = data
	return nil
}
