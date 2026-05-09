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
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"cloud.google.com/go/firestore"
	"github.com/firebase/genkit/go/core/x/session"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/firebase"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// SessionStoreOption configures a FirestoreSessionStore.
// Implemented by firestoreOptions (WithCollection, WithTTL).
type SessionStoreOption interface {
	applySessionStore(*sessionStoreOptions) error
}

// sessionStoreOptions holds configuration for FirestoreSessionStore.
type sessionStoreOptions struct {
	firestoreOptions
}

// applySessionStore implements SessionStoreOption for sessionStoreOptions.
func (o *sessionStoreOptions) applySessionStore(opts *sessionStoreOptions) error {
	return o.firestoreOptions.applyFirestore(&opts.firestoreOptions)
}

// FirestoreSessionStore implements [session.Store[S]] using Firestore as the backend.
// Session state is persisted in Firestore documents, allowing sessions to survive
// server restarts and be accessible across multiple instances.
type FirestoreSessionStore[S any] struct {
	client     *firestore.Client
	collection string
	ttl        time.Duration
}

// sessionDocument represents the structure of a session document in Firestore.
type sessionDocument struct {
	State     json.RawMessage `firestore:"state"`
	CreatedAt time.Time       `firestore:"createdAt"`
	UpdatedAt time.Time       `firestore:"updatedAt"`
	ExpiresAt *time.Time      `firestore:"expiresAt,omitempty"`
}

// NewFirestoreSessionStore creates a Firestore-backed session store.
// Requires the Firebase plugin to be initialized in the Genkit instance.
func NewFirestoreSessionStore[S any](ctx context.Context, g *genkit.Genkit, opts ...SessionStoreOption) (*FirestoreSessionStore[S], error) {
	storeOpts := &sessionStoreOptions{}
	for _, opt := range opts {
		if err := opt.applySessionStore(storeOpts); err != nil {
			return nil, fmt.Errorf("firebase.NewFirestoreSessionStore: error applying options: %w", err)
		}
	}
	if storeOpts.Collection == "" {
		return nil, errors.New("firebase.NewFirestoreSessionStore: Collection name is required.\n" +
			"  Specify the Firestore collection where session documents will be stored:\n" +
			"    firebase.NewFirestoreSessionStore[MyState](ctx, g, firebase.WithCollection(\"genkit-sessions\"))")
	}
	if storeOpts.TTL == 0 {
		storeOpts.TTL = DefaultTTL
	}

	plugin := genkit.LookupPlugin(g, "firebase")
	if plugin == nil {
		return nil, errors.New("firebase.NewFirestoreSessionStore: Firebase plugin not found.\n" +
			"  Pass the Firebase plugin to genkit.Init():\n" +
			"    g := genkit.Init(ctx, genkit.WithPlugins(&firebase.Firebase{ProjectId: \"your-project\"}))")
	}
	f, ok := plugin.(*firebase.Firebase)
	if !ok {
		return nil, fmt.Errorf("firebase.NewFirestoreSessionStore: unexpected plugin type %T", plugin)
	}

	client, err := f.Firestore(ctx)
	if err != nil {
		return nil, fmt.Errorf("firebase.NewFirestoreSessionStore: %w", err)
	}

	return &FirestoreSessionStore[S]{
		client:     client,
		collection: storeOpts.Collection,
		ttl:        storeOpts.TTL,
	}, nil
}

// Get retrieves session data by ID from Firestore.
// Returns nil if the session does not exist.
func (s *FirestoreSessionStore[S]) Get(ctx context.Context, sessionID string) (*session.Data[S], error) {
	docRef := s.client.Collection(s.collection).Doc(sessionID)

	snapshot, err := docRef.Get(ctx)
	if err != nil {
		if status.Code(err) == codes.NotFound {
			return nil, nil
		}
		return nil, fmt.Errorf("firebase.FirestoreSessionStore.Get: %w", err)
	}
	if !snapshot.Exists() {
		return nil, nil
	}

	var doc sessionDocument
	if err := snapshot.DataTo(&doc); err != nil {
		return nil, fmt.Errorf("firebase.FirestoreSessionStore.Get: failed to parse document: %w", err)
	}

	var state S
	if len(doc.State) > 0 {
		if err := json.Unmarshal(doc.State, &state); err != nil {
			return nil, fmt.Errorf("firebase.FirestoreSessionStore.Get: failed to unmarshal state: %w", err)
		}
	}

	return &session.Data[S]{
		ID:    sessionID,
		State: state,
	}, nil
}

// Save persists session data to Firestore, creating or updating as needed.
// CreatedAt is only set when the document is first created; subsequent saves
// only update UpdatedAt and ExpiresAt.
func (s *FirestoreSessionStore[S]) Save(ctx context.Context, sessionID string, data *session.Data[S]) error {
	docRef := s.client.Collection(s.collection).Doc(sessionID)

	stateJSON, err := json.Marshal(data.State)
	if err != nil {
		return fmt.Errorf("firebase.FirestoreSessionStore.Save: failed to marshal state: %w", err)
	}

	now := time.Now()
	expiresAt := now.Add(s.ttl)

	err = s.client.RunTransaction(ctx, func(ctx context.Context, tx *firestore.Transaction) error {
		snapshot, err := tx.Get(docRef)
		if err != nil && status.Code(err) != codes.NotFound {
			return err
		}

		if !snapshot.Exists() {
			// Document doesn't exist - create it with CreatedAt
			return tx.Create(docRef, sessionDocument{
				State:     stateJSON,
				CreatedAt: now,
				UpdatedAt: now,
				ExpiresAt: &expiresAt,
			})
		}

		// Document exists - update without modifying CreatedAt
		return tx.Update(docRef, []firestore.Update{
			{Path: "state", Value: stateJSON},
			{Path: "updatedAt", Value: now},
			{Path: "expiresAt", Value: &expiresAt},
		})
	})
	if err != nil {
		return fmt.Errorf("firebase.FirestoreSessionStore.Save: %w", err)
	}

	return nil
}
