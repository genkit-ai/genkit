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

// Package x contains experimental Firebase features.
//
// APIs in this package are under active development and may change in any
// minor version release. Use with caution in production environments.
package x

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"

	"cloud.google.com/go/firestore"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/x/streaming"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/firebase"
	"github.com/google/uuid"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

const (
	streamBufferSize = 100
	defaultTimeout   = 60 * time.Second
	streamEventChunk = "chunk"
	streamEventDone  = "done"
	streamEventError = "error"
)

// StreamManagerOption configures a FirestoreStreamManager.
// Implemented by firestoreOptions (WithCollection, WithTTL) and streamManagerOptions (WithTimeout).
type StreamManagerOption interface {
	applyStreamManager(*streamManagerOptions) error
}

// streamManagerOptions holds configuration for FirestoreStreamManager.
type streamManagerOptions struct {
	firestoreOptions
	Timeout time.Duration
}

// applyStreamManager implements StreamManagerOption for streamManagerOptions.
func (o *streamManagerOptions) applyStreamManager(opts *streamManagerOptions) error {
	if err := o.firestoreOptions.applyFirestore(&opts.firestoreOptions); err != nil {
		return err
	}

	if o.Timeout > 0 {
		if opts.Timeout > 0 {
			return errors.New("cannot set timeout more than once (WithTimeout)")
		}
		opts.Timeout = o.Timeout
	}

	return nil
}

// WithTimeout sets how long a subscriber waits for new events before giving up.
// If no activity occurs within this duration, subscribers receive a DEADLINE_EXCEEDED error.
// Default is 60 seconds.
func WithTimeout(timeout time.Duration) StreamManagerOption {
	return &streamManagerOptions{Timeout: timeout}
}

// FirestoreStreamManager implements [streaming.StreamManager] using Firestore as the backend.
// Stream state is persisted in Firestore documents, allowing streams to survive server
// restarts and be accessible across multiple instances.
type FirestoreStreamManager struct {
	client     *firestore.Client
	collection string
	timeout    time.Duration
	ttl        time.Duration
}

// streamDocument represents the structure of a stream document in Firestore.
type streamDocument struct {
	Stream    []streamEntry `firestore:"stream"`
	CreatedAt time.Time     `firestore:"createdAt"`
	UpdatedAt time.Time     `firestore:"updatedAt"`
	ExpiresAt *time.Time    `firestore:"expiresAt,omitempty"`
}

// streamEntry represents a single entry in the stream array.
type streamEntry struct {
	Type   string          `firestore:"type"`
	Chunk  json.RawMessage `firestore:"chunk,omitempty"`
	Output json.RawMessage `firestore:"output,omitempty"`
	Err    *streamError    `firestore:"err,omitempty"`
	UUID   string          `firestore:"uuid,omitempty"`
}

// streamError represents a serializable error for Firestore storage.
type streamError struct {
	Status  string `firestore:"status"`
	Message string `firestore:"message"`
}

// NewFirestoreStreamManager creates a [FirestoreStreamManager] for durable streaming.
// Requires the Firebase plugin to be initialized in the Genkit instance.
func NewFirestoreStreamManager(ctx context.Context, g *genkit.Genkit, opts ...StreamManagerOption) (*FirestoreStreamManager, error) {
	streamOpts := &streamManagerOptions{}
	for _, opt := range opts {
		if err := opt.applyStreamManager(streamOpts); err != nil {
			return nil, fmt.Errorf("firebase.NewFirestoreStreamManager: error applying options: %w", err)
		}
	}
	if streamOpts.Collection == "" {
		return nil, errors.New("firebase.NewFirestoreStreamManager: Collection name is required.\n" +
			"  Specify the Firestore collection where stream documents will be stored:\n" +
			"    firebase.NewFirestoreStreamManager(ctx, g, firebase.WithCollection(\"genkit-streams\"))")
	}
	if streamOpts.Timeout == 0 {
		streamOpts.Timeout = defaultTimeout
	}
	if streamOpts.TTL == 0 {
		streamOpts.TTL = DefaultTTL
	}

	plugin := genkit.LookupPlugin(g, "firebase")
	if plugin == nil {
		return nil, errors.New("firebase.NewFirestoreStreamManager: Firebase plugin not found.\n" +
			"  Pass the Firebase plugin to genkit.Init():\n" +
			"    g := genkit.Init(ctx, genkit.WithPlugins(&firebase.Firebase{ProjectId: \"your-project\"}))")
	}
	f, ok := plugin.(*firebase.Firebase)
	if !ok {
		return nil, fmt.Errorf("firebase.NewFirestoreStreamManager: unexpected plugin type %T", plugin)
	}

	client, err := f.Firestore(ctx)
	if err != nil {
		return nil, fmt.Errorf("firebase.NewFirestoreStreamManager: %w", err)
	}

	return &FirestoreStreamManager{
		client:     client,
		collection: streamOpts.Collection,
		timeout:    streamOpts.Timeout,
		ttl:        streamOpts.TTL,
	}, nil
}

// Open creates a new stream for writing.
// Returns ALREADY_EXISTS error if a stream with the given ID already exists.
func (m *FirestoreStreamManager) Open(ctx context.Context, streamID string) (streaming.StreamInput, error) {
	docRef := m.client.Collection(m.collection).Doc(streamID)
	now := time.Now()
	expiresAt := now.Add(m.timeout + m.ttl)
	_, err := docRef.Create(ctx, streamDocument{
		Stream:    []streamEntry{},
		CreatedAt: now,
		UpdatedAt: now,
		ExpiresAt: &expiresAt,
	})
	if err != nil {
		if status.Code(err) == codes.AlreadyExists {
			return nil, core.NewPublicError(core.ALREADY_EXISTS, "stream already exists", nil)
		}
		return nil, err
	}
	return &firestoreStreamInput{
		manager:  m,
		streamID: streamID,
		docRef:   docRef,
	}, nil
}

// Subscribe subscribes to an existing stream.
func (m *FirestoreStreamManager) Subscribe(ctx context.Context, streamID string) (<-chan streaming.StreamEvent, func(), error) {
	docRef := m.client.Collection(m.collection).Doc(streamID)

	snapshot, err := docRef.Get(ctx)
	if err != nil {
		if isNotFound(err) {
			return nil, nil, core.NewPublicError(core.NOT_FOUND, "stream not found", nil)
		}
		return nil, nil, err
	}
	if !snapshot.Exists() {
		return nil, nil, core.NewPublicError(core.NOT_FOUND, "stream not found", nil)
	}

	ch := make(chan streaming.StreamEvent, streamBufferSize)
	var mu sync.Mutex
	var lastIndex int = -1
	var unsubscribed bool
	var cancelSnapshot context.CancelFunc

	snapshotCtx, cancelSnapshot := context.WithCancel(ctx)

	var timeoutTimer *time.Timer
	resetTimeout := func() {
		mu.Lock()
		defer mu.Unlock()
		if timeoutTimer != nil {
			timeoutTimer.Stop()
		}
		timeoutTimer = time.AfterFunc(m.timeout, func() {
			mu.Lock()
			defer mu.Unlock()
			if !unsubscribed {
				unsubscribed = true
				ch <- streaming.StreamEvent{
					Type: streaming.StreamEventError,
					Err:  core.NewPublicError(core.DEADLINE_EXCEEDED, "stream timed out", nil),
				}
				close(ch)
				cancelSnapshot()
			}
		})
	}

	unsubscribe := func() {
		mu.Lock()
		defer mu.Unlock()
		if !unsubscribed {
			unsubscribed = true
			if timeoutTimer != nil {
				timeoutTimer.Stop()
			}
			close(ch)
			cancelSnapshot()
		}
	}

	resetTimeout()

	go func() {
		snapshots := docRef.Snapshots(snapshotCtx)
		defer snapshots.Stop()

		for {
			snap, err := snapshots.Next()
			if err != nil {
				mu.Lock()
				if !unsubscribed {
					if snapshotCtx.Err() == nil {
						ch <- streaming.StreamEvent{
							Type: streaming.StreamEventError,
							Err:  err,
						}
					}
					unsubscribed = true
					if timeoutTimer != nil {
						timeoutTimer.Stop()
					}
					close(ch)
				}
				mu.Unlock()
				return
			}

			resetTimeout()

			if !snap.Exists() {
				continue
			}

			var doc streamDocument
			if err := snap.DataTo(&doc); err != nil {
				mu.Lock()
				if !unsubscribed {
					ch <- streaming.StreamEvent{
						Type: streaming.StreamEventError,
						Err:  err,
					}
					unsubscribed = true
					if timeoutTimer != nil {
						timeoutTimer.Stop()
					}
					close(ch)
				}
				mu.Unlock()
				return
			}

			mu.Lock()
			for i := lastIndex + 1; i < len(doc.Stream); i++ {
				entry := doc.Stream[i]
				switch entry.Type {
				case streamEventChunk:
					if !unsubscribed {
						select {
						case ch <- streaming.StreamEvent{Type: streaming.StreamEventChunk, Chunk: entry.Chunk}:
						default:
						}
					}
				case streamEventDone:
					if !unsubscribed {
						select {
						case ch <- streaming.StreamEvent{Type: streaming.StreamEventDone, Output: entry.Output}:
						default:
						}
						unsubscribed = true
						if timeoutTimer != nil {
							timeoutTimer.Stop()
						}
						close(ch)
					}
					mu.Unlock()
					return
				case streamEventError:
					if !unsubscribed {
						var errStatus core.StatusName = core.UNKNOWN
						var errMsg string
						if entry.Err != nil {
							errMsg = entry.Err.Message
							if entry.Err.Status != "" {
								errStatus = core.StatusName(entry.Err.Status)
							}
						}
						select {
						case ch <- streaming.StreamEvent{
							Type: streaming.StreamEventError,
							Err:  core.NewPublicError(errStatus, errMsg, nil),
						}:
						default:
						}
						unsubscribed = true
						if timeoutTimer != nil {
							timeoutTimer.Stop()
						}
						close(ch)
					}
					mu.Unlock()
					return
				}
			}
			lastIndex = len(doc.Stream) - 1
			mu.Unlock()
		}
	}()

	return ch, unsubscribe, nil
}

// isNotFound checks if the error is a not found error.
func isNotFound(err error) bool {
	if err == nil {
		return false
	}
	if grpcErr, ok := status.FromError(err); ok {
		return grpcErr.Code() == codes.NotFound
	}
	return false
}

// firestoreStreamInput implements streaming.StreamInput for Firestore.
type firestoreStreamInput struct {
	manager  *FirestoreStreamManager
	streamID string
	docRef   *firestore.DocumentRef
	closed   bool
	mu       sync.Mutex
}

func (s *firestoreStreamInput) Write(ctx context.Context, chunk json.RawMessage) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return core.NewPublicError(core.FAILED_PRECONDITION, "stream writer is closed", nil)
	}

	_, err := s.docRef.Update(ctx, []firestore.Update{
		{
			Path: "stream",
			Value: firestore.ArrayUnion(streamEntry{
				Type:  streamEventChunk,
				Chunk: chunk,
				UUID:  uuid.New().String(),
			}),
		},
		{
			Path:  "updatedAt",
			Value: firestore.ServerTimestamp,
		},
	})
	return err
}

func (s *firestoreStreamInput) Done(ctx context.Context, output json.RawMessage) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return core.NewPublicError(core.FAILED_PRECONDITION, "stream writer is closed", nil)
	}
	s.closed = true

	expiresAt := time.Now().Add(s.manager.ttl)
	_, err := s.docRef.Update(ctx, []firestore.Update{
		{
			Path: "stream",
			Value: firestore.ArrayUnion(streamEntry{
				Type:   streamEventDone,
				Output: output,
			}),
		},
		{
			Path:  "updatedAt",
			Value: firestore.ServerTimestamp,
		},
		{
			Path:  "expiresAt",
			Value: expiresAt,
		},
	})
	return err
}

func (s *firestoreStreamInput) Error(ctx context.Context, err error) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return core.NewPublicError(core.FAILED_PRECONDITION, "stream writer is closed", nil)
	}
	s.closed = true

	streamErr := &streamError{
		Status:  string(core.UNKNOWN),
		Message: err.Error(),
	}
	var ufErr *core.UserFacingError
	if errors.As(err, &ufErr) {
		streamErr.Status = string(ufErr.Status)
	}

	expiresAt := time.Now().Add(s.manager.ttl)
	_, updateErr := s.docRef.Update(ctx, []firestore.Update{
		{
			Path: "stream",
			Value: firestore.ArrayUnion(streamEntry{
				Type: streamEventError,
				Err:  streamErr,
			}),
		},
		{
			Path:  "updatedAt",
			Value: firestore.ServerTimestamp,
		},
		{
			Path:  "expiresAt",
			Value: expiresAt,
		},
	})
	return updateErr
}

func (s *firestoreStreamInput) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.closed = true
	return nil
}
