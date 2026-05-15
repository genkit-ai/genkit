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
	"errors"
	"time"
)

const (
	// DefaultTTL is the default time-to-live for Firestore documents.
	DefaultTTL = 5 * time.Minute
)

// firestoreOptions holds common configuration for Firestore-based services.
type firestoreOptions struct {
	Collection string
	TTL        time.Duration
}

// applyFirestore applies common Firestore options.
func (o *firestoreOptions) applyFirestore(opts *firestoreOptions) error {
	if o.Collection != "" {
		if opts.Collection != "" {
			return errors.New("cannot set collection more than once (WithCollection)")
		}
		opts.Collection = o.Collection
	}

	if o.TTL > 0 {
		if opts.TTL > 0 {
			return errors.New("cannot set TTL more than once (WithTTL)")
		}
		opts.TTL = o.TTL
	}

	return nil
}

// applyStreamManager implements StreamManagerOption for firestoreOptions.
func (o *firestoreOptions) applyStreamManager(opts *streamManagerOptions) error {
	return o.applyFirestore(&opts.firestoreOptions)
}

// applySessionStore implements SessionStoreOption for firestoreOptions.
func (o *firestoreOptions) applySessionStore(opts *sessionStoreOptions) error {
	return o.applyFirestore(&opts.firestoreOptions)
}

// WithCollection sets the Firestore collection name where documents are stored.
// This option is required for all Firestore-based services.
func WithCollection(collection string) *firestoreOptions {
	return &firestoreOptions{Collection: collection}
}

// WithTTL sets how long documents are retained before Firestore auto-deletes them.
// Requires a TTL policy on the collection for the "expiresAt" field.
// Default is 5 minutes.
// See: https://firebase.google.com/docs/firestore/ttl
func WithTTL(ttl time.Duration) *firestoreOptions {
	return &firestoreOptions{TTL: ttl}
}
