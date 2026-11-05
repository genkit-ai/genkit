// Copyright 2026 Google LLC
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

// Package sessionstoretest checks [exp.SessionStore] implementations against
// the contract the agent runtime relies on.
//
// Run the suite from a test in the store's package:
//
//	func TestConformance(t *testing.T) {
//		sessionstoretest.Run(t, func(t *testing.T) aix.SessionStore[MyState] {
//			return newEmptyStore[MyState](t)
//		}, nil)
//	}
//
// The suite works only through the store's exported methods, so it suits any
// implementation, including one outside this module. It checks the optional
// capabilities ([exp.SnapshotMetadataReader] and [exp.SnapshotSubscriber]) when
// the store implements them, and [Options] turns on the checks that need more
// than one store: a second instance over the same storage, and a store that
// partitions its rows by a key from the context.
//
// The test data lives in the conversation (messages, artifacts, and usage) and
// the custom state keeps its zero value, so the suite runs with any State type,
// including against a store that supports only one State type.
//
// APIs in this package are under active development and may change in any
// minor version release.
package sessionstoretest

import (
	"context"
	"testing"
	"time"

	"github.com/firebase/genkit/go/ai/exp"
)

const (
	// eventTimeout bounds the wait for a subscription event the suite expects.
	// It is generous because a store may deliver through a remote listener or
	// a poll; a store that delivers at once is not slowed down by it.
	eventTimeout = 15 * time.Second
	// writers is how many goroutines a concurrency check runs at once.
	writers = 8
	// writeAttempts is how many times a concurrent writer tries one save. A
	// store with optimistic transactions may give up under contention, and
	// callers retry such a save, so the suite does too; what it checks is that
	// no committed write is lost or applied twice.
	writeAttempts = 25
	// chainLength is how many rows the linear chain check writes: enough to
	// cross several boundaries of a store that writes diffs with a full
	// checkpoint every few dozen rows.
	chainLength = 60
	// largeStateSize is the approximate JSON size of the large state, beyond
	// the single-document limits of common databases.
	largeStateSize = 5 << 19 // 2.5 MiB
)

// Options turns on the checks that need more than the store under test. A nil
// *Options runs the single-store checks only.
type Options[State any] struct {
	// Reopen returns a second store over the same storage as store, the way
	// another process would open it. With it, the suite checks that rows
	// written through one instance read back through the other, and that a
	// status change written through one reaches a subscriber on the other,
	// which is how one process aborts a detached run another process executes.
	// The suite subscribes through the reopened store and writes through the
	// original. Leave it nil when the storage is private to one instance.
	Reopen func(t *testing.T, store exp.SessionStore[State]) exp.SessionStore[State]

	// Scoped returns a fresh, empty store that partitions its rows by a key it
	// derives from each call's context, such as a tenant prefix, and two
	// contexts the store maps to different partitions. With it, the suite
	// checks that neither partition can read, change, or subscribe to the
	// other's rows, including under a snapshot ID both use. Leave it nil when
	// the store has no such partitioning.
	Scoped func(t *testing.T) (store exp.SessionStore[State], a, b context.Context)
}

// Run checks the stores newStore returns against the session store contract,
// as subtests of t grouped by area: Save, Lifecycle, Chains, Latest, Metadata,
// Subscriber, SecondInstance, and Scopes. A group whose capability the store
// lacks, or whose [Options] hook is nil, is skipped with the reason.
//
// newStore must return a fresh, empty store on every call, with its default
// behavior: retention that deletes old rows, for example, must be off, because
// the suite reads back every row it writes. Stores from separate calls may
// share storage as long as neither can see the other's rows. Run calls newStore
// once more than its subtests need, to probe the optional capabilities.
//
// The suite expects timestamps to round-trip to at least millisecond precision
// (it compares instants, so the time zone may change) and the state to
// round-trip as JSON. Its snapshot and session IDs contain only ASCII letters,
// digits, and hyphens. A store that cannot hold one of the larger cases (about
// 2.5 MiB of state) can leave it out with go test -skip.
func Run[State any](t *testing.T, newStore func(t *testing.T) exp.SessionStore[State], opts *Options[State]) {
	t.Helper()
	s := &suite[State]{newStore: newStore}
	if opts != nil {
		s.opts = *opts
	}
	// Every store from newStore has the same type, so one probes the optional
	// capabilities for all.
	probe := newStore(t)
	if probe == nil {
		t.Fatal("sessionstoretest.Run: newStore returned nil")
	}
	_, s.metadata = probe.(exp.SnapshotMetadataReader[State])
	_, s.subscriber = probe.(exp.SnapshotSubscriber)
	t.Run("Save", s.testSave)
	t.Run("Lifecycle", s.testLifecycle)
	t.Run("Chains", s.testChains)
	t.Run("Latest", s.testLatest)
	t.Run("Metadata", s.testMetadata)
	t.Run("Subscriber", s.testSubscriber)
	t.Run("SecondInstance", s.testSecondInstance)
	t.Run("Scopes", s.testScopes)
}

// suite holds what every check needs: the store factory, the options, and
// which optional capabilities the stores have.
type suite[State any] struct {
	newStore   func(t *testing.T) exp.SessionStore[State]
	opts       Options[State]
	metadata   bool // the stores implement exp.SnapshotMetadataReader
	subscriber bool // the stores implement exp.SnapshotSubscriber
}
