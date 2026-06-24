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

package localstore

import (
	"context"
	"errors"
	"time"
)

// fileStoreOptions holds the optional settings for a [FileSessionStore].
type fileStoreOptions struct {
	maxChain *int                         // Retention window (>= 1) when set; nil means unset.
	prefixFn func(context.Context) string // Derives the per-call subdirectory.
	poll     *time.Duration               // Cross-process poll interval when set; nil means unset.
}

// FileStoreOption configures a [FileSessionStore] at construction.
// It applies only to [NewFileSessionStore].
type FileStoreOption interface {
	applyFileStore(*fileStoreOptions) error
}

// applyFileStore merges o into opts, rejecting an option set more than once.
func (o *fileStoreOptions) applyFileStore(opts *fileStoreOptions) error {
	if o.maxChain != nil {
		if *o.maxChain < 1 {
			return errors.New("max persisted chain length must be at least 1 (WithMaxPersistedChainLength)")
		}
		if opts.maxChain != nil {
			return errors.New("cannot set max persisted chain length more than once (WithMaxPersistedChainLength)")
		}
		opts.maxChain = o.maxChain
	}

	if o.prefixFn != nil {
		if opts.prefixFn != nil {
			return errors.New("cannot set snapshot path prefix more than once (WithSnapshotPathPrefix)")
		}
		opts.prefixFn = o.prefixFn
	}

	if o.poll != nil {
		if opts.poll != nil {
			return errors.New("cannot set poll interval more than once (WithPollInterval)")
		}
		opts.poll = o.poll
	}

	return nil
}

// WithMaxPersistedChainLength bounds how many snapshots a single conversation
// chain keeps on disk. Each save walks the new snapshot's parentId chain and
// unlinks every row older than the newest n, capping per-conversation disk use.
//
// n must be at least 1; [NewFileSessionStore] rejects 0 or a negative value. A
// window of 1 retains only the latest snapshot (each save prunes its
// predecessor). Omitting the option entirely (the default) leaves pruning
// disabled, and chains grow without bound.
//
// Snapshots are self-contained (each carries the full session state), so
// dropping an old ancestor only removes it as a resume point; every surviving
// row remains fully loadable. Retention follows parentId links, whereas
// [FileSessionStore.GetLatestSnapshot] resolves recency by CreatedAt, so pruning
// only ever removes rows reachable from the saved snapshot's own chain; a
// sibling branch (e.g. after a regenerate) is pruned independently when it is
// itself extended.
func WithMaxPersistedChainLength(n int) FileStoreOption {
	return &fileStoreOptions{maxChain: &n}
}

// WithSnapshotPathPrefix derives a per-call subdirectory from the operation's
// context, isolating snapshots by tenant: a snapshot written under one prefix is
// visible only to calls that derive the same prefix. A typical fn pulls a stable
// identity (e.g. an authenticated user or org ID) out of ctx.
//
// The returned value may contain "/" to nest several levels
// (e.g. "org-42/user-7"); empty and separator-only results place snapshots
// directly under the store root. The value must be stable for a given
// snapshot's lifetime, since every read recomputes it - derive it from stable
// identity, not from per-request state. A value that would escape the store
// directory (contains "..", a backslash, or a segment starting with ".") is
// rejected at call time.
func WithSnapshotPathPrefix(fn func(ctx context.Context) string) FileStoreOption {
	return &fileStoreOptions{prefixFn: fn}
}

// WithPollInterval sets how often the store re-reads subscribed snapshot files
// to surface status changes that other processes (or other store instances)
// sharing the directory write through [FileSessionStore.OnSnapshotStatusChange].
// That cross-process visibility is what lets one process observe an abort (or
// any status change) another process commits, for example to stop a detached
// turn it is running.
//
// The default is one second. A value <= 0 disables cross-process polling:
// subscriptions then observe only changes written through this same store
// instance.
func WithPollInterval(d time.Duration) FileStoreOption {
	return &fileStoreOptions{poll: &d}
}
