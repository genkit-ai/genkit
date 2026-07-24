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

package exp

import "github.com/firebase/genkit/go/core/status"

// Failure modes agents and session stores report. Match them with errors.Is.
var (
	// ErrSnapshotNotFound means no snapshot exists under the given ID, or the
	// session has none yet.
	ErrSnapshotNotFound = status.ErrNotFound.Subtype("snapshot not found")

	// ErrNoSessionStore means the operation needs server-managed state but the
	// agent was defined without WithSessionStore.
	ErrNoSessionStore = status.ErrFailedPrecondition.Subtype("no session store")

	// ErrSessionIDRequired means a snapshot reached a store without a session ID.
	// Every store implementation rejects this: a snapshot with no session cannot
	// be resolved back to a conversation.
	ErrSessionIDRequired = status.ErrInvalidArgument.Subtype("session ID is required")
)
