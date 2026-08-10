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

package status

import (
	"errors"
	"fmt"
	"io"
	"testing"
)

// Error exposes a single cause through Unwrap and matches its sentinel through
// Is, rather than putting both in an Unwrap() []error. The stdlib errors.Unwrap
// returns nil for a multi-cause unwrapper, so the []error form would silently
// truncate the hand-rolled chain walks that logging, telemetry, and third-party
// error tooling do. These tests pin that the single-cause form loses nothing.
func TestUnwrapIsWalkableByStdlib(t *testing.T) {
	cause := errors.New("boom")
	err := error(Errorf(ErrNotFound, "wrapped: %w", cause))

	if got := errors.Unwrap(err); got != cause {
		t.Errorf("errors.Unwrap = %v, want %v", got, cause)
	}

	depth := 0
	for e := fmt.Errorf("outer: %w", err); e != nil; e = errors.Unwrap(e) {
		depth++
	}
	if depth != 3 { // fmt wrapper -> Error -> cause
		t.Errorf("chain walk depth = %d, want 3", depth)
	}
}

func TestUnwrapIsNilWithoutCause(t *testing.T) {
	if got := errors.Unwrap(error(Errorf(ErrNotFound, "no cause here"))); got != nil {
		t.Errorf("errors.Unwrap = %v, want nil", got)
	}
}

func TestIsMatchesSentinelAtEveryLevel(t *testing.T) {
	mid := ErrAborted.Subtype("mid")
	leaf := mid.Subtype("leaf")
	err := error(Errorf(leaf, "boom"))

	for _, target := range []*Sentinel{leaf, mid, ErrAborted} {
		if !errors.Is(err, target) {
			t.Errorf("errors.Is(err, %v) = false", target)
		}
	}
	if errors.Is(err, ErrNotFound) {
		t.Error("errors.Is(err, ErrNotFound) = true, want false")
	}
	// A sibling subtype of the same parent must not match.
	if sibling := ErrAborted.Subtype("sibling"); errors.Is(err, sibling) {
		t.Error("errors.Is(err, sibling) = true, want false")
	}
}

// Sentinel matching and cause matching are independent: adding a cause must not
// shadow the sentinel, and classifying must not hide the cause.
func TestIsMatchesSentinelAndCauseTogether(t *testing.T) {
	err := error(Errorf(ErrNotFound.Subtype("model not found"), "model %q: %w", "x", io.EOF))
	if !errors.Is(err, io.EOF) {
		t.Error("errors.Is(io.EOF) = false")
	}
	if !errors.Is(err, ErrNotFound) {
		t.Error("errors.Is(ErrNotFound) = false")
	}

	// Through an intervening fmt wrapper as well.
	wrapped := fmt.Errorf("context: %w", err)
	if !errors.Is(wrapped, io.EOF) || !errors.Is(wrapped, ErrNotFound) {
		t.Error("matching broke through fmt.Errorf")
	}
}

func TestWithCauseRecordsWithoutChangingMessage(t *testing.T) {
	cause := errors.New("boom")
	err := Errorf(ErrInternal, "tool %q failed", "weather").WithCause(cause)

	if got, want := err.Error(), `tool "weather" failed`; got != want {
		t.Errorf("Error() = %q, want %q", got, want)
	}
	if !errors.Is(err, cause) {
		t.Error("errors.Is(cause) = false")
	}
	if !errors.Is(err, ErrInternal) {
		t.Error("errors.Is(ErrInternal) = false")
	}
	// A second call is a no-op: the first cause wins.
	if err.WithCause(errors.New("other")); errors.Unwrap(error(err)) != cause {
		t.Error("second WithCause overwrote the first cause")
	}
}

// A literal-constructed Error (no sentinel) must not panic in Is, and still
// reports its status.
func TestZeroSentinelIsSafe(t *testing.T) {
	err := &Error{Status: NotFound, Message: "decoded from the wire"}
	if errors.Is(err, ErrNotFound) {
		t.Error("errors.Is = true for an Error with no sentinel")
	}
	if got := Of(err); got != NotFound {
		t.Errorf("Of = %q, want %q", got, NotFound)
	}
	if got := errors.Unwrap(error(err)); got != nil {
		t.Errorf("Unwrap = %v, want nil", got)
	}
}
