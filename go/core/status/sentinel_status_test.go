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

import "testing"

// Every base sentinel must carry the status it is named for. A mismatch would
// silently change the HTTP code and the retry/fallback decision for every site
// classified with it.
func TestBaseSentinelStatuses(t *testing.T) {
	want := map[*Sentinel]Name{
		ErrCancelled:          Cancelled,
		ErrUnknown:            Unknown,
		ErrInvalidArgument:    InvalidArgument,
		ErrDeadlineExceeded:   DeadlineExceeded,
		ErrNotFound:           NotFound,
		ErrAlreadyExists:      AlreadyExists,
		ErrPermissionDenied:   PermissionDenied,
		ErrUnauthenticated:    Unauthenticated,
		ErrResourceExhausted:  ResourceExhausted,
		ErrFailedPrecondition: FailedPrecondition,
		ErrAborted:            Aborted,
		ErrOutOfRange:         OutOfRange,
		ErrUnimplemented:      Unimplemented,
		ErrInternal:           Internal,
		ErrUnavailable:        Unavailable,
		ErrDataLoss:           DataLoss,
	}
	for s, n := range want {
		if got := s.Status(); got != n {
			t.Errorf("%v.Status() = %q, want %q", s, got, n)
		}
		if Base(n) != s {
			t.Errorf("Base(%q) is not the sentinel named for it", n)
		}
	}
	if len(want) != len(baseSentinels) {
		t.Errorf("checked %d sentinels, table has %d", len(want), len(baseSentinels))
	}
	// Every canonical status needs a base sentinel, or code classifying at
	// runtime through Base silently degrades that status to UNKNOWN. OK is the
	// exception: success is not a failure to classify.
	for n := range statuses {
		if n == OK {
			continue
		}
		if _, ok := baseSentinels[n]; !ok {
			t.Errorf("status %q has no base sentinel; Base(%q) would answer ErrUnknown", n, n)
		}
	}
}

// The framework sentinels must keep the statuses their call sites sent before
// they were classified, or the migration silently changed HTTP codes.
func TestFrameworkSentinelStatuses(t *testing.T) {
	for _, tt := range []struct {
		name string
		s    *Sentinel
		want Name
	}{
		{"ErrInvalidSchema", ErrInvalidSchema, InvalidArgument},
		{"ErrInvalidInput", ErrInvalidInput, InvalidArgument},
		{"ErrInvalidOutput", ErrInvalidOutput, Internal},
		{"ErrActionNotFound", ErrActionNotFound, NotFound},
		{"ErrPanic", ErrPanic, Internal},
	} {
		if got := tt.s.Status(); got != tt.want {
			t.Errorf("%s.Status() = %q, want %q", tt.name, got, tt.want)
		}
	}
}

// A subtype inherits its parent's status. This is what lets a call site swap a
// base sentinel for a domain one without changing what the client sees.
func TestSubtypeInheritsStatusTransitively(t *testing.T) {
	a := ErrFailedPrecondition.Subtype("a")
	b := a.Subtype("b")
	for _, s := range []*Sentinel{a, b} {
		if got := s.Status(); got != FailedPrecondition {
			t.Errorf("%v.Status() = %q, want %q", s, got, FailedPrecondition)
		}
	}
	if got := Errorf(b, "x").Status; got != FailedPrecondition {
		t.Errorf("Errorf(subtype).Status = %q, want %q", got, FailedPrecondition)
	}
}
