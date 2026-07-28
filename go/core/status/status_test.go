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
	"net/http"
	"testing"
)

func TestNameHTTPCode(t *testing.T) {
	tests := []struct {
		name     string
		status   Name
		wantCode int
	}{
		{"OK", OK, http.StatusOK},
		{"CANCELLED", Cancelled, 499},
		{"UNKNOWN", Unknown, http.StatusInternalServerError},
		{"INVALID_ARGUMENT", InvalidArgument, http.StatusBadRequest},
		{"DEADLINE_EXCEEDED", DeadlineExceeded, http.StatusGatewayTimeout},
		{"NOT_FOUND", NotFound, http.StatusNotFound},
		{"ALREADY_EXISTS", AlreadyExists, http.StatusConflict},
		{"PERMISSION_DENIED", PermissionDenied, http.StatusForbidden},
		{"UNAUTHENTICATED", Unauthenticated, http.StatusUnauthorized},
		{"RESOURCE_EXHAUSTED", ResourceExhausted, http.StatusTooManyRequests},
		{"FAILED_PRECONDITION", FailedPrecondition, http.StatusBadRequest},
		{"ABORTED", Aborted, http.StatusConflict},
		{"OUT_OF_RANGE", OutOfRange, http.StatusBadRequest},
		{"UNIMPLEMENTED", Unimplemented, http.StatusNotImplemented},
		{"INTERNAL", Internal, http.StatusInternalServerError},
		{"UNAVAILABLE", Unavailable, http.StatusServiceUnavailable},
		{"DATA_LOSS", DataLoss, http.StatusInternalServerError},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.status.HTTPCode()
			if got != tt.wantCode {
				t.Errorf("%q.HTTPCode() = %d, want %d", tt.status, got, tt.wantCode)
			}
		})
	}

	t.Run("unknown status returns 500", func(t *testing.T) {
		got := Name("UNKNOWN_STATUS").HTTPCode()
		if got != http.StatusInternalServerError {
			t.Errorf("unknown Name.HTTPCode() = %d, want %d", got, http.StatusInternalServerError)
		}
	})
}

func TestFromHTTPCode(t *testing.T) {
	tests := []struct {
		code int
		want Name
	}{
		{http.StatusOK, OK},
		{499, Cancelled},
		{http.StatusBadRequest, InvalidArgument},
		{http.StatusUnauthorized, Unauthenticated},
		{http.StatusForbidden, PermissionDenied},
		{http.StatusNotFound, NotFound},
		{http.StatusConflict, Aborted},
		{http.StatusTooManyRequests, ResourceExhausted},
		{http.StatusInternalServerError, Internal},
		{http.StatusNotImplemented, Unimplemented},
		{http.StatusServiceUnavailable, Unavailable},
		{http.StatusGatewayTimeout, DeadlineExceeded},
		// Unmapped 5xx codes fall through to Internal.
		{http.StatusBadGateway, Internal},
		{599, Internal},
		// Unmapped non-5xx codes land on Unknown.
		{http.StatusTeapot, Unknown},
		{0, Unknown},
	}
	for _, tt := range tests {
		t.Run(http.StatusText(tt.code), func(t *testing.T) {
			if got := FromHTTPCode(tt.code); got != tt.want {
				t.Errorf("FromHTTPCode(%d) = %q, want %q", tt.code, got, tt.want)
			}
		})
	}
}

func TestNameIsValid(t *testing.T) {
	if !InvalidArgument.IsValid() {
		t.Error("INVALID_ARGUMENT.IsValid() = false")
	}
	if Name("NOPE").IsValid() {
		t.Error(`Name("NOPE").IsValid() = true`)
	}
	if got := Name("NOPE").Code(); got != 2 {
		t.Errorf(`Name("NOPE").Code() = %d, want 2 (Unknown)`, got)
	}
}

func TestNameCode(t *testing.T) {
	t.Run("every canonical name maps to its gRPC code", func(t *testing.T) {
		expectedMappings := map[Name]int{
			OK: 0, Cancelled: 1, Unknown: 2, InvalidArgument: 3,
			DeadlineExceeded: 4, NotFound: 5, AlreadyExists: 6,
			PermissionDenied: 7, ResourceExhausted: 8,
			FailedPrecondition: 9, Aborted: 10, OutOfRange: 11,
			Unimplemented: 12, Internal: 13, Unavailable: 14,
			DataLoss: 15, Unauthenticated: 16,
		}

		for name, wantCode := range expectedMappings {
			if got := name.Code(); got != wantCode {
				t.Errorf("%q.Code() = %d, want %d", name, got, wantCode)
			}
		}
	})
}

// The Go identifiers are Go-cased but the values they carry are the wire
// format, shared verbatim with the JS and Python runtimes. Drift here breaks
// cross-runtime compatibility silently, so pin every one of them.
func TestWireValues(t *testing.T) {
	want := map[Name]string{
		OK:                "OK",
		Cancelled:         "CANCELLED",
		Unknown:           "UNKNOWN",
		InvalidArgument:   "INVALID_ARGUMENT",
		DeadlineExceeded:  "DEADLINE_EXCEEDED",
		NotFound:          "NOT_FOUND",
		AlreadyExists:     "ALREADY_EXISTS",
		PermissionDenied:  "PERMISSION_DENIED",
		Unauthenticated:   "UNAUTHENTICATED",
		ResourceExhausted: "RESOURCE_EXHAUSTED",

		FailedPrecondition: "FAILED_PRECONDITION",
		Aborted:            "ABORTED",
		OutOfRange:         "OUT_OF_RANGE",
		Unimplemented:      "UNIMPLEMENTED",
		Internal:           "INTERNAL",
		Unavailable:        "UNAVAILABLE",
		DataLoss:           "DATA_LOSS",
	}
	for name, wire := range want {
		if string(name) != wire {
			t.Errorf("wire value drift: %q, want %q", string(name), wire)
		}
	}
	if len(want) != len(statuses) {
		t.Errorf("statuses has %d entries but only %d wire values are pinned", len(statuses), len(want))
	}
}
