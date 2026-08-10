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

package core

import (
	"net/http"
	"testing"
)

func TestHTTPStatusCode(t *testing.T) {
	tests := []struct {
		name     string
		status   StatusName
		wantCode int
	}{
		{"OK", OK, http.StatusOK},
		{"CANCELLED", CANCELLED, 499},
		{"UNKNOWN", UNKNOWN, http.StatusInternalServerError},
		{"INVALID_ARGUMENT", INVALID_ARGUMENT, http.StatusBadRequest},
		{"DEADLINE_EXCEEDED", DEADLINE_EXCEEDED, http.StatusGatewayTimeout},
		{"NOT_FOUND", NOT_FOUND, http.StatusNotFound},
		{"ALREADY_EXISTS", ALREADY_EXISTS, http.StatusConflict},
		{"PERMISSION_DENIED", PERMISSION_DENIED, http.StatusForbidden},
		{"UNAUTHENTICATED", UNAUTHENTICATED, http.StatusUnauthorized},
		{"RESOURCE_EXHAUSTED", RESOURCE_EXHAUSTED, http.StatusTooManyRequests},
		{"FAILED_PRECONDITION", FAILED_PRECONDITION, http.StatusBadRequest},
		{"ABORTED", ABORTED, http.StatusConflict},
		{"OUT_OF_RANGE", OUT_OF_RANGE, http.StatusBadRequest},
		{"UNIMPLEMENTED", UNIMPLEMENTED, http.StatusNotImplemented},
		{"INTERNAL", INTERNAL, http.StatusInternalServerError},
		{"UNAVAILABLE", UNAVAILABLE, http.StatusServiceUnavailable},
		{"DATA_LOSS", DATA_LOSS, http.StatusInternalServerError},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := HTTPStatusCode(tt.status)
			if got != tt.wantCode {
				t.Errorf("HTTPStatusCode(%q) = %d, want %d", tt.status, got, tt.wantCode)
			}
		})
	}

	t.Run("unknown status returns 500", func(t *testing.T) {
		got := HTTPStatusCode(StatusName("UNKNOWN_STATUS"))
		if got != http.StatusInternalServerError {
			t.Errorf("HTTPStatusCode(unknown) = %d, want %d", got, http.StatusInternalServerError)
		}
	})
}

func TestStatusFromHTTPCode(t *testing.T) {
	tests := []struct {
		code int
		want StatusName
	}{
		{http.StatusOK, OK},
		{499, CANCELLED},
		{http.StatusBadRequest, INVALID_ARGUMENT},
		{http.StatusUnauthorized, UNAUTHENTICATED},
		{http.StatusForbidden, PERMISSION_DENIED},
		{http.StatusNotFound, NOT_FOUND},
		{http.StatusConflict, ABORTED},
		{http.StatusTooManyRequests, RESOURCE_EXHAUSTED},
		{http.StatusInternalServerError, INTERNAL},
		{http.StatusNotImplemented, UNIMPLEMENTED},
		{http.StatusServiceUnavailable, UNAVAILABLE},
		{http.StatusGatewayTimeout, DEADLINE_EXCEEDED},
		// Unmapped 5xx codes fall through to INTERNAL.
		{http.StatusBadGateway, INTERNAL},
		{599, INTERNAL},
		// Unmapped non-5xx codes land on UNKNOWN.
		{http.StatusTeapot, UNKNOWN},
		{0, UNKNOWN},
	}
	for _, tt := range tests {
		t.Run(http.StatusText(tt.code), func(t *testing.T) {
			if got := StatusFromHTTPCode(tt.code); got != tt.want {
				t.Errorf("StatusFromHTTPCode(%d) = %q, want %q", tt.code, got, tt.want)
			}
		})
	}
}

func TestNewStatus(t *testing.T) {
	t.Run("creates status with name and message", func(t *testing.T) {
		s := NewStatus(NOT_FOUND, "resource not found")

		if s.Name != NOT_FOUND {
			t.Errorf("Name = %q, want %q", s.Name, NOT_FOUND)
		}
		if s.Message != "resource not found" {
			t.Errorf("Message = %q, want %q", s.Message, "resource not found")
		}
	})

	t.Run("creates status with empty message", func(t *testing.T) {
		s := NewStatus(OK, "")

		if s.Name != OK {
			t.Errorf("Name = %q, want %q", s.Name, OK)
		}
		if s.Message != "" {
			t.Errorf("Message = %q, want empty string", s.Message)
		}
	})
}

func TestStatusNameToCode(t *testing.T) {
	t.Run("maps all status names to codes", func(t *testing.T) {
		expectedMappings := map[StatusName]int{
			OK:                  CodeOK,
			CANCELLED:           CodeCancelled,
			UNKNOWN:             CodeUnknown,
			INVALID_ARGUMENT:    CodeInvalidArgument,
			DEADLINE_EXCEEDED:   CodeDeadlineExceeded,
			NOT_FOUND:           CodeNotFound,
			ALREADY_EXISTS:      CodeAlreadyExists,
			PERMISSION_DENIED:   CodePermissionDenied,
			UNAUTHENTICATED:     CodeUnauthenticated,
			RESOURCE_EXHAUSTED:  CodeResourceExhausted,
			FAILED_PRECONDITION: CodeFailedPrecondition,
			ABORTED:             CodeAborted,
			OUT_OF_RANGE:        CodeOutOfRange,
			UNIMPLEMENTED:       CodeUnimplemented,
			INTERNAL:            CodeInternal,
			UNAVAILABLE:         CodeUnavailable,
			DATA_LOSS:           CodeDataLoss,
		}

		for name, wantCode := range expectedMappings {
			got, ok := StatusNameToCode[name]
			if !ok {
				t.Errorf("StatusNameToCode missing mapping for %q", name)
				continue
			}
			if got != wantCode {
				t.Errorf("StatusNameToCode[%q] = %d, want %d", name, got, wantCode)
			}
		}
	})
}
