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

// Package status defines canonical status codes, names, and related types
// inspired by gRPC status codes.
package core

import "net/http" // Import standard http package for status codes

// StatusName defines the set of canonical status names.
type StatusName string

// Constants for canonical status names.
const (
	OK                  StatusName = "OK"
	CANCELLED           StatusName = "CANCELLED"
	UNKNOWN             StatusName = "UNKNOWN"
	INVALID_ARGUMENT    StatusName = "INVALID_ARGUMENT"
	DEADLINE_EXCEEDED   StatusName = "DEADLINE_EXCEEDED"
	NOT_FOUND           StatusName = "NOT_FOUND"
	ALREADY_EXISTS      StatusName = "ALREADY_EXISTS"
	PERMISSION_DENIED   StatusName = "PERMISSION_DENIED"
	UNAUTHENTICATED     StatusName = "UNAUTHENTICATED"
	RESOURCE_EXHAUSTED  StatusName = "RESOURCE_EXHAUSTED"
	FAILED_PRECONDITION StatusName = "FAILED_PRECONDITION"
	ABORTED             StatusName = "ABORTED"
	OUT_OF_RANGE        StatusName = "OUT_OF_RANGE"
	UNIMPLEMENTED       StatusName = "UNIMPLEMENTED"
	INTERNAL            StatusName = "INTERNAL_SERVER_ERROR"
	UNAVAILABLE         StatusName = "UNAVAILABLE"
	DATA_LOSS           StatusName = "DATA_LOSS"
)

// Constants for canonical status codes (integer values).
const (
	// CodeOK means not an error; returned on success.
	CodeOK = 0
	// CodeCancelled means the operation was cancelled, typically by the caller.
	CodeCancelled = 1
	// CodeUnknown means an unknown error occurred.
	CodeUnknown = 2
	// CodeInvalidArgument means the client specified an invalid argument.
	CodeInvalidArgument = 3
	// CodeDeadlineExceeded means the deadline expired before the operation could complete.
	CodeDeadlineExceeded = 4
	// CodeNotFound means some requested entity (e.g., file or directory) was not found.
	CodeNotFound = 5
	// CodeAlreadyExists means the entity that a client attempted to create already exists.
	CodeAlreadyExists = 6
	// CodePermissionDenied means the caller does not have permission to execute the operation.
	CodePermissionDenied = 7
	// CodeUnauthenticated means the request does not have valid authentication credentials.
	CodeUnauthenticated = 16
	// CodeResourceExhausted means some resource has been exhausted.
	CodeResourceExhausted = 8
	// CodeFailedPrecondition means the operation was rejected because the system is not in a state required.
	CodeFailedPrecondition = 9
	// CodeAborted means the operation was aborted, typically due to some issue.
	CodeAborted = 10
	// CodeOutOfRange means the operation was attempted past the valid range.
	CodeOutOfRange = 11
	// CodeUnimplemented means the operation is not implemented or not supported/enabled.
	CodeUnimplemented = 12
	// CodeInternal means internal errors. Some invariants expected by the underlying system were broken.
	CodeInternal = 13
	// CodeUnavailable means the service is currently unavailable.
	CodeUnavailable = 14
	// CodeDataLoss means unrecoverable data loss or corruption.
	CodeDataLoss = 15
)

// StatusNameToCode maps status names to their integer code values.
// Exported for potential use elsewhere if needed.
var StatusNameToCode = map[StatusName]int{
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

// statusNameToHTTPCode maps status names to HTTP status codes.
// Kept unexported as it's primarily used by the HTTPStatusCode function.
var statusNameToHTTPCode = map[StatusName]int{
	OK:                  http.StatusOK,                  // 200
	CANCELLED:           499,                            // Client Closed Request (non-standard but common)
	UNKNOWN:             http.StatusInternalServerError, // 500
	INVALID_ARGUMENT:    http.StatusBadRequest,          // 400
	DEADLINE_EXCEEDED:   http.StatusGatewayTimeout,      // 504
	NOT_FOUND:           http.StatusNotFound,            // 404
	ALREADY_EXISTS:      http.StatusConflict,            // 409
	PERMISSION_DENIED:   http.StatusForbidden,           // 403
	UNAUTHENTICATED:     http.StatusUnauthorized,        // 401
	RESOURCE_EXHAUSTED:  http.StatusTooManyRequests,     // 429
	FAILED_PRECONDITION: http.StatusBadRequest,          // 400
	ABORTED:             http.StatusConflict,            // 409
	OUT_OF_RANGE:        http.StatusBadRequest,          // 400
	UNIMPLEMENTED:       http.StatusNotImplemented,      // 501
	INTERNAL:            http.StatusInternalServerError, // 500
	UNAVAILABLE:         http.StatusServiceUnavailable,  // 503
	DATA_LOSS:           http.StatusInternalServerError, // 500
}

// httpCodeToStatusName is the canonical reverse of [statusNameToHTTPCode].
// Several StatusNames share an HTTP code (e.g. 400 maps from INVALID_ARGUMENT,
// FAILED_PRECONDITION, and OUT_OF_RANGE); this table picks the canonical
// gRPC choice in each case, matching
// https://cloud.google.com/apis/design/errors.
var httpCodeToStatusName = map[int]StatusName{
	http.StatusOK:                  OK,
	499:                            CANCELLED,
	http.StatusBadRequest:          INVALID_ARGUMENT,
	http.StatusGatewayTimeout:      DEADLINE_EXCEEDED,
	http.StatusNotFound:            NOT_FOUND,
	http.StatusConflict:            ABORTED,
	http.StatusForbidden:           PERMISSION_DENIED,
	http.StatusUnauthorized:        UNAUTHENTICATED,
	http.StatusTooManyRequests:     RESOURCE_EXHAUSTED,
	http.StatusNotImplemented:      UNIMPLEMENTED,
	http.StatusInternalServerError: INTERNAL,
	http.StatusServiceUnavailable:  UNAVAILABLE,
}

// HTTPStatusCode gets the corresponding HTTP status code for a given Genkit status name.
func HTTPStatusCode(name StatusName) int {
	if code, ok := statusNameToHTTPCode[name]; ok {
		return code
	}

	return http.StatusInternalServerError
}

// StatusFromHTTPCode returns the canonical [StatusName] for an HTTP status
// code, following the gRPC / Google API reverse mapping. Any 5xx code with no
// explicit entry falls through to INTERNAL; unmapped 4xx codes return UNKNOWN.
//
// This is intended for plugins wrapping HTTP-based SDK errors so that
// status-aware middleware (retry, fallback, ...) can reason about them.
func StatusFromHTTPCode(code int) StatusName {
	if s, ok := httpCodeToStatusName[code]; ok {
		return s
	}
	if code >= 500 {
		return INTERNAL
	}
	return UNKNOWN
}

// Status represents a status condition, typically used in responses or errors.
type Status struct {
	Name    StatusName `json:"name"`
	Message string     `json:"message,omitempty"`
}

// NewStatus creates a new Status object.
func NewStatus(name StatusName, message string) *Status {
	return &Status{
		Name:    name,
		Message: message,
	}
}
