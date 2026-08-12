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

import "github.com/firebase/genkit/go/core/status"

// StatusName defines the set of canonical status names.
//
// Deprecated: use [status.Name]. This is an alias for it, so the two are the
// same type and values are interchangeable.
type StatusName = status.Name

// Constants for canonical status names.
//
// Deprecated: use the Go-cased constants in [github.com/firebase/genkit/go/core/status]
// ([status.InvalidArgument], [status.NotFound], ...). These alias them, so the
// values are identical and the wire format is unchanged.
const (
	OK                  = status.OK
	CANCELLED           = status.Cancelled
	UNKNOWN             = status.Unknown
	INVALID_ARGUMENT    = status.InvalidArgument
	DEADLINE_EXCEEDED   = status.DeadlineExceeded
	NOT_FOUND           = status.NotFound
	ALREADY_EXISTS      = status.AlreadyExists
	PERMISSION_DENIED   = status.PermissionDenied
	UNAUTHENTICATED     = status.Unauthenticated
	RESOURCE_EXHAUSTED  = status.ResourceExhausted
	FAILED_PRECONDITION = status.FailedPrecondition
	ABORTED             = status.Aborted
	OUT_OF_RANGE        = status.OutOfRange
	UNIMPLEMENTED       = status.Unimplemented
	INTERNAL            = status.Internal
	UNAVAILABLE         = status.Unavailable
	DATA_LOSS           = status.DataLoss
)

// Constants for canonical status codes (integer values).
//
// Deprecated: use [status.Name.Code].
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
//
// Deprecated: use [status.Name.Code], which is correct for every name rather
// than only the ones present in this map.
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

// HTTPStatusCode gets the corresponding HTTP status code for a given Genkit status name.
//
// Deprecated: use [status.Name.HTTPCode].
func HTTPStatusCode(name StatusName) int { return name.HTTPCode() }

// StatusFromHTTPCode returns the canonical [StatusName] for an HTTP status
// code, following the gRPC / Google API reverse mapping.
//
// Deprecated: use [status.FromHTTPCode].
func StatusFromHTTPCode(code int) StatusName { return status.FromHTTPCode(code) }

// Status represents a status condition, typically used in responses or errors.
//
// Deprecated: use [status.Error], which carries a status alongside the message
// and participates in errors.Is and errors.As.
type Status struct {
	Name    StatusName `json:"name"`
	Message string     `json:"message,omitempty"`
}

// NewStatus creates a new Status object.
//
// Deprecated: use [status.Errorf].
func NewStatus(name StatusName, message string) *Status {
	return &Status{
		Name:    name,
		Message: message,
	}
}
