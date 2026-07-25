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

import "net/http"

// Name is a canonical status name, drawn from the gRPC status codes. It is the
// value Genkit puts on the wire, shared by the Go, JS, and Python runtimes.
type Name string

// The canonical status names.
const (
	OK                 Name = "OK"
	Cancelled          Name = "CANCELLED"
	Unknown            Name = "UNKNOWN"
	InvalidArgument    Name = "INVALID_ARGUMENT"
	DeadlineExceeded   Name = "DEADLINE_EXCEEDED"
	NotFound           Name = "NOT_FOUND"
	AlreadyExists      Name = "ALREADY_EXISTS"
	PermissionDenied   Name = "PERMISSION_DENIED"
	Unauthenticated    Name = "UNAUTHENTICATED"
	ResourceExhausted  Name = "RESOURCE_EXHAUSTED"
	FailedPrecondition Name = "FAILED_PRECONDITION"
	Aborted            Name = "ABORTED"
	OutOfRange         Name = "OUT_OF_RANGE"
	Unimplemented      Name = "UNIMPLEMENTED"
	Internal           Name = "INTERNAL"
	Unavailable        Name = "UNAVAILABLE"
	DataLoss           Name = "DATA_LOSS"
)

// statuses is the canonical table: for each status name, the gRPC integer code
// and the HTTP status it maps to. Membership defines [Name.IsValid].
//
// Codes and HTTP mappings both follow https://cloud.google.com/apis/design/errors.
var statuses = map[Name]struct{ code, httpCode int }{
	OK:                 {0, http.StatusOK},                   // 200
	Cancelled:          {1, 499},                             // Client Closed Request (non-standard but common)
	Unknown:            {2, http.StatusInternalServerError},  // 500
	InvalidArgument:    {3, http.StatusBadRequest},           // 400
	DeadlineExceeded:   {4, http.StatusGatewayTimeout},       // 504
	NotFound:           {5, http.StatusNotFound},             // 404
	AlreadyExists:      {6, http.StatusConflict},             // 409
	PermissionDenied:   {7, http.StatusForbidden},            // 403
	ResourceExhausted:  {8, http.StatusTooManyRequests},      // 429
	FailedPrecondition: {9, http.StatusBadRequest},           // 400
	Aborted:            {10, http.StatusConflict},            // 409
	OutOfRange:         {11, http.StatusBadRequest},          // 400
	Unimplemented:      {12, http.StatusNotImplemented},      // 501
	Internal:           {13, http.StatusInternalServerError}, // 500
	Unavailable:        {14, http.StatusServiceUnavailable},  // 503
	DataLoss:           {15, http.StatusInternalServerError}, // 500
	Unauthenticated:    {16, http.StatusUnauthorized},        // 401
}

// httpCodeToName is the canonical reverse of the HTTP column above. Several
// names share an HTTP code (400 maps from InvalidArgument, FailedPrecondition,
// and OutOfRange); this table picks the canonical gRPC choice in each case.
var httpCodeToName = map[int]Name{
	http.StatusOK:                  OK,
	499:                            Cancelled,
	http.StatusBadRequest:          InvalidArgument,
	http.StatusGatewayTimeout:      DeadlineExceeded,
	http.StatusNotFound:            NotFound,
	http.StatusConflict:            Aborted,
	http.StatusForbidden:           PermissionDenied,
	http.StatusUnauthorized:        Unauthenticated,
	http.StatusTooManyRequests:     ResourceExhausted,
	http.StatusNotImplemented:      Unimplemented,
	http.StatusInternalServerError: Internal,
	http.StatusServiceUnavailable:  Unavailable,
}

// IsValid reports whether n is one of the canonical status names.
func (n Name) IsValid() bool {
	_, ok := statuses[n]
	return ok
}

// Code returns the gRPC integer code for n, or 2 (Unknown) if n is not
// canonical.
func (n Name) Code() int {
	if s, ok := statuses[n]; ok {
		return s.code
	}
	return statuses[Unknown].code
}

// HTTPCode returns the HTTP status code for n, or 500 if n is not canonical.
func (n Name) HTTPCode() int {
	if s, ok := statuses[n]; ok {
		return s.httpCode
	}
	return http.StatusInternalServerError
}

// FromHTTPCode returns the canonical status name for an HTTP status code,
// following the gRPC / Google API reverse mapping. Any 5xx code with no
// explicit entry falls through to Internal; unmapped 4xx codes return Unknown.
//
// This is intended for plugins wrapping HTTP-based SDK errors so that
// status-aware middleware (retry, fallback, ...) can reason about them.
func FromHTTPCode(code int) Name {
	if n, ok := httpCodeToName[code]; ok {
		return n
	}
	if code >= 500 {
		return Internal
	}
	return Unknown
}
