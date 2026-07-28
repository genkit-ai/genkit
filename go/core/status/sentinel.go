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

// Sentinel classifies a failure. It pairs a status name with a short,
// stable label and is the first argument to [Errorf] and [PublicErrorf].
//
// Sentinels are comparable with [errors.Is], which is how callers branch on a
// failure mode instead of matching on message text:
//
//	if errors.Is(err, ai.ErrMaxTurnsExceeded) { ... }
//
// A sentinel created with [Sentinel.Subtype] inherits its parent's status and also
// matches the parent under [errors.Is], so callers can match at whichever
// granularity they need.
type Sentinel struct {
	status Name
	label  string
	parent *Sentinel
}

// NewSentinel returns a base sentinel carrying status. Prefer deriving from an
// existing sentinel with [Sentinel.Subtype]; use NewSentinel only when introducing
// a classification that no existing sentinel covers.
func NewSentinel(status Name, label string) *Sentinel {
	return &Sentinel{status: status, label: label}
}

// Subtype returns a more specific sentinel that inherits s's status and matches s
// under [errors.Is]. It is how packages declare domain failure modes:
//
//	var ErrMaxTurnsExceeded = status.ErrAborted.Subtype("max turns exceeded")
//
//	errors.Is(err, ErrMaxTurnsExceeded) // specific
//	errors.Is(err, status.ErrAborted)   // broad
func (s *Sentinel) Subtype(label string) *Sentinel {
	return &Sentinel{status: s.status, label: label, parent: s}
}

// Status returns the status name s carries.
func (s *Sentinel) Status() Name { return s.status }

// Error implements error so a sentinel can be returned or wrapped directly.
func (s *Sentinel) Error() string { return s.label }

// Unwrap returns the sentinel s was derived from, or nil for a base sentinel.
func (s *Sentinel) Unwrap() error {
	if s.parent == nil {
		return nil
	}
	return s.parent
}

// Base sentinels, one per status name. Reach for these when no more specific
// sentinel fits; otherwise prefer (or declare) a domain sentinel via
// [Sentinel.Subtype] so callers can branch on the actual failure mode.
var (
	ErrCancelled          = NewSentinel(Cancelled, "cancelled")
	ErrUnknown            = NewSentinel(Unknown, "unknown")
	ErrInvalidArgument    = NewSentinel(InvalidArgument, "invalid argument")
	ErrDeadlineExceeded   = NewSentinel(DeadlineExceeded, "deadline exceeded")
	ErrNotFound           = NewSentinel(NotFound, "not found")
	ErrAlreadyExists      = NewSentinel(AlreadyExists, "already exists")
	ErrPermissionDenied   = NewSentinel(PermissionDenied, "permission denied")
	ErrUnauthenticated    = NewSentinel(Unauthenticated, "unauthenticated")
	ErrResourceExhausted  = NewSentinel(ResourceExhausted, "resource exhausted")
	ErrFailedPrecondition = NewSentinel(FailedPrecondition, "failed precondition")
	ErrAborted            = NewSentinel(Aborted, "aborted")
	ErrOutOfRange         = NewSentinel(OutOfRange, "out of range")
	ErrUnimplemented      = NewSentinel(Unimplemented, "unimplemented")
	ErrInternal           = NewSentinel(Internal, "internal")
	ErrUnavailable        = NewSentinel(Unavailable, "unavailable")
	ErrDataLoss           = NewSentinel(DataLoss, "data loss")
)

// Framework-level sentinels for failures the action machinery raises. Domain
// sentinels live with the package that raises them (see ai.ErrModelNotFound,
// streaming.ErrStreamNotFound, and friends).
var (
	// ErrInvalidSchema means an action's declared input or output schema could
	// not be resolved or compiled. The schema itself is wrong, not the value.
	ErrInvalidSchema = ErrInvalidArgument.Subtype("invalid schema")

	// ErrInvalidInput means a value failed validation against an action's input
	// schema, e.g. a model produced malformed tool arguments.
	ErrInvalidInput = ErrInvalidArgument.Subtype("invalid input")

	// ErrInvalidOutput means an action produced a value that does not match its
	// declared output schema. This is a bug in the action, hence Internal.
	ErrInvalidOutput = ErrInternal.Subtype("invalid output")

	// ErrActionNotFound means no action is registered under the requested key.
	ErrActionNotFound = ErrNotFound.Subtype("action not found")

	// ErrPanic means a user-supplied function panicked and the framework
	// recovered at an action boundary.
	ErrPanic = ErrInternal.Subtype("panic")
)

// baseSentinels indexes the base sentinels by status name for [Base].
var baseSentinels = map[Name]*Sentinel{
	Cancelled:          ErrCancelled,
	Unknown:            ErrUnknown,
	InvalidArgument:    ErrInvalidArgument,
	DeadlineExceeded:   ErrDeadlineExceeded,
	NotFound:           ErrNotFound,
	AlreadyExists:      ErrAlreadyExists,
	PermissionDenied:   ErrPermissionDenied,
	Unauthenticated:    ErrUnauthenticated,
	ResourceExhausted:  ErrResourceExhausted,
	FailedPrecondition: ErrFailedPrecondition,
	Aborted:            ErrAborted,
	OutOfRange:         ErrOutOfRange,
	Unimplemented:      ErrUnimplemented,
	Internal:           ErrInternal,
	Unavailable:        ErrUnavailable,
	DataLoss:           ErrDataLoss,
}

// Base returns the base sentinel for a status name, or [ErrUnknown] if the name
// is not canonical. Use it when the status is only known at runtime, such as a
// plugin translating a provider's error code:
//
//	return status.Errorf(status.Base(status.FromHTTPCode(resp.StatusCode)),
//		"%s: %s", provider, body)
func Base(n Name) *Sentinel {
	if s, ok := baseSentinels[n]; ok {
		return s
	}
	return ErrUnknown
}
