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

// Package core provides base error types and utilities for Genkit.
//
// The error surface in this file is deprecated in favour of
// [github.com/firebase/genkit/go/core/status], which unifies the two error
// types below into one and adds sentinel classification so callers can branch
// with errors.Is instead of matching on message text. Everything here is an
// alias or a thin wrapper over that package: [GenkitError] and [status.Error]
// are the same type, so an errors.As for either finds errors raised by any part
// of Genkit, old or new.
package core

import (
	"fmt"

	"github.com/firebase/genkit/go/core/status"
)

// GenkitError is the base error type for Genkit errors.
//
// Deprecated: use [status.Error]. This is an alias for it, so the two are the
// same type: an errors.As for a *GenkitError still matches every error Genkit
// raises, and a *status.Error can be used anywhere a *GenkitError is expected.
// Note that [status.Error] classifies failures with a sentinel, so prefer
// errors.Is against the sentinels in core/status (and the domain sentinels in
// ai, exp, and friends) over comparing the Status field.
type GenkitError = status.Error

// AsGenkitError returns err as a *GenkitError, wrapping it in a fresh
// one with status INTERNAL if it isn't one already. Returns nil for a
// nil input.
//
// Deprecated: use [status.Convert], or [status.Of] when you only need the
// status. Note that Convert derives the status from the error (mapping a
// cancelled context to CANCELLED, for instance) rather than always using
// INTERNAL.
func AsGenkitError(err error) *GenkitError { return status.Convert(err) }

// UserFacingError is the base error type for user facing errors.
//
// Deprecated: use [status.PublicErrorf], which produces a [status.Error] with
// Public set. Unlike this type, the result carries a sentinel and its status
// reaches HTTP transports, so a public INVALID_ARGUMENT returns 400 rather than
// falling through to 500.
type UserFacingError struct {
	Message string         `json:"message"` // Exclude from default JSON if embedded elsewhere
	Status  StatusName     `json:"status"`
	Details map[string]any `json:"details"` // Use map for arbitrary details
}

// NewPublicError allows a web framework handler to know it
// is safe to return the message in a request. Other kinds of errors will
// result in a generic 500 message to avoid the possibility of internal
// exceptions being leaked to attackers.
//
// Deprecated: use [status.PublicErrorf].
func NewPublicError(status StatusName, message string, details map[string]any) *UserFacingError {
	return &UserFacingError{
		Status:  status,
		Details: details,
		Message: message,
	}
}

// Error implements the standard error interface for UserFacingError.
func (e *UserFacingError) Error() string {
	return fmt.Sprintf("%s: %s", e.Status, e.Message)
}

// Unwrap returns the base sentinel for the error's status, so a UserFacingError
// classifies the same way a [status.Error] does: [status.Of] reports its Status
// rather than defaulting to INTERNAL, and errors.Is matches the corresponding
// base sentinel.
func (e *UserFacingError) Unwrap() error { return status.Base(e.Status) }

// PublicMessage reports the error's message as safe to return to clients.
// Transports call this to decide what reaches a client; implementing it keeps
// a UserFacingError public now that publicness is a property of the error
// rather than of its type.
func (e *UserFacingError) PublicMessage() (string, bool) { return e.Message, true }

// NewError creates a new GenkitError with a stack trace.
//
// Deprecated: use [status.Errorf] with a sentinel, which classifies the failure
// so callers can match it with errors.Is:
//
//	status.Errorf(status.ErrNotFound, "model %q not found", name)
//
// Record a cause with %w rather than relying on the implicit wrapping of the
// last error argument that this function performs.
func NewError(name StatusName, message string, args ...any) *GenkitError {
	ge := status.Errorf(status.Base(name), message, args...)
	// status.Base has no sentinel for names outside the canonical set (they
	// coerce to UNKNOWN) or for OK (an error cannot classify as success), but
	// v1 put whatever it was given on the wire. Restore it: this constructor's
	// contract is to behave exactly as it did, and the sentinel it was
	// classified with stays ErrUnknown, which is the honest classification.
	if ge.Status != name {
		ge.Status = name
		ge.HTTPCode = name.HTTPCode()
	}
	// v1 scanned args for the last error and wrapped it implicitly, with no %w
	// in the format. Preserve that so errors.Is and errors.As still reach it.
	for i := len(args) - 1; i >= 0; i-- {
		if err, ok := args[i].(error); ok {
			ge.WithCause(err)
			break
		}
	}
	// v1 recorded the stack in Details; format the one Errorf already captured
	// rather than capturing a second with debug.Stack.
	ge.Details = map[string]any{"stack": ge.Stack()}
	return ge
}
