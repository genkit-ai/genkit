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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"runtime"
	"strings"

	"github.com/firebase/genkit/go/internal/base"
	"github.com/invopop/jsonschema"
)

// Error is Genkit's error type. It carries a canonical [Name] status, the
// [Sentinel] that classified it, and any wrapped causes.
//
// On the wire an Error marshals to the canonical Genkit error shape
// ({status, message, details}), which mirrors the RuntimeError definition in
// the shared JSON schema. Fields that exist only in-process (Public, the
// sentinel, the causes, the stack) are not serialized.
//
// Construct one with [Errorf] or [PublicErrorf]. To add context to an existing
// error without reclassifying it, use fmt.Errorf with %w instead.
type Error struct {
	// Status is the canonical status name for this failure. Wire field "status".
	Status Name
	// Message describes the failure. Wire field "message".
	Message string
	// Public reports whether Message is safe to return to a client. Transports
	// replace the message of a non-public error with a generic one so internal
	// details do not leak. Not serialized.
	Public bool
	// Details is optional structured information about the failure.
	// Wire field "details" (omitted when empty).
	Details map[string]any

	sentinel *Sentinel
	// unwrap is the sentinel followed by any %w causes, precomputed because
	// errors.Is and errors.As call Unwrap once per node per traversal.
	unwrap []error
	stack  []uintptr
}

// Errorf returns an [Error] classified by sentinel, with a message built as by
// fmt.Errorf. Use %w in format to record a cause: the cause stays reachable
// through [errors.Is] and [errors.As] alongside the sentinel.
//
//	return status.Errorf(status.ErrNotFound, "model %q not found", name)
//	return status.Errorf(ai.ErrToolFailed, "tool %q: %w", tool, err)
//
// A nil sentinel is treated as [ErrInternal].
func Errorf(sentinel *Sentinel, format string, args ...any) *Error {
	return newError(sentinel, false, format, args...)
}

// PublicErrorf is [Errorf] for a message that is safe to return to clients.
// Transports may surface the message verbatim, so it must not contain internal
// details. Everything else is a generic message and the status code alone.
func PublicErrorf(sentinel *Sentinel, format string, args ...any) *Error {
	return newError(sentinel, true, format, args...)
}

func newError(sentinel *Sentinel, public bool, format string, args ...any) *Error {
	if sentinel == nil {
		sentinel = ErrInternal
	}
	formatted := fmt.Errorf(format, args...)
	msg := formatted.Error()
	if msg == "" {
		msg = sentinel.label
	}
	return &Error{
		Status:   sentinel.status,
		Message:  msg,
		Public:   public,
		sentinel: sentinel,
		unwrap:   append([]error{sentinel}, unwrapAll(formatted)...),
		stack:    callers(4),
	}
}

// WithDetails attaches structured details and returns e, for chaining onto a
// constructor. Details are serialized and reach clients, so keep them free of
// internal information unless the error is public.
func (e *Error) WithDetails(details map[string]any) *Error {
	e.Details = details
	return e
}

// Error implements error. It returns Message alone: the sentinel is a
// classification label, not a message prefix, so callers control the wording.
func (e *Error) Error() string { return e.Message }

// Unwrap returns the classifying sentinel followed by any causes recorded via
// %w, so [errors.Is] matches the sentinel (and its parents) as well as anything
// the caller wrapped.
func (e *Error) Unwrap() []error { return e.unwrap }

// Sentinel returns the sentinel that classified e, or nil if it was decoded
// from the wire rather than constructed in this process.
func (e *Error) Sentinel() *Sentinel { return e.sentinel }

// HTTPCode returns the HTTP status code for e's status.
func (e *Error) HTTPCode() int { return e.Status.HTTPCode() }

// Stack returns the call stack captured when e was constructed, formatted like
// a panic trace, or "" for an error decoded from the wire. It is formatted on
// demand: construction only records program counters.
func (e *Error) Stack() string { return formatStack(e.stack) }

// Of returns the status of err.
//
// It reports the status of the outermost [Error] in the chain, so a boundary
// that deliberately reclassifies with [Errorf] wins over anything beneath it. A
// bare [Sentinel] reports its own status. Context cancellation and deadline
// errors map to Cancelled and DeadlineExceeded. Anything else is Internal: an
// unclassified failure is a failure of ours, not of the caller's request.
//
// Of(nil) is OK.
func Of(err error) Name {
	if err == nil {
		return OK
	}
	var e *Error
	if errors.As(err, &e) {
		return e.Status
	}
	var s *Sentinel
	if errors.As(err, &s) {
		return s.status
	}
	switch {
	case errors.Is(err, context.Canceled):
		return Cancelled
	case errors.Is(err, context.DeadlineExceeded):
		return DeadlineExceeded
	}
	return Internal
}

// Convert returns err as an [Error], converting it if it is not one already.
// The converted error takes its status from [Of] and is never public. Returns
// nil for a nil err.
//
// Prefer errors.As when you need to know whether err really is an [Error]; this
// is for boundaries that must produce one either way.
func Convert(err error) *Error {
	if err == nil {
		return nil
	}
	var e *Error
	if errors.As(err, &e) {
		return e
	}
	return &Error{Status: Of(err), Message: err.Error(), unwrap: []error{err}}
}

// PublicMessage returns a message for err that is safe to show a client, and
// whether it came from the error itself. When the outermost [Error] is public
// its Message is returned verbatim; otherwise the result is a generic string
// derived from the status, so internal details never reach the client.
//
// Transports should use this instead of err.Error(). Note that the fallback is
// deliberately uninformative: log err separately for diagnosis.
func PublicMessage(err error) (msg string, public bool) {
	if err == nil {
		return "", false
	}
	var e *Error
	if errors.As(err, &e) && e.Public {
		return e.Message, true
	}
	return genericMessage(Of(err)), false
}

func genericMessage(n Name) string {
	if s, ok := baseSentinels[n]; ok {
		return s.label
	}
	return "internal"
}

// unwrapAll returns the errors err wraps, handling both the single- and
// multi-cause forms fmt.Errorf produces.
func unwrapAll(err error) []error {
	switch x := err.(type) {
	case interface{ Unwrap() error }:
		if c := x.Unwrap(); c != nil {
			return []error{c}
		}
	case interface{ Unwrap() []error }:
		return x.Unwrap()
	}
	return nil
}

// maxStackDepth bounds the frames recorded per error. Deep enough to reach a
// user's own code from anywhere in the framework.
const maxStackDepth = 64

// callers records the stack starting at the frame skip levels above
// runtime.Callers itself (4 == the caller of Errorf/PublicErrorf).
func callers(skip int) []uintptr {
	pcs := make([]uintptr, maxStackDepth)
	return pcs[:runtime.Callers(skip, pcs)]
}

func formatStack(pcs []uintptr) string {
	if len(pcs) == 0 {
		return ""
	}
	var b strings.Builder
	frames := runtime.CallersFrames(pcs)
	for {
		f, more := frames.Next()
		fmt.Fprintf(&b, "%s\n\t%s:%d\n", f.Function, f.File, f.Line)
		if !more {
			break
		}
	}
	return b.String()
}

// MarshalJSON encodes e in the canonical Genkit error wire format
// ({status, message, details}). The wire shape ([errorWire]) is generated from
// the shared JSON schema's RuntimeError definition.
//
// The captured stack is in-process diagnostics, not wire data, so errors
// embedded in values (a failed agent invocation's output, say) do not leak
// process internals to clients. Consumers that want the stack read
// [Error.Stack] directly.
func (e *Error) MarshalJSON() ([]byte, error) {
	return json.Marshal(errorWire{
		Status:  e.Status,
		Message: e.Message,
		Details: e.Details,
	})
}

// UnmarshalJSON decodes an Error from the canonical wire format. The result
// carries no sentinel, causes, or stack: those do not cross the wire.
func (e *Error) UnmarshalJSON(data []byte) error {
	var w errorWire
	if err := json.Unmarshal(data, &w); err != nil {
		return err
	}
	e.Status = w.Status
	e.Message = w.Message
	e.Details = w.Details
	return nil
}

// JSONSchema describes the error's wire format for schema inference. Without
// it, inference would reflect over the struct fields, requiring in-process
// fields that MarshalJSON never emits, so values embedding an Error would fail
// validation against their own inferred schema.
func (Error) JSONSchema() *jsonschema.Schema {
	return base.InferJSONSchema(errorWire{})
}
