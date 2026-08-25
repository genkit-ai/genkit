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
	"maps"
	"runtime"
	"strings"

	"github.com/firebase/genkit/go/internal/base"
	"github.com/invopop/jsonschema"
)

// Error is Genkit's error type. It carries a canonical [Name] status, the
// [Sentinel] that classified it, and any wrapped cause.
//
// On the wire an Error marshals to the canonical Genkit error shape
// ({status, message, details}), which mirrors the RuntimeError definition in
// the shared JSON schema. Fields that exist only in-process (Public, the
// sentinel, the cause, the stack) are not serialized.
//
// Construct one with [Errorf] or [PublicErrorf]. To add context to an existing
// error without reclassifying it, use fmt.Errorf with %w instead.
//
// # Nil receivers
//
// Error's methods, and the package functions that inspect an error, tolerate a
// nil *Error. This matters because Genkit hands out *Error in places that are
// nil in the ordinary case: [Convert] returns nil for a nil error, and the
// generated AgentOutput.Error and SessionSnapshot.Error fields are nil whenever
// nothing failed. Assigning one of those to an error variable produces an
// interface that is non-nil but holds a nil pointer, and without these guards
// the first errors.Is or transport call on it would panic, typically inside a
// request handler.
//
// Field access cannot be guarded the same way: e.Status on a nil *Error panics
// like any other nil dereference. Read fields only after checking for nil, or
// go through [Of] and [PublicMessage], which handle it.
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

	// HTTPCode is the HTTP status for Status, recorded at construction.
	//
	// Deprecated: use Status.HTTPCode(), which is correct for every Error
	// including ones built as a struct literal. This field exists so
	// core.GenkitError can alias Error, and will be removed with it.
	HTTPCode int

	// Source names the component that raised the error.
	//
	// Deprecated: never populated. It exists so core.GenkitError can alias
	// Error, and will be removed with it.
	Source *string

	sentinel *Sentinel
	// cause is the error recorded via %w, if any. Keeping it a single error
	// (rather than an Unwrap() []error holding the sentinel too) means the
	// stdlib errors.Unwrap and hand-rolled chain walks still see through an
	// Error; sentinel matching goes through [Error.Is] instead.
	cause error
	stack []uintptr
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
		HTTPCode: sentinel.status.HTTPCode(),
		sentinel: sentinel,
		cause:    causeOf(formatted),
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

// WithCause records err as e's cause without folding it into the message, and
// returns e. Use it when the cause is worth keeping reachable through
// [errors.Is] and [errors.As] but not worth repeating in the text:
//
//	return status.Errorf(ai.ErrToolFailed, "tool %q failed", name).WithCause(err)
//
// Prefer %w in the format string when the cause belongs in the message. A nil
// err, or a second call, is a no-op.
func (e *Error) WithCause(err error) *Error {
	if err != nil && e.cause == nil {
		e.cause = err
	}
	return e
}

// Error implements error. It returns Message alone: the sentinel is a
// classification label, not a message prefix, so callers control the wording.
// A nil *Error renders as "<nil>", matching how fmt prints a nil error, rather
// than as "" which would be indistinguishable from an empty message.
func (e *Error) Error() string {
	if e == nil {
		return "<nil>"
	}
	return e.Message
}

// Unwrap returns the cause recorded via %w or [Error.WithCause], or nil. The
// classifying sentinel is deliberately not part of the unwrap chain, so
// errors.Unwrap and hand-rolled chain walks behave the way they do for any
// fmt.Errorf result; [Error.Is] handles sentinel matching.
func (e *Error) Unwrap() error {
	if e == nil {
		return nil
	}
	return e.cause
}

// Is reports whether e was classified by target or by a sentinel derived from
// it. errors.Is consults this before walking [Error.Unwrap], so both
// granularities match:
//
//	errors.Is(err, ai.ErrMaxTurnsExceeded) // the specific sentinel
//	errors.Is(err, status.ErrAborted)      // the base it derives from
func (e *Error) Is(target error) bool {
	// errors.Is calls this whenever the interface is non-nil, including when it
	// holds a nil *Error, so the nil check has to be here rather than at the
	// call site.
	if e == nil || e.sentinel == nil {
		return false
	}
	return errors.Is(e.sentinel, target)
}

// Sentinel returns the sentinel that classified e, or nil if it was decoded
// from the wire rather than constructed in this process.
func (e *Error) Sentinel() *Sentinel {
	if e == nil {
		return nil
	}
	return e.sentinel
}

// Stack returns the call stack captured when e was constructed, formatted like
// a panic trace, or "" for an error decoded from the wire. It is formatted on
// demand: construction only records program counters.
func (e *Error) Stack() string {
	if e == nil {
		return ""
	}
	return formatStack(e.stack)
}

// Of returns the status of err.
//
// It reports the status of the outermost [Error] in the chain, so a boundary
// that deliberately reclassifies with [Errorf] wins over anything beneath it. A
// bare [Sentinel] reports its own status. Context cancellation and deadline
// errors map to Cancelled and DeadlineExceeded. Anything else is Internal: an
// unclassified failure is a failure of ours, not of the caller's request.
//
// A typed-nil *Error carries no classification: when err itself is one, Of is
// OK (nothing failed, the nil merely escaped through an error variable), and
// when one appears inside a chain it is skipped so it cannot mask the rest of
// the chain.
//
// Of(nil) is OK.
func Of(err error) Name {
	s, _ := Classified(err)
	return s
}

// Classified returns err's status and whether anything in its chain actually
// carries one. It is [Of] with the one distinction Of cannot make: an
// unclassified failure and one deliberately classified Internal both report
// Internal, and only the second is a decision someone made.
//
// Middleware that acts on a status needs that distinction, since the action it
// takes on an unclassified error (a network blip from a provider SDK, say)
// should not follow from INTERNAL happening to be in a configured list:
//
//	if s, ok := status.Classified(err); ok && slices.Contains(retryOn, s) { ... }
//
// Cancellation and deadline expiry count as classified. Classified(nil) is
// (OK, false).
func Classified(err error) (Name, bool) {
	if err == nil {
		return OK, false
	}
	if e := firstError(err); e != nil {
		return e.Status, true
	}
	if e, ok := err.(*Error); ok && e == nil {
		return OK, false // a non-nil interface holding a nil *Error is not a failure
	}
	// A typed-nil *Sentinel matches errors.As but carries nothing; skip it the
	// way firstError skips a typed-nil *Error.
	var s *Sentinel
	if errors.As(err, &s) && s != nil {
		return s.status, true
	}
	switch {
	case errors.Is(err, context.Canceled):
		return Cancelled, true
	case errors.Is(err, context.DeadlineExceeded):
		return DeadlineExceeded, true
	}
	return Internal, false
}

// firstError returns the first non-nil [Error] in err's chain, or nil. It is
// errors.As with one refinement: a typed-nil *Error node does not count as a
// match and does not end the search, so a nil that escaped through an error
// variable cannot mask a real classification elsewhere in the chain.
func firstError(err error) *Error {
	var e *Error
	if !errors.As(err, &e) {
		return nil
	}
	if e != nil {
		return e
	}
	// errors.As stopped at a typed-nil node. Nothing unwraps out of a nil
	// *Error, but a multi-error wrapper can hold a real one in a sibling
	// branch, so walk the tree skipping nil nodes.
	return walkPastNil(err)
}

func walkPastNil(err error) *Error {
	if e, ok := err.(*Error); ok {
		if e != nil {
			return e
		}
		return nil
	}
	switch x := err.(type) {
	case interface{ Unwrap() error }:
		if u := x.Unwrap(); u != nil {
			return walkPastNil(u)
		}
	case interface{ Unwrap() []error }:
		for _, u := range x.Unwrap() {
			if u == nil {
				continue
			}
			if e := walkPastNil(u); e != nil {
				return e
			}
		}
	}
	return nil
}

// Convert returns err as an [Error], converting it if it is not one already.
// The converted error takes its status from [Of] and is never public. Returns
// nil for a nil err, and also for an err that is itself a non-nil interface
// holding a nil *Error, so callers must check the result rather than assume it
// is non-nil. A typed-nil *Error inside a larger chain is skipped instead: the
// chain is a real error and converts like any other.
//
// Prefer errors.As when you need to know whether err really is an [Error]; this
// is for boundaries that must produce one either way.
func Convert(err error) *Error {
	if err == nil {
		return nil
	}
	if e := firstError(err); e != nil {
		return e
	}
	if e, ok := err.(*Error); ok && e == nil {
		return nil
	}
	n := Of(err)
	return &Error{Status: n, Message: err.Error(), HTTPCode: n.HTTPCode(), cause: err}
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
	if e := firstError(err); e != nil {
		if e.Public {
			return e.Message, true
		}
		return genericMessage(e.Status), false
	}
	if e, ok := err.(*Error); ok && e == nil {
		return "", false
	}
	// No Error in the chain: fall back to the interface, which the deprecated
	// core.UserFacingError implements so its message still reaches clients.
	var pm publicMessager
	if errors.As(err, &pm) {
		if m, ok := pm.PublicMessage(); ok {
			return m, true
		}
	}
	return genericMessage(Of(err)), false
}

// publicMessager lets a type declared outside this package mark its message
// safe to return to clients. It exists for the deprecated core.UserFacingError,
// whose whole purpose was to be public but which predates [Error.Public].
type publicMessager interface {
	PublicMessage() (string, bool)
}

func genericMessage(n Name) string {
	if s, ok := baseSentinels[n]; ok {
		return s.label
	}
	return "internal"
}

// causeOf returns the error fmt.Errorf recorded via %w, or nil when the format
// had none. A format with several %w verbs yields a wrapper holding all of
// them; that wrapper is returned whole so errors.Is and errors.As still reach
// every branch through it.
func causeOf(err error) error {
	switch x := err.(type) {
	case interface{ Unwrap() error }:
		return x.Unwrap()
	case interface{ Unwrap() []error }:
		return err
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
// A captured stack is in-process diagnostics, not wire data, so errors
// embedded in values (a failed agent invocation's output, say) do not leak
// process internals to clients. Consumers that want the stack read
// [Error.Stack] directly. [Error.Stack] keeps it off Details to begin with;
// a "stack" entry put there by hand (as the deprecated core.NewError does for
// compatibility) is dropped here too.
func (e *Error) MarshalJSON() ([]byte, error) {
	details := e.Details
	if _, ok := details["stack"]; ok {
		details = maps.Clone(details)
		delete(details, "stack")
		if len(details) == 0 {
			details = nil
		}
	}
	return json.Marshal(errorWire{
		Status:  e.Status,
		Message: e.Message,
		Details: details,
	})
}

// UnmarshalJSON decodes an Error from the canonical wire format. The result
// carries no sentinel, cause, or stack: those do not cross the wire.
func (e *Error) UnmarshalJSON(data []byte) error {
	var w errorWire
	if err := json.Unmarshal(data, &w); err != nil {
		return err
	}
	e.Status = w.Status
	e.Message = w.Message
	e.Details = w.Details
	e.HTTPCode = w.Status.HTTPCode()
	return nil
}

// JSONSchema describes the error's wire format for schema inference. Without
// it, inference would reflect over the struct fields, requiring in-process
// fields that MarshalJSON never emits, so values embedding an Error would fail
// validation against their own inferred schema.
func (Error) JSONSchema() *jsonschema.Schema {
	return base.InferJSONSchema(errorWire{})
}
