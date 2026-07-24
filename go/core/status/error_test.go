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
	"net/http"
	"strings"
	"testing"
)

// errMaxTurns stands in for a domain sentinel a feature package would declare.
var errMaxTurns = ErrAborted.Subtype("max turns exceeded")

func TestErrorf(t *testing.T) {
	err := Errorf(ErrNotFound, "model %q not found", "gemini")

	if got, want := err.Error(), `model "gemini" not found`; got != want {
		t.Errorf("Error() = %q, want %q", got, want)
	}
	if got, want := err.Status, NotFound; got != want {
		t.Errorf("Status = %q, want %q", got, want)
	}
	if err.Public {
		t.Error("Errorf produced a public error")
	}
	if got, want := err.HTTPCode(), http.StatusNotFound; got != want {
		t.Errorf("HTTPCode() = %d, want %d", got, want)
	}
}

func TestErrorfEmptyMessageFallsBackToSentinel(t *testing.T) {
	if got, want := Errorf(ErrAborted, "").Error(), "aborted"; got != want {
		t.Errorf("Error() = %q, want the sentinel label %q", got, want)
	}
}

func TestErrorfNilSentinelIsInternal(t *testing.T) {
	err := Errorf(nil, "boom")
	if got, want := err.Status, Internal; got != want {
		t.Errorf("Status = %q, want %q", got, want)
	}
	if !errors.Is(err, ErrInternal) {
		t.Error("errors.Is(err, ErrInternal) = false")
	}
}

func TestPublicErrorf(t *testing.T) {
	err := PublicErrorf(ErrInvalidArgument, "invalid %q parameter", "stream")
	if !err.Public {
		t.Error("PublicErrorf produced a non-public error")
	}
	msg, public := PublicMessage(err)
	if !public || msg != `invalid "stream" parameter` {
		t.Errorf("PublicMessage() = (%q, %v), want the message verbatim", msg, public)
	}
}

func TestPublicMessageHidesNonPublicText(t *testing.T) {
	err := Errorf(ErrPermissionDenied, "user alice lacks role admin on project p-42")

	msg, public := PublicMessage(err)
	if public {
		t.Error("PublicMessage reported a non-public error as public")
	}
	if strings.Contains(msg, "alice") || strings.Contains(msg, "p-42") {
		t.Errorf("PublicMessage() = %q, leaked internal detail", msg)
	}
	if msg != "permission denied" {
		t.Errorf("PublicMessage() = %q, want the generic status label", msg)
	}
}

func TestPublicMessageOfPlainError(t *testing.T) {
	msg, public := PublicMessage(errors.New("connection string: postgres://user:pw@host"))
	if public {
		t.Error("a plain error was reported as public")
	}
	if strings.Contains(msg, "postgres") {
		t.Errorf("PublicMessage() = %q, leaked the underlying message", msg)
	}
}

func TestSentinelMatchingIsTwoLevel(t *testing.T) {
	err := Errorf(errMaxTurns, "stopped after %d turns", 5)

	if !errors.Is(err, errMaxTurns) {
		t.Error("errors.Is(err, errMaxTurns) = false, want a specific match")
	}
	if !errors.Is(err, ErrAborted) {
		t.Error("errors.Is(err, ErrAborted) = false, want a broad match via the parent")
	}
	if errors.Is(err, ErrInternal) {
		t.Error("errors.Is(err, ErrInternal) = true, want no match on an unrelated sentinel")
	}
	if got, want := err.Status, Aborted; got != want {
		t.Errorf("Status = %q, want the parent's %q", got, want)
	}
}

func TestSubInheritsStatus(t *testing.T) {
	sub := ErrNotFound.Subtype("model not found")
	if got, want := sub.Status(), NotFound; got != want {
		t.Errorf("Sub().Status() = %q, want %q", got, want)
	}
	if !errors.Is(sub, ErrNotFound) {
		t.Error("a sub-sentinel does not match its parent")
	}
	// A bare sentinel is a usable error value on its own.
	if got := Of(sub); got != NotFound {
		t.Errorf("Of(sentinel) = %q, want NOT_FOUND", got)
	}
}

func TestErrorfWrapsCause(t *testing.T) {
	cause := errors.New("dial tcp: connection refused")
	err := Errorf(ErrUnavailable, "reaching provider: %w", cause)

	if !errors.Is(err, cause) {
		t.Error("errors.Is(err, cause) = false, want the %w cause reachable")
	}
	if !errors.Is(err, ErrUnavailable) {
		t.Error("errors.Is(err, ErrUnavailable) = false, want the sentinel reachable alongside the cause")
	}
	if got, want := err.Error(), "reaching provider: dial tcp: connection refused"; got != want {
		t.Errorf("Error() = %q, want %q", got, want)
	}
}

func TestErrorfWrapsMultipleCauses(t *testing.T) {
	a, b := errors.New("a"), errors.New("b")
	err := Errorf(ErrInternal, "%w and %w", a, b)

	for _, target := range []error{a, b, ErrInternal} {
		if !errors.Is(err, target) {
			t.Errorf("errors.Is(err, %v) = false", target)
		}
	}
}

// Adding context with fmt.Errorf must leave the classification alone: this is
// the common case as an error travels up the stack.
func TestContextWrappingPreservesStatus(t *testing.T) {
	err := error(Errorf(errMaxTurns, "stopped after 5 turns"))
	err = fmt.Errorf("agent %q: %w", "planner", err)
	err = fmt.Errorf("flow %q: %w", "chat", err)

	if got, want := Of(err), Aborted; got != want {
		t.Errorf("Of(err) = %q, want %q", got, want)
	}
	if !errors.Is(err, errMaxTurns) {
		t.Error("the sentinel did not survive two layers of fmt.Errorf")
	}
	if got, want := err.Error(), `flow "chat": agent "planner": stopped after 5 turns`; got != want {
		t.Errorf("Error() = %q, want %q", got, want)
	}
}

// Reclassifying at a boundary is deliberate, and the outermost classification
// is the one transports report.
func TestReclassificationWins(t *testing.T) {
	inner := Errorf(ErrNotFound, "row not found")
	outer := Errorf(ErrInternal, "tool %q: %w", "lookup", inner)

	if got, want := Of(outer), Internal; got != want {
		t.Errorf("Of(outer) = %q, want the outermost status %q", got, want)
	}
	// The original classification is still reachable for callers that want it.
	if !errors.Is(outer, ErrNotFound) {
		t.Error("the inner sentinel is no longer reachable")
	}

	var got *Error
	if !errors.As(outer, &got) {
		t.Fatal("errors.As found no *Error")
	}
	if got != outer {
		t.Error("errors.As returned the inner *Error, want the outermost")
	}
}

func TestOf(t *testing.T) {
	canceled, cancel := context.WithCancel(context.Background())
	cancel()

	tests := []struct {
		name string
		err  error
		want Name
	}{
		{"nil", nil, OK},
		{"plain error", errors.New("boom"), Internal},
		{"status error", Errorf(ErrAlreadyExists, "dup"), AlreadyExists},
		{"bare sentinel", ErrUnimplemented, Unimplemented},
		{"wrapped sentinel", fmt.Errorf("x: %w", ErrDataLoss), DataLoss},
		{"context canceled", canceled.Err(), Cancelled},
		{"wrapped cancel", fmt.Errorf("x: %w", context.Canceled), Cancelled},
		{"deadline", context.DeadlineExceeded, DeadlineExceeded},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := Of(tt.err); got != tt.want {
				t.Errorf("Of() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestConvert(t *testing.T) {
	t.Run("nil", func(t *testing.T) {
		if got := Convert(nil); got != nil {
			t.Errorf("Convert(nil) = %+v, want nil", got)
		}
	})
	t.Run("passes through an existing Error", func(t *testing.T) {
		orig := Errorf(ErrOutOfRange, "index 9")
		if got := Convert(fmt.Errorf("x: %w", orig)); got != orig {
			t.Errorf("Convert() = %+v, want the wrapped *Error itself", got)
		}
	})
	t.Run("converts a plain error", func(t *testing.T) {
		cause := errors.New("boom")
		got := Convert(cause)
		if got.Status != Internal || got.Message != "boom" {
			t.Errorf("Convert() = %+v, want INTERNAL/boom", got)
		}
		if got.Public {
			t.Error("a converted error must not be public")
		}
		if !errors.Is(got, cause) {
			t.Error("the original error is no longer reachable")
		}
	})
}

func TestStack(t *testing.T) {
	stack := Errorf(ErrInternal, "boom").Stack()
	if !strings.Contains(stack, "TestStack") {
		t.Errorf("Stack() does not name the calling test:\n%s", stack)
	}
	if strings.Contains(strings.SplitN(stack, "\n", 2)[0], "status.Errorf") {
		t.Errorf("Stack() starts inside the status package:\n%s", stack)
	}
	// A wire-decoded error has no stack.
	var decoded Error
	if err := json.Unmarshal([]byte(`{"status":"INTERNAL","message":"x"}`), &decoded); err != nil {
		t.Fatal(err)
	}
	if got := decoded.Stack(); got != "" {
		t.Errorf("Stack() on a decoded error = %q, want empty", got)
	}
}

func TestJSONRoundTrip(t *testing.T) {
	err := Errorf(ErrFailedPrecondition, "not ready").WithDetails(map[string]any{"retryAfter": "5s"})

	data, mErr := json.Marshal(err)
	if mErr != nil {
		t.Fatal(mErr)
	}
	// The stack and the public flag are in-process only.
	if got, want := string(data), `{"details":{"retryAfter":"5s"},"message":"not ready","status":"FAILED_PRECONDITION"}`; got != want {
		t.Errorf("MarshalJSON() = %s, want %s", got, want)
	}

	var back Error
	if err := json.Unmarshal(data, &back); err != nil {
		t.Fatal(err)
	}
	if back.Status != FailedPrecondition || back.Message != "not ready" {
		t.Errorf("round trip = %+v", back)
	}
	if back.Sentinel() != nil {
		t.Error("a decoded error should carry no sentinel")
	}
}

func TestMarshalOmitsStack(t *testing.T) {
	data, err := json.Marshal(Errorf(ErrInternal, "boom"))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(data), "stack") || strings.Contains(string(data), "status_test.go") {
		t.Errorf("MarshalJSON() leaked the stack: %s", data)
	}
}

func TestBase(t *testing.T) {
	if got := Base(ResourceExhausted); got != ErrResourceExhausted {
		t.Errorf("Base(RESOURCE_EXHAUSTED) = %v, want ErrResourceExhausted", got)
	}
	if got := Base("NOT_A_STATUS"); got != ErrUnknown {
		t.Errorf("Base(unknown) = %v, want ErrUnknown", got)
	}
	// Every canonical name except OK has a base sentinel that agrees with it.
	for name := range statuses {
		if name == OK {
			continue
		}
		if got := Base(name); got.Status() != name {
			t.Errorf("Base(%q).Status() = %q", name, got.Status())
		}
	}
}
