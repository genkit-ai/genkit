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
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"testing"
)

func TestNewPublicError(t *testing.T) {
	t.Run("creates error with all fields", func(t *testing.T) {
		details := map[string]any{"field": "username"}
		err := NewPublicError(INVALID_ARGUMENT, "invalid username", details)

		if err.Status != INVALID_ARGUMENT {
			t.Errorf("Status = %q, want %q", err.Status, INVALID_ARGUMENT)
		}
		if err.Message != "invalid username" {
			t.Errorf("Message = %q, want %q", err.Message, "invalid username")
		}
		if err.Details["field"] != "username" {
			t.Errorf("Details[field] = %v, want %q", err.Details["field"], "username")
		}
	})

	t.Run("creates error with nil details", func(t *testing.T) {
		err := NewPublicError(NOT_FOUND, "resource not found", nil)

		if err.Status != NOT_FOUND {
			t.Errorf("Status = %q, want %q", err.Status, NOT_FOUND)
		}
		if err.Details != nil {
			t.Errorf("Details = %v, want nil", err.Details)
		}
	})
}

func TestUserFacingErrorError(t *testing.T) {
	t.Run("formats error message correctly", func(t *testing.T) {
		err := NewPublicError(PERMISSION_DENIED, "access denied", nil)
		got := err.Error()
		want := "PERMISSION_DENIED: access denied"

		if got != want {
			t.Errorf("Error() = %q, want %q", got, want)
		}
	})
}

func TestNewError(t *testing.T) {
	t.Run("creates error with simple message", func(t *testing.T) {
		err := NewError(INTERNAL, "internal error")

		if err.Status != INTERNAL {
			t.Errorf("Status = %q, want %q", err.Status, INTERNAL)
		}
		if err.Message != "internal error" {
			t.Errorf("Message = %q, want %q", err.Message, "internal error")
		}
	})

	t.Run("creates error with formatted message", func(t *testing.T) {
		err := NewError(INVALID_ARGUMENT, "field %q has invalid value %d", "count", 42)

		want := `field "count" has invalid value 42`
		if err.Message != want {
			t.Errorf("Message = %q, want %q", err.Message, want)
		}
	})

	t.Run("captures stack trace", func(t *testing.T) {
		err := NewError(INTERNAL, "error with stack")

		if err.Details == nil {
			t.Fatal("Details is nil, expected stack trace")
		}
		stack, ok := err.Details["stack"].(string)
		if !ok {
			t.Fatal("stack is not a string")
		}
		if !strings.Contains(stack, "TestNewError") {
			t.Errorf("stack trace does not contain test function name")
		}
	})
}

func TestGenkitErrorError(t *testing.T) {
	t.Run("returns message as error string", func(t *testing.T) {
		err := NewError(INTERNAL, "something went wrong")
		got := err.Error()

		if got != "something went wrong" {
			t.Errorf("Error() = %q, want %q", got, "something went wrong")
		}
	})
}

func TestGenkitErrorJSONRoundtrip(t *testing.T) {
	t.Run("marshals canonical wire shape", func(t *testing.T) {
		ge := &GenkitError{
			Status:   NOT_FOUND,
			Message:  "missing",
			Details:  map[string]any{"id": "abc"},
			HTTPCode: 999,                                      // not on the wire
			Source:   func() *string { s := "x"; return &s }(), // not on the wire
		}
		got, err := json.Marshal(ge)
		if err != nil {
			t.Fatalf("Marshal: %v", err)
		}
		// Key order follows the generated wire struct's field order.
		want := `{"details":{"id":"abc"},"message":"missing","status":"NOT_FOUND"}`
		if string(got) != want {
			t.Errorf("Marshal = %s, want %s", got, want)
		}
	})

	t.Run("omits empty details", func(t *testing.T) {
		ge := &GenkitError{Status: NOT_FOUND, Message: "missing"}
		got, err := json.Marshal(ge)
		if err != nil {
			t.Fatalf("Marshal: %v", err)
		}
		want := `{"message":"missing","status":"NOT_FOUND"}`
		if string(got) != want {
			t.Errorf("Marshal = %s, want %s", got, want)
		}
	})

	t.Run("omits the auto-captured stack detail", func(t *testing.T) {
		ge := NewError(NOT_FOUND, "missing")
		ge.Details["id"] = "abc"
		got, err := json.Marshal(ge)
		if err != nil {
			t.Fatalf("Marshal: %v", err)
		}
		// The stack is in-process diagnostics; only the other details
		// cross the wire.
		want := `{"details":{"id":"abc"},"message":"missing","status":"NOT_FOUND"}`
		if string(got) != want {
			t.Errorf("Marshal = %s, want %s", got, want)
		}
		if _, ok := ge.Details["stack"]; !ok {
			t.Error("marshaling must not mutate the in-process Details")
		}
	})

	t.Run("omits details entirely when stack is the only entry", func(t *testing.T) {
		ge := NewError(NOT_FOUND, "missing")
		got, err := json.Marshal(ge)
		if err != nil {
			t.Fatalf("Marshal: %v", err)
		}
		want := `{"message":"missing","status":"NOT_FOUND"}`
		if string(got) != want {
			t.Errorf("Marshal = %s, want %s", got, want)
		}
	})

	t.Run("unmarshals and derives HTTPCode", func(t *testing.T) {
		raw := `{"status":"NOT_FOUND","message":"missing","details":{"id":"abc"}}`
		var ge GenkitError
		if err := json.Unmarshal([]byte(raw), &ge); err != nil {
			t.Fatalf("Unmarshal: %v", err)
		}
		if ge.Status != NOT_FOUND {
			t.Errorf("Status = %q, want %q", ge.Status, NOT_FOUND)
		}
		if ge.Message != "missing" {
			t.Errorf("Message = %q, want %q", ge.Message, "missing")
		}
		if ge.HTTPCode != http.StatusNotFound {
			t.Errorf("HTTPCode = %d, want %d", ge.HTTPCode, http.StatusNotFound)
		}
		if ge.Details["id"] != "abc" {
			t.Errorf("Details[id] = %v, want %q", ge.Details["id"], "abc")
		}
	})
}

func TestAsGenkitError(t *testing.T) {
	t.Run("nil returns nil", func(t *testing.T) {
		if got := AsGenkitError(nil); got != nil {
			t.Errorf("AsGenkitError(nil) = %+v, want nil", got)
		}
	})

	t.Run("returns existing GenkitError unchanged", func(t *testing.T) {
		orig := &GenkitError{Status: NOT_FOUND, Message: "missing"}
		if got := AsGenkitError(orig); got != orig {
			t.Errorf("expected same pointer, got %+v", got)
		}
	})

	t.Run("unwraps nested GenkitError", func(t *testing.T) {
		orig := &GenkitError{Status: NOT_FOUND, Message: "missing"}
		wrapped := fmt.Errorf("wrap: %w", orig)
		got := AsGenkitError(wrapped)
		if got != orig {
			t.Errorf("expected unwrapped pointer, got %+v", got)
		}
	})

	t.Run("wraps plain error with INTERNAL", func(t *testing.T) {
		got := AsGenkitError(errors.New("boom"))
		if got.Status != INTERNAL {
			t.Errorf("Status = %q, want INTERNAL", got.Status)
		}
		if got.Message != "boom" {
			t.Errorf("Message = %q, want boom", got.Message)
		}
		if got.HTTPCode != http.StatusInternalServerError {
			t.Errorf("HTTPCode = %d, want %d", got.HTTPCode, http.StatusInternalServerError)
		}
	})
}

// testCustomError is a helper type for the errors.As subtest.
type testCustomError struct {
	code int
}

func (e *testCustomError) Error() string {
	return fmt.Sprintf("custom error %d", e.code)
}

func TestGenkitErrorUnwrap(t *testing.T) {
	t.Run("errors.Is matches original cause", func(t *testing.T) {
		original := errors.New("original failure")
		gErr := NewError(INTERNAL, "something happened: %v", original)

		if !errors.Is(gErr, original) {
			t.Errorf("expected errors.Is to return true, but got false")
		}
		if gErr.Unwrap() != original {
			t.Errorf("Unwrap() returned wrong error")
		}
	})

	t.Run("errors.As extracts typed cause", func(t *testing.T) {
		cause := &testCustomError{code: 42}
		ge := NewError(INTERNAL, "failed: %v", cause)

		var target *testCustomError
		if !errors.As(ge, &target) {
			t.Fatal("errors.As failed to find *testCustomError")
		}
		if target.code != 42 {
			t.Errorf("target.code = %d, want 42", target.code)
		}
	})

	t.Run("no args returns nil", func(t *testing.T) {
		ge := NewError(INTERNAL, "no args error")

		if ge.Unwrap() != nil {
			t.Errorf("Unwrap() = %v, want nil", ge.Unwrap())
		}
	})

	t.Run("multiple errors preserves the last one", func(t *testing.T) {
		first := errors.New("first")
		second := errors.New("second")
		ge := NewError(INTERNAL, "two errors: %v %v", first, second)

		if ge.Unwrap() != second {
			t.Errorf("Unwrap() = %v, want %v (last error)", ge.Unwrap(), second)
		}
		if !errors.Is(ge, second) {
			t.Error("errors.Is(ge, second) = false, want true")
		}
	})

	t.Run("non-error args returns nil", func(t *testing.T) {
		ge := NewError(INTERNAL, "value is %d and %s", 42, "hello")

		if ge.Unwrap() != nil {
			t.Errorf("Unwrap() = %v, want nil", ge.Unwrap())
		}
	})
}
