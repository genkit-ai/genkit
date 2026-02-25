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

func TestGenkitErrorToReflectionError(t *testing.T) {
	t.Run("converts error with stack", func(t *testing.T) {
		ge := NewError(NOT_FOUND, "resource not found")
		re := ge.ToReflectionError()

		if re.Message != "resource not found" {
			t.Errorf("Message = %q, want %q", re.Message, "resource not found")
		}
		if re.Code != http.StatusNotFound {
			t.Errorf("Code = %d, want %d", re.Code, http.StatusNotFound)
		}
		if re.Details == nil || re.Details.Stack == nil {
			t.Error("expected stack in details")
		}
	})

	t.Run("converts error with traceId", func(t *testing.T) {
		ge := &GenkitError{
			Status:  INTERNAL,
			Message: "internal error",
			Details: map[string]any{
				"traceId": "trace-123",
			},
		}
		re := ge.ToReflectionError()

		if re.Details == nil || re.Details.TraceID == nil {
			t.Fatal("expected traceId in details")
		}
		if *re.Details.TraceID != "trace-123" {
			t.Errorf("TraceID = %q, want %q", *re.Details.TraceID, "trace-123")
		}
	})

	t.Run("handles empty details", func(t *testing.T) {
		ge := &GenkitError{
			Status:  OK,
			Message: "success",
			Details: nil,
		}
		re := ge.ToReflectionError()

		if re.Message != "success" {
			t.Errorf("Message = %q, want %q", re.Message, "success")
		}
		if re.Details != nil {
			t.Error("expected nil details")
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

func TestToReflectionError(t *testing.T) {
	t.Run("handles GenkitError directly", func(t *testing.T) {
		ge := NewError(INVALID_ARGUMENT, "bad input")
		re := ToReflectionError(ge)

		if re.Message != "bad input" {
			t.Errorf("Message = %q, want %q", re.Message, "bad input")
		}
		if re.Code != http.StatusBadRequest {
			t.Errorf("Code = %d, want %d", re.Code, http.StatusBadRequest)
		}
	})

	t.Run("handles wrapped GenkitError", func(t *testing.T) {
		ge := NewError(NOT_FOUND, "not found")
		wrapped := fmt.Errorf("context: %w", ge)
		re := ToReflectionError(wrapped)

		if re.Message != "not found" {
			t.Errorf("Message = %q, want %q", re.Message, "not found")
		}
		if re.Code != http.StatusNotFound {
			t.Errorf("Code = %d, want %d", re.Code, http.StatusNotFound)
		}
	})

	t.Run("handles plain error", func(t *testing.T) {
		plainErr := errors.New("plain error")
		re := ToReflectionError(plainErr)

		if re.Message != "plain error" {
			t.Errorf("Message = %q, want %q", re.Message, "plain error")
		}
		if re.Code != http.StatusInternalServerError {
			t.Errorf("Code = %d, want %d", re.Code, http.StatusInternalServerError)
		}
	})

	t.Run("handles doubly wrapped GenkitError", func(t *testing.T) {
		ge := NewError(PERMISSION_DENIED, "denied")
		wrapped1 := fmt.Errorf("layer1: %w", ge)
		wrapped2 := fmt.Errorf("layer2: %w", wrapped1)
		re := ToReflectionError(wrapped2)

		if re.Message != "denied" {
			t.Errorf("Message = %q, want %q", re.Message, "denied")
		}
		if re.Code != http.StatusForbidden {
			t.Errorf("Code = %d, want %d", re.Code, http.StatusForbidden)
		}
	})
}
