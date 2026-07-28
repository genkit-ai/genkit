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
	"errors"
	"fmt"
	"testing"
)

// agentOutput mirrors the generated AgentOutput: an exported *Error field that
// is nil whenever nothing failed. Returning that field from a function whose
// result is `error` is the ordinary way a typed nil escapes into Genkit's error
// handling, so every path below has to survive it.
type agentOutput struct {
	Error *Error
}

// failedRun is the shape of the bug: the field is nil, but the returned
// interface is not, because it carries the *Error type.
func failedRun() error {
	out := &agentOutput{} // success: Error is nil
	return out.Error
}

func TestTypedNilIsNotMistakenForAFailure(t *testing.T) {
	err := failedRun()
	if err == nil {
		t.Fatal("test is not exercising a typed nil")
	}

	if got := Of(err); got != OK {
		t.Errorf("Of = %q, want %q", got, OK)
	}
	msg, public := PublicMessage(err)
	if public || msg != "" {
		t.Errorf("PublicMessage = (%q, %v), want (\"\", false)", msg, public)
	}
	if got := Convert(err); got != nil {
		t.Errorf("Convert = %v, want nil", got)
	}
}

// Each method has to tolerate the nil receiver on its own: errors.Is and the
// transports call them through a non-nil interface, so a guard at the call site
// would not help.
func TestNilReceiverMethodsDoNotPanic(t *testing.T) {
	var e *Error

	if got := e.Error(); got != "<nil>" {
		t.Errorf("Error() = %q, want %q", got, "<nil>")
	}
	if got := e.Unwrap(); got != nil {
		t.Errorf("Unwrap() = %v, want nil", got)
	}
	if e.Is(ErrNotFound) {
		t.Error("Is() = true on a nil receiver")
	}
	if got := e.Sentinel(); got != nil {
		t.Errorf("Sentinel() = %v, want nil", got)
	}
	if got := e.Stack(); got != "" {
		t.Errorf("Stack() = %q, want empty", got)
	}
}

// errors.Is and errors.As are the two that would panic first in practice, since
// any handler inspecting an error reaches for them before anything else.
func TestErrorsIsAndAsSurviveATypedNil(t *testing.T) {
	err := failedRun()

	if errors.Is(err, ErrNotFound) {
		t.Error("errors.Is = true for a typed nil")
	}
	var e *Error
	if !errors.As(err, &e) {
		t.Fatal("errors.As = false; the typed nil should still match its own type")
	}
	if e != nil {
		t.Errorf("errors.As set e = %v, want nil", e)
	}

	// And through a wrapper, which is how it usually arrives.
	wrapped := fmt.Errorf("running agent: %w", err)
	if errors.Is(wrapped, ErrNotFound) {
		t.Error("errors.Is = true through a wrapped typed nil")
	}
	if got := Of(wrapped); got != OK {
		t.Errorf("Of(wrapped) = %q, want %q", got, OK)
	}
}

// A nil *Error must not be confused with a real Error carrying an empty
// message: they are different situations and Error() distinguishes them.
func TestNilRendersDistinctlyFromAnEmptyMessage(t *testing.T) {
	var nilErr *Error
	empty := &Error{Status: Internal}
	if nilErr.Error() == empty.Error() {
		t.Errorf("nil and empty-message errors both render as %q", empty.Error())
	}
}
