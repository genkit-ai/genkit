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

package core_test

import (
	"errors"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/status"
)

// errModelNotFound stands in for a domain sentinel like ai.ErrModelNotFound,
// declared here so core's tests do not import ai.
var errModelNotFound = status.ErrNotFound.Subtype("model not found")

// internalError is what a migrated internal call site now returns.
func internalError() error {
	return status.Errorf(errModelNotFound, "model %q not found", "googleai/gemini-flash-latest")
}

// TestV1ErrorsAsStillMatches is the load-bearing compatibility guarantee: code
// written against *core.GenkitError keeps matching errors that Genkit now
// raises as *status.Error, because the two are the same type.
func TestV1ErrorsAsStillMatches(t *testing.T) {
	for name, err := range map[string]error{
		"direct":  internalError(),
		"wrapped": fmt.Errorf("resolving model: %w", internalError()),
	} {
		t.Run(name, func(t *testing.T) {
			var ge *core.GenkitError
			if !errors.As(err, &ge) {
				t.Fatalf("errors.As(*core.GenkitError) = false for %T", err)
			}
			if ge.Status != core.NOT_FOUND {
				t.Errorf("Status = %q, want %q", ge.Status, core.NOT_FOUND)
			}
			if ge.HTTPCode != http.StatusNotFound {
				t.Errorf("HTTPCode = %d, want %d", ge.HTTPCode, http.StatusNotFound)
			}
			if got := core.HTTPStatusCode(ge.Status); got != http.StatusNotFound {
				t.Errorf("core.HTTPStatusCode = %d, want %d", got, http.StatusNotFound)
			}

			// The same error also answers to the v2 surface.
			if !errors.Is(err, errModelNotFound) {
				t.Error("errors.Is(specific sentinel) = false")
			}
			if !errors.Is(err, status.ErrNotFound) {
				t.Error("errors.Is(base sentinel) = false")
			}
			if got := status.Of(err); got != status.NotFound {
				t.Errorf("status.Of = %q, want %q", got, status.NotFound)
			}
		})
	}
}

// TestErrorsUnwrapStillWalks guards the reason Error keeps a single-cause
// Unwrap: the stdlib errors.Unwrap returns nil for an Unwrap() []error, which
// would silently truncate hand-rolled chain walks in logging and telemetry
// middleware.
func TestErrorsUnwrapStillWalks(t *testing.T) {
	cause := errors.New("boom")
	err := fmt.Errorf("outer: %w", status.Errorf(status.ErrInternal, "inner: %w", cause))

	depth := 0
	for e := err; e != nil; e = errors.Unwrap(e) {
		depth++
	}
	if depth != 3 { // fmt wrapper -> status.Error -> cause
		t.Errorf("chain walk depth = %d, want 3", depth)
	}
	if got := errors.Unwrap(errors.Unwrap(err)); got != cause {
		t.Errorf("errors.Unwrap through status.Error = %v, want %v", got, cause)
	}
}

// TestV1ConstructorsPreserveV1Behaviour: anything built through the deprecated
// constructors behaves exactly as it did before, including the two behaviours
// status.Errorf deliberately drops.
func TestV1ConstructorsPreserveV1Behaviour(t *testing.T) {
	t.Run("NewError implicitly wraps the last error argument", func(t *testing.T) {
		cause := errors.New("boom")
		// Note %v, not %w: v1 wrapped by scanning args, not by verb.
		err := core.NewError(core.INVALID_ARGUMENT, "bad input: %v", cause)
		if !errors.Is(err, cause) {
			t.Error("errors.Is(cause) = false; implicit wrapping lost")
		}
		if got := err.Error(); got != "bad input: boom" {
			t.Errorf("Error() = %q, want %q", got, "bad input: boom")
		}
		// And it now classifies for v2 consumers too.
		if !errors.Is(err, status.ErrInvalidArgument) {
			t.Error("v1-constructed error does not match its base sentinel")
		}
	})

	t.Run("NewError keeps a non-canonical status name on the wire", func(t *testing.T) {
		// v1 put whatever StatusName it was handed on the wire rather than
		// coercing it, and the shim's contract is to behave the same.
		weird := core.StatusName("NOT_A_REAL_STATUS")
		err := core.NewError(weird, "boom")
		if err.Status != weird {
			t.Errorf("Status = %q, want %q", err.Status, weird)
		}
		if err.HTTPCode != http.StatusInternalServerError {
			t.Errorf("HTTPCode = %d, want 500", err.HTTPCode)
		}
	})

	t.Run("NewError keeps OK on the wire", func(t *testing.T) {
		// OK is the one canonical name status.Base has no sentinel for (an
		// error cannot classify as success), so like a non-canonical name it
		// must be restored rather than surfacing as UNKNOWN/500.
		err := core.NewError(core.OK, "done")
		if err.Status != core.OK {
			t.Errorf("Status = %q, want %q", err.Status, core.OK)
		}
		if err.HTTPCode != http.StatusOK {
			t.Errorf("HTTPCode = %d, want 200", err.HTTPCode)
		}
	})

	t.Run("NewError records a stack in Details", func(t *testing.T) {
		err := core.NewError(core.INTERNAL, "boom")
		stack, ok := err.Details["stack"].(string)
		if !ok || stack == "" {
			t.Fatal(`Details["stack"] missing`)
		}
		if !strings.Contains(stack, "TestV1ConstructorsPreserveV1Behaviour") {
			t.Errorf("stack does not reach the caller:\n%s", stack)
		}
	})

	t.Run("UserFacingError keeps its shape and text", func(t *testing.T) {
		err := core.NewPublicError(core.INVALID_ARGUMENT, "invalid email", map[string]any{"field": "email"})
		var uf *core.UserFacingError
		if !errors.As(error(err), &uf) {
			t.Fatal("errors.As(*core.UserFacingError) = false")
		}
		if got, want := err.Error(), "INVALID_ARGUMENT: invalid email"; got != want {
			t.Errorf("Error() = %q, want %q", got, want)
		}
		if uf.Details["field"] != "email" {
			t.Errorf("Details = %v, want field=email", uf.Details)
		}
		// It stays a distinct type from GenkitError.
		if errors.As(error(err), new(*core.GenkitError)) {
			t.Error("UserFacingError matched *core.GenkitError; the two must stay distinct")
		}
	})

	t.Run("UserFacingError now carries a usable status", func(t *testing.T) {
		err := core.NewPublicError(core.INVALID_ARGUMENT, "invalid email", nil)
		// v1 bug: transports could not read this status, so a public
		// INVALID_ARGUMENT went out as HTTP 500.
		if got := status.Of(err); got != status.InvalidArgument {
			t.Errorf("status.Of = %q, want %q", got, status.InvalidArgument)
		}
		if !errors.Is(err, status.ErrInvalidArgument) {
			t.Error("errors.Is(base sentinel) = false")
		}
		msg, public := status.PublicMessage(err)
		if !public || msg != "invalid email" {
			t.Errorf("PublicMessage = (%q, %v), want (%q, true)", msg, public, "invalid email")
		}
	})

	t.Run("an invalid-input error classifies and keeps its cause", func(t *testing.T) {
		cause := errors.New("field x: expected string")
		err := error(status.Errorf(status.ErrInvalidInput, "invalid input to action %q: %w", "/flow/foo", cause))

		var ge *core.GenkitError
		if !errors.As(err, &ge) {
			t.Fatal("errors.As(*core.GenkitError) = false")
		}
		if !errors.Is(err, cause) {
			t.Error("errors.Is(cause) = false")
		}
		if !errors.Is(err, status.ErrInvalidInput) {
			t.Error("errors.Is(status.ErrInvalidInput) = false")
		}
		if got := status.Of(err); got != status.InvalidArgument {
			t.Errorf("status.Of = %q, want %q", got, status.InvalidArgument)
		}
	})
}

// TestUnclassifiedErrorsStayRetryable pins the classification that keeps the
// retry middleware's default behaviour unchanged: an error nobody classified is
// INTERNAL, which is in the default retry set. A cancelled context is not.
func TestUnclassifiedErrorsStayRetryable(t *testing.T) {
	if got := status.Of(errors.New("transient network blip")); got != status.Internal {
		t.Errorf("status.Of(unclassified) = %q, want %q", got, status.Internal)
	}
}

// TestStatusNameAliasIsInterchangeable covers the exported plugin config that
// is typed []core.StatusName (retry, fallback): the alias must let callers use
// either spelling.
func TestStatusNameAliasIsInterchangeable(t *testing.T) {
	v1 := []core.StatusName{core.UNAVAILABLE, core.NOT_FOUND}
	v2 := []status.Name{status.Unavailable, status.NotFound}
	for i := range v1 {
		if v1[i] != v2[i] {
			t.Errorf("v1[%d] = %q, v2[%d] = %q; alias is not transparent", i, v1[i], i, v2[i])
		}
	}
	var n status.Name = core.INVALID_ARGUMENT
	if n.HTTPCode() != http.StatusBadRequest {
		t.Errorf("HTTPCode = %d, want %d", n.HTTPCode(), http.StatusBadRequest)
	}
}
