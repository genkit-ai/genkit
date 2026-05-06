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

package googlegenai

import (
	"errors"
	"fmt"
	"testing"

	"github.com/firebase/genkit/go/core"
	"google.golang.org/genai"
)

func TestWrapAPIError_NilPassthrough(t *testing.T) {
	if got := wrapAPIError(nil); got != nil {
		t.Fatalf("wrapAPIError(nil) = %v, want nil", got)
	}
}

func TestWrapAPIError_NonAPIErrorPassthrough(t *testing.T) {
	plain := errors.New("some non-SDK error")
	got := wrapAPIError(plain)
	var ge *core.GenkitError
	if errors.As(got, &ge) {
		t.Fatalf("wrapAPIError wrapped a non-APIError into a GenkitError; got %#v", ge)
	}
	if got != plain {
		t.Fatalf("wrapAPIError mutated a non-APIError; got %v, want %v", got, plain)
	}
}

func TestWrapAPIError_MapsCanonicalStatus(t *testing.T) {
	cases := []struct {
		status string
		want   core.StatusName
	}{
		{"NOT_FOUND", core.NOT_FOUND},
		{"UNAVAILABLE", core.UNAVAILABLE},
		{"DEADLINE_EXCEEDED", core.DEADLINE_EXCEEDED},
		{"RESOURCE_EXHAUSTED", core.RESOURCE_EXHAUSTED},
		{"INTERNAL", core.INTERNAL},
		{"INVALID_ARGUMENT", core.INVALID_ARGUMENT},
		{"PERMISSION_DENIED", core.PERMISSION_DENIED},
		{"UNAUTHENTICATED", core.UNAUTHENTICATED},
		{"UNIMPLEMENTED", core.UNIMPLEMENTED},
	}
	for _, tc := range cases {
		t.Run(tc.status, func(t *testing.T) {
			api := genai.APIError{Code: 500, Status: tc.status, Message: "boom"}
			wrapped := wrapAPIError(api)
			var ge *core.GenkitError
			if !errors.As(wrapped, &ge) {
				t.Fatalf("wrapAPIError did not produce a GenkitError; got %T: %v", wrapped, wrapped)
			}
			if ge.Status != tc.want {
				t.Fatalf("status = %q, want %q", ge.Status, tc.want)
			}
			var underlying genai.APIError
			if !errors.As(wrapped, &underlying) {
				t.Fatalf("wrapped error does not unwrap back to genai.APIError")
			}
		})
	}
}

func TestWrapAPIError_FallsBackToHTTPCode(t *testing.T) {
	// Mirrors the SDK's fallback path in api_client.go where a non-JSON
	// error body produces APIError{Code: resp.StatusCode, Status: resp.Status, ...}
	// with Status being the raw HTTP status line ("404 Not Found") rather
	// than a canonical gRPC name.
	cases := []struct {
		code int
		want core.StatusName
	}{
		{400, core.INVALID_ARGUMENT},
		{401, core.UNAUTHENTICATED},
		{403, core.PERMISSION_DENIED},
		{404, core.NOT_FOUND},
		{409, core.ABORTED},
		{429, core.RESOURCE_EXHAUSTED},
		{500, core.INTERNAL},
		{501, core.UNIMPLEMENTED},
		{502, core.INTERNAL},
		{503, core.UNAVAILABLE},
		{504, core.DEADLINE_EXCEEDED},
	}
	for _, tc := range cases {
		t.Run(fmt.Sprintf("%d", tc.code), func(t *testing.T) {
			api := genai.APIError{Code: tc.code, Status: fmt.Sprintf("%d plain text", tc.code)}
			wrapped := wrapAPIError(api)
			var ge *core.GenkitError
			if !errors.As(wrapped, &ge) {
				t.Fatalf("wrapAPIError did not produce a GenkitError; got %T", wrapped)
			}
			if ge.Status != tc.want {
				t.Fatalf("status = %q, want %q", ge.Status, tc.want)
			}
		})
	}
}

func TestWrapAPIError_PreservesOriginalMessage(t *testing.T) {
	api := genai.APIError{Code: 404, Status: "NOT_FOUND", Message: "models/foo not found"}
	wrapped := wrapAPIError(api)
	if wrapped.Error() != api.Error() {
		t.Fatalf("wrapped message = %q, want %q", wrapped.Error(), api.Error())
	}
}

func TestWrapAPIError_HandlesWrappedAPIError(t *testing.T) {
	// Verifies wrapAPIError still works when the APIError is already
	// wrapped by another error (e.g., context boundary).
	api := genai.APIError{Code: 503, Status: "UNAVAILABLE", Message: "temporary outage"}
	outer := fmt.Errorf("while calling model: %w", api)
	wrapped := wrapAPIError(outer)
	var ge *core.GenkitError
	if !errors.As(wrapped, &ge) {
		t.Fatalf("wrapAPIError did not produce a GenkitError; got %T", wrapped)
	}
	if ge.Status != core.UNAVAILABLE {
		t.Fatalf("status = %q, want UNAVAILABLE", ge.Status)
	}
}
