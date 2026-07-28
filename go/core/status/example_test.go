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

package status_test

import (
	"errors"
	"fmt"

	"github.com/firebase/genkit/go/core/status"
)

// A package declares its domain failure modes from a base sentinel, so callers
// can match at either granularity.
var errMaxTurnsExceeded = status.ErrAborted.Subtype("max turns exceeded")

func ExampleErrorf() {
	err := status.Errorf(errMaxTurnsExceeded, "stopped after %d turns", 5)

	fmt.Println(err)
	fmt.Println("specific:", errors.Is(err, errMaxTurnsExceeded))
	fmt.Println("broad:   ", errors.Is(err, status.ErrAborted))
	fmt.Println("status:  ", status.Of(err))
	// Output:
	// stopped after 5 turns
	// specific: true
	// broad:    true
	// status:   ABORTED
}

// Adding context as an error travels up the stack must not reclassify it.
func ExampleOf_wrapping() {
	err := error(status.Errorf(status.ErrNotFound, "model %q not found", "gemini"))
	err = fmt.Errorf("agent %q: %w", "planner", err)

	fmt.Println(err)
	fmt.Println("status:", status.Of(err))
	fmt.Println("still not found:", errors.Is(err, status.ErrNotFound))
	// Output:
	// agent "planner": model "gemini" not found
	// status: NOT_FOUND
	// still not found: true
}

// A tool's own NOT_FOUND is not a NOT_FOUND for the request that invoked it.
// Reclassify deliberately at the boundary; the original stays reachable.
func ExampleErrorf_reclassify() {
	inner := status.Errorf(status.ErrNotFound, "no row for id 42")
	err := status.Errorf(status.ErrInternal, "tool %q: %w", "lookup", inner)

	fmt.Println(err)
	fmt.Println("status:", status.Of(err))
	fmt.Println("cause still reachable:", errors.Is(err, status.ErrNotFound))
	// Output:
	// tool "lookup": no row for id 42
	// status: INTERNAL
	// cause still reachable: true
}

// Transports report the message only when it was marked safe to return.
func ExamplePublicMessage() {
	internal := status.Errorf(status.ErrPermissionDenied, "user alice lacks role admin")
	public := status.PublicErrorf(status.ErrInvalidArgument, "invalid %q parameter", "stream")

	for _, err := range []error{internal, public} {
		msg, ok := status.PublicMessage(err)
		fmt.Printf("%d %q (public=%v)\n", status.Of(err).HTTPCode(), msg, ok)
	}
	// Output:
	// 403 "permission denied" (public=false)
	// 400 "invalid \"stream\" parameter" (public=true)
}
