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

package anthropic

import (
	"errors"
	"fmt"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/firebase/genkit/go/core/status"
)

func TestWrapAPIErrorNil(t *testing.T) {
	if got := WrapAPIError(nil); got != nil {
		t.Errorf("WrapAPIError(nil) = %v, want nil", got)
	}
}

func TestWrapAPIErrorPassesThroughNonAPIError(t *testing.T) {
	// A transport failure the SDK does not wrap stays unclassified, which the
	// retry middleware treats as retryable. That is the wanted behavior: a
	// connection reset is worth another attempt.
	plain := errors.New("connection reset by peer")
	got := WrapAPIError(plain)
	if got != plain {
		t.Errorf("WrapAPIError returned %v, want the original error unchanged", got)
	}
	if _, ok := status.Classified(got); ok {
		t.Error("WrapAPIError classified an error that did not come from the API")
	}
}

func TestWrapAPIErrorMapsHTTPStatus(t *testing.T) {
	tests := []struct {
		code int
		want status.Name
	}{
		{400, status.InvalidArgument},
		{401, status.Unauthenticated},
		{403, status.PermissionDenied},
		{404, status.NotFound},
		{429, status.ResourceExhausted},
		{500, status.Internal},
		{503, status.Unavailable},
		{529, status.Internal}, // Anthropic's "overloaded"; any unmapped 5xx is Internal
	}
	for _, tt := range tests {
		t.Run(fmt.Sprint(tt.code), func(t *testing.T) {
			got := WrapAPIError(&anthropic.Error{StatusCode: tt.code})
			s, ok := status.Classified(got)
			if !ok {
				t.Fatalf("WrapAPIError left a %d API error unclassified", tt.code)
			}
			if s != tt.want {
				t.Errorf("status = %v, want %v", s, tt.want)
			}
		})
	}
}

func TestWrapAPIErrorFindsWrappedAPIError(t *testing.T) {
	// Classification must survive the context wrapping callers add, since the
	// SDK error is often several frames below where it is returned.
	err := fmt.Errorf("generating content: %w", &anthropic.Error{StatusCode: 400})
	if got := status.Of(WrapAPIError(err)); got != status.InvalidArgument {
		t.Errorf("status = %v, want %v", got, status.InvalidArgument)
	}
}

// TestWrapAPIErrorClientErrorsAreNotRetried pins the property that motivates
// the wrapper: a request the server rejected must not be reissued unchanged by
// the retry middleware, whose default set is UNAVAILABLE, DEADLINE_EXCEEDED,
// RESOURCE_EXHAUSTED, ABORTED, and INTERNAL.
func TestWrapAPIErrorClientErrorsAreNotRetried(t *testing.T) {
	retried := map[status.Name]bool{
		status.Unavailable:       true,
		status.DeadlineExceeded:  true,
		status.ResourceExhausted: true,
		status.Aborted:           true,
		status.Internal:          true,
	}
	for _, code := range []int{400, 401, 403, 404} {
		s, ok := status.Classified(WrapAPIError(&anthropic.Error{StatusCode: code}))
		if !ok {
			t.Fatalf("a %d API error is unclassified, so retry would reissue it", code)
		}
		if retried[s] {
			t.Errorf("a %d API error maps to %v, which the retry middleware reissues", code, s)
		}
	}
}
