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

package compat_oai

import (
	"errors"
	"fmt"
	"net/http"
	"testing"

	"github.com/openai/openai-go"

	"github.com/firebase/genkit/go/core/status"
)

// apiError builds an SDK error the way the SDK does. Request and Response must
// both be set: (*openai.Error).Error() dereferences them unconditionally, so a
// zero value panics when formatted.
func apiError(code int) *openai.Error {
	req, _ := http.NewRequest(http.MethodPost, "https://api.openai.com/v1/chat/completions", nil)
	return &openai.Error{
		StatusCode: code,
		Request:    req,
		Response:   &http.Response{StatusCode: code},
	}
}

func TestWrapAPIErrorNil(t *testing.T) {
	if got := WrapAPIError(nil); got != nil {
		t.Errorf("WrapAPIError(nil) = %v, want nil", got)
	}
}

func TestWrapAPIErrorPassesThroughNonAPIError(t *testing.T) {
	// The SDK returns transport failures unwrapped. Those stay unclassified,
	// which the retry middleware treats as retryable: a dial timeout is worth
	// another attempt.
	plain := errors.New("dial tcp: i/o timeout")
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
		{http.StatusBadRequest, status.InvalidArgument},
		{http.StatusUnauthorized, status.Unauthenticated},
		{http.StatusForbidden, status.PermissionDenied},
		{http.StatusNotFound, status.NotFound},
		{http.StatusTooManyRequests, status.ResourceExhausted},
		{http.StatusInternalServerError, status.Internal},
		{http.StatusServiceUnavailable, status.Unavailable},
	}
	for _, tt := range tests {
		t.Run(http.StatusText(tt.code), func(t *testing.T) {
			s, ok := status.Classified(WrapAPIError(apiError(tt.code)))
			if !ok {
				t.Fatalf("WrapAPIError left a %d unclassified", tt.code)
			}
			if s != tt.want {
				t.Errorf("status = %v, want %v", s, tt.want)
			}
		})
	}
}

func TestWrapAPIErrorFindsWrappedAPIError(t *testing.T) {
	// The SDK error is usually several frames below where it is returned, so
	// classification has to survive the context callers wrap around it.
	err := fmt.Errorf("failed to create completion: %w", apiError(http.StatusBadRequest))
	if got := status.Of(WrapAPIError(err)); got != status.InvalidArgument {
		t.Errorf("status = %v, want %v", got, status.InvalidArgument)
	}
}

// TestWrapAPIErrorClientErrorsAreNotRetried pins the property that motivates
// the wrapper: a request the provider rejected must not be reissued unchanged
// by the retry middleware, whose default set is UNAVAILABLE, DEADLINE_EXCEEDED,
// RESOURCE_EXHAUSTED, ABORTED, and INTERNAL.
func TestWrapAPIErrorClientErrorsAreNotRetried(t *testing.T) {
	retried := map[status.Name]bool{
		status.Unavailable:       true,
		status.DeadlineExceeded:  true,
		status.ResourceExhausted: true,
		status.Aborted:           true,
		status.Internal:          true,
	}
	for _, code := range []int{
		http.StatusBadRequest,
		http.StatusUnauthorized,
		http.StatusForbidden,
		http.StatusNotFound,
	} {
		s, ok := status.Classified(WrapAPIError(apiError(code)))
		if !ok {
			t.Fatalf("a %d is unclassified, so retry would reissue it", code)
		}
		if retried[s] {
			t.Errorf("a %d maps to %v, which the retry middleware reissues", code, s)
		}
	}
	// A rate limit stays retryable: that one is worth waiting out.
	if s, _ := status.Classified(WrapAPIError(apiError(http.StatusTooManyRequests))); !retried[s] {
		t.Errorf("429 maps to %v, which retry does not reissue", s)
	}
}
