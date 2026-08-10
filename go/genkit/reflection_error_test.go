// Copyright 2026 Google LLC
// SPDX-License-Identifier: Apache-2.0

package genkit

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/tracing"
)

// TestRunActionPlainErrorResponse pins that an action failing with an error the
// framework never classified still produces an error response.
//
// ToReflectionError fills in the details envelope only from a stack or trace
// the error itself carried, and leaves the pointer nil otherwise. A plain error
// returned by a plugin or a user's own function hits that case, and the handler
// then wrote the run's trace ID into the nil envelope, panicking the reflection
// server mid-response. That is every unclassified failure, which is the common
// one: a provider SDK rejecting a request reaches here as its own error type.
func TestRunActionPlainErrorResponse(t *testing.T) {
	tc := tracing.NewTestOnlyTelemetryClient()
	tracing.WriteTelemetryImmediate(tc)

	g := Init(context.Background())
	defineTestAction(g.reg, "test/boom", api.ActionTypeCustom, nil, nil,
		func(_ context.Context, x int) (int, error) {
			// Deliberately unclassified, as a provider SDK's error would be.
			return 0, errors.New("provider rejected the request")
		})

	s := &reflectionServer{Server: &http.Server{}, activeActions: newActiveActionsMap()}
	ts := httptest.NewServer(serveMux(g, s))
	s.Addr = strings.TrimPrefix(ts.URL, "http://")
	defer ts.Close()

	for _, tt := range []struct {
		name string
		path string
	}{
		{"non-streaming", "/api/runAction"},
		{"streaming", "/api/runAction?stream=true"},
	} {
		t.Run(tt.name, func(t *testing.T) {
			res, err := http.Post(ts.URL+tt.path, "application/json",
				strings.NewReader(`{"key":"/custom/test/boom","input":3}`))
			if err != nil {
				// A panic in the handler closes the connection mid-response,
				// so it surfaces here rather than as a 500.
				t.Fatalf("request failed, the handler likely panicked: %v", err)
			}
			defer res.Body.Close()

			body, err := io.ReadAll(res.Body)
			if err != nil {
				t.Fatalf("reading body: %v", err)
			}

			var got struct {
				Error struct {
					Message string `json:"message"`
					Code    int    `json:"code"`
					Details *struct {
						TraceID *string `json:"traceId"`
					} `json:"details"`
				} `json:"error"`
			}
			if err := json.Unmarshal(body, &got); err != nil {
				t.Fatalf("response is not the error envelope: %v\nbody: %s", err, body)
			}
			if !strings.Contains(got.Error.Message, "provider rejected the request") {
				t.Errorf("message = %q, want the action's error", got.Error.Message)
			}
			// The trace ID is why the envelope gets written to at all, so an
			// error that arrived without one must still come back carrying it.
			if got.Error.Details == nil || got.Error.Details.TraceID == nil {
				t.Errorf("no trace ID on the error response: %s", body)
			}
		})
	}
}
