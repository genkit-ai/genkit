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

package genkit

import (
	"context"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/core/status"
)

// Each case below is a path that used to put internal failure detail on the
// wire. The flow HTTP boundary wrote err.Error() verbatim on every branch, so
// whatever an error happened to say reached the client.
func TestHandlerDoesNotLeakInternalDetail(t *testing.T) {
	g := Init(context.Background())

	// A provider error relayed verbatim. googlegenai wraps the raw SDK error,
	// whose text can carry project and model resource paths, quota specifics,
	// and permission detail.
	providerFlow := DefineFlow(g, "leakProvider", func(ctx context.Context, in string) (string, error) {
		return "", status.Errorf(status.Base(status.FromHTTPCode(429)),
			"googleapi: Error 429: Quota exceeded for project 12345678, model projects/acme-prod/locations/us/models/x")
	})

	// An output schema violation. The message embeds the action key and a
	// field-by-field dump of the shape the action failed to produce.
	outputFlow := DefineFlow(g, "leakOutput", func(ctx context.Context, in string) (string, error) {
		return "", status.Errorf(status.ErrInvalidOutput,
			"invalid output from action %q: data did not match expected schema:\n- ssn: Invalid type. Expected: string", "/flow/leakOutput")
	})

	// An unclassified error from user code: the default path for anything not
	// deliberately classified.
	unclassifiedFlow := DefineFlow(g, "leakUnclassified", func(ctx context.Context, in string) (string, error) {
		return "", errors.New("dial tcp 10.0.0.7:5432: connect: connection refused")
	})

	tests := []struct {
		name    string
		flow    api.Action
		code    int
		secrets []string
	}{
		{"provider error", providerFlow, http.StatusTooManyRequests, []string{"acme-prod", "12345678", "googleapi"}},
		{"output schema violation", outputFlow, http.StatusInternalServerError, []string{"ssn", "leakOutput", "expected schema"}},
		{"unclassified error", unclassifiedFlow, http.StatusInternalServerError, []string{"10.0.0.7", "connection refused"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			handler := Handler(tt.flow)

			req := httptest.NewRequest("POST", "/", strings.NewReader(`{"data":"x"}`))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()
			handler(w, req)

			resp := w.Result()
			body, _ := io.ReadAll(resp.Body)

			if resp.StatusCode != tt.code {
				t.Errorf("status = %d, want %d", resp.StatusCode, tt.code)
			}
			for _, s := range tt.secrets {
				if strings.Contains(string(body), s) {
					t.Errorf("response leaked %q; body = %q", s, string(body))
				}
			}
		})
	}
}

// The generic message must still name the status so a client can act on it,
// even though the specific message is withheld.
func TestGenericMessageStillCarriesStatus(t *testing.T) {
	for _, tt := range []struct {
		err  error
		want string
	}{
		{status.Errorf(status.ErrNotFound, "model %q not found", "secret-internal-name"), "not found"},
		{status.Errorf(status.ErrPermissionDenied, "caller lacks roles/aiplatform.user"), "permission denied"},
		{errors.New("raw"), "internal"},
	} {
		msg, public := status.PublicMessage(tt.err)
		if public {
			t.Errorf("PublicMessage(%v) reported public", tt.err)
		}
		if msg != tt.want {
			t.Errorf("PublicMessage = %q, want %q", msg, tt.want)
		}
	}
}

// GENKIT_ENV=dev keeps the full message, so local development is not blinded by
// the redaction that protects deployed servers.
func TestDevEnvironmentKeepsFullMessage(t *testing.T) {
	t.Setenv("GENKIT_ENV", "dev")

	err := status.Errorf(status.ErrNotFound, "model %q not found", "googleai/nope")
	msg, code := clientError(err)
	if !strings.Contains(msg, "googleai/nope") {
		t.Errorf("dev message = %q, want the full text", msg)
	}
	if code != http.StatusNotFound {
		t.Errorf("code = %d, want %d", code, http.StatusNotFound)
	}
}

// A public error keeps its own status code. It used to fall through to 500
// because *core.UserFacingError is structurally unrelated to *core.GenkitError,
// so the handler's errors.As could never match it.
func TestPublicErrorKeepsItsStatusCode(t *testing.T) {
	for _, tt := range []struct {
		err  error
		code int
	}{
		{status.PublicErrorf(status.ErrUnauthenticated, "authorization header is required"), http.StatusUnauthorized},
		{status.PublicErrorf(status.ErrInvalidArgument, "field %q is required", "email"), http.StatusBadRequest},
	} {
		msg, code := clientError(tt.err)
		if code != tt.code {
			t.Errorf("code = %d, want %d", code, tt.code)
		}
		if msg != tt.err.Error() {
			t.Errorf("msg = %q, want the public message %q", msg, tt.err.Error())
		}
	}
}
