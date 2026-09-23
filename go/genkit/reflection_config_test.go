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
	"net/http"
	"net/http/httptest"
	"testing"
)

// envFunc turns a map into the getenv signature resolveReflectionConfig takes.
func envFunc(env map[string]string) func(string) string {
	return func(key string) string { return env[key] }
}

func TestResolveReflectionConfig(t *testing.T) {
	tests := []struct {
		name    string
		env     map[string]string
		optPort int
		want    reflectionConfig
	}{
		{
			name: "off when nothing is set",
			env:  map[string]string{},
			want: reflectionConfig{mode: reflectionOff},
		},
		{
			name: "disabled beats everything",
			env: map[string]string{
				"GENKIT_REFLECTION_DISABLED":  "true",
				"GENKIT_ENV":                  "dev",
				"GENKIT_REFLECTION_PORT":      "3100",
				"GENKIT_REFLECTION_V2_SERVER": "ws://127.0.0.1:3200",
			},
			want: reflectionConfig{mode: reflectionDisabled},
		},
		{
			name: "only the exact string true disables",
			env:  map[string]string{"GENKIT_REFLECTION_DISABLED": "1", "GENKIT_ENV": "dev"},
			want: reflectionConfig{mode: reflectionV1, host: defaultReflectionHost, port: defaultReflectionPort},
		},
		{
			name: "v2 beats a configured v1 port",
			env: map[string]string{
				"GENKIT_REFLECTION_V2_SERVER":    "ws://127.0.0.1:3200",
				"GENKIT_REFLECTION_PORT":         "3100",
				"GENKIT_REFLECTION_SECRET_TOKEN": "s3cret",
			},
			want: reflectionConfig{mode: reflectionV2, v2URL: "ws://127.0.0.1:3200", secret: "s3cret"},
		},
		{
			name: "host alone turns it on",
			env:  map[string]string{"GENKIT_REFLECTION_HOST": "0.0.0.0"},
			want: reflectionConfig{mode: reflectionV1, host: "0.0.0.0", port: defaultReflectionPort},
		},
		{
			name: "port alone turns it on and pins",
			env:  map[string]string{"GENKIT_REFLECTION_PORT": "4200"},
			want: reflectionConfig{mode: reflectionV1, host: defaultReflectionHost, port: 4200, pinned: true},
		},
		{
			name: "dev probes from 3100",
			env:  map[string]string{"GENKIT_ENV": "dev"},
			want: reflectionConfig{mode: reflectionV1, host: defaultReflectionHost, port: defaultReflectionPort},
		},
		{
			name:    "env port beats the option",
			env:     map[string]string{"GENKIT_REFLECTION_PORT": "4200"},
			optPort: 9999,
			want:    reflectionConfig{mode: reflectionV1, host: defaultReflectionHost, port: 4200, pinned: true},
		},
		{
			name:    "option is the probe start when the env has no port",
			env:     map[string]string{"GENKIT_ENV": "dev"},
			optPort: 9999,
			want:    reflectionConfig{mode: reflectionV1, host: defaultReflectionHost, port: 9999},
		},
		{
			name: "port 0 is valid and pinned",
			env:  map[string]string{"GENKIT_REFLECTION_PORT": "0"},
			want: reflectionConfig{mode: reflectionV1, host: defaultReflectionHost, port: 0, pinned: true},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := resolveReflectionConfig(envFunc(tt.env), tt.optPort)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got != tt.want {
				t.Errorf("got %+v, want %+v", got, tt.want)
			}
		})
	}
}

func TestResolveReflectionConfigInvalidPort(t *testing.T) {
	// An invalid port must fail startup rather than quietly fall back to
	// probing, which would hide a typo in a deployment config.
	for _, port := range []string{"abc", "-1", "70000", "3100.5", " "} {
		t.Run(port, func(t *testing.T) {
			_, err := resolveReflectionConfig(
				envFunc(map[string]string{"GENKIT_REFLECTION_PORT": port}), 0)
			if err == nil {
				t.Fatalf("expected an error for %q", port)
			}
		})
	}
}

func TestIsLoopbackHost(t *testing.T) {
	for _, host := range []string{"127.0.0.1", "127.1.2.3", "localhost", "::1", "[::1]"} {
		if !isLoopbackHost(host) {
			t.Errorf("%q should be loopback", host)
		}
	}
	for _, host := range []string{"0.0.0.0", "192.168.1.5", "10.0.0.1", "example.com"} {
		if isLoopbackHost(host) {
			t.Errorf("%q should not be loopback", host)
		}
	}
}

func TestSecretsEqual(t *testing.T) {
	if !secretsEqual("abc", "abc") {
		t.Error("equal secrets should compare equal")
	}
	if secretsEqual("abc", "abd") {
		t.Error("different secrets should not compare equal")
	}
	if secretsEqual("abc", "much-longer-secret") {
		t.Error("different-length secrets should not compare equal")
	}
}

func TestRequireReflectionSecret(t *testing.T) {
	ok := http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	tests := []struct {
		name     string
		secret   string
		path     string
		provided string
		want     int
	}{
		{name: "no secret configured", secret: "", path: "/api/actions", want: http.StatusOK},
		{name: "missing header", secret: "s3cret", path: "/api/actions", want: http.StatusUnauthorized},
		{name: "wrong secret", secret: "s3cret", path: "/api/actions", provided: "nope", want: http.StatusUnauthorized},
		{name: "right secret", secret: "s3cret", path: "/api/actions", provided: "s3cret", want: http.StatusOK},
		{name: "health is exempt", secret: "s3cret", path: "/api/__health", want: http.StatusOK},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, tt.path, nil)
			if tt.provided != "" {
				req.Header.Set(reflectionSecretHeader, tt.provided)
			}
			rec := httptest.NewRecorder()
			requireReflectionSecret(tt.secret, ok).ServeHTTP(rec, req)
			if rec.Code != tt.want {
				t.Errorf("got %d, want %d", rec.Code, tt.want)
			}
			if tt.want == http.StatusUnauthorized && rec.Body.Len() != 0 {
				t.Errorf("401 body should be empty, got %q", rec.Body.String())
			}
		})
	}
}
