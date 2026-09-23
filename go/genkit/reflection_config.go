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
	"crypto/sha256"
	"crypto/subtle"
	"fmt"
	"net/http"
	"strconv"
	"strings"
)

const (
	// reflectionSecretHeader carries the reflection secret on v1 requests.
	reflectionSecretHeader = "x-genkit-reflection-secret"
	// defaultReflectionHost is the interface used when none is configured.
	defaultReflectionHost = "127.0.0.1"
	// defaultReflectionPort is the first port tried when none is pinned.
	defaultReflectionPort = 3100
	// reflectionAuthErrorCode is the JSON-RPC code the CLI returns when a v2
	// register fails auth. Terminal: the secret will not change, so a runtime
	// that sees it must stop reconnecting.
	reflectionAuthErrorCode = -32001
)

// reflectionMode is how the reflection API runs, if at all.
type reflectionMode int

const (
	// reflectionDisabled is an explicit kill switch, honoured everywhere.
	reflectionDisabled reflectionMode = iota
	// reflectionOff means nothing asked for a server.
	reflectionOff
	// reflectionV1 listens for the CLI.
	reflectionV1
	// reflectionV2 dials out to the CLI.
	reflectionV2
)

// reflectionConfig is the resolved reflection API configuration.
type reflectionConfig struct {
	mode reflectionMode
	// v2URL is set when mode is reflectionV2.
	v2URL string
	// host and port apply when mode is reflectionV1.
	host string
	port int
	// pinned reports whether port must be bound exactly, with no probing.
	pinned bool
	// secret is required on every call when non-empty.
	secret string
}

// resolveReflectionConfig decides how the reflection API should run.
//
// First match wins:
//  1. GENKIT_REFLECTION_DISABLED == "true" turns everything off.
//  2. GENKIT_REFLECTION_V2_SERVER dials out instead of listening.
//  3. GENKIT_REFLECTION_PORT or GENKIT_REFLECTION_HOST starts the v1 server.
//  4. GENKIT_ENV == "dev" starts the v1 server with defaults.
//  5. Otherwise off.
//
// Setting a host or port is itself the on-switch, so there is no way to
// configure a server and then wonder why it never started. The environment
// beats optPort on purpose: whoever set the variable is typically the
// supervisor that already published that port.
//
// getenv is injected so this is testable without touching the process
// environment.
func resolveReflectionConfig(getenv func(string) string, optPort int) (reflectionConfig, error) {
	if getenv("GENKIT_REFLECTION_DISABLED") == "true" {
		return reflectionConfig{mode: reflectionDisabled}, nil
	}
	secret := getenv("GENKIT_REFLECTION_SECRET_TOKEN")
	if v2URL := getenv("GENKIT_REFLECTION_V2_SERVER"); v2URL != "" {
		return reflectionConfig{mode: reflectionV2, v2URL: v2URL, secret: secret}, nil
	}

	envPort := getenv("GENKIT_REFLECTION_PORT")
	host := getenv("GENKIT_REFLECTION_HOST")
	if envPort == "" && host == "" && getenv("GENKIT_ENV") != "dev" {
		return reflectionConfig{mode: reflectionOff}, nil
	}

	cfg := reflectionConfig{
		mode:   reflectionV1,
		host:   host,
		port:   optPort,
		secret: secret,
	}
	if cfg.host == "" {
		cfg.host = defaultReflectionHost
	}
	if cfg.port == 0 {
		cfg.port = defaultReflectionPort
	}
	if envPort != "" {
		// An invalid value fails startup rather than falling back to probing:
		// a typo in a deployment config should be loud.
		port, err := strconv.Atoi(envPort)
		if err != nil || port < 0 || port > 65535 {
			return reflectionConfig{}, fmt.Errorf(
				"GENKIT_REFLECTION_PORT must be an integer between 0 and 65535, got %q", envPort)
		}
		cfg.port = port
		cfg.pinned = true
	}
	return cfg, nil
}

// requireReflectionSecret rejects requests that do not carry secret.
//
// /api/__health is exempt: it carries no registry content and is what
// orchestrators and the CLI probe before they have any reason to know a
// secret. The 401 body is empty on purpose.
func requireReflectionSecret(secret string, next http.Handler) http.Handler {
	if secret == "" {
		return next
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/__health" {
			next.ServeHTTP(w, r)
			return
		}
		provided := r.Header.Get(reflectionSecretHeader)
		if provided == "" || !secretsEqual(provided, secret) {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// isLoopbackHost reports whether a host is unreachable from other machines.
func isLoopbackHost(host string) bool {
	return host == "localhost" || host == "::1" || host == "[::1]" ||
		strings.HasPrefix(host, "127.")
}

// secretsEqual compares in constant time. Hashing first gives equal-length
// inputs without leaking the expected length.
func secretsEqual(a, b string) bool {
	ha := sha256.Sum256([]byte(a))
	hb := sha256.Sum256([]byte(b))
	return subtle.ConstantTimeCompare(ha[:], hb[:]) == 1
}
