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
	"net"
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

// advertisedReflectionAddr is the host:port to write into the runtime
// discovery file for a server listening on addr. A wildcard bind is reachable
// on loopback, and 0.0.0.0 is not a valid destination everywhere, so it is
// advertised as 127.0.0.1.
func advertisedReflectionAddr(addr string) string {
	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		return addr
	}
	if ip := net.ParseIP(host); ip != nil && ip.IsUnspecified() {
		host = defaultReflectionHost
	}
	return net.JoinHostPort(host, port)
}

// reflectionMode is how the reflection API runs, if at all.
type reflectionMode int

const (
	// reflectionOff means no server: either nothing asked for one, or
	// GENKIT_REFLECTION_ENABLED=false turned it off. The zero value.
	reflectionOff reflectionMode = iota
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

// enabled reports whether a reflection server (v1 or v2) should run.
func (c reflectionConfig) enabled() bool {
	return c.mode != reflectionOff
}

// resolveReflectionConfig decides how the reflection API should run.
//
// Whether it runs:
//   - GENKIT_REFLECTION_ENABLED=false turns it off, even under dev.
//   - GENKIT_REFLECTION_ENABLED=true turns it on in any environment.
//   - Unset, it runs only under GENKIT_ENV=dev, as it always has.
//
// How it runs, once on: GENKIT_REFLECTION_V2_SERVER dials out; otherwise the
// v1 server listens on GENKIT_REFLECTION_HOST/GENKIT_REFLECTION_PORT. Those
// are settings, not on-switches: a stray value in a production env does not
// expose the API, and is not even parsed while reflection is off.
//
// A chosen port, from the environment or optPort, is bound exactly; only an
// unchosen port probes upward from 3100. The environment beats optPort on
// purpose: whoever set the variable is typically the supervisor that already
// published that port. A nil optPort means unset; 0 lets the OS pick, the
// same as GENKIT_REFLECTION_PORT=0. optPort is validated even when unused so
// a bad value fails early.
//
// getenv is injected so this is testable without touching the process
// environment.
func resolveReflectionConfig(getenv func(string) string, optPort *int) (reflectionConfig, error) {
	if optPort != nil && (*optPort < 0 || *optPort > 65535) {
		return reflectionConfig{}, fmt.Errorf(
			"reflection port must be an integer between 0 and 65535, got %d", *optPort)
	}
	switch enabled := getenv("GENKIT_REFLECTION_ENABLED"); enabled {
	case "false":
		return reflectionConfig{mode: reflectionOff}, nil
	case "true":
	case "":
		if getenv("GENKIT_ENV") != "dev" {
			return reflectionConfig{mode: reflectionOff}, nil
		}
	default:
		return reflectionConfig{}, fmt.Errorf(
			"GENKIT_REFLECTION_ENABLED must be \"true\" or \"false\", got %q", enabled)
	}

	secret := getenv("GENKIT_REFLECTION_SECRET_TOKEN")
	if v2URL := getenv("GENKIT_REFLECTION_V2_SERVER"); v2URL != "" {
		return reflectionConfig{mode: reflectionV2, v2URL: v2URL, secret: secret}, nil
	}

	envPort := getenv("GENKIT_REFLECTION_PORT")
	host := getenv("GENKIT_REFLECTION_HOST")
	cfg := reflectionConfig{
		mode:   reflectionV1,
		host:   host,
		port:   defaultReflectionPort,
		secret: secret,
	}
	if cfg.host == "" {
		cfg.host = defaultReflectionHost
	}
	if optPort != nil {
		cfg.port, cfg.pinned = *optPort, true
	}
	if envPort != "" {
		// An invalid value fails startup rather than falling back to probing:
		// a typo in a deployment config should be loud.
		// Atoi alone accepts a leading sign ("+7"); require plain decimal digits.
		port, err := strconv.Atoi(envPort)
		if !isDecimalDigits(envPort) || err != nil || port < 0 || port > 65535 {
			return reflectionConfig{}, fmt.Errorf(
				"GENKIT_REFLECTION_PORT must be an integer between 0 and 65535, got %q", envPort)
		}
		cfg.port = port
		cfg.pinned = true
	}
	return cfg, nil
}

// isDecimalDigits reports whether s is non-empty and made only of ASCII digits.
func isDecimalDigits(s string) bool {
	if s == "" {
		return false
	}
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
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
// Only literal loopback IPs (any spelling of ::1, dotted-quad 127.x) and
// "localhost" count; a hostname like 127.internal.example does not.
func isLoopbackHost(host string) bool {
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(strings.TrimSuffix(strings.TrimPrefix(host, "["), "]"))
	return ip != nil && ip.IsLoopback()
}

// secretsEqual compares in constant time. Hashing first gives equal-length
// inputs without leaking the expected length.
func secretsEqual(a, b string) bool {
	ha := sha256.Sum256([]byte(a))
	hb := sha256.Sum256([]byte(b))
	return subtle.ConstantTimeCompare(ha[:], hb[:]) == 1
}
