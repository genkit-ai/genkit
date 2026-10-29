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

// Package tracingtest captures the spans Genkit reports, in memory and with no
// OpenTelemetry SDK, for tests across the module. core/tracing's own tests
// cannot import it (import cycle) and keep local helpers.
//
// The instrumentation registry is process-wide, so a test that calls Capture
// must not run in parallel with other tests that read spans.
package tracingtest

import (
	"sync"
	"testing"

	"github.com/firebase/genkit/go/core/tracing"
)

// Recorder is an in-memory capture of the spans Genkit reports.
type Recorder struct {
	*tracing.TestOnlyTelemetryClient
}

var (
	mu sync.Mutex
	// fallback is the provider Capture restores on cleanup: the one
	// InstallDefault set, or nil for the implicit default.
	fallback tracing.Instrumentation
)

// InstallDefault routes every span in the test binary to an in-memory
// recorder, so runs carry real trace ids without an OpenTelemetry SDK. Call it
// from TestMain; Capture restores it after each capturing test.
func InstallDefault() *Recorder {
	r := newRecorder()
	inst := tracing.NewDirectTelemetryInstrumentation(r.TestOnlyTelemetryClient)
	mu.Lock()
	fallback = inst
	mu.Unlock()
	tracing.SetInstrumentation(inst)
	return r
}

// Capture routes spans to a fresh recorder for the rest of the test, then
// restores the InstallDefault provider (or the implicit default) on cleanup.
func Capture(t testing.TB) *Recorder {
	t.Helper()
	r := newRecorder()
	tracing.SetInstrumentation(tracing.NewDirectTelemetryInstrumentation(r.TestOnlyTelemetryClient))
	t.Cleanup(func() {
		mu.Lock()
		restore := fallback
		mu.Unlock()
		tracing.SetInstrumentation(restore) // nil restores the implicit default
	})
	return r
}

func newRecorder() *Recorder {
	return &Recorder{TestOnlyTelemetryClient: tracing.NewTestOnlyTelemetryClient()}
}

// ByName returns every recorded span with the given display name.
func (r *Recorder) ByName(name string) []*tracing.SpanData {
	var out []*tracing.SpanData
	for _, s := range r.Spans() {
		if s.DisplayName == name {
			out = append(out, s)
		}
	}
	return out
}

// First returns the first recorded span with the given display name, or nil.
func (r *Recorder) First(name string) *tracing.SpanData {
	for _, s := range r.Spans() {
		if s.DisplayName == name {
			return s
		}
	}
	return nil
}

// Attr returns the string value of the span attribute key, if present.
func Attr(span *tracing.SpanData, key string) (string, bool) {
	s, ok := span.Attributes[key].(string)
	return s, ok
}
