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

package otelgenai

import (
	"context"
	"sync"
	"testing"

	"github.com/google/go-cmp/cmp"
	"go.opentelemetry.io/otel/attribute"
	otellog "go.opentelemetry.io/otel/log"
	"go.opentelemetry.io/otel/log/embedded"

	"github.com/firebase/genkit/go/core/tracing"
	"github.com/firebase/genkit/go/plugins/otelgenai/internal/genai"
)

// recordingLoggerProvider keeps emitted log records in memory. A fake rather
// than sdk/log, which the library module does not otherwise depend on.
type recordingLoggerProvider struct {
	embedded.LoggerProvider
	mu      sync.Mutex
	records []otellog.Record
}

func (p *recordingLoggerProvider) Logger(string, ...otellog.LoggerOption) otellog.Logger {
	return &recordingLogger{p: p}
}

type recordingLogger struct {
	embedded.Logger
	p *recordingLoggerProvider
}

func (l *recordingLogger) Emit(_ context.Context, r otellog.Record) {
	l.p.mu.Lock()
	defer l.p.mu.Unlock()
	l.p.records = append(l.p.records, r.Clone())
}

func (l *recordingLogger) Enabled(context.Context, otellog.EnabledParameters) bool { return true }

// TestEventOnlyStructuredContent checks the operation.details event carries
// content as structured values, and that it goes to the configured provider.
func TestEventOnlyStructuredContent(t *testing.T) {
	setup(t, Options{})
	lp := &recordingLoggerProvider{}
	tracing.SetInstrumentation(New(Options{ContentCapturingMode: EventOnly, LoggerProvider: lp}))

	runModel(t)

	if len(lp.records) != 1 {
		t.Fatalf("records = %d, want 1", len(lp.records))
	}
	rec := lp.records[0]
	if rec.EventName() != genai.OperationDetailsEvent {
		t.Errorf("event name = %q, want %q", rec.EventName(), genai.OperationDetailsEvent)
	}
	got := map[string]any{}
	rec.WalkAttributes(func(kv attribute.KeyValue) bool {
		got[string(kv.Key)] = plain(kv.Value)
		return true
	})
	want := map[string]any{
		genai.AttrSystemInstructions: []any{
			map[string]any{"type": "text", "content": "be nice"},
		},
		genai.AttrInputMessages: []any{
			map[string]any{"role": "user", "parts": []any{map[string]any{"type": "text", "content": "hello"}}},
		},
		genai.AttrOutputMessages: []any{
			map[string]any{
				"role":          "assistant",
				"finish_reason": "stop",
				"parts":         []any{map[string]any{"type": "text", "content": "hi there"}},
			},
		},
	}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("event attributes mismatch (-want +got):\n%s", diff)
	}
}

// plain unwraps a structured attribute value into Go maps and slices.
func plain(v attribute.Value) any {
	switch v.Type() {
	case attribute.MAP:
		m := map[string]any{}
		for _, kv := range v.AsMap() {
			m[string(kv.Key)] = plain(kv.Value)
		}
		return m
	case attribute.SLICE:
		var s []any
		for _, e := range v.AsSlice() {
			s = append(s, plain(e))
		}
		return s
	default:
		return v.AsInterface()
	}
}

func TestStructuredValueScalars(t *testing.T) {
	got := plain(structuredValue(map[string]any{"n": 3, "f": 1.5, "b": true, "nil": nil, "s": "x"}))
	want := map[string]any{"n": int64(3), "f": 1.5, "b": true, "nil": nil, "s": "x"}
	if diff := cmp.Diff(want, got); diff != "" {
		t.Errorf("structuredValue mismatch (-want +got):\n%s", diff)
	}
}
