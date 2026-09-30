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

package tracing

import (
	"context"
	"errors"
	"testing"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/codes"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace/noop"
)

// TestOTelInstrumentation_EmptyIDsWhenUnconfigured verifies that
// OTelInstrumentation installs no provider of its own. With OTel's no-op
// provider (nothing configured) its span ids are invalid, so TraceInfo reports
// empty rather than an all-zero span.
func TestOTelInstrumentation_EmptyIDsWhenUnconfigured(t *testing.T) {
	prev := otel.GetTracerProvider()
	t.Cleanup(func() { otel.SetTracerProvider(prev) })
	otel.SetTracerProvider(noop.NewTracerProvider())

	o := &OTelInstrumentation{}
	info := &SpanInfo{metadata: &spanMetadata{Name: "n"}}
	_, span := o.StartSpan(context.Background(), info)
	got := span.TraceInfo()
	span.End(&SpanResult{})
	if got.TraceID != "" || got.SpanID != "" {
		t.Errorf("TraceInfo = %+v, want empty (no provider configured)", got)
	}
}

// TestOTelInstrumentation_SkipsEncodingWhenNotRecording checks that a span the
// no-op provider discards is not paid for: the end write, which JSON-encodes
// input and output, is skipped. Realtime is pinned off: with it on, input is
// seeded as a start option, before recording is known (and is then reused by
// the other providers through the cache).
func TestOTelInstrumentation_SkipsEncodingWhenNotRecording(t *testing.T) {
	prevRealtime := realtimeTelemetryActive
	realtimeTelemetryActive = false
	t.Cleanup(func() { realtimeTelemetryActive = prevRealtime })

	prev := otel.GetTracerProvider()
	t.Cleanup(func() { otel.SetTracerProvider(prev) })
	otel.SetTracerProvider(noop.NewTracerProvider())

	sm := &spanMetadata{Name: "n", Input: "in"}
	o := &OTelInstrumentation{}
	_, span := o.StartSpan(context.Background(), &SpanInfo{metadata: sm})
	sm.Output = "out"
	span.End(&SpanResult{output: "out"})
	if sm.inputJSON != nil || sm.outputJSON != nil {
		t.Error("input/output were JSON-encoded for a non-recording span")
	}
}

// TestOTelInstrumentation_RealIDsWhenConfigured verifies that once a real SDK
// provider is registered (a user's setup, or a plugin), OTelInstrumentation
// reads it and yields real ids.
func TestOTelInstrumentation_RealIDsWhenConfigured(t *testing.T) {
	prev := otel.GetTracerProvider()
	t.Cleanup(func() { otel.SetTracerProvider(prev) })
	otel.SetTracerProvider(sdktrace.NewTracerProvider())

	o := &OTelInstrumentation{}
	info := &SpanInfo{metadata: &spanMetadata{Name: "n"}}
	_, span := o.StartSpan(context.Background(), info)
	got := span.TraceInfo()
	span.End(&SpanResult{})
	if got.TraceID == "" || got.SpanID == "" {
		t.Errorf("TraceInfo = %+v, want real ids (provider configured)", got)
	}
}

// TestOTelInstrumentation_DedicatedTracerProvider checks that a provider set
// on the field receives Genkit spans and the global one does not.
func TestOTelInstrumentation_DedicatedTracerProvider(t *testing.T) {
	globalExp := tracetest.NewInMemoryExporter()
	prev := otel.GetTracerProvider()
	t.Cleanup(func() { otel.SetTracerProvider(prev) })
	otel.SetTracerProvider(sdktrace.NewTracerProvider(sdktrace.WithSyncer(globalExp)))

	exp := tracetest.NewInMemoryExporter()
	useInstrumentation(t, &OTelInstrumentation{
		TracerProvider: sdktrace.NewTracerProvider(sdktrace.WithSyncer(exp)),
	})

	wantErr := errors.New("boom")
	_, _ = RunInNewSpan(context.Background(), &SpanMetadata{Name: "root", Type: "action"}, "in",
		func(ctx context.Context, _ string) (string, error) { return "", wantErr })

	spans := exp.GetSpans()
	if len(spans) != 1 || spans[0].Name != "root" {
		t.Fatalf("dedicated provider got %v, want one span named root", spans)
	}
	if spans[0].Status.Code != codes.Error || spans[0].Status.Description != "boom" {
		t.Errorf("status = %+v, want error boom", spans[0].Status)
	}
	if n := len(globalExp.GetSpans()); n != 0 {
		t.Errorf("global provider got %d spans, want 0", n)
	}
}
