// Copyright 2025 Google LLC
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
	"testing"

	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace/noop"
)

// TestOTelInstrumentation_EmptyIDsWhenUnconfigured verifies the redesign's core
// property: OTelInstrumentation has no opinion about the provider and installs
// none. With OTel's no-op provider (nothing configured) its span ids are
// invalid, so TraceInfo reports empty rather than an all-zero span.
func TestOTelInstrumentation_EmptyIDsWhenUnconfigured(t *testing.T) {
	prev := otel.GetTracerProvider()
	t.Cleanup(func() { otel.SetTracerProvider(prev) })
	otel.SetTracerProvider(noop.NewTracerProvider())

	o := &OTelInstrumentation{}
	info := &SpanInfo{metadata: &spanMetadata{Name: "n"}}
	var got TraceInfo
	_, err := o.RunInNewSpan(context.Background(), info, func(_ context.Context, span Span) (any, error) {
		got = span.TraceInfo()
		return nil, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.TraceID != "" || got.SpanID != "" {
		t.Errorf("TraceInfo = %+v, want empty (no provider configured)", got)
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
	var got TraceInfo
	_, err := o.RunInNewSpan(context.Background(), info, func(_ context.Context, span Span) (any, error) {
		got = span.TraceInfo()
		return nil, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if got.TraceID == "" || got.SpanID == "" {
		t.Errorf("TraceInfo = %+v, want real ids (provider configured)", got)
	}
}
