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

package otel

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"go.opentelemetry.io/otel/attribute"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/status"
	"github.com/firebase/genkit/go/core/tracing"
	"github.com/firebase/genkit/go/plugins/otel/genai"
)

// setupMetrics configures the instrumentation with a meter backed by a manual
// reader, on top of the span recorder from setup.
func setupMetrics(t *testing.T) *sdkmetric.ManualReader {
	t.Helper()
	setup(t, GenAiInstrumentationOptions{})
	reader := sdkmetric.NewManualReader()
	mp := sdkmetric.NewMeterProvider(sdkmetric.WithReader(reader))
	tracing.ConfigureInstrumentation(NewGenAiInstrumentation(GenAiInstrumentationOptions{Meter: mp.Meter("test")}))
	return reader
}

func collect(t *testing.T, reader *sdkmetric.ManualReader) map[string]metricdata.Aggregation {
	t.Helper()
	var rm metricdata.ResourceMetrics
	if err := reader.Collect(context.Background(), &rm); err != nil {
		t.Fatal(err)
	}
	got := map[string]metricdata.Aggregation{}
	for _, sm := range rm.ScopeMetrics {
		for _, m := range sm.Metrics {
			got[m.Name] = m.Data
		}
	}
	return got
}

// point is a histogram data point reduced to what the tests assert on.
type point struct {
	Attrs map[string]string
	Count uint64
	Sum   float64
}

func points[N int64 | float64](t *testing.T, agg metricdata.Aggregation) []point {
	t.Helper()
	h, ok := agg.(metricdata.Histogram[N])
	if !ok {
		t.Fatalf("aggregation = %T, want histogram", agg)
	}
	var out []point
	for _, dp := range h.DataPoints {
		attrs := map[string]string{}
		for _, kv := range dp.Attributes.ToSlice() {
			attrs[string(kv.Key)] = kv.Value.Emit()
		}
		out = append(out, point{Attrs: attrs, Count: dp.Count, Sum: float64(dp.Sum)})
	}
	return out
}

func runModelWith(t *testing.T, resp *ai.ModelResponse, err error) {
	t.Helper()
	_, _ = tracing.RunInNewSpan(context.Background(),
		&tracing.SpanMetadata{Name: "googleai/gemini-flash-latest", Type: "action", Subtype: "model"},
		modelRequest(),
		func(ctx context.Context, _ *ai.ModelRequest) (*ai.ModelResponse, error) { return resp, err })
}

var baseMetricAttrs = map[string]string{
	genai.AttrOperationName: genai.OperationChat,
	genai.AttrRequestModel:  "gemini-flash-latest",
	genai.AttrProviderName:  genai.ProviderGCPGemini,
}

func withAttrs(extra ...attribute.KeyValue) map[string]string {
	m := map[string]string{}
	for k, v := range baseMetricAttrs {
		m[k] = v
	}
	for _, kv := range extra {
		m[string(kv.Key)] = kv.Value.Emit()
	}
	return m
}

func TestModelMetrics(t *testing.T) {
	reader := setupMetrics(t)
	runModelWith(t, modelResponse(), nil) // 3 input, 7 output tokens

	got := collect(t, reader)
	wantTokens := []point{
		{Attrs: withAttrs(attribute.String(genai.AttrTokenType, genai.TokenTypeInput)), Count: 1, Sum: 3},
		{Attrs: withAttrs(attribute.String(genai.AttrTokenType, genai.TokenTypeOutput)), Count: 1, Sum: 7},
	}
	if diff := cmp.Diff(wantTokens, points[int64](t, got[genai.MetricTokenUsage]), sortPoints); diff != "" {
		t.Errorf("token usage mismatch (-want +got):\n%s", diff)
	}
	dur := points[float64](t, got[genai.MetricOperationDuration])
	if len(dur) != 1 || dur[0].Count != 1 || !cmp.Equal(dur[0].Attrs, baseMetricAttrs) {
		t.Errorf("duration points = %+v, want one point with %v", dur, baseMetricAttrs)
	}
}

// TestModelMetricsError covers the failure path: the duration point carries
// error.type, and tokens from a partial response are still counted.
func TestModelMetricsError(t *testing.T) {
	reader := setupMetrics(t)
	partial := &ai.ModelResponse{
		Message: ai.NewModelTextMessage("not json"),
		Usage:   &ai.GenerationUsage{InputTokens: 2000, OutputTokens: 100},
	}
	runModelWith(t, partial, status.Errorf(status.ErrInvalidOutput, "invalid output"))
	runModelWith(t, nil, status.Errorf(status.ErrResourceExhausted, "quota"))

	got := collect(t, reader)
	wantTokens := []point{
		{Attrs: withAttrs(attribute.String(genai.AttrTokenType, genai.TokenTypeInput)), Count: 1, Sum: 2000},
		{Attrs: withAttrs(attribute.String(genai.AttrTokenType, genai.TokenTypeOutput)), Count: 1, Sum: 100},
	}
	if diff := cmp.Diff(wantTokens, points[int64](t, got[genai.MetricTokenUsage]), sortPoints); diff != "" {
		t.Errorf("token usage mismatch (-want +got):\n%s", diff)
	}

	var gotErrTypes []string
	for _, p := range points[float64](t, got[genai.MetricOperationDuration]) {
		gotErrTypes = append(gotErrTypes, p.Attrs[genai.AttrErrorType])
	}
	wantErrTypes := []string{string(status.Internal), string(status.ResourceExhausted)}
	if diff := cmp.Diff(wantErrTypes, gotErrTypes, cmpopts.SortSlices(func(a, b string) bool { return a < b })); diff != "" {
		t.Errorf("duration error.type mismatch (-want +got):\n%s", diff)
	}
}

func TestMetricsDisabled(t *testing.T) {
	setup(t, GenAiInstrumentationOptions{})
	reader := sdkmetric.NewManualReader()
	mp := sdkmetric.NewMeterProvider(sdkmetric.WithReader(reader))
	tracing.ConfigureInstrumentation(NewGenAiInstrumentation(GenAiInstrumentationOptions{Meter: mp.Meter("test"), DisableMetrics: true}))

	runModelWith(t, modelResponse(), nil)
	if got := collect(t, reader); len(got) != 0 {
		t.Errorf("metrics = %v, want none", got)
	}
}

var sortPoints = cmpopts.SortSlices(func(a, b point) bool {
	return a.Attrs[genai.AttrTokenType] < b.Attrs[genai.AttrTokenType]
})

// TestEventOnlyKeepsContentOffSpan verifies EVENT_ONLY does not attach content
// to span attributes (it emits a log event instead, which needs no SDK here).
func TestEventOnlyKeepsContentOffSpan(t *testing.T) {
	sr := setup(t, GenAiInstrumentationOptions{ContentCapturingMode: genai.EventOnly})
	runModel(t)
	a := attrMap(findSpan(t, sr, "chat gemini-flash-latest"))
	if _, ok := a[genai.AttrInputMessages]; ok {
		t.Error("EVENT_ONLY should not set input messages on the span")
	}
	if _, ok := a[genai.AttrOutputMessages]; ok {
		t.Error("EVENT_ONLY should not set output messages on the span")
	}
}
