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

	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/metric/metricdata"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/tracing"
	"github.com/firebase/genkit/go/plugins/otel/genai"
)

// TestModelMetrics drives a model span with an injected meter backed by a manual
// reader and asserts both spec metrics are recorded.
func TestModelMetrics(t *testing.T) {
	sr := setup(t, GenAiInstrumentationOptions{})

	reader := sdkmetric.NewManualReader()
	mp := sdkmetric.NewMeterProvider(sdkmetric.WithReader(reader))
	inst := NewGenAiInstrumentation(GenAiInstrumentationOptions{Meter: mp.Meter("test")})
	tracing.ConfigureInstrumentation(inst)

	_, err := tracing.RunInNewSpan(context.Background(),
		&tracing.SpanMetadata{Name: "googleai/gemini-flash-latest", Type: "action", Subtype: "model"},
		modelRequest(),
		func(ctx context.Context, _ *ai.ModelRequest) (*ai.ModelResponse, error) {
			return modelResponse(), nil
		})
	if err != nil {
		t.Fatal(err)
	}
	_ = sr // span recorder cleanup is handled by setup

	var rm metricdata.ResourceMetrics
	if err := reader.Collect(context.Background(), &rm); err != nil {
		t.Fatal(err)
	}
	got := map[string]bool{}
	for _, sm := range rm.ScopeMetrics {
		for _, m := range sm.Metrics {
			got[m.Name] = true
		}
	}
	if !got[genai.MetricTokenUsage] {
		t.Errorf("missing %s metric; got %v", genai.MetricTokenUsage, got)
	}
	if !got[genai.MetricOperationDuration] {
		t.Errorf("missing %s metric; got %v", genai.MetricOperationDuration, got)
	}
}

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
