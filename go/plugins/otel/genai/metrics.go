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

package genai

import (
	"context"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
)

// tokenBuckets are the explicit token-count buckets recommended by the spec for
// the token-usage histogram; the duration histogram uses default seconds buckets.
var tokenBuckets = []float64{
	1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304,
	16777216, 67108864,
}

// Metrics holds the two spec-defined GenAI client metrics, recorded per model
// operation.
//
// See the spec:
// https://github.com/open-telemetry/semantic-conventions-genai
//
// Instruments are created up front from the meter. When no MeterProvider is
// configured, go.opentelemetry.io/otel returns no-op instruments, so this stays
// a no-op until the app wires up metrics collection.
type Metrics struct {
	tokenUsage        metric.Int64Histogram
	operationDuration metric.Float64Histogram
}

// NewMetrics builds the GenAI client metric instruments from meter.
func NewMetrics(meter metric.Meter) (*Metrics, error) {
	tokenUsage, err := meter.Int64Histogram(
		MetricTokenUsage,
		metric.WithUnit("{token}"),
		metric.WithDescription("Number of input and output tokens used by the model."),
		metric.WithExplicitBucketBoundaries(tokenBuckets...),
	)
	if err != nil {
		return nil, err
	}
	operationDuration, err := meter.Float64Histogram(
		MetricOperationDuration,
		metric.WithUnit("s"),
		metric.WithDescription("Duration of a GenAI model operation."),
	)
	if err != nil {
		return nil, err
	}
	return &Metrics{tokenUsage: tokenUsage, operationDuration: operationDuration}, nil
}

// RecordTokenUsage records input/output token counts, one point per provided
// count, tagged with gen_ai.token.type.
func (m *Metrics) RecordTokenUsage(ctx context.Context, base []attribute.KeyValue, inputTokens, outputTokens *int) {
	if inputTokens != nil {
		attrs := append(append([]attribute.KeyValue{}, base...), attribute.String(AttrTokenType, "input"))
		m.tokenUsage.Record(ctx, int64(*inputTokens), metric.WithAttributes(attrs...))
	}
	if outputTokens != nil {
		attrs := append(append([]attribute.KeyValue{}, base...), attribute.String(AttrTokenType, "output"))
		m.tokenUsage.Record(ctx, int64(*outputTokens), metric.WithAttributes(attrs...))
	}
}

// RecordDuration records the operation duration in seconds. Recorded for both
// successful and failed operations (failures carry error.type in attrs).
func (m *Metrics) RecordDuration(ctx context.Context, seconds float64, attrs []attribute.KeyValue) {
	m.operationDuration.Record(ctx, seconds, metric.WithAttributes(attrs...))
}
