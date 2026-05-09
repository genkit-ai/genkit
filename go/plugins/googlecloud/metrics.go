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

package googlecloud

import (
	"context"
	"fmt"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/metric"
)

// MetricCounterOptions holds configuration for a counter metric
type MetricCounterOptions struct {
	Description string
	Unit        string
}

// MetricHistogramOptions holds configuration for a histogram metric
type MetricHistogramOptions struct {
	Description string
	Unit        string
}

// MetricCounter wraps OpenTelemetry counter with Genkit conventions
type MetricCounter struct {
	counter metric.Int64Counter
}

// NewMetricCounter creates a new counter metric with the given name and options
func NewMetricCounter(name string, opts MetricCounterOptions) *MetricCounter {
	meter := otel.Meter("genkit")
	counter, err := meter.Int64Counter(name,
		metric.WithDescription(opts.Description),
		metric.WithUnit(opts.Unit))
	if err != nil {
		panic(fmt.Sprintf("failed to create counter %s: %v", name, err))
	}
	return &MetricCounter{counter: counter}
}

// Add records a value to the counter with the given attributes
func (m *MetricCounter) Add(value int64, attributes map[string]interface{}) {
	if m.counter == nil {
		return
	}
	attrs := convertToOTelAttributes(attributes)
	m.counter.Add(context.Background(), value, metric.WithAttributes(attrs...))
}

// MetricHistogram wraps OpenTelemetry histogram with Genkit conventions
type MetricHistogram struct {
	histogram metric.Float64Histogram
}

// NewMetricHistogram creates a new histogram metric with the given name and options
func NewMetricHistogram(name string, opts MetricHistogramOptions) *MetricHistogram {
	meter := otel.Meter("genkit")
	histogram, err := meter.Float64Histogram(name,
		metric.WithDescription(opts.Description),
		metric.WithUnit(opts.Unit))
	if err != nil {
		panic(fmt.Sprintf("failed to create histogram %s: %v", name, err))
	}
	return &MetricHistogram{histogram: histogram}
}

// Record records a value to the histogram with the given attributes
func (m *MetricHistogram) Record(value float64, attributes map[string]interface{}) {
	if m.histogram == nil {
		return
	}
	attrs := convertToOTelAttributes(attributes)
	m.histogram.Record(context.Background(), value, metric.WithAttributes(attrs...))
}

// convertToOTelAttributes converts a map of string interfaces to OpenTelemetry attributes
func convertToOTelAttributes(attrs map[string]interface{}) []attribute.KeyValue {
	if attrs == nil {
		return nil
	}

	result := make([]attribute.KeyValue, 0, len(attrs))
	for key, value := range attrs {
		if value == nil {
			continue
		}

		switch v := value.(type) {
		case string:
			result = append(result, attribute.String(key, v))
		case int:
			result = append(result, attribute.Int(key, v))
		case int64:
			result = append(result, attribute.Int64(key, v))
		case float64:
			result = append(result, attribute.Float64(key, v))
		case bool:
			result = append(result, attribute.Bool(key, v))
		default:
			result = append(result, attribute.String(key, fmt.Sprintf("%v", v)))
		}
	}
	return result
}

// internalMetricNamespaceWrap prefixes metric names with genkit namespace
func internalMetricNamespaceWrap(namespace, metricName string) string {
	return fmt.Sprintf("genkit/%s/%s", namespace, metricName)
}
