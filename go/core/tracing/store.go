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

// Data is information about a trace.
type Data struct {
	TraceID     string               `json:"traceId"`
	DisplayName string               `json:"displayName"`
	StartTime   Milliseconds         `json:"startTime"`
	EndTime     Milliseconds         `json:"endTime"`
	Spans       map[string]*SpanData `json:"spans"`
}

// SpanData is information about a trace span.
// Most of this information comes from OpenTelemetry.
// See https://pkg.go.dev/go.opentelemetry.io/otel/sdk/trace#ReadOnlySpan.
// SpanData can be passed to json.Marshal, whereas most of the OpenTelemetry
// types make no claims about JSON serializability.
type SpanData struct {
	SpanID               string               `json:"spanId"`
	TraceID              string               `json:"traceId,omitempty"`
	ParentSpanID         string               `json:"parentSpanId,omitempty"`
	StartTime            Milliseconds         `json:"startTime"`
	EndTime              Milliseconds         `json:"endTime"`
	Attributes           map[string]any       `json:"attributes"`
	DisplayName          string               `json:"displayName"`
	Links                []*Link              `json:"links,omitempty"`
	InstrumentationScope InstrumentationScope `json:"instrumentationLibrary"` // TODO: update json tag when JS runtime gets updated
	SpanKind             string               `json:"spanKind"`               // trace.SpanKind as a string
	// This bool is in a separate struct, to match the js (and presumably the OTel) formats.
	SameProcessAsParentSpan BoolValue  `json:"sameProcessAsParentSpan"`
	Status                  Status     `json:"status"`
	TimeEvents              TimeEvents `json:"timeEvents"`
}

type TimeEvents struct {
	TimeEvent []TimeEvent `json:"timeEvent,omitempty"`
}

type BoolValue struct {
	Value bool `json:"value"`
}

type TimeEvent struct {
	Time       Milliseconds `json:"time,omitempty"`
	Annotation Annotation   `json:"annotation"`
}

type Annotation struct {
	Attributes  map[string]any `json:"attributes,omitempty"`
	Description string         `json:"description,omitempty"`
}

// A SpanContext contains identifying trace information about a Span.
type SpanContext struct {
	TraceID    string `json:"traceId,omitempty"`
	SpanID     string `json:"spanId"`
	IsRemote   bool   `json:"isRemote"`
	TraceFlags int    `json:"traceFlags"`
}

// A Link describes the relationship between two Spans.
type Link struct {
	SpanContext            SpanContext    `json:"context"`
	Attributes             map[string]any `json:"attributes,omitempty"`
	DroppedAttributesCount int            `json:"droppedAttributesCount"`
}

// InstrumentationScope is a copy of [go.opentelemetry.io/otel/sdk/instrumentation.Library],
// with added struct tags to match the javascript JSON field names.
type InstrumentationScope struct {
	Name      string `json:"name"`
	Version   string `json:"version"`
	SchemaURL string `json:"schemaUrl,omitempty"`
}

// Status is a copy of [go.opentelemetry.io/otel/sdk/trace.Status],
// with added struct tags to match the javascript JSON field names.
type Status struct {
	Code        uint32 `json:"code"` // avoid the MarshalJSON method on codes.Code
	Description string `json:"message,omitempty"`
}
