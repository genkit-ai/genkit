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
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"reflect"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/firebase/genkit/go/internal"
	otrace "go.opentelemetry.io/otel/trace"
)

// The dev log exporter forwards slog records to the telemetry server used by
// the Genkit Dev UI, correlated with the active trace span so each record
// shows up in the UI on the span that emitted it. It is the log-side
// counterpart of the trace exporters in this package and mirrors the JS
// runtime's LogServerExporter, which POSTs the same OTLP-JSON payload to the
// same /api/otlp endpoint.

const (
	// logQueueSize bounds the export queue. Logging never blocks: when the
	// queue is full, records are dropped (and counted) rather than stalling
	// the caller.
	logQueueSize = 1024
	// logBatchSize and logBatchDelay shape the batches: a batch is sent when
	// it reaches logBatchSize records or when logBatchDelay has passed since
	// its first record, whichever comes first.
	logBatchSize  = 64
	logBatchDelay = 100 * time.Millisecond
	// logExportTimeout bounds a single POST to the telemetry server.
	logExportTimeout = 10 * time.Second
)

// logExportDisabled reports whether the user explicitly opted out of log
// export via GENKIT_OTEL_ENABLE_LOGS. The same variable gates the JS
// runtime's log export, but with an opt-in default; Go exports whenever a
// telemetry server is configured, which only the dev tooling does, so a false
// value is the only meaningful setting. It is read when export is enabled,
// not at package initialization, so a value set programmatically before
// genkit.Init (os.Setenv in main, t.Setenv in tests) is honored.
func logExportDisabled() bool {
	v, err := strconv.ParseBool(os.Getenv("GENKIT_OTEL_ENABLE_LOGS"))
	return err == nil && !v
}

// logExporter converts slog records to OTLP-JSON log records and ships them
// to the telemetry server from a single background worker.
type logExporter struct {
	client  atomic.Pointer[httpTelemetryClient]
	queue   chan otlpLogRecord
	start   sync.Once
	dropped atomic.Int64

	// warnDropped fires once so a full queue is reported without flooding the
	// console. An unreachable server is reported through the package-wide
	// [warnTelemetryUnreachable], shared with the trace export path.
	warnDropped sync.Once
}

// exporter is the process-wide dev log exporter. Its handler is installed by
// genkit.Init in dev mode; it stays inert until [EnableLogExport] gives it a
// telemetry server to talk to.
var exporter = &logExporter{queue: make(chan otlpLogRecord, logQueueSize)}

// diag reports the exporter's own problems. It writes straight to stderr
// rather than through the default logger: the default handler includes the
// export handler itself, so an exporter warning routed through it would
// re-enter the exporter. On the overflow path that re-entry is a deadlock
// (enqueue's sync.Once would be entered recursively on the same goroutine).
var diag = slog.New(slog.NewTextHandler(os.Stderr, nil))

// EnableLogExport starts forwarding log records to the telemetry server at
// url, correlating each record with the active span at the time of logging.
// It is called during dev-mode initialization, either with the value of
// GENKIT_TELEMETRY_SERVER or with the URL the Genkit CLI supplies when it
// notifies the reflection server. Calling it again replaces the destination;
// an empty url is ignored. Setting GENKIT_OTEL_ENABLE_LOGS=false disables
// export entirely.
//
// Records reach the telemetry server only if a handler created by
// [LogExportHandler] is registered, which genkit.Init does in dev mode.
func EnableLogExport(url string) {
	if url == "" || logExportDisabled() {
		return
	}
	exporter.client.Store(NewHTTPTelemetryClient(url))
	exporter.start.Do(func() { go exporter.run() })
}

// LogExportHandler returns a slog.Handler that forwards every record at
// slog.LevelDebug or above to the telemetry server configured with
// [EnableLogExport]. Until then the handler reports itself disabled and adds
// no overhead. The handler never blocks: records beyond the export queue's
// capacity are dropped.
func LogExportHandler() slog.Handler {
	return &logExportHandler{exporter: exporter}
}

// run is the export worker: it drains the queue into batches and POSTs them.
func (e *logExporter) run() {
	for first := range e.queue {
		batch := []otlpLogRecord{first}
		timer := time.NewTimer(logBatchDelay)
	collect:
		for len(batch) < logBatchSize {
			select {
			case rec := <-e.queue:
				batch = append(batch, rec)
			case <-timer.C:
				break collect
			}
		}
		timer.Stop()
		e.send(batch)
	}
}

// send delivers one batch to the telemetry server.
func (e *logExporter) send(batch []otlpLogRecord) {
	client := e.client.Load()
	if client == nil {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), logExportTimeout)
	defer cancel()
	if err := client.SaveLogs(ctx, &otlpLogsPayload{
		ResourceLogs: []otlpResourceLogs{{
			Resource: otlpResource{Attributes: []otlpKeyValue{}},
			ScopeLogs: []otlpScopeLogs{{
				Scope:      otlpScope{Name: "genkit-go", Version: internal.Version},
				LogRecords: batch,
			}},
		}},
	}); err != nil {
		warnTelemetryUnreachable(err)
	}
}

// enqueue adds rec to the export queue without blocking, dropping it if the
// queue is full.
func (e *logExporter) enqueue(rec otlpLogRecord) {
	select {
	case e.queue <- rec:
	default:
		e.dropped.Add(1)
		e.warnDropped.Do(func() {
			diag.Warn("log export queue is full; dropping records from the Dev UI log view")
		})
	}
}

// logExportHandler is the slog.Handler side of the exporter. WithAttrs and
// WithGroup accumulate into pre-converted attributes and a dotted key prefix,
// since the telemetry server stores attributes as a flat list.
type logExportHandler struct {
	exporter *logExporter
	attrs    []otlpKeyValue
	prefix   string
}

func (h *logExportHandler) Enabled(ctx context.Context, level slog.Level) bool {
	return level >= slog.LevelDebug && h.exporter.client.Load() != nil
}

func (h *logExportHandler) Handle(ctx context.Context, r slog.Record) error {
	rec := otlpLogRecord{
		TimeUnixNano:   strconv.FormatInt(r.Time.UnixNano(), 10),
		SeverityNumber: severityNumber(r.Level),
		SeverityText:   severityText(r.Level),
		Body:           otlpStringValue(r.Message),
		Attributes:     append(make([]otlpKeyValue, 0, len(h.attrs)+r.NumAttrs()), h.attrs...),
	}
	r.Attrs(func(a slog.Attr) bool {
		rec.Attributes = appendOtlpAttr(rec.Attributes, h.prefix, a)
		return true
	})
	// The span active when the log statement ran determines where the record
	// appears in the Dev UI, which looks logs up by exact trace and span ID.
	// A record logged outside any span is still stored, just not shown on a
	// span.
	if sc := otrace.SpanContextFromContext(ctx); sc.IsValid() {
		rec.TraceID = sc.TraceID().String()
		rec.SpanID = sc.SpanID().String()
	}
	h.exporter.enqueue(rec)
	return nil
}

func (h *logExportHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	if len(attrs) == 0 {
		return h
	}
	nh := &logExportHandler{exporter: h.exporter, prefix: h.prefix}
	nh.attrs = append(append([]otlpKeyValue{}, h.attrs...), convertOtlpAttrs(h.prefix, attrs)...)
	return nh
}

func (h *logExportHandler) WithGroup(name string) slog.Handler {
	if name == "" {
		return h
	}
	return &logExportHandler{exporter: h.exporter, attrs: h.attrs, prefix: h.prefix + name + "."}
}

// severityNumber maps a slog level to an OpenTelemetry severity number.
// The ranges line up exactly: slog Debug (-4) through Error (+8) map to OTel
// DEBUG (5) through ERROR (17) with a constant offset of 9.
func severityNumber(level slog.Level) int {
	return min(max(int(level)+9, 1), 24)
}

// severityText maps a slog level to the OTel severity text the Dev UI colors
// by. In-between levels take the text of the level they are at or above.
func severityText(level slog.Level) string {
	switch {
	case level >= slog.LevelError:
		return "ERROR"
	case level >= slog.LevelWarn:
		return "WARN"
	case level >= slog.LevelInfo:
		return "INFO"
	default:
		return "DEBUG"
	}
}

// OTLP-JSON wire types, matching the subset the telemetry server parses. The
// server understands only string, int, and bool attribute values, so every
// other kind is rendered to a string.

type otlpLogsPayload struct {
	ResourceLogs []otlpResourceLogs `json:"resourceLogs"`
}

type otlpResourceLogs struct {
	Resource  otlpResource    `json:"resource"`
	ScopeLogs []otlpScopeLogs `json:"scopeLogs"`
}

type otlpResource struct {
	Attributes             []otlpKeyValue `json:"attributes"`
	DroppedAttributesCount int            `json:"droppedAttributesCount"`
}

type otlpScopeLogs struct {
	Scope      otlpScope       `json:"scope"`
	LogRecords []otlpLogRecord `json:"logRecords"`
}

type otlpScope struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

type otlpLogRecord struct {
	TimeUnixNano   string         `json:"timeUnixNano"`
	SeverityNumber int            `json:"severityNumber"`
	SeverityText   string         `json:"severityText"`
	Body           otlpValue      `json:"body"`
	Attributes     []otlpKeyValue `json:"attributes"`
	TraceID        string         `json:"traceId,omitempty"`
	SpanID         string         `json:"spanId,omitempty"`
}

type otlpKeyValue struct {
	Key   string    `json:"key"`
	Value otlpValue `json:"value"`
}

type otlpValue struct {
	StringValue *string `json:"stringValue,omitempty"`
	IntValue    *int64  `json:"intValue,omitempty"`
	BoolValue   *bool   `json:"boolValue,omitempty"`
}

func otlpStringValue(s string) otlpValue { return otlpValue{StringValue: &s} }

func convertOtlpAttrs(prefix string, attrs []slog.Attr) []otlpKeyValue {
	var kvs []otlpKeyValue
	for _, a := range attrs {
		kvs = appendOtlpAttr(kvs, prefix, a)
	}
	return kvs
}

// appendOtlpAttr converts one slog attribute (flattening groups into dotted
// keys) and appends the result to kvs.
func appendOtlpAttr(kvs []otlpKeyValue, prefix string, a slog.Attr) []otlpKeyValue {
	v := a.Value.Resolve()
	if a.Key == "" && v.Kind() != slog.KindGroup {
		return kvs
	}
	key := prefix + a.Key
	switch v.Kind() {
	case slog.KindGroup:
		groupPrefix := prefix
		if a.Key != "" {
			groupPrefix += a.Key + "."
		}
		for _, ga := range v.Group() {
			kvs = appendOtlpAttr(kvs, groupPrefix, ga)
		}
		return kvs
	case slog.KindString:
		return append(kvs, otlpKeyValue{Key: key, Value: otlpStringValue(v.String())})
	case slog.KindInt64:
		n := v.Int64()
		return append(kvs, otlpKeyValue{Key: key, Value: otlpValue{IntValue: &n}})
	case slog.KindUint64:
		if u := v.Uint64(); u <= uint64(1<<63-1) {
			n := int64(u)
			return append(kvs, otlpKeyValue{Key: key, Value: otlpValue{IntValue: &n}})
		}
		return append(kvs, otlpKeyValue{Key: key, Value: otlpStringValue(strconv.FormatUint(v.Uint64(), 10))})
	case slog.KindBool:
		b := v.Bool()
		return append(kvs, otlpKeyValue{Key: key, Value: otlpValue{BoolValue: &b}})
	default:
		return append(kvs, otlpKeyValue{Key: key, Value: otlpStringValue(otlpRenderValue(v))})
	}
}

// otlpRenderValue renders a value the wire format has no native type for.
// Floats, durations, and times use their standard string forms; errors their
// message; named string types their string value; everything else its JSON
// form when it has one.
func otlpRenderValue(v slog.Value) string {
	switch v.Kind() {
	case slog.KindFloat64:
		return strconv.FormatFloat(v.Float64(), 'g', -1, 64)
	case slog.KindDuration:
		return v.Duration().String()
	case slog.KindTime:
		return v.Time().Format(time.RFC3339Nano)
	default:
		av := v.Any()
		if err, ok := av.(error); ok {
			return err.Error()
		}
		// A named string type (e.g. ai.FinishReason) has slog kind Any; JSON
		// would wrap it in an extra pair of quotes.
		if rv := reflect.ValueOf(av); rv.IsValid() && rv.Kind() == reflect.String {
			return rv.String()
		}
		if b, err := json.Marshal(av); err == nil {
			return string(b)
		}
		return fmt.Sprintf("%v", av)
	}
}
