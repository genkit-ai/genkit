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
	"errors"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	otrace "go.opentelemetry.io/otel/trace"
)

// startLogCollector runs a test telemetry server that accumulates OTLP log
// records POSTed to /api/otlp and returns the exporter wired to it.
func startLogCollector(t *testing.T) (*logExporter, func() []otlpLogRecord) {
	t.Helper()
	var mu sync.Mutex
	var records []otlpLogRecord
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/otlp" {
			t.Errorf("unexpected path %q", r.URL.Path)
		}
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Errorf("reading body: %v", err)
		}
		var payload otlpLogsPayload
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Errorf("unmarshaling payload: %v", err)
		}
		mu.Lock()
		for _, rl := range payload.ResourceLogs {
			for _, sl := range rl.ScopeLogs {
				if sl.Scope.Name != "genkit-go" {
					t.Errorf("scope name = %q, want genkit-go", sl.Scope.Name)
				}
				records = append(records, sl.LogRecords...)
			}
		}
		mu.Unlock()
		w.WriteHeader(http.StatusOK)
	}))
	t.Cleanup(srv.Close)

	e := &logExporter{queue: make(chan otlpLogRecord, logQueueSize)}
	e.client.Store(NewHTTPTelemetryClient(srv.URL))
	go e.run()

	return e, func() []otlpLogRecord {
		mu.Lock()
		defer mu.Unlock()
		return append([]otlpLogRecord{}, records...)
	}
}

// waitForRecords polls until the collector has at least n records.
func waitForRecords(t *testing.T, collect func() []otlpLogRecord, n int) []otlpLogRecord {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		if recs := collect(); len(recs) >= n {
			return recs
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %d log records, have %d", n, len(collect()))
	return nil
}

// namedString exercises the named-string-type rendering path.
type namedString string

func attrMap(rec otlpLogRecord) map[string]otlpValue {
	m := map[string]otlpValue{}
	for _, kv := range rec.Attributes {
		m[kv.Key] = kv.Value
	}
	return m
}

func TestLogExportHandler(t *testing.T) {
	e, collect := startLogCollector(t)
	l := slog.New(&logExportHandler{exporter: e})

	sc := otrace.NewSpanContext(otrace.SpanContextConfig{
		TraceID: otrace.TraceID{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
		SpanID:  otrace.SpanID{1, 2, 3, 4, 5, 6, 7, 8},
	})
	ctx := otrace.ContextWithSpanContext(context.Background(), sc)

	l.Log(ctx, slog.LevelInfo, "correlated message",
		"str", "value",
		"num", 42,
		"flag", true,
		"pi", 3.5,
		"took", 250*time.Millisecond,
		"cause", errors.New("boom"),
		"kind", namedString("stop"),
		slog.Group("req", "id", 7),
	)

	recs := waitForRecords(t, collect, 1)
	rec := recs[0]

	if got, want := rec.TraceID, sc.TraceID().String(); got != want {
		t.Errorf("traceId = %q, want %q", got, want)
	}
	if got, want := rec.SpanID, sc.SpanID().String(); got != want {
		t.Errorf("spanId = %q, want %q", got, want)
	}
	if rec.SeverityText != "INFO" || rec.SeverityNumber != 9 {
		t.Errorf("severity = %q/%d, want INFO/9", rec.SeverityText, rec.SeverityNumber)
	}
	if rec.Body.StringValue == nil || *rec.Body.StringValue != "correlated message" {
		t.Errorf("body = %+v, want correlated message", rec.Body)
	}
	if rec.TimeUnixNano == "" || rec.TimeUnixNano == "0" {
		t.Errorf("timeUnixNano = %q, want a real timestamp", rec.TimeUnixNano)
	}

	attrs := attrMap(rec)
	if v := attrs["str"]; v.StringValue == nil || *v.StringValue != "value" {
		t.Errorf(`attr str = %+v, want "value"`, v)
	}
	if v := attrs["num"]; v.IntValue == nil || *v.IntValue != 42 {
		t.Errorf("attr num = %+v, want 42", v)
	}
	if v := attrs["flag"]; v.BoolValue == nil || !*v.BoolValue {
		t.Errorf("attr flag = %+v, want true", v)
	}
	// The wire format has no float, duration, or error types; they render as strings.
	if v := attrs["pi"]; v.StringValue == nil || *v.StringValue != "3.5" {
		t.Errorf(`attr pi = %+v, want "3.5"`, v)
	}
	if v := attrs["took"]; v.StringValue == nil || *v.StringValue != "250ms" {
		t.Errorf(`attr took = %+v, want "250ms"`, v)
	}
	if v := attrs["cause"]; v.StringValue == nil || *v.StringValue != "boom" {
		t.Errorf(`attr cause = %+v, want "boom"`, v)
	}
	if v := attrs["kind"]; v.StringValue == nil || *v.StringValue != "stop" {
		t.Errorf(`attr kind = %+v, want "stop" (named string type unwrapped)`, v)
	}
	if v := attrs["req.id"]; v.IntValue == nil || *v.IntValue != 7 {
		t.Errorf("attr req.id = %+v, want 7 (group flattened)", v)
	}
}

func TestLogExportHandlerWithoutSpan(t *testing.T) {
	e, collect := startLogCollector(t)
	l := slog.New(&logExportHandler{exporter: e})

	l.Warn("unattached")

	rec := waitForRecords(t, collect, 1)[0]
	if rec.TraceID != "" || rec.SpanID != "" {
		t.Errorf("traceId/spanId = %q/%q, want empty for a record logged outside a span", rec.TraceID, rec.SpanID)
	}
	if rec.SeverityText != "WARN" || rec.SeverityNumber != 13 {
		t.Errorf("severity = %q/%d, want WARN/13", rec.SeverityText, rec.SeverityNumber)
	}
}

func TestLogExportHandlerBoundAttrsAndGroups(t *testing.T) {
	e, collect := startLogCollector(t)
	base := slog.New(&logExportHandler{exporter: e})
	l := base.With("bound", "b").WithGroup("g")

	l.Error("grouped", "inner", 1)

	rec := waitForRecords(t, collect, 1)[0]
	attrs := attrMap(rec)
	if v := attrs["bound"]; v.StringValue == nil || *v.StringValue != "b" {
		t.Errorf("attr bound = %+v, want b", v)
	}
	if v := attrs["g.inner"]; v.IntValue == nil || *v.IntValue != 1 {
		t.Errorf("attr g.inner = %+v, want 1 (group prefix applied)", v)
	}
	if rec.SeverityText != "ERROR" || rec.SeverityNumber != 17 {
		t.Errorf("severity = %q/%d, want ERROR/17", rec.SeverityText, rec.SeverityNumber)
	}
}

func TestLogExportHandlerDisabledWithoutClient(t *testing.T) {
	e := &logExporter{queue: make(chan otlpLogRecord, 4)}
	h := &logExportHandler{exporter: e}
	if h.Enabled(context.Background(), slog.LevelError) {
		t.Error("handler reports enabled with no telemetry client configured")
	}
}

func TestLogExportDisabledByEnv(t *testing.T) {
	// The opt-out is read when export is enabled, not at package init, so a
	// value set with t.Setenv (or os.Setenv before genkit.Init) takes effect.
	cases := []struct {
		value string
		want  bool
	}{
		{"", false},
		{"true", false},
		{"1", false},
		{"false", true},
		{"0", true},
		{"FALSE", true},
		{"off", false}, // not a strconv.ParseBool value; export stays enabled
	}
	for _, c := range cases {
		t.Run("value="+c.value, func(t *testing.T) {
			t.Setenv("GENKIT_OTEL_ENABLE_LOGS", c.value)
			if got := logExportDisabled(); got != c.want {
				t.Errorf("logExportDisabled() with %q = %v, want %v", c.value, got, c.want)
			}
		})
	}
}

func TestLogExportOverflowDoesNotDeadlock(t *testing.T) {
	// A full queue with no worker draining it: enqueue must drop, and the
	// drop warning must not travel back through the export handler. Before
	// diag existed, slog.Warn inside warnDropped.Do re-entered enqueue on
	// the same goroutine and recursed into the sync.Once, deadlocking.
	e := &logExporter{queue: make(chan otlpLogRecord, 1)}
	e.client.Store(NewHTTPTelemetryClient("http://127.0.0.1:1"))
	l := slog.New(&logExportHandler{exporter: e})
	// The re-entry only happens when the warning's logger routes back into
	// the export handler, i.e. when it is part of the process default.
	prev := slog.Default()
	t.Cleanup(func() { slog.SetDefault(prev) })
	slog.SetDefault(l)

	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := range 3 {
			l.Info("overflow", "i", i)
		}
	}()
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("logging deadlocked on queue overflow")
	}
	if got := e.dropped.Load(); got != 2 {
		t.Errorf("dropped = %d, want 2", got)
	}
}

func TestLogExportBatching(t *testing.T) {
	e, collect := startLogCollector(t)
	l := slog.New(&logExportHandler{exporter: e})

	const n = 10
	for i := range n {
		l.Info("batch item", "i", i)
	}

	recs := waitForRecords(t, collect, n)
	if len(recs) != n {
		t.Errorf("got %d records, want %d", len(recs), n)
	}
}
