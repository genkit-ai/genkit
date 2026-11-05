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

package logger

import (
	"context"
	"log/slog"
	"slices"
	"strings"
	"testing"

	otrace "go.opentelemetry.io/otel/trace"
)

// recordHandler collects records for assertions.
type recordHandler struct {
	level   slog.Level
	records *[]slog.Record
	attrs   []slog.Attr
}

func newRecordHandler(level slog.Level) *recordHandler {
	return &recordHandler{level: level, records: &[]slog.Record{}}
}

func (h *recordHandler) Enabled(ctx context.Context, l slog.Level) bool { return l >= h.level }

func (h *recordHandler) Handle(ctx context.Context, r slog.Record) error {
	nr := r.Clone()
	nr.AddAttrs(h.attrs...)
	*h.records = append(*h.records, nr)
	return nil
}

func (h *recordHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	return &recordHandler{level: h.level, records: h.records, attrs: append(slices.Clip(h.attrs), attrs...)}
}

func (h *recordHandler) WithGroup(name string) slog.Handler { return h }

func (h *recordHandler) messages() []string {
	var msgs []string
	for _, r := range *h.records {
		msgs = append(msgs, r.Message)
	}
	return msgs
}

// resetGlobalState restores the package and slog globals a test mutates.
func resetGlobalState(t *testing.T) {
	t.Helper()
	prev := slog.Default()
	t.Cleanup(func() {
		mu.Lock()
		sinks = nil
		console = nil
		level.Set(slog.LevelInfo)
		mu.Unlock()
		slog.SetDefault(prev)
	})
}

func TestWithContext(t *testing.T) {
	resetGlobalState(t)

	stored := newRecordHandler(slog.LevelDebug)
	ctx := WithContext(context.Background(), slog.New(stored))
	def := newRecordHandler(slog.LevelDebug)
	slog.SetDefault(slog.New(def))

	FromContext(ctx).Info("stored")
	FromContext(context.Background()).Info("default")

	if got := stored.messages(); !slices.Equal(got, []string{"stored"}) {
		t.Errorf("stored logger messages = %v, want [stored]", got)
	}
	if got := def.messages(); !slices.Equal(got, []string{"default"}) {
		t.Errorf("default logger messages = %v, want [default]", got)
	}
}

// spanHandler records the span context carried by each record's context.
type spanHandler struct {
	spans *[]otrace.SpanContext
}

func (h *spanHandler) Enabled(context.Context, slog.Level) bool { return true }

func (h *spanHandler) Handle(ctx context.Context, _ slog.Record) error {
	*h.spans = append(*h.spans, otrace.SpanContextFromContext(ctx))
	return nil
}

func (h *spanHandler) WithAttrs([]slog.Attr) slog.Handler { return h }
func (h *spanHandler) WithGroup(string) slog.Handler      { return h }

func spanContext(id byte) otrace.SpanContext {
	return otrace.NewSpanContext(otrace.SpanContextConfig{
		TraceID: otrace.TraceID{id},
		SpanID:  otrace.SpanID{id},
	})
}

func TestFromContextBindsSpan(t *testing.T) {
	resetGlobalState(t)

	var spans []otrace.SpanContext
	slog.SetDefault(slog.New(&spanHandler{spans: &spans}))

	scA, scB := spanContext(1), spanContext(2)
	ctxA := otrace.ContextWithSpanContext(context.Background(), scA)
	ctxB := otrace.ContextWithSpanContext(context.Background(), scB)

	l := FromContext(ctxA)
	l.Info("context-free")                          // bound context supplies span A
	l.InfoContext(ctxB, "explicit span")            // call-site span B wins
	l.InfoContext(context.Background(), "spanless") // bound span A fills in

	// Re-fetching a stored bound logger under a new context rebinds it to
	// that context rather than nesting a stale binding.
	FromContext(WithContext(ctxB, l)).Info("rebound")

	want := []otrace.SpanContext{scA, scB, scA, scB}
	if len(spans) != len(want) {
		t.Fatalf("got %d records, want %d", len(spans), len(want))
	}
	for i := range want {
		if !spans[i].Equal(want[i]) {
			t.Errorf("record %d span = %v, want %v", i, spans[i], want[i])
		}
	}
}

func TestFromContextKeepsBoundAttrs(t *testing.T) {
	resetGlobalState(t)

	h := newRecordHandler(slog.LevelDebug)
	slog.SetDefault(slog.New(h))

	FromContext(context.Background()).With("requestId", "r1").Info("m")

	if len(*h.records) != 1 {
		t.Fatalf("got %d records, want 1", len(*h.records))
	}
	found := false
	(*h.records)[0].Attrs(func(a slog.Attr) bool {
		if a.Key == "requestId" {
			found = true
		}
		return true
	})
	if !found {
		t.Error("record lost the attribute bound with With")
	}
}

func TestPackageLevelHelpers(t *testing.T) {
	resetGlobalState(t)

	h := newRecordHandler(slog.LevelDebug)
	ctx := WithContext(context.Background(), slog.New(h))

	Debug(ctx, "debug msg", "k", 1)
	Info(ctx, "info msg")
	Warn(ctx, "warn msg")
	Error(ctx, "error msg")

	want := []string{"debug msg", "info msg", "warn msg", "error msg"}
	if got := h.messages(); !slices.Equal(got, want) {
		t.Errorf("messages = %v, want %v", got, want)
	}
	wantLevels := []slog.Level{slog.LevelDebug, slog.LevelInfo, slog.LevelWarn, slog.LevelError}
	for i, r := range *h.records {
		if r.Level != wantLevels[i] {
			t.Errorf("record %d level = %v, want %v", i, r.Level, wantLevels[i])
		}
	}
}

func TestHelpersUseDefaultWithoutContextLogger(t *testing.T) {
	resetGlobalState(t)

	h := newRecordHandler(slog.LevelDebug)
	slog.SetDefault(slog.New(h))

	Info(context.Background(), "through default")

	if got := h.messages(); !slices.Equal(got, []string{"through default"}) {
		t.Errorf("messages = %v, want [through default]", got)
	}
}

func TestAddHandlerTees(t *testing.T) {
	resetGlobalState(t)

	console := newRecordHandler(slog.LevelInfo)
	slog.SetDefault(slog.New(console))
	sink := newRecordHandler(slog.LevelDebug)
	AddHandler(sink)

	slog.Info("visible to both")
	slog.Debug("sink only")

	if got := console.messages(); !slices.Equal(got, []string{"visible to both"}) {
		t.Errorf("console messages = %v, want [visible to both]", got)
	}
	if got := sink.messages(); !slices.Equal(got, []string{"visible to both", "sink only"}) {
		t.Errorf("sink messages = %v, want both records", got)
	}
}

func TestSetLevelPreservesSinks(t *testing.T) {
	resetGlobalState(t)

	sink := newRecordHandler(slog.LevelDebug)
	AddHandler(sink)
	SetLevel(slog.LevelWarn)

	if got := GetLevel(); got != slog.LevelWarn {
		t.Errorf("GetLevel = %v, want %v", got, slog.LevelWarn)
	}

	slog.Debug("after set level")

	if got := sink.messages(); !slices.Equal(got, []string{"after set level"}) {
		t.Errorf("sink messages = %v, want [after set level]", got)
	}
}

func TestAddHandlerOverStdlibDefault(t *testing.T) {
	resetGlobalState(t)

	// Canary for the type-name detection: the process default here is the
	// stdlib handler (resetGlobalState restored it). If this fails on a Go
	// upgrade, the stdlib renamed its default handler type and
	// isStdlibDefaultHandler must learn the new name.
	if !isStdlibDefaultHandler(slog.Default().Handler()) {
		t.Fatalf("isStdlibDefaultHandler does not recognize the process default handler (%T)", slog.Default().Handler())
	}

	// Teeing the stdlib handler back in would re-enter the tee through the
	// log package on every record and deadlock, so AddHandler must
	// substitute the managed console handler.
	sink := newRecordHandler(slog.LevelDebug)
	AddHandler(sink)

	// Completing at all is the regression check: with the stdlib handler as
	// a tee member this call would deadlock on the log package's mutex.
	slog.Info("no deadlock")

	if got := sink.messages(); !slices.Equal(got, []string{"no deadlock"}) {
		t.Errorf("sink messages = %v, want [no deadlock]", got)
	}
	tee, ok := slog.Default().Handler().(*teeHandler)
	if !ok {
		t.Fatalf("default handler is %T, want *teeHandler", slog.Default().Handler())
	}
	if isStdlibDefaultHandler(tee.handlers[0]) {
		t.Error("tee kept the stdlib default handler as its console member")
	}
}

func TestCustomDefaultInstalledBeforeAddHandlerIsKept(t *testing.T) {
	resetGlobalState(t)

	// A custom default handler must never be mistaken for the stdlib one and
	// substituted away, no matter when it was installed.
	custom := newRecordHandler(slog.LevelInfo)
	slog.SetDefault(slog.New(custom))
	AddHandler(newRecordHandler(slog.LevelDebug))

	slog.Info("kept")

	if got := custom.messages(); !slices.Equal(got, []string{"kept"}) {
		t.Errorf("custom handler messages = %v, want [kept]", got)
	}
}

func TestAddHandlerKeepsCustomDefaultAfterSetLevel(t *testing.T) {
	resetGlobalState(t)

	// SetLevel creates the managed console; a custom default installed
	// afterwards must still be the base AddHandler tees onto, not the stale
	// managed console.
	SetLevel(slog.LevelInfo)
	custom := newRecordHandler(slog.LevelInfo)
	slog.SetDefault(slog.New(custom))
	AddHandler(newRecordHandler(slog.LevelDebug))

	slog.Info("kept")

	if got := custom.messages(); !slices.Equal(got, []string{"kept"}) {
		t.Errorf("custom handler messages = %v, want [kept]", got)
	}
}

func TestSetLevelKeepsCustomDefault(t *testing.T) {
	resetGlobalState(t)

	custom := newRecordHandler(slog.LevelInfo)
	SetDefaultHandler(custom)
	SetLevel(slog.LevelDebug)

	slog.Info("still through custom")

	// The application's handler stays the default; SetLevel warns through it
	// instead of replacing it with the managed console.
	msgs := custom.messages()
	if len(msgs) != 2 || !strings.Contains(msgs[0], "SetLevel") || msgs[1] != "still through custom" {
		t.Errorf("custom handler messages = %v, want the SetLevel warning followed by [still through custom]", msgs)
	}
	if got := GetLevel(); got != slog.LevelDebug {
		t.Errorf("GetLevel = %v, want %v", got, slog.LevelDebug)
	}
}

func TestSetDefaultHandlerPreservesSinks(t *testing.T) {
	resetGlobalState(t)

	sink := newRecordHandler(slog.LevelDebug)
	AddHandler(sink)
	replacement := newRecordHandler(slog.LevelInfo)
	SetDefaultHandler(replacement)

	slog.Info("to both")

	if got := replacement.messages(); !slices.Equal(got, []string{"to both"}) {
		t.Errorf("replacement messages = %v, want [to both]", got)
	}
	if got := sink.messages(); !slices.Equal(got, []string{"to both"}) {
		t.Errorf("sink messages = %v, want [to both]", got)
	}
}

func TestHasCustomDefault(t *testing.T) {
	resetGlobalState(t)

	if HasCustomDefault() {
		t.Error("HasCustomDefault = true with the stdlib default handler")
	}
	SetLevel(slog.LevelInfo)
	if HasCustomDefault() {
		t.Error("HasCustomDefault = true with the managed console installed")
	}
	AddHandler(newRecordHandler(slog.LevelDebug))
	if HasCustomDefault() {
		t.Error("HasCustomDefault = true with the managed console teed with a sink")
	}
	slog.SetDefault(slog.New(newRecordHandler(slog.LevelInfo)))
	if !HasCustomDefault() {
		t.Error("HasCustomDefault = false after the application installed its own handler")
	}
	// A component-supplied base handler also counts as custom: console
	// configuration such as GENKIT_LOG_LEVEL must not clobber it.
	SetDefaultHandler(newRecordHandler(slog.LevelInfo))
	if !HasCustomDefault() {
		t.Error("HasCustomDefault = false after SetDefaultHandler installed a component handler")
	}
}

func TestStdlibLogLoggerLevelCarriesOver(t *testing.T) {
	resetGlobalState(t)

	// Verbosity raised on the stdlib default handler via SetLogLoggerLevel
	// must survive its substitution with the managed console.
	prev := slog.SetLogLoggerLevel(slog.LevelDebug)
	t.Cleanup(func() { slog.SetLogLoggerLevel(prev) })

	AddHandler(newRecordHandler(slog.LevelDebug))

	if got := GetLevel(); got != slog.LevelDebug {
		t.Errorf("GetLevel = %v, want %v (seeded from SetLogLoggerLevel)", got, slog.LevelDebug)
	}
}

func TestAddHandlerTwiceDoesNotNest(t *testing.T) {
	resetGlobalState(t)

	console := newRecordHandler(slog.LevelInfo)
	slog.SetDefault(slog.New(console))
	AddHandler(newRecordHandler(slog.LevelDebug))
	AddHandler(newRecordHandler(slog.LevelDebug))

	slog.Info("once")

	if got := console.messages(); !slices.Equal(got, []string{"once"}) {
		t.Errorf("console messages = %v, want exactly one record", got)
	}
}

func TestTeeWithAttrsAndGroups(t *testing.T) {
	resetGlobalState(t)

	sink := newRecordHandler(slog.LevelDebug)
	tee := &teeHandler{handlers: []slog.Handler{sink}}
	l := slog.New(tee).With("bound", "yes")

	l.Info("with attrs", "k", "v")

	if len(*sink.records) != 1 {
		t.Fatalf("got %d records, want 1", len(*sink.records))
	}
	var keys []string
	(*sink.records)[0].Attrs(func(a slog.Attr) bool {
		keys = append(keys, a.Key)
		return true
	})
	joined := strings.Join(keys, ",")
	if !strings.Contains(joined, "bound") || !strings.Contains(joined, "k") {
		t.Errorf("record attrs = %v, want bound and k", keys)
	}
}
