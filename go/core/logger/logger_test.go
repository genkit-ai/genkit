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

	h := newRecordHandler(slog.LevelDebug)
	l := slog.New(h)
	ctx := WithContext(context.Background(), l)

	if got := FromContext(ctx); got != l {
		t.Errorf("FromContext returned %v, want the logger from WithContext", got)
	}
	if got := FromContext(context.Background()); got != slog.Default() {
		t.Errorf("FromContext without a logger returned %v, want slog.Default()", got)
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

	// The process default is the stdlib handler here (resetGlobalState
	// restored it). Teeing that handler back in would re-enter the tee
	// through the log package on every record and deadlock, so AddHandler
	// must substitute the managed console handler.
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
	if tee.handlers[0] == initialDefaultHandler {
		t.Error("tee kept the stdlib default handler as its console member")
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
