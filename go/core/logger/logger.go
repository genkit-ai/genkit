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

// Package logger provides a context-scoped slog.Logger.
package logger

import (
	"context"
	"log/slog"
	"os"
	"sync"

	"github.com/firebase/genkit/go/internal/base"
)

var (
	mu sync.Mutex
	// level is the minimum level for the console handler that SetLevel manages.
	// The zero value is slog.LevelInfo, which is also the effective minimum of
	// the stdlib default handler that is in place before SetLevel is called.
	level slog.LevelVar
	// console is the handler SetLevel installs, created on first use and wired
	// to level so later SetLevel calls only adjust the level.
	console slog.Handler
	// sinks are additional handlers installed by AddHandler. Records logged
	// through the default logger are delivered to every sink whose Enabled
	// accepts them, alongside the console handler.
	sinks     []slog.Handler
	loggerKey = base.NewContextKey[*slog.Logger]()
	// initialDefaultHandler is the stdlib default handler that is in place
	// before anyone calls slog.SetDefault. It is not a real sink: it writes
	// through the log package, whose output slog.SetDefault redirects back to
	// the current default slog handler. Making it a member of an installed
	// tee would therefore re-enter the tee on every record and deadlock on
	// the log package's mutex, so it is detected and substituted instead.
	initialDefaultHandler = slog.Default().Handler()
)

// SetLevel sets the minimum level of Genkit's console log handler and installs
// it as the process-wide default logger, replacing the current one. Handlers
// previously installed with [AddHandler] are preserved.
//
// Applications that manage their own [slog] handler should set their handler's
// level directly instead of calling SetLevel.
func SetLevel(l slog.Level) {
	mu.Lock()
	defer mu.Unlock()
	level.Set(l)
	ensureConsole()
	install(console)
}

// ensureConsole creates the managed console handler if it does not exist yet.
// Callers must hold mu.
func ensureConsole() {
	if console == nil {
		console = slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: &level})
	}
}

// GetLevel returns the level most recently passed to [SetLevel], or
// slog.LevelInfo if SetLevel has not been called.
func GetLevel() slog.Level {
	mu.Lock()
	defer mu.Unlock()
	return level.Level()
}

// AddHandler registers h as an additional destination for records logged
// through the default logger: every record whose level passes h's Enabled
// method is handed to h in addition to the current default handler. Genkit
// uses this in dev mode to stream logs to the Dev UI; applications can use it
// to mirror logs to a file or a test recorder.
//
// The registration survives [SetLevel], but a later call to [slog.SetDefault]
// replaces the composed handler entirely.
func AddHandler(h slog.Handler) {
	mu.Lock()
	defer mu.Unlock()
	sinks = append(sinks, h)
	base := console
	if base == nil {
		base = currentBase()
	}
	install(base)
}

// currentBase returns the handler that console output should continue to flow
// through: the current default handler, unwrapped if it is a tee this package
// installed earlier (so re-installation never nests tees). The stdlib default
// handler is substituted with the managed console handler, since teeing it
// would deadlock (see initialDefaultHandler). Callers must hold mu.
func currentBase() slog.Handler {
	h := slog.Default().Handler()
	if t, ok := h.(*teeHandler); ok {
		h = t.handlers[0]
	}
	if h == initialDefaultHandler {
		ensureConsole()
		return console
	}
	return h
}

// install makes base, teed with any registered sinks, the default slog
// handler. Callers must hold mu.
func install(base slog.Handler) {
	if len(sinks) == 0 {
		slog.SetDefault(slog.New(base))
		return
	}
	handlers := make([]slog.Handler, 0, len(sinks)+1)
	handlers = append(handlers, base)
	handlers = append(handlers, sinks...)
	slog.SetDefault(slog.New(&teeHandler{handlers: handlers}))
}

// FromContext returns the Logger in ctx, or the default Logger
// if there is none.
func FromContext(ctx context.Context) *slog.Logger {
	if l := loggerKey.FromContext(ctx); l != nil {
		return l
	}
	return slog.Default()
}

// WithContext returns a copy of ctx carrying l. [FromContext] and the
// package-level logging functions use l for anything logged under the
// returned context, so attributes bound with l.With flow to every log
// statement downstream:
//
//	ctx = logger.WithContext(ctx, logger.FromContext(ctx).With("requestId", id))
func WithContext(ctx context.Context, l *slog.Logger) context.Context {
	return loggerKey.NewContext(ctx, l)
}

// Debug logs at slog.LevelDebug using the logger in ctx.
//
// The package-level logging functions are the preferred way to log within
// Genkit: they use the context's logger (with any attributes bound via
// [WithContext]) and pass ctx through to the handler, which is what lets a
// context-aware handler correlate the record with the active trace span.
func Debug(ctx context.Context, msg string, args ...any) {
	FromContext(ctx).Log(ctx, slog.LevelDebug, msg, args...)
}

// Info logs at slog.LevelInfo using the logger in ctx. See [Debug].
func Info(ctx context.Context, msg string, args ...any) {
	FromContext(ctx).Log(ctx, slog.LevelInfo, msg, args...)
}

// Warn logs at slog.LevelWarn using the logger in ctx. See [Debug].
func Warn(ctx context.Context, msg string, args ...any) {
	FromContext(ctx).Log(ctx, slog.LevelWarn, msg, args...)
}

// Error logs at slog.LevelError using the logger in ctx. See [Debug].
func Error(ctx context.Context, msg string, args ...any) {
	FromContext(ctx).Log(ctx, slog.LevelError, msg, args...)
}
