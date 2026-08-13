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
	"reflect"
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
)

// isStdlibDefaultHandler reports whether h is the standard library's default
// slog handler, detected by its (unexported) type. That handler is not a real
// sink: it writes through the log package, whose output slog.SetDefault
// redirects back to the current default slog handler, so making it a member
// of an installed tee would re-enter the tee on every record and deadlock on
// the log package's mutex. It is detected by type rather than by capturing
// slog.Default() at package initialization, since another package's init can
// legitimately install a custom default before this package initializes, and
// that handler must not be mistaken for the stdlib one. The type is matched
// by package path and name, which are exact where Type.String's short-name
// form could collide with a third-party package also called slog. A test
// pins the identification against stdlib renames.
func isStdlibDefaultHandler(h slog.Handler) bool {
	t := reflect.TypeOf(h)
	if t == nil || t.Kind() != reflect.Pointer {
		return false
	}
	return t.Elem().PkgPath() == "log/slog" && t.Elem().Name() == "defaultHandler"
}

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
// slog.LevelInfo if SetLevel has not been called. slog.LevelVar is safe for
// concurrent use, so no lock is needed.
func GetLevel() slog.Level {
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
	install(currentBase())
}

// SetDefaultHandler installs h as the base handler of the process-wide
// default logger, replacing the current base while keeping every sink
// registered with [AddHandler]. Components that bring their own destination
// handler (for example a plugin that ships logs to a cloud service) should
// use this instead of [slog.SetDefault], which would silently disconnect the
// registered sinks, including dev-mode streaming to the Dev UI.
func SetDefaultHandler(h slog.Handler) {
	mu.Lock()
	defer mu.Unlock()
	install(h)
}

// HasCustomDefault reports whether the process-wide default logger is built
// on a handler the application installed itself, rather than the stdlib
// default handler or the managed console handler this package installs.
// Genkit uses it to decide whether console logging configuration (such as the
// GENKIT_LOG_LEVEL environment variable) is Genkit's to apply.
func HasCustomDefault() bool {
	mu.Lock()
	defer mu.Unlock()
	h := slog.Default().Handler()
	if t, ok := h.(*teeHandler); ok {
		h = t.handlers[0]
	}
	return !isStdlibDefaultHandler(h) && h != console
}

// currentBase returns the handler that console output should continue to flow
// through: the current default handler, unwrapped if it is a tee this package
// installed earlier (so re-installation never nests tees). The stdlib default
// handler is substituted with the managed console handler, since teeing it
// would deadlock (see [isStdlibDefaultHandler]); its effective level is
// carried over so the substitution does not change verbosity. Callers must
// hold mu.
func currentBase() slog.Handler {
	h := slog.Default().Handler()
	if t, ok := h.(*teeHandler); ok {
		h = t.handlers[0]
	}
	if isStdlibDefaultHandler(h) {
		if console == nil {
			seedLevelFromStdlib(h)
		}
		ensureConsole()
		return console
	}
	return h
}

// seedLevelFromStdlib copies the stdlib default handler's effective minimum
// level into the managed level, so substituting the managed console for that
// handler preserves verbosity raised with [slog.SetLogLoggerLevel] (which has
// no getter; the handler is probed through Enabled instead). It runs only
// before the console handler exists, so an explicit [SetLevel] always wins.
// Callers must hold mu.
func seedLevelFromStdlib(h slog.Handler) {
	ctx := context.Background()
	for _, l := range []slog.Level{slog.LevelDebug, slog.LevelInfo, slog.LevelWarn, slog.LevelError} {
		if h.Enabled(ctx, l) {
			level.Set(l)
			return
		}
	}
	level.Set(slog.LevelError + 1)
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
