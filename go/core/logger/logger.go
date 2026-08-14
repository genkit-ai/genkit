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
	otrace "go.opentelemetry.io/otel/trace"
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
// it as the process-wide default logger. Handlers previously installed with
// [AddHandler] are preserved.
//
// A default handler the application installed itself (via [slog.SetDefault]
// or [SetDefaultHandler]) is not Genkit's to manage: SetLevel records the
// level for the managed console handler but leaves the application's handler
// in place and warns, exactly as GENKIT_LOG_LEVEL does, since that handler's
// own level is what governs output. Set the application handler's level
// directly instead.
func SetLevel(l slog.Level) {
	mu.Lock()
	level.Set(l)
	custom := hasCustomDefaultLocked()
	if !custom {
		ensureConsole()
		install(console)
	}
	mu.Unlock()
	if custom {
		// Logged after unlocking: the warning flows through the application's
		// handler, which must be free to call back into this package.
		slog.Warn("logger.SetLevel: the application installed its own default log handler; set its level directly", "level", l)
	}
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
	return hasCustomDefaultLocked()
}

// hasCustomDefaultLocked is [HasCustomDefault] for callers already holding mu.
func hasCustomDefaultLocked() bool {
	h := baseHandler(slog.Default().Handler())
	return !isStdlibDefaultHandler(h) && h != console
}

// baseHandler strips the composition layers this package wraps around a base
// handler: the tee that install builds, and the context binding [FromContext]
// applies (in case an application handed a FromContext logger to
// slog.SetDefault). What remains is the handler whose identity decides
// whether the base is the stdlib default, the managed console, or
// application-owned.
func baseHandler(h slog.Handler) slog.Handler {
	for {
		switch v := h.(type) {
		case *teeHandler:
			h = v.handlers[0]
		case *ctxHandler:
			h = v.inner
		default:
			return h
		}
	}
}

// currentBase returns the handler that console output should continue to flow
// through: the current default handler, unwrapped if it is a tee this package
// installed earlier (so re-installation never nests tees). The stdlib default
// handler is substituted with the managed console handler, since teeing it
// would deadlock (see [isStdlibDefaultHandler]); its effective level is
// carried over so the substitution does not change verbosity. Callers must
// hold mu.
func currentBase() slog.Handler {
	h := baseHandler(slog.Default().Handler())
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

// FromContext returns the logger carried by ctx, or the process default
// logger if there is none, bound to ctx: records logged even through the
// returned logger's context-free methods (Info, Error, ...) reach the handler
// carrying ctx, where those methods would otherwise hand the handler a
// background context. The binding is what lets a context-aware handler, such
// as the one that streams logs to the Dev UI, correlate such records with the
// span that was active when the logger was obtained. A context passed
// explicitly at the call site (InfoContext, Log) wins whenever it carries a
// span of its own.
func FromContext(ctx context.Context) *slog.Logger {
	h := fromContext(ctx).Handler()
	if c, ok := h.(*ctxHandler); ok {
		h = c.inner // rebind to ctx rather than nest bindings
	}
	return slog.New(&ctxHandler{ctx: ctx, inner: h})
}

// fromContext returns the logger carried by ctx, or the process default,
// without the context binding [FromContext] adds. The package-level logging
// functions use it because they pass the call-site context through
// explicitly, which makes the binding redundant work.
func fromContext(ctx context.Context) *slog.Logger {
	if l := loggerKey.FromContext(ctx); l != nil {
		return l
	}
	return slog.Default()
}

// ctxHandler binds a context to a handler. slog's context-free logging
// methods hand handlers context.Background(), stripping the trace span that
// context-aware handlers correlate records by; the bound context fills that
// gap.
type ctxHandler struct {
	ctx   context.Context
	inner slog.Handler
}

// resolve picks the context the inner handler sees: the call-site context
// when it carries a span, since a log statement under a deeper span must
// attribute to that span, and the bound context otherwise.
func (h *ctxHandler) resolve(ctx context.Context) context.Context {
	if h.ctx == nil || otrace.SpanContextFromContext(ctx).IsValid() {
		return ctx
	}
	return h.ctx
}

func (h *ctxHandler) Enabled(ctx context.Context, l slog.Level) bool {
	return h.inner.Enabled(h.resolve(ctx), l)
}

func (h *ctxHandler) Handle(ctx context.Context, r slog.Record) error {
	return h.inner.Handle(h.resolve(ctx), r)
}

func (h *ctxHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	if len(attrs) == 0 {
		return h
	}
	return &ctxHandler{ctx: h.ctx, inner: h.inner.WithAttrs(attrs)}
}

func (h *ctxHandler) WithGroup(name string) slog.Handler {
	if name == "" {
		return h
	}
	return &ctxHandler{ctx: h.ctx, inner: h.inner.WithGroup(name)}
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
	fromContext(ctx).Log(ctx, slog.LevelDebug, msg, args...)
}

// Info logs at slog.LevelInfo using the logger in ctx. See [Debug].
func Info(ctx context.Context, msg string, args ...any) {
	fromContext(ctx).Log(ctx, slog.LevelInfo, msg, args...)
}

// Warn logs at slog.LevelWarn using the logger in ctx. See [Debug].
func Warn(ctx context.Context, msg string, args ...any) {
	fromContext(ctx).Log(ctx, slog.LevelWarn, msg, args...)
}

// Error logs at slog.LevelError using the logger in ctx. See [Debug].
func Error(ctx context.Context, msg string, args ...any) {
	fromContext(ctx).Log(ctx, slog.LevelError, msg, args...)
}
