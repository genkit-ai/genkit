// Copyright 2026 Google LLC
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
	"fmt"
	"maps"
	"slices"
	"sync"
	"sync/atomic"

	"github.com/firebase/genkit/go/internal/base"
	"github.com/firebase/genkit/go/internal/tracingbridge"
)

// This file holds the pluggable instrumentation abstraction: span creation is
// decoupled from collection/export. [RunInNewSpan] is the entry point and owns
// Genkit semantics (path, isRoot, state/output, debug logging) as well as the
// run itself; it asks each configured [Instrumentation] provider to start a
// span, runs the caller's function once, and ends every span with the result.
// Providers only encode.
//
// Extending the API: Instrumentation and Span are implemented outside this
// package, so methods are never added to them. New per-span data arrives as new
// accessors on SpanInfo and SpanResult (Genkit-owned structs), and new
// capabilities as optional interfaces a Span may implement, which the
// dispatcher detects with a type assertion. Both are additive and do not break
// existing providers.

// SpanInfo is the backend-independent view of a span the dispatcher hands to
// each [Instrumentation] provider. Everything it exposes is known when the span
// starts and is read-only: maps are returned as copies. The run's outcome
// arrives separately, as the [SpanResult] passed to [Span.End].
type SpanInfo struct {
	// labels are the raw TelemetryLabels, set directly as span attributes.
	// Often shared with other spans (the reflection server attaches one map
	// per request), so it is never handed out uncopied.
	labels map[string]string

	// metadata is the dispatcher-owned running metadata. In-package providers
	// (OTel, Direct) encode it via its attributes()/startAttributes() methods.
	// It is fully populated (state, output, error) by the time spans end.
	metadata *spanMetadata
}

// Labels returns a copy of the span's telemetry labels (set directly as span
// attributes, with no prefix), or nil if there are none.
func (i *SpanInfo) Labels() map[string]string {
	if i == nil {
		return nil
	}
	return maps.Clone(i.labels)
}

// The accessors tolerate a nil receiver and nil metadata, so a provider (or a
// test) holding a zero-value *SpanInfo does not panic.

// Name is the span name. For a model span it is the fully qualified model name
// (e.g. "googleai/gemini-flash-latest").
func (i *SpanInfo) Name() string {
	if i == nil || i.metadata == nil {
		return ""
	}
	return i.metadata.Name
}

// Type is the Genkit span type ("action", "flowStep", "util", ...).
func (i *SpanInfo) Type() string {
	if i == nil || i.metadata == nil {
		return ""
	}
	return i.metadata.Type
}

// Subtype is the finer categorization ("model", "tool", "flow", ...), or "".
func (i *SpanInfo) Subtype() string {
	if i == nil || i.metadata == nil {
		return ""
	}
	return i.metadata.Subtype
}

// Path is the type-annotated span path, e.g.
// "/{chatFlow,t:flow}/{googleai/gemini-flash-latest,t:action,s:model}".
func (i *SpanInfo) Path() string {
	if i == nil || i.metadata == nil {
		return ""
	}
	return i.metadata.Path
}

// IsRoot reports whether this span is the root of a Genkit trace.
func (i *SpanInfo) IsRoot() bool {
	if i == nil || i.metadata == nil {
		return false
	}
	return i.metadata.IsRoot
}

// Input is the raw input the action was invoked with.
func (i *SpanInfo) Input() any {
	if i == nil || i.metadata == nil {
		return nil
	}
	return i.metadata.Input
}

// Init is the initialization data the action was invoked with (recorded as
// genkit:init), or nil. Kept apart from Input so per-call input and session
// initialization stay distinguishable.
func (i *SpanInfo) Init() any {
	if i == nil || i.metadata == nil {
		return nil
	}
	return i.metadata.Init
}

// Metadata returns a copy of the span's custom metadata (recorded as
// genkit:metadata:<key> attributes), or nil if there is none. Metadata added
// mid-run arrives through [Span.SetMetadata] instead.
func (i *SpanInfo) Metadata() map[string]string {
	if i == nil || i.metadata == nil {
		return nil
	}
	return maps.Clone(i.metadata.Metadata)
}

// spanMeta returns the dispatcher metadata, or an empty one for a SpanInfo not
// built by the dispatcher (a zero value handed to a provider in a test).
func (i *SpanInfo) spanMeta() *spanMetadata {
	if i == nil || i.metadata == nil {
		return &spanMetadata{}
	}
	return i.metadata
}

// SpanResult is how a span's run ended. The dispatcher builds one per span and
// passes the same value to every provider's [Span.End]; it is read-only.
type SpanResult struct {
	output any
	err    error
}

// Output is the run's output. It can be non-nil alongside a non-nil Err: a
// failed run may still return a partial result (e.g. the generate loop returns
// the conversation it completed with its error).
func (r *SpanResult) Output() any {
	if r == nil {
		return nil
	}
	return r.output
}

// Err is the run's error, or nil on success. A panic in the run is reported
// here as an error (the panic itself keeps propagating after spans end).
func (r *SpanResult) Err() error {
	if r == nil {
		return nil
	}
	return r.err
}

// Span is the backend-independent handle a provider returns for the span it
// started. The dispatcher reads TraceInfo to resolve the span's ids, fans
// SetMetadata out for mid-run custom metadata, and calls End exactly once.
type Span interface {
	// TraceInfo returns the provider's ids, or empty strings if it tracks none.
	TraceInfo() TraceInfo
	// SetMetadata records mid-run custom metadata on the span. It may be
	// called from goroutines other than the one running the span.
	SetMetadata(md map[string]string)
	// End finalizes the span with the run's result. Called exactly once, on
	// the goroutine that started the span, in reverse start order across
	// providers; also called when the run panics.
	End(res *SpanResult)
}

// Instrumentation is a pluggable span-creation provider. For every Genkit span
// the dispatcher calls StartSpan on each configured provider in order, runs the
// span's function once under the returned context, then ends each Span. A
// provider cannot skip, repeat, or alter the run.
type Instrumentation interface {
	// StartSpan opens a backend span and returns the context children should
	// observe (e.g. carrying the backend's parent span) along with its handle.
	StartSpan(ctx context.Context, info *SpanInfo) (context.Context, Span)
}

// ---------------------------------------------------------------------------
// Provider registry.
// ---------------------------------------------------------------------------

var (
	// instrumentationMu serializes writers of the registry; readers on the
	// span hot path only load activeChain.
	instrumentationMu sync.Mutex
	// configured replaces the implicit OTel default when non-empty; set via
	// SetInstrumentation.
	configured []Instrumentation
	// defaultOTel is the implicit, back-compat default. Removed in the next
	// major to reach the shared "not instrumented by default" goal.
	defaultOTel Instrumentation = &OTelInstrumentation{}
	// direct feeds the Dev UI without OTel; set by setDevTelemetryServer.
	direct *DirectTelemetryInstrumentation
	// activeChain is the resolved chain every span dispatches through,
	// rebuilt by each registry write.
	activeChain atomic.Pointer[[]Instrumentation]
)

func init() {
	rebuildChainLocked()
	tracingbridge.SetDevTelemetryServer = setDevTelemetryServer
}

// SetInstrumentation sets the providers every Genkit span is reported to, in
// order, replacing the implicit default ([OTelInstrumentation] over the global
// OpenTelemetry TracerProvider). Calling it with no providers restores the
// default. Nil providers are ignored.
//
// To add a provider while keeping OpenTelemetry, list both:
//
//	tracing.SetInstrumentation(&tracing.OTelInstrumentation{}, myProvider)
//
// Configuration is process-wide, like otel.SetTracerProvider: Genkit actions
// can run without a Genkit instance and are instrumented too. In the dev
// environment the Developer UI's provider is always kept at the front of the
// chain, whatever is set here.
func SetInstrumentation(providers ...Instrumentation) {
	var ps []Instrumentation
	for _, p := range providers {
		if p != nil {
			ps = append(ps, p)
		}
	}
	instrumentationMu.Lock()
	defer instrumentationMu.Unlock()
	configured = ps
	rebuildChainLocked()
}

// setDevTelemetryServer points the Developer UI's Direct provider at url,
// installing it at the front of the chain. An empty url is ignored: genkit.Init
// passes GENKIT_TELEMETRY_SERVER through unconditionally, and an unset
// variable must not undo a URL the reflection handshake already supplied.
// Reached through [tracingbridge.SetDevTelemetryServer] by genkit.Init and the
// reflection server, so it is not public API.
//
// The instance is kept across calls: the reflection server calls this on every
// (re)connect, and replacing the instance would orphan children of spans still
// open at that moment (they track parentage through the instance's context
// key), splitting the trace. A repeated url is a no-op; a new one retargets
// the existing instance.
func setDevTelemetryServer(url string) {
	if url == "" {
		return
	}
	instrumentationMu.Lock()
	defer instrumentationMu.Unlock()
	if direct != nil {
		direct.retarget(url)
		return // chain unchanged
	}
	direct = newDirectTelemetryInstrumentation(url)
	rebuildChainLocked()
}

// resetInstrumentation restores the implicit default and removes the dev
// provider. For tests.
func resetInstrumentation() {
	instrumentationMu.Lock()
	defer instrumentationMu.Unlock()
	configured = nil
	direct = nil
	rebuildChainLocked()
}

// activeInstrumentations returns the chain for a new span: the Direct provider
// first when a dev telemetry server is configured (so its ids win), then the
// configured providers or the implicit OTel default. Lock-free: this runs for
// every span.
func activeInstrumentations() []Instrumentation {
	return *activeChain.Load()
}

// scopedKey holds the providers [WithInstrumentation] added to a context, in
// the order they were added.
var scopedKey = base.NewContextKey[[]Instrumentation]()

// WithInstrumentation returns ctx carrying providers in addition to any it
// already carries. Every span started under the returned context is reported
// to them, after the process-wide providers (see [SetInstrumentation]): they
// start after those and end before them, the most recently added first. A
// scoped provider cannot be removed, so an enclosing one sees every span
// started under it, nested runs and subagents included. Nil providers are
// ignored.
//
// [SetInstrumentation] configures telemetry for the process. WithInstrumentation
// is for code that reacts to what runs under one context, such as middleware
// that counts the model calls of its run, and adds its provider itself, so
// it needs no setup by the application.
func WithInstrumentation(ctx context.Context, providers ...Instrumentation) context.Context {
	prev := scopedKey.FromContext(ctx)
	next := slices.Clip(prev)
	for _, p := range providers {
		if p != nil {
			next = append(next, p)
		}
	}
	if len(next) == len(prev) {
		return ctx
	}
	return scopedKey.NewContext(ctx, next)
}

// spanChain returns the chain for a span started under ctx: the process-wide
// providers, then the providers ctx carries.
func spanChain(ctx context.Context) []Instrumentation {
	chain := activeInstrumentations()
	scoped := scopedKey.FromContext(ctx)
	if len(scoped) == 0 {
		return chain
	}
	return append(slices.Clip(chain), scoped...)
}

// rebuildChainLocked publishes the chain for the current registry state.
// Caller holds instrumentationMu (or is init).
func rebuildChainLocked() {
	var chain []Instrumentation
	if direct != nil {
		chain = append(chain, direct)
	}
	if len(configured) > 0 {
		chain = append(chain, configured...)
	} else {
		chain = append(chain, defaultOTel)
	}
	activeChain.Store(&chain)
}

// ---------------------------------------------------------------------------
// Dispatcher.
// ---------------------------------------------------------------------------

// startSpans starts a span on each provider in chain, threading the context
// through them in order, and appends each Span to *spans as it is started so a
// caller's deferred endSpans sees every span even if a later StartSpan panics.
// A provider returning a nil context or Span is tolerated.
func startSpans(ctx context.Context, chain []Instrumentation, info *SpanInfo, spans *[]Span) context.Context {
	for _, p := range chain {
		c, s := p.StartSpan(ctx, info)
		if c != nil {
			ctx = c
		}
		if s != nil {
			*spans = append(*spans, s)
		}
	}
	return ctx
}

// endSpans ends spans in reverse start order, the nesting order a backend
// expects.
func endSpans(spans []Span, res *SpanResult) {
	for i := len(spans) - 1; i >= 0; i-- {
		spans[i].End(res)
	}
}

// panicError describes a run that panicked, for SpanResult.Err.
func panicError(r any) error {
	if err, ok := r.(error); ok {
		return fmt.Errorf("panic: %w", err)
	}
	return fmt.Errorf("panic: %v", r)
}

// spanHandle is the part of a Span the running code reaches through the
// context (ids and mid-run metadata); ending stays with the dispatcher.
type spanHandle interface {
	TraceInfo() TraceInfo
	SetMetadata(md map[string]string)
}

// handleFor returns a single handle over spans: the span itself when there is
// one (the common case, no allocation), otherwise a composite.
func handleFor(spans []Span) spanHandle {
	if len(spans) == 1 {
		return spans[0]
	}
	return compositeSpan(spans)
}

// compositeSpan resolves TraceInfo as the first non-empty ids across the chain
// (Direct wins in dev) and fans SetMetadata out to every provider.
type compositeSpan []Span

func (cs compositeSpan) TraceInfo() TraceInfo {
	var out TraceInfo
	for _, s := range cs {
		ti := s.TraceInfo()
		if out.TraceID == "" && ti.TraceID != "" {
			out.TraceID = ti.TraceID
		}
		if out.SpanID == "" && ti.SpanID != "" {
			out.SpanID = ti.SpanID
		}
	}
	return out
}

func (cs compositeSpan) SetMetadata(md map[string]string) {
	for _, s := range cs {
		s.SetMetadata(md)
	}
}
