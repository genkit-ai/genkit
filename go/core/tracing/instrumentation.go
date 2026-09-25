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
	"maps"
	"sync"
	"sync/atomic"
)

// This file holds the pluggable instrumentation abstraction: span creation is
// decoupled from collection/export. [RunInNewSpan] is the entry point and owns
// Genkit semantics (path, isRoot, state/output, debug logging); it dispatches
// to a chain of Instrumentation providers that each only encode the span into
// their backend.

// SpanInfo is the backend-independent view of a span the dispatcher hands to
// each [Instrumentation] provider. The dispatcher owns Genkit semantics; a
// provider only encodes this into its backend, reading it through Labels and
// the accessor methods.
//
// Everything the accessors return is known when the span starts. The run's
// result is not on SpanInfo: a provider reads the output and error from what
// next returns, the only point they are known.
type SpanInfo struct {
	// Labels are the raw TelemetryLabels set directly as span attributes.
	Labels map[string]string

	// metadata is the dispatcher-owned running metadata. In-package providers
	// (OTel, Direct) encode it via its attributes()/startAttributes() methods.
	// It is fully populated (state, output, error) only after next returns.
	metadata *spanMetadata
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

// Metadata returns a copy of the span's custom metadata (recorded as
// genkit:metadata:<key> attributes), or nil if there is none. Metadata added
// mid-run arrives through [Span.SetMetadata] instead.
func (i *SpanInfo) Metadata() map[string]string {
	if i == nil || i.metadata == nil {
		return nil
	}
	return maps.Clone(i.metadata.Metadata)
}

// Span is the backend-independent handle a provider exposes for the span it
// opened. The dispatcher reads TraceInfo to build the composite ids and fans
// SetMetadata out for mid-run custom metadata.
type Span interface {
	// TraceInfo returns the provider's ids, or empty strings if it tracks none.
	TraceInfo() TraceInfo
	// SetMetadata records mid-run custom metadata on the span.
	SetMetadata(md map[string]string)
}

// NextFunc continues the instrumentation chain. A provider must call it exactly
// once, passing the context children should observe and the Span it opened, and
// return its result.
type NextFunc func(ctx context.Context, span Span) (any, error)

// Instrumentation is a pluggable span-creation provider: middleware over span
// creation. Implementations open a backend span, call next, and finalize from
// info once it returns.
type Instrumentation interface {
	RunInNewSpan(ctx context.Context, info *SpanInfo, next NextFunc) (any, error)
}

// ---------------------------------------------------------------------------
// Provider registry.
// ---------------------------------------------------------------------------

var (
	// instrumentationMu serializes writers of the registry; readers on the
	// span hot path only load activeChain.
	instrumentationMu sync.Mutex
	// configured replaces the implicit OTel default when set via
	// ConfigureInstrumentation.
	configured Instrumentation
	// defaultOTel is the implicit, back-compat default. Removed in the next
	// major to reach the shared "not instrumented by default" goal.
	defaultOTel Instrumentation = &OTelInstrumentation{}
	// direct feeds the Dev UI without OTel; set by EnableDevInstrumentation.
	direct Instrumentation
	// activeChain is the resolved chain every span dispatches through,
	// rebuilt by each registry write.
	activeChain atomic.Pointer[[]Instrumentation]
)

func init() {
	rebuildChainLocked()
}

// ConfigureInstrumentation replaces the implicit default
// ([OTelInstrumentation]) with i. DirectTelemetryInstrumentation is still
// prepended in dev (when a telemetry server is configured), so a typical dev
// chain becomes [Direct, i].
func ConfigureInstrumentation(i Instrumentation) {
	instrumentationMu.Lock()
	defer instrumentationMu.Unlock()
	configured = i
	rebuildChainLocked()
}

// EnableDevInstrumentation installs the Direct provider, feeding the Dev UI's
// telemetry server at url, at the front of the chain. An empty url is ignored;
// a later call updates the destination. In the dev environment genkit.Init
// calls it with GENKIT_TELEMETRY_SERVER, and the reflection server calls it
// with the URL the Genkit CLI supplies.
func EnableDevInstrumentation(url string) {
	if url == "" {
		return
	}
	instrumentationMu.Lock()
	defer instrumentationMu.Unlock()
	direct = newDirectTelemetryInstrumentation(url)
	rebuildChainLocked()
}

// ResetInstrumentation clears configured and dev instrumentation, restoring
// the implicit OTel default. Intended for tests that call
// [ConfigureInstrumentation] and want to undo it (typically via t.Cleanup).
func ResetInstrumentation() {
	instrumentationMu.Lock()
	defer instrumentationMu.Unlock()
	configured = nil
	direct = nil
	rebuildChainLocked()
}

// activeInstrumentations returns the chain for a new span: the Direct provider
// first when a dev telemetry server is configured (so its ids win), then the
// configured provider or the implicit OTel default. Lock-free: this runs for
// every span.
func activeInstrumentations() []Instrumentation {
	return *activeChain.Load()
}

// rebuildChainLocked publishes the chain for the current registry state.
// Caller holds instrumentationMu (or is init).
func rebuildChainLocked() {
	base := configured
	if base == nil {
		base = defaultOTel
	}
	chain := []Instrumentation{base}
	if direct != nil {
		chain = []Instrumentation{direct, base}
	}
	activeChain.Store(&chain)
}

// ---------------------------------------------------------------------------
// Dispatcher.
// ---------------------------------------------------------------------------

// dispatch nests each provider's RunInNewSpan (outermost first), collecting the
// Span each hands to next, and invokes runBody at the center with a composite
// span over them all.
func dispatch(ctx context.Context, chain []Instrumentation, info *SpanInfo, runBody NextFunc) (any, error) {
	spans := make([]Span, len(chain))
	var at func(i int) NextFunc
	at = func(i int) NextFunc {
		return func(ctx context.Context, span Span) (any, error) {
			spans[i] = span
			if i+1 < len(chain) {
				return chain[i+1].RunInNewSpan(ctx, info, at(i+1))
			}
			return runBody(ctx, compositeSpan(spans))
		}
	}
	return chain[0].RunInNewSpan(ctx, info, at(0))
}

// compositeSpan resolves TraceInfo as the first non-empty ids across the chain
// (Direct wins in dev) and fans SetMetadata out to every provider.
type compositeSpan []Span

func (cs compositeSpan) TraceInfo() TraceInfo {
	var out TraceInfo
	// A third-party provider may call next with a nil span, so guard each.
	for _, s := range cs {
		if s == nil {
			continue
		}
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
		if s != nil {
			s.SetMetadata(md)
		}
	}
}
