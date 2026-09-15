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
	"os"
	"sync"
)

// This file holds the pluggable instrumentation abstraction: span creation is
// decoupled from collection/export. [RunInNewSpan] stays the entry point and
// owns Genkit semantics (path, isRoot, state/output, debug logging); it
// dispatches to a chain of Instrumentation providers that each only encode the
// span into their backend. Collection stays with the explicit WriteTelemetry*
// helpers and the GCP / Firebase plugins.

// SpanInfo is the backend-independent view of a span the dispatcher hands to
// each [Instrumentation] provider. The dispatcher owns Genkit semantics; a
// provider only encodes this into its backend.
//
// The running metadata (name, path, type, input/output/state, ...) is kept in
// the unexported spanMetadata. Built-in providers reach it directly; an
// out-of-package provider reads it through the exported Labels field and the
// Name/Type/Subtype/Input accessors below.
type SpanInfo struct {
	// Labels are the raw TelemetryLabels set directly as span attributes.
	Labels map[string]string

	// metadata is the dispatcher-owned running metadata. In-package providers
	// (OTel, Direct) encode it via its attributes()/startAttributes() methods.
	// It is fully populated (state, output, error) only after next returns.
	metadata *spanMetadata
}

// The methods below are the read-only view an out-of-package Instrumentation
// (e.g. the OTel GenAI plugin) needs to encode a span into its backend. Output
// is deliberately absent: a provider reads it from the value next returns,
// which is the only point it is known.

// The accessors guard a nil receiver and nil metadata so an out-of-package
// provider (or a test) that holds a zero-value *SpanInfo does not panic.

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

// Input is the raw Genkit input the action was invoked with, before next runs.
func (i *SpanInfo) Input() any {
	if i == nil || i.metadata == nil {
		return nil
	}
	return i.metadata.Input
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
	instrumentationMu sync.Mutex
	// configured replaces the implicit OTel default when set via
	// ConfigureInstrumentation.
	configured Instrumentation
	// defaultOTel is the implicit, back-compat default. Removed in the next
	// major to reach the shared "not instrumented by default" goal.
	defaultOTel Instrumentation = &OTelInstrumentation{}
	// direct feeds the Dev UI without OTel; prepended when a dev telemetry
	// server is configured (env var or the reflection handshake).
	direct Instrumentation
	// devServerURL is the dev telemetry server the Direct provider talks to.
	devServerURL string
)

// ConfigureInstrumentation replaces the implicit default
// ([OTelInstrumentation]) with i. DirectTelemetryInstrumentation is still
// prepended in dev (when a telemetry server is configured), so a typical dev
// chain becomes [Direct, i].
func ConfigureInstrumentation(i Instrumentation) {
	instrumentationMu.Lock()
	defer instrumentationMu.Unlock()
	configured = i
}

// EnableDevInstrumentation installs the Direct provider (feeding the Dev UI at
// url) as the primary dev instrumentation. It replaces the old telemetry-server
// span-processor auto-registration. An empty url is ignored; a later call
// updates the destination. Called during dev-mode init with GENKIT_TELEMETRY_SERVER
// or the URL the Genkit CLI supplies over the reflection API.
func EnableDevInstrumentation(url string) {
	if url == "" {
		return
	}
	instrumentationMu.Lock()
	defer instrumentationMu.Unlock()
	devServerURL = url
	direct = newDirectTelemetryInstrumentation(url)
}

// ResetInstrumentation clears configured and auto-injected instrumentation,
// restoring the implicit OTel default. Intended for tests that call
// [ConfigureInstrumentation] and want to undo it (typically via t.Cleanup).
func ResetInstrumentation() {
	instrumentationMu.Lock()
	defer instrumentationMu.Unlock()
	configured = nil
	direct = nil
	devServerURL = ""
}

// activeInstrumentations resolves the chain for this call: base is the
// configured provider or the implicit OTel default; the Direct provider is
// prepended when a dev telemetry server is configured, so its ids win and the
// Dev UI is fed without OTel.
func activeInstrumentations() []Instrumentation {
	instrumentationMu.Lock()
	defer instrumentationMu.Unlock()
	base := configured
	if base == nil {
		base = defaultOTel
	}
	if d := resolveDirectLocked(); d != nil {
		return []Instrumentation{d, base}
	}
	return []Instrumentation{base}
}

// resolveDirectLocked lazily builds the Direct provider from the dev server URL
// (explicitly set, or read once from GENKIT_TELEMETRY_SERVER). The env fallback
// mirrors the pre-refactor TracerProvider bootstrap. Caller holds
// instrumentationMu.
func resolveDirectLocked() Instrumentation {
	if direct != nil {
		return direct
	}
	url := devServerURL
	if url == "" {
		url = os.Getenv("GENKIT_TELEMETRY_SERVER")
	}
	if url == "" {
		return nil
	}
	direct = newDirectTelemetryInstrumentation(url)
	return direct
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
