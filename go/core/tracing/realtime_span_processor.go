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
	"log/slog"
	"sync"

	sdktrace "go.opentelemetry.io/otel/sdk/trace"
)

// realtimeSpanProcessor is an OpenTelemetry SpanProcessor that writes each span
// to the telemetry server both when it starts and when it ends, so a trace
// shows up in the dev UI while it is still running.
//
// The default dev processor (a SimpleSpanProcessor, registered by
// [WriteTelemetryImmediate]) exports a span only when it ends. That is fine for
// a short flow, but an agent holds a single long-lived root span open for the
// whole bidirectional connection; the telemetry server only adds a trace to its
// listing once it receives that parentless span, so the trace would not appear
// in the "all traces" view until the connection closed. Exporting on start
// delivers the root span immediately and mirrors the JS RealtimeSpanProcessor,
// which the `genkit start` CLI enables via GENKIT_ENABLE_REALTIME_TELEMETRY.
type realtimeSpanProcessor struct {
	client TelemetryClient
	// mu guards closed and serializes it with the wg.Add in save, so a span
	// starting concurrently with Shutdown cannot add to the WaitGroup after
	// draining has begun. Without that ordering, a positive Add from a zero
	// counter could race wait's wg.Wait and either panic ("WaitGroup is reused
	// before previous Wait has returned") or let the save be silently dropped.
	mu     sync.Mutex
	closed bool
	// wg tracks in-flight start-of-span saves so ForceFlush/Shutdown can wait
	// for them; end-of-span saves are synchronous and need no tracking.
	wg sync.WaitGroup
}

func newRealtimeSpanProcessor(client TelemetryClient) *realtimeSpanProcessor {
	return &realtimeSpanProcessor{client: client}
}

// OnStart exports the span as soon as it begins. The span has no end time yet,
// so convertSpan records it as in progress. The network save runs on a
// goroutine: OnStart sits on the hot path of every action, flow, and model
// call, and a live preview is best-effort, so span creation must not block on
// telemetry I/O. The span is converted synchronously first (see save).
func (p *realtimeSpanProcessor) OnStart(parent context.Context, s sdktrace.ReadWriteSpan) {
	p.save(s, true)
}

// OnEnd exports the completed span synchronously, matching the durability of
// the immediate (SimpleSpanProcessor) path: once End returns, the final span
// has been handed to the telemetry server. A late end save also corrects any
// start save that the server merged out of order.
func (p *realtimeSpanProcessor) OnEnd(s sdktrace.ReadOnlySpan) {
	p.save(s, false)
}

// save converts s to trace data and sends it to the telemetry server, either
// asynchronously (on start) or synchronously (on end).
func (p *realtimeSpanProcessor) save(s sdktrace.ReadOnlySpan, async bool) {
	if p.client == nil {
		return
	}
	// Convert while the caller still owns the span: OnStart runs inline inside
	// Tracer().Start, before the span is shared with any other goroutine, and
	// OnEnd receives an immutable snapshot. Reading the live span here is
	// therefore race-free, and the resulting *Data is a plain value safe to hand
	// to a goroutine. A start export carries a single span, so convertTrace
	// never sees more than one parentless span.
	td, err := convertTrace([]sdktrace.ReadOnlySpan{s})
	if err != nil {
		slog.Debug("realtime telemetry: failed to convert span", "error", err)
		return
	}
	td.TraceID = s.SpanContext().TraceID().String()
	if !async {
		// Use a fresh context, not the action's: a span often ends because its
		// action context was canceled (the dev UI canceled the run, or the agent
		// connection closed), and the final trace write must still land.
		if err := p.client.Save(context.Background(), td); err != nil {
			slog.Debug("realtime telemetry: failed to save trace", "error", err)
		}
		return
	}
	// Track this save so ForceFlush/Shutdown can wait for it, unless Shutdown has
	// begun: adding to a WaitGroup that wait is already draining is racy, so once
	// closed we drop the save instead. The lock pairs the closed check with the
	// Add so it can't interleave with Shutdown setting closed.
	p.mu.Lock()
	if p.closed {
		p.mu.Unlock()
		return
	}
	p.wg.Add(1)
	p.mu.Unlock()
	go func() {
		defer p.wg.Done()
		if err := p.client.Save(context.Background(), td); err != nil {
			slog.Debug("realtime telemetry: failed to save trace", "error", err)
		}
	}()
}

// ForceFlush waits for in-flight start-of-span saves to complete, or until ctx
// is done. Unlike Shutdown it keeps accepting new saves; the dev SDK only calls
// it at flush/teardown points where no new spans are starting, so its wait does
// not race a fresh save.
func (p *realtimeSpanProcessor) ForceFlush(ctx context.Context) error {
	return p.wait(ctx)
}

// Shutdown stops accepting new start-of-span saves, then waits for in-flight
// ones to complete, or until ctx is done. Marking the processor closed before
// waiting is what makes the wait safe: no new save can Add to the WaitGroup once
// draining has begun.
func (p *realtimeSpanProcessor) Shutdown(ctx context.Context) error {
	p.mu.Lock()
	p.closed = true
	p.mu.Unlock()
	return p.wait(ctx)
}

func (p *realtimeSpanProcessor) wait(ctx context.Context) error {
	done := make(chan struct{})
	go func() {
		p.wg.Wait()
		close(done)
	}()
	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}
