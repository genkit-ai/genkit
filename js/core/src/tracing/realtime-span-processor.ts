/**
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { Context } from '@opentelemetry/api';
import {
  ReadableSpan,
  Span,
  SpanExporter,
  SpanProcessor,
} from '@opentelemetry/sdk-trace-base';

/**
 * RealtimeSpanProcessor exports spans both when they start and when they end.
 * This enables real-time trace visualization in development.
 */
export class RealtimeSpanProcessor implements SpanProcessor {
  constructor(private readonly exporter: SpanExporter) {}

  /**
   * Called when a span is started. Exports immediately for real-time updates.
   */
  onStart(span: Span, _parentContext: Context): void {
    // Export the span immediately (it won't have endTime yet)
    this.exporter.export([span], () => {
      // Ignore result - we don't want to block span creation
    });
  }

  /**
   * Called when a span ends. Exports again with complete data.
   */
  onEnd(span: ReadableSpan): void {
    // Export the completed span
    this.exporter.export([span], () => {
      // Ignore result
    });
  }

  /**
   * Forces the exporter to flush any buffered spans.
   */
  async forceFlush(): Promise<void> {
    if (this.exporter.forceFlush) {
      return this.exporter.forceFlush();
    }
  }

  /**
   * Shuts down the processor and exporter.
   */
  async shutdown(): Promise<void> {
    return this.exporter.shutdown();
  }
}
