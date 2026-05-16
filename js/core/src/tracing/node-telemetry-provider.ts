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

import {
  BatchLogRecordProcessor,
  SimpleLogRecordProcessor,
  type LogRecordProcessor,
} from '@opentelemetry/sdk-logs';
import { NodeSDK } from '@opentelemetry/sdk-node';
import {
  BatchSpanProcessor,
  SimpleSpanProcessor,
  type SpanProcessor,
} from '@opentelemetry/sdk-trace-base';
import { logger } from '../logging.js';
import type { TelemetryConfig } from '../telemetryTypes.js';
import { setTelemetryProvider } from '../tracing.js';
import { isDevEnv } from '../utils.js';
import {
  LogServerExporter,
  TraceServerExporter,
  setTelemetryServerUrl,
} from './exporter.js';
import { RealtimeSpanProcessor } from './realtime-span-processor.js';

let telemetrySDK: NodeSDK | null = null;
let nodeOtelConfig: TelemetryConfig | null = null;

export function initNodeTelemetryProvider() {
  setTelemetryProvider({
    enableTelemetry,
    flushTracing,
  });
}

/**
 * Enables tracing and metrics open telemetry configuration.
 */
async function enableTelemetry(
  telemetryConfig: TelemetryConfig | Promise<TelemetryConfig>
) {
  if (process.env.GENKIT_TELEMETRY_SERVER) {
    setTelemetryServerUrl(process.env.GENKIT_TELEMETRY_SERVER);
  }

  telemetryConfig =
    telemetryConfig instanceof Promise
      ? await telemetryConfig
      : telemetryConfig;

  nodeOtelConfig = telemetryConfig || {};

  const processors: SpanProcessor[] = [createTelemetryServerProcessor()];
  if (nodeOtelConfig.traceExporter) {
    throw new Error('Please specify spanProcessors instead.');
  }
  if (nodeOtelConfig.spanProcessors) {
    processors.push(...nodeOtelConfig.spanProcessors);
  }
  if (nodeOtelConfig.spanProcessor) {
    processors.push(nodeOtelConfig.spanProcessor);
    delete nodeOtelConfig.spanProcessor;
  }
  nodeOtelConfig.spanProcessors = processors;

  // Add LogRecordProcessors
  if (process.env.GENKIT_OTEL_ENABLE_LOGS === 'true') {
    const enableRealTimeTelemetry =
      process.env.GENKIT_ENABLE_REALTIME_TELEMETRY === 'true';
    const logExporter = new LogServerExporter();
    const logProcessor: LogRecordProcessor =
      isDevEnv() || enableRealTimeTelemetry
        ? new SimpleLogRecordProcessor(logExporter)
        : new BatchLogRecordProcessor(logExporter);
    nodeOtelConfig.logRecordProcessor = logProcessor;
  }

  telemetrySDK = new NodeSDK(nodeOtelConfig);
  telemetrySDK.start();
  process.on('SIGTERM', async () => await cleanUpTracing());
}

async function cleanUpTracing(): Promise<void> {
  if (!telemetrySDK) {
    return;
  }

  // Metrics are not flushed as part of the shutdown operation. If metrics
  // are enabled, we need to manually flush them *before* the reader
  // receives shutdown order.
  await maybeFlushMetrics();
  await telemetrySDK.shutdown();
  logger.debug('OpenTelemetry SDK shut down.');
  telemetrySDK = null;
}

/**
 * Creates a new SpanProcessor for exporting data to the telemetry server.
 */
function createTelemetryServerProcessor(): SpanProcessor {
  const exporter = new TraceServerExporter();
  // Use RealtimeSpanProcessor in dev environment (unless disabled), or when explicitly enabled
  const enableRealTimeTelemetry =
    process.env.GENKIT_ENABLE_REALTIME_TELEMETRY === 'true';
  if (isDevEnv() && enableRealTimeTelemetry) {
    return new RealtimeSpanProcessor(exporter);
  } else if (isDevEnv()) {
    return new SimpleSpanProcessor(exporter);
  }
  return new BatchSpanProcessor(exporter);
}

/** Flush metrics if present. */
function maybeFlushMetrics(): Promise<void> {
  if (nodeOtelConfig?.metricReader) {
    return nodeOtelConfig.metricReader.forceFlush();
  }
  return Promise.resolve();
}

/**
 * Flushes all configured span and log processors.
 */
async function flushTracing() {
  const promises: Promise<void>[] = [];
  if (nodeOtelConfig?.spanProcessors) {
    promises.push(...nodeOtelConfig.spanProcessors.map((p) => p.forceFlush()));
  }
  if (nodeOtelConfig?.logRecordProcessor) {
    promises.push(nodeOtelConfig.logRecordProcessor.forceFlush());
  }
  await Promise.all(promises);
}
