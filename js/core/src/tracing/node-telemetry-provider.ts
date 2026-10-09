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

import { NodeSDK } from '@opentelemetry/sdk-node';
import { type SpanProcessor } from '@opentelemetry/sdk-trace-base';
import { logger } from '../logging.js';
import type { TelemetryConfig } from '../telemetryTypes.js';
import { setTelemetryProvider } from '../tracing.js';
import { setTelemetryServerUrl } from './exporter.js';

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
 *
 * This is the collection side: export to the exporters the caller (or a plugin
 * such as GCP / Firebase) configured. It no longer auto-wires the telemetry
 * server exporter or a default log processor to feed the Developer UI; that is
 * now the instrumentation side's job (`DirectTelemetryInstrumentation`). Only
 * user-supplied span/log processors and metric readers are honored here.
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

  const processors: SpanProcessor[] = [];
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
