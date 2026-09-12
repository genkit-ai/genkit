/**
 * Copyright 2025 Google LLC
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

import { googleAI } from '@genkit-ai/google-genai';
import { GenAiInstrumentation } from '@genkit-ai/otel';
import { OTLPMetricExporter } from '@opentelemetry/exporter-metrics-otlp-proto';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-proto';
import { PeriodicExportingMetricReader } from '@opentelemetry/sdk-metrics';
import { NodeSDK } from '@opentelemetry/sdk-node';
import { genkit } from 'genkit';
import { configureInstrumentation } from 'genkit/tracing';

// The application owns the OTel SDK. With no OTEL_* env vars it defaults to
// http://localhost:4318 (OTLP http/proto), which a local collector accepts.
// See README.md for a Docker-free local Jaeger + collector setup.
const sdk = new NodeSDK({
  traceExporter: new OTLPTraceExporter(),
  metricReader: new PeriodicExportingMetricReader({
    exporter: new OTLPMetricExporter(),
  }),
});
sdk.start();

// captureContent is opt-in (it may contain PII). 'span' mode is the easiest to
// eyeball in Jaeger; switch to 'event' to keep bodies off the span.
configureInstrumentation(
  new GenAiInstrumentation({
    captureContent: true,
    contentMode: 'span',
    emitToolSpans: true,
  })
);

const ai = genkit({ plugins: [googleAI()] });

async function main() {
  const { text } = await ai.generate({
    model: googleAI.model('gemini-flash-latest'),
    prompt: 'Explain OpenTelemetry in one sentence.',
  });
  console.log(text);

  // Flush and shut down so spans and metrics reach the collector.
  await sdk.shutdown();
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
