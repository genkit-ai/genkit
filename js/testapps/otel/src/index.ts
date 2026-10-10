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
import { NodeSDK } from '@opentelemetry/sdk-node';
import { genkit, z } from 'genkit';
import { configureInstrumentation } from 'genkit/tracing';

// The application owns the OTel SDK. With no arguments NodeSDK configures
// traces, metrics, and logs from the standard OTEL_* env vars, defaulting to
// the OTLP http/proto exporter at http://localhost:4318, which a local
// collector accepts. See README.md for a Docker-free local Jaeger + collector
// setup.
const sdk = new NodeSDK();
sdk.start();

// Content capture is opt-in (it may contain PII). SPAN_ONLY is the easiest to
// eyeball in Jaeger; use EVENT_ONLY to keep bodies off the span (preferred for
// production), or SPAN_AND_EVENT for both.
configureInstrumentation(
  new GenAiInstrumentation({
    contentCapturingMode: 'SPAN_ONLY',
    emitToolSpans: true,
  })
);

const ai = genkit({ plugins: [googleAI()] });

// With emitToolSpans enabled, each call becomes an `execute_tool getWeather`
// span nested under the flow.
const getWeather = ai.defineTool(
  {
    name: 'getWeather',
    description: 'Gets the current weather for a city.',
    inputSchema: z.object({ city: z.string() }),
    outputSchema: z.string(),
  },
  async ({ city }) => `It is 21C and sunny in ${city}.`
);

// Produces a flow span with `chat gemini-flash-latest` client spans (one per
// model turn) and the tool span in between.
export const weatherFlow = ai.defineFlow(
  {
    name: 'weatherFlow',
    inputSchema: z.string().default('Paris'),
    outputSchema: z.string(),
  },
  async (city) => {
    const { text } = await ai.generate({
      model: googleAI.model('gemini-flash-latest'),
      prompt: `What's the weather in ${city}? Answer in one sentence.`,
      tools: [getWeather],
    });
    return text;
  }
);

// Best-effort flush on exit. Genkit registers its own SIGTERM/SIGINT handler
// that calls process.exit, so this can lose the race; the batch span processor
// exports periodically anyway, so at most the last few seconds are dropped.
for (const signal of ['SIGTERM', 'SIGINT'] as const) {
  process.once(signal, () => {
    sdk.shutdown().catch((e) => console.error('OTel SDK shutdown failed', e));
  });
}
