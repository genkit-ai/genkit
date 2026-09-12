# @genkit-ai/otel

OpenTelemetry GenAI semantic-conventions instrumentation for Genkit.

This plugin maps Genkit actions to the [OpenTelemetry GenAI semantic
conventions](https://github.com/open-telemetry/semantic-conventions-genai):

- Model actions become `chat <model>` client spans carrying `gen_ai.*`
  attributes (provider, model, request config, response finish reasons, token
  usage).
- Tool actions can optionally become `execute_tool <name>` spans.
- Every other Genkit action type becomes a generic span tagged with
  `genkit.action.type` so the trace tree stays connected.
- The two spec metrics `gen_ai.client.token.usage` and
  `gen_ai.client.operation.duration` are emitted per model call.

The application owns the OpenTelemetry SDK: configure your TracerProvider /
MeterProvider / LoggerProvider (e.g. via `@opentelemetry/sdk-node`) before
constructing Genkit. When no SDK is configured, `@opentelemetry/api` returns
non-recording spans / no-op instruments and the provider is inert.

## Installation

```bash
npm i @genkit-ai/otel
```

## Usage

```ts
import { NodeSDK } from '@opentelemetry/sdk-node';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-proto';
import { genkit } from 'genkit';
import { configureInstrumentation } from 'genkit/tracing';
import { googleAI } from '@genkit-ai/google-genai';
import { GenAiInstrumentation } from '@genkit-ai/otel';

// 1. The app owns the OTel SDK.
const sdk = new NodeSDK({ traceExporter: new OTLPTraceExporter() });
sdk.start();

// 2. Route Genkit telemetry through the GenAI provider. It composes with the
//    built-in dev instrumentation that feeds the Developer UI.
configureInstrumentation(new GenAiInstrumentation());

// 3. Use Genkit as usual.
const ai = genkit({ plugins: [googleAI()] });
const { text } = await ai.generate({
  model: googleAI.model('gemini-flash-latest'),
  prompt: 'Explain OpenTelemetry in one sentence.',
});
console.log(text);
```

## Options

`new GenAiInstrumentation(options)` accepts:

| Option            | Default   | Description                                                                 |
| ----------------- | --------- | --------------------------------------------------------------------------- |
| `captureContent`  | env-gated | Capture spec-shaped message content. Off unless `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true`. |
| `contentMode`     | `'event'` | Where captured content lands: an operation-details event or span attributes. |
| `captureActionIO` | `false`   | Record raw Genkit input/output as `genkit.input` / `genkit.output`.         |
| `emitToolSpans`   | `false`   | Emit `execute_tool` spans for tool actions.                                 |
| `emitMetrics`     | `true`    | Emit the GenAI client metrics.                                              |
| `scopeName`       | `'genkit-genai'` | Instrumentation scope for the tracer/meter/logger.                  |
| `tracer`/`meter`  | resolved  | Escape hatches to inject explicit instruments.                             |

Content and raw IO may contain PII, so both are opt-in.

The sources of Genkit are available on
[GitHub](https://github.com/genkit-ai/genkit).
