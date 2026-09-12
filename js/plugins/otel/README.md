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

| Option                 | Default          | Description                                                         |
| ---------------------- | ---------------- | ------------------------------------------------------------------- |
| `contentCapturingMode` | env-gated        | Where spec-shaped message content is recorded (see below).          |
| `captureActionIO`      | `false`          | Record raw Genkit input/output as `genkit.input` / `genkit.output`. |
| `emitToolSpans`        | `false`          | Emit `execute_tool` spans for tool actions.                         |
| `emitMetrics`          | `true`           | Emit the GenAI client metrics.                                      |
| `scopeName`            | `'genkit-genai'` | Instrumentation scope for the tracer/meter/logger.                  |
| `tracer`/`meter`       | resolved         | Escape hatches to inject explicit instruments.                      |

### Content capture

`contentCapturingMode` mirrors the OTel GenAI `ContentCapturingMode` and takes
one of:

| Value            | Effect                                                                                                                                              |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `NO_CONTENT`     | Default. No message content is recorded.                                                                                                            |
| `SPAN_ONLY`      | Content on span attributes as JSON strings. Easy to eyeball in Jaeger, but subject to backend attribute/envelope size limits. Best for development. |
| `EVENT_ONLY`     | Content on a dedicated `gen_ai.client.inference.operation.details` log event in structured form. Preferred for production.                          |
| `SPAN_AND_EVENT` | Both of the above.                                                                                                                                  |

When `contentCapturingMode` is omitted, the env var
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` is consulted (same enum
names); an explicit option overrides the env var. Content and raw IO may
contain PII, so both are opt-in.

> [!NOTE] > `EVENT_ONLY` emits content on the OpenTelemetry **logs** signal (a
> `gen_ai.client.inference.operation.details` log record), not on the span.
> Trace-only backends like Jaeger cannot display it: the GenAI tab reads span
> attributes, and the "Trace Logs" tab reads span events, neither of which is
> the logs signal. Use `SPAN_ONLY` or `SPAN_AND_EVENT` to see content in
> Jaeger, or send logs to a logs-capable backend (e.g. Grafana Loki,
> Elasticsearch/OpenSearch) for `EVENT_ONLY`.

The sources of Genkit are available on
[GitHub](https://github.com/genkit-ai/genkit).
