# Genkit + OpenTelemetry GenAI sample

A minimal Genkit app wired to `@genkit-ai/otel`, which emits OpenTelemetry
GenAI semantic-convention traces and metrics.

## Run

Set your API key and point the OTLP exporters at a collector (defaults to
`http://localhost:4318`):

```bash
export GEMINI_API_KEY=...
# Optional, defaults to http://localhost:4318:
# export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318

pnpm dev
```

## Local telemetry stack

The repo ships a Docker-free helper that downloads Jaeger v2 and an
`otelcol-contrib` collector, wires them together, and serves the Jaeger UI on
http://localhost:16686. Run it from the repo root in a separate terminal:

```bash
pnpm local-telemetry
```

It listens for OTLP on `http://localhost:4318` (http) and `localhost:4317`
(grpc), forwards traces to Jaeger, and debug-logs metrics/logs to
`.otel/collector.log`. Leave it running, then start the sample.

Open http://localhost:16686 and look for `chat gemini-flash-latest` client
spans carrying `gen_ai.*` attributes. Metrics
(`gen_ai.client.token.usage`, `gen_ai.client.operation.duration`) show up in
the collector log (`tail -f .otel/collector.log`).

## What to expect

- A `chat gemini-flash-latest` CLIENT span with `gen_ai.request.*` config and
  `gen_ai.usage.*` token attributes.
- `execute_tool <name>` spans if the model calls tools (this sample enables
  `emitToolSpans`).
- The two GenAI client metrics per model call.
- With `captureContent: true` and `contentMode: 'span'`, `gen_ai.input.messages`
  / `gen_ai.output.messages` on the span (may contain PII).
