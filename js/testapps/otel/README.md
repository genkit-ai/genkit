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

## Genkit Dev UI (`genkit start --use-otel`)

The Genkit Dev UI can render these OTel traces too, no Jaeger or collector
needed. `genkit start --use-otel` points the app's own OTel SDK at the dev
telemetry server's OTLP endpoint (via standard `OTEL_EXPORTER_OTLP_*` env vars)
instead of enabling Genkit's native dev instrumentation:

```bash
export GEMINI_API_KEY=...
genkit start --use-otel -- npx tsx src/index.ts
```

This is the "what would my traces look like in prod" view: only what the app's
own OTel instrumentation emits shows up (the `gen_ai.*` client spans), not
Genkit's more detailed native dev spans. For the richer native view, run
`genkit start` without `--use-otel`.

Note: the dev telemetry server ingests OTLP traces and logs (`http/json`) but
ignores metrics, so the two GenAI client metrics won't appear in the Dev UI.

## What to expect

- A `chat gemini-flash-latest` CLIENT span with `gen_ai.request.*` config and
  `gen_ai.usage.*` token attributes.
- `execute_tool <name>` spans if the model calls tools (this sample enables
  `emitToolSpans`).
- The two GenAI client metrics per model call.
- This sample uses `contentCapturingMode: 'SPAN_ONLY'`, so
  `gen_ai.input.messages` / `gen_ai.output.messages` land on the span (may
  contain PII) and render in Jaeger's GenAI tab.

> [!NOTE]
> Switching to `EVENT_ONLY` moves content to the OpenTelemetry logs signal, so
> it no longer appears in Jaeger (neither the GenAI tab nor "Trace Logs"). The
> collector still logs it via its debug exporter, so `tail -f
.otel/collector.log` to see the `gen_ai.client.inference.operation.details`
> record. Use `SPAN_ONLY` or `SPAN_AND_EVENT` to keep content visible in
> Jaeger.
