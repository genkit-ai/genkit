# Genkit + OpenTelemetry GenAI sample (Go)

A minimal Genkit Go app wired to the `go/plugins/otel` GenAI instrumentation,
which emits OpenTelemetry GenAI semantic-convention traces and metrics.

The app owns the OTel SDK. `setupOTel` initializes it from the standard `OTEL_*`
env vars, the Go analog of JS's `new NodeSDK()`: with nothing set it defaults to
OTLP `http/protobuf` at `localhost:4318` for traces, metrics, and logs, and env
vars override that. Point it at a collector (below) or use `--use-otel`; an
unconfigured run will log OTLP connection errors, same as JS `NodeSDK()`.

## Run against a local collector + Jaeger

The repo ships a Docker-free helper that downloads Jaeger v2 and an
`otelcol-contrib` collector and serves the Jaeger UI on http://localhost:16686.
Run it from the repo root in a separate terminal:

```bash
npx tsx scripts/local-telemetry.ts
```

It listens for OTLP on `http://localhost:4318` (http) and `localhost:4317`
(grpc). Then run the sample, which makes a single generate call and exits:

```bash
export GEMINI_API_KEY=...
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
# Optional, for a nicer service name in Jaeger (defaults to unknown_service):
export OTEL_SERVICE_NAME=otel-genai
go run .
```

Open http://localhost:16686 and look for `chat gemini-flash-latest` client spans
carrying `gen_ai.*` attributes. Metrics (`gen_ai.client.token.usage`,
`gen_ai.client.operation.duration`) show up in the collector log
(`tail -f .otel/collector.log`).

## Genkit Dev UI (`genkit start --use-otel`)

The Genkit Dev UI can render these OTel traces too, no Jaeger or collector
needed. `genkit start --use-otel` points the app's OTel SDK at the dev telemetry
server's OTLP endpoint (via `OTEL_EXPORTER_OTLP_*` env vars, `http/json`) instead
of Genkit's native dev instrumentation:

```bash
GEMINI_API_KEY=... genkit start --use-otel -- go run .
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
  `EmitToolSpans`).
- The two GenAI client metrics per model call.
- This sample uses `ContentCapturingMode: SpanOnly`, so `gen_ai.input.messages` /
  `gen_ai.output.messages` land on the span (may contain PII) and render in
  Jaeger's GenAI tab. Switch to `EventOnly` to move content to the logs signal
  instead (needs a logs endpoint, e.g. `--use-otel`).
