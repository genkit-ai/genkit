# Genkit + OpenTelemetry GenAI sample (Go)

A minimal Genkit Go app wired to the `go/plugins/otel` GenAI instrumentation,
which emits OpenTelemetry GenAI semantic-convention traces and metrics.

The app owns the OTel SDK. `setupOTel` initializes it from the standard `OTEL_*`
env vars, the Go analog of JS's `new NodeSDK()`: with nothing set it defaults to
OTLP `http/protobuf` at `localhost:4318` for traces, metrics, and logs, and env
vars override that. Point it at a collector (below); an unconfigured run will
log OTLP connection errors, same as JS `NodeSDK()`.

The sample is its own Go module (with a `replace` pointing at the local Genkit
checkout) so the OTLP exporter dependencies it pulls in via
`go.opentelemetry.io/contrib/exporters/autoexport` stay out of the Genkit
library module.

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

## Genkit Dev UI (`genkit start --experimental-use-otel`)

`genkit start --experimental-use-otel` is meant to render the app's own OTel
traces in the Dev UI: it points the app's OTel SDK at the dev telemetry server
via `OTEL_EXPORTER_OTLP_*` env vars instead of enabling Genkit's native dev
instrumentation.

It does not work with this sample yet. The CLI sets
`OTEL_EXPORTER_OTLP_*_PROTOCOL=http/json`, the only encoding the dev telemetry
server ingests today. Go's OTLP trace exporter supports `http/json` (since
v1.46.0), but `autoexport`, which `setupOTel` uses to read the env vars, only
accepts `grpc` and `http/protobuf` and fails at startup. The Go OTLP log
exporter has no JSON support at all. Protobuf ingest in the telemetry server is
planned as a follow-up, which makes the default `http/protobuf` work. Until
then, use a local collector (above), or plain `genkit start` for Genkit's native
dev traces.

## What to expect

- A `chat gemini-flash-latest` CLIENT span with `gen_ai.request.*` config and
  `gen_ai.usage.*` token attributes.
- `execute_tool <name>` spans if the model calls tools (this sample enables
  `EmitToolSpans`).
- The two GenAI client metrics per model call.
- This sample uses `ContentCapturingMode: SpanOnly`, so `gen_ai.input.messages` /
  `gen_ai.output.messages` land on the span (may contain PII) and render in
  Jaeger's GenAI tab. Switch to `EventOnly` to move content to the logs signal
  instead (needs a logs endpoint, e.g. the local collector above).
