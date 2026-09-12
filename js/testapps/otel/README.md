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

The quickest Docker-free option is Jaeger v2, which ingests OTLP directly and
serves a UI on http://localhost:16686:

```bash
jaeger   # from a Jaeger v2 release binary; listens on OTLP 4317/4318
```

Then open http://localhost:16686 and look for `chat gemini-flash-latest` client
spans carrying `gen_ai.*` attributes. To also inspect metrics
(`gen_ai.client.token.usage`, `gen_ai.client.operation.duration`), run an
`otelcol-contrib` collector with a `debug` exporter in front of Jaeger.

## What to expect

- A `chat gemini-flash-latest` CLIENT span with `gen_ai.request.*` config and
  `gen_ai.usage.*` token attributes.
- `execute_tool <name>` spans if the model calls tools (this sample enables
  `emitToolSpans`).
- The two GenAI client metrics per model call.
- With `captureContent: true` and `contentMode: 'span'`, `gen_ai.input.messages`
  / `gen_ai.output.messages` on the span (may contain PII).
