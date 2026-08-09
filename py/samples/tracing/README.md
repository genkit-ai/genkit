# Tracing

Spans show up in Dev UI as they start, not when they finish. For long flows with many steps.

```bash
export GEMINI_API_KEY=your-api-key
uv sync
uv run src/main.py
```

To watch it in Dev UI instead:

```bash
genkit start -- uv run src/main.py
```

Run `trace_steps_live` and watch the Traces tab.

## Langfuse via plain OpenTelemetry (no custom Genkit handler)

Genkit instruments with the OTel API. Your app owns the SDK / exporters.
Point OTLP at Langfuse — Genkit spans show up. See `src/otel_to_langfuse.py`.

```bash
# AUTH=$(echo -n "$LANGFUSE_PUBLIC_KEY:$LANGFUSE_SECRET_KEY" | base64)
export OTEL_EXPORTER_OTLP_ENDPOINT="https://cloud.langfuse.com/api/public/otel"
export OTEL_EXPORTER_OTLP_HEADERS="Authorization=Basic ${AUTH},x-langfuse-ingestion-version=4"
uv run src/otel_to_langfuse.py
```
