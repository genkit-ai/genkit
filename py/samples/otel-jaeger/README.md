# Genkit OpenTelemetry sample (Jaeger, no Docker)

Demonstrates `genkit-otel`: OpenTelemetry GenAI semantic-conventions
instrumentation for Genkit. Traces go to Jaeger, metrics (token usage and
operation duration) go to a local collector's debug log.

The telemetry stack runs from downloaded release binaries, so no Docker is
required.

## What it wires up

```
python src/main.py --OTLP:4318--> otelcol-contrib --OTLP:14317--> Jaeger (UI :16686)
                                         \--debug--> .otel/collector.log (metrics + logs)
```

The collector receives OTLP over both HTTP (`:4318`, the SDK's zero-config
default) and gRPC (`:4317`).

Jaeger v2's binary is itself an OTel collector (it ingests OTLP directly); the
separate `otelcol-contrib` also debug-logs metrics and logs so you can watch
`gen_ai.client.token.usage` and `gen_ai.client.operation.duration`. Jaeger's own
OTLP receiver is moved to `14317`/`14318` so the collector can own the
app-facing `4317`/`4318`.

## Run it

1. Start the telemetry stack (downloads Jaeger + otelcol-contrib on first run
   into `.otel/`, then reuses them):

   ```sh
   python tool/telemetry.py
   ```

   It prints the Jaeger UI URL and the collector log path, then stays running.
   Press Ctrl+C to stop both processes.

2. In another terminal, run the sample:

   ```sh
   export GEMINI_API_KEY=your-key
   export OTEL_SERVICE_NAME=genkit-otel-sample   # optional; names the service
   python src/main.py
   ```

3. Open Jaeger at http://localhost:16686, pick your service, and inspect the
   `chat gemini-flash-latest` span (with request config, usage, and captured
   content attributes).

4. Watch metrics in the collector log:

   ```sh
   tail -f .otel/collector.log
   ```

## Locked-down networks

If your environment blocks GitHub release downloads, provide the binaries
yourself:

```sh
# Use binaries already on disk (skip download):
export JAEGER_BIN=/path/to/jaeger
export OTEL_COLLECTOR_BIN=/path/to/otelcol-contrib

# ...or pin specific release tags instead of "latest":
export JAEGER_VERSION=v2.11.0
export OTEL_COLLECTOR_VERSION=v0.140.0
```

## The instrumentation, in short

```python
from genkit.telemetry import configure_instrumentation
from genkit_otel import ContentCapturingMode, GenAiInstrumentation

# Own the OpenTelemetry SDK, then:
configure_instrumentation(
    GenAiInstrumentation(
        # SPAN_ONLY is easiest to read in Jaeger; may contain PII.
        content_capturing_mode=ContentCapturingMode.SPAN_ONLY,
    ),
)
```
