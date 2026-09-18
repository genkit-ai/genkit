# genkit-otel

OpenTelemetry GenAI semantic-conventions instrumentation for
[Genkit](https://github.com/genkit-ai/genkit). It plugs into Genkit's
pluggable instrumentation system and emits telemetry that follows the
[OTel GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai):

- `gen_ai.*` client spans for model operations (`chat <model>`), with request
  config, token usage, and finish reasons.
- The GenAI client metrics `gen_ai.client.token.usage` (split by
  `gen_ai.token.type`) and `gen_ai.client.operation.duration`.
- Optional message-content capture, either as span attributes or a dedicated
  `gen_ai.client.inference.operation.details` event.
- Optional `execute_tool` spans, and generic spans for other Genkit action
  types so the trace tree stays connected.

## Usage

The application owns the OpenTelemetry SDK. Initialize it, then register the
provider before creating `Genkit`:

```python
from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from genkit import Genkit
from genkit.telemetry import configure_instrumentation
from genkit_otel import GenAiInstrumentation

resource = Resource.create({SERVICE_NAME: 'my-service'})
provider = TracerProvider(resource=resource)
provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
trace.set_tracer_provider(provider)

configure_instrumentation(GenAiInstrumentation())

ai = Genkit()
# ...
await ai.aclose()
```

When the SDK is not initialized, the provider is effectively a no-op.

## Content capture (PII)

Prompt and response content may contain PII, so capture is off by default.
`content_capturing_mode` mirrors the OTel GenAI `ContentCapturingMode`:

| Mode | Where content goes |
| --- | --- |
| `NO_CONTENT` (default) | not captured |
| `SPAN_ONLY` | span attributes (`gen_ai.*.messages`) as JSON strings |
| `EVENT_ONLY` | a `gen_ai.client.inference.operation.details` log event |
| `SPAN_AND_EVENT` | both |

```python
from genkit_otel import ContentCapturingMode, GenAiInstrumentation

GenAiInstrumentation(
    content_capturing_mode=ContentCapturingMode.SPAN_ONLY,
)
```

When not supplied, the env var `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`
is consulted using the spec's UPPER_SNAKE tokens (`NO_CONTENT`, `SPAN_ONLY`,
`EVENT_ONLY`, `SPAN_AND_EVENT`); an explicit value overrides it. An unknown
token logs a one-time warning and falls back to `NO_CONTENT`.

> `EVENT_ONLY` emits content on the OpenTelemetry logs signal (a
> `gen_ai.client.inference.operation.details` log record), not on the span.
> Trace-only backends like Jaeger cannot display it (its GenAI tab reads span
> attributes; its "Trace Logs" tab reads span events, neither is the logs
> signal). Use `SPAN_ONLY` or `SPAN_AND_EVENT` for Jaeger, or a logs backend
> (e.g. Loki, Elasticsearch/OpenSearch) for `EVENT_ONLY`.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `content_capturing_mode` | env or `NO_CONTENT` | Where spec-shaped `gen_ai.*` message content is recorded. |
| `capture_action_io` | `False` | Capture raw Genkit input/output as `genkit.input`/`genkit.output` on every span (debugging / Dev UI). |
| `emit_metrics` | `True` | Emit token-usage and operation-duration metrics. |
| `emit_tool_spans` | `False` | Emit `execute_tool` spans for tool actions. |
| `scope_name` | `genkit-genai` | Instrumentation scope for tracer/meter/logger. |
| `tracer` / `meter` / `otel_logger` | resolved lazily | Escape hatches to inject explicit instances. |

## Example

See [`samples/otel-jaeger`](../../samples/otel-jaeger) for a runnable sample
that exports traces to Jaeger and metrics to a local collector, with a
Docker-free script that downloads and runs both.
