# genkit-otel

OpenTelemetry instrumentation and backend for [Genkit](https://github.com/genkit-ai/genkit).

It provides `GenAiInstrumentation`: OpenTelemetry GenAI semantic-conventions instrumentation emitting `gen_ai.*` spans and metrics.

Pass `tracer` to mint on that tracer instead of the process-global one.

## GenAI Semantic Conventions Instrumentation

Plugs into Genkit's pluggable instrumentation system and emits telemetry that follows the
[OTel GenAI semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai):

- `gen_ai.*` client spans for model operations (`chat <model>`), with request
  config, token usage, and finish reasons.
- The GenAI client metrics `gen_ai.client.token.usage` (split by
  `gen_ai.token.type`) and `gen_ai.client.operation.duration`.
- Optional `execute_tool` spans, and generic spans for other Genkit action
  types so the trace tree stays connected.

Prompt and reply text are not recorded.

### Usage

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
```

When the SDK is not initialized, the provider is effectively a no-op.

### Options

| Option | Default | Description |
| --- | --- | --- |
| `capture_action_io` | `False` | Capture raw Genkit input/output as `genkit.input`/`genkit.output` on every span (debugging / Dev UI). |
| `emit_metrics` | `True` | Emit token-usage and operation-duration metrics. |
| `emit_tool_spans` | `False` | Emit `execute_tool` spans for tool actions. |
| `scope_name` | `genkit-genai` | Instrumentation scope for tracer/meter. |
| `tracer` / `meter` | resolved lazily | Escape hatches to inject explicit instances. |
