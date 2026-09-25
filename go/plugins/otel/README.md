# otel (GenAI instrumentation)

OpenTelemetry GenAI semantic-conventions instrumentation for Genkit Go.

Import path: `github.com/firebase/genkit/go/plugins/otel`

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
MeterProvider / LoggerProvider before constructing Genkit. When no SDK is
configured, `go.opentelemetry.io/otel` returns non-recording spans and no-op
instruments and the provider is inert.

## Usage

```go
import (
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"

	"github.com/firebase/genkit/go/core/tracing"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	genaiotel "github.com/firebase/genkit/go/plugins/otel"
)

// 1. The app owns the OTel SDK (point the exporter at your OTLP collector).
otel.SetTracerProvider(sdktrace.NewTracerProvider(/* ... */))

// 2. Route Genkit telemetry through the GenAI provider. It composes with the
//    built-in dev instrumentation that feeds the Developer UI.
tracing.ConfigureInstrumentation(genaiotel.NewGenAiInstrumentation(genaiotel.GenAiInstrumentationOptions{}))

// 3. Use Genkit as usual.
g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))
```

The configuration is process-wide on purpose: Genkit actions can run without
a Genkit instance (for example a model obtained directly from a plugin), and
those calls are instrumented too.

See `go/samples/otel-genai` for a runnable example that exports over OTLP.

## Google Cloud / Firebase telemetry

Do not combine this provider with the `googlecloud` or `firebase` telemetry
plugins. Configuring it replaces Genkit's default OpenTelemetry encoding (the
`genkit:*` span attributes) that those plugins read, so their Genkit metrics
stop, and their input/output redaction does not cover `gen_ai.*` content
attributes.

## What gets recorded

- `gen_ai.request.*` params are read from the request config, whether it is
  Genkit's `GenerationCommonConfig` or a provider SDK's native type (camelCase
  and snake_case names are both recognized). A param set to `0` is recorded.
- A call that fails after the model responded (for example, output schema
  validation) still records the response: token usage, content, and a
  `gen_ai.response.finish_reasons` of `error`.
- `error.type` is the Genkit status name, such as `INVALID_ARGUMENT` or
  `RESOURCE_EXHAUSTED` (`INTERNAL` when unclassified).
- Inline media (`data:` URIs) is captured as a `blob` part with its decoded
  size in `size_bytes`, never the payload. Remote media becomes a `uri` part.
- Telemetry labels (`tracing.WithTelemetryLabels`) are set as span attributes.

## Options

`GenAiInstrumentationOptions`:

| Option                 | Default     | Description                                                                    |
| ---------------------- | ----------- | ------------------------------------------------------------------------------ |
| `ContentCapturingMode` | `NoContent` | Where spec-shaped message content is recorded (span/event). May contain PII.   |
| `CaptureActionIO`      | `false`     | Record raw Genkit input/output as `genkit.input` / `genkit.output` JSON.       |
| `EmitToolSpans`        | `false`     | Emit `execute_tool` spans for tool actions.                                    |
| `DisableMetrics`       | `false`     | Turn off the GenAI client metrics (on by default).                             |
| `ScopeName`            | `genkit-genai` | Instrumentation scope name for the tracer/logger/meter.                     |
| `Tracer` / `Meter`     | global      | Optional explicit instruments (escape hatch).                                  |

When `ContentCapturingMode` is left at its zero value, the env var
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` is consulted (spec enum
names: `NO_CONTENT`, `SPAN_ONLY`, `EVENT_ONLY`, `SPAN_AND_EVENT`).
