# Realtime Tracing Demo

This sample demonstrates Genkit's realtime tracing feature, which exports spans to the DevUI as they **start** (not just when they complete). This enables live visualization of in-progress operations.

## What is Realtime Tracing?

Standard OpenTelemetry processors only export spans when they finish. This means long-running operations (like multi-turn LLM calls or complex workflows) don't appear in the DevUI until they're complete.

With realtime tracing enabled:
- Spans appear in the DevUI **immediately** when they start
- You can see operations **in progress** (without endTime)
- The span is exported again when it completes with full data

```
Standard Tracing:          Realtime Tracing:
                           
Start ─────────────────►   Start ────► [Span appears!]
      (nothing visible)           │
      ...                         │ (visible as "in progress")
      ...                         │
End ──────► [Span appears]  End ──┴─► [Span updated]
```

## How It Works

The `RealtimeSpanProcessor` wraps a standard exporter and calls `export()` twice:
1. On `on_start()` - exports immediately (no endTime)
2. On `on_end()` - exports again with complete data

## Usage

### Enable via Environment Variable

```bash
# Enable realtime telemetry
export GENKIT_ENABLE_REALTIME_TELEMETRY=true

# Run with genkit start
genkit start -- python src/main.py
```

### Enable Programmatically

```python
from opentelemetry.sdk.trace import TracerProvider

from genkit.core.trace import RealtimeSpanProcessor, TelemetryServerSpanExporter

# Create exporter
exporter = TelemetryServerSpanExporter(
    telemetry_server_url='http://localhost:4000'
)

# Wrap with RealtimeSpanProcessor
processor = RealtimeSpanProcessor(exporter)

# Add to tracer provider
provider = TracerProvider()
provider.add_span_processor(processor)
```

## Running the Demo

1. Run with realtime tracing enabled (with hot reload):
   ```bash
   ./run.sh
   ```

   You'll be prompted for `GEMINI_API_KEY` if not set.

2. Open the DevUI at http://localhost:4000

3. Trigger flows and watch spans appear **immediately** as operations start!

4. Edit code and it will automatically reload.

## Key APIs Demonstrated

| API | Description |
|-----|-------------|
| `RealtimeSpanProcessor` | SpanProcessor that exports on start AND end |
| `is_realtime_telemetry_enabled()` | Check if realtime mode is enabled |
| `create_span_processor(exporter)` | Auto-selects processor based on env |
| `GENKIT_ENABLE_REALTIME_TELEMETRY` | Environment variable to enable |

## When to Use

- **Development**: Great for debugging and understanding flow execution
- **Long-running operations**: See progress of complex workflows
- **DevUI demos**: Showcase live updates

## When NOT to Use

- **Production**: Doubles network traffic (each span exported twice)
- **High-throughput**: May impact performance
- **Simple flows**: Standard tracing is sufficient

## Related Samples

- `session-demo/` - Multi-turn conversations
- `chat-demo/` - Chat application with streaming
- `tool-interrupts/` - Human-in-the-loop workflows

## Testing This Demo

1. **Prerequisites**:
   ```bash
   export GEMINI_API_KEY=your_api_key
   ```
   Or the demo will prompt for the key interactively.

2. **Run the demo**:
   ```bash
   cd py/samples/framework-realtime-tracing-demo
   ./run.sh  # This sets GENKIT_ENABLE_REALTIME_TELEMETRY=true
   ```

3. **Open DevUI** at http://localhost:4000

4. **Test realtime tracing**:
   - [ ] Open the Traces tab in DevUI
   - [ ] Trigger a multi-step flow
   - [ ] Watch spans appear IMMEDIATELY as they start
   - [ ] Compare to non-realtime (spans appear at end)

5. **Test flows**:
   - [ ] `multi_step_flow` - See each step appear in order
   - [ ] `nested_flow` - See parent/child span hierarchy
   - [ ] `long_running_flow` - Watch progress of slow tasks

6. **Expected behavior**:
   - Spans appear in DevUI as soon as they START
   - You see "in progress" spans while they're running
   - Nested spans show proper parent/child relationships
   - Long-running spans show duration updating in real-time
