# @genkit-ai/otel

## 0.1.0

- Initial release: `GenAiInstrumentation`, an OpenTelemetry GenAI
  semantic-conventions provider for Genkit built on `@opentelemetry/api`.
- Emits `gen_ai.*` client spans for model operations, optional `execute_tool`
  spans, and generic spans for other Genkit action types.
- Emits the GenAI client metrics `gen_ai.client.token.usage` and
  `gen_ai.client.operation.duration`.
- Optional message-content capture (span attributes or a dedicated event), off
  by default and gated by `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`.
