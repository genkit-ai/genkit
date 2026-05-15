// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

/*
Package tracing provides execution trace support for Genkit operations.

This package implements OpenTelemetry-based tracing for Genkit actions and flows.
Traces capture the execution path, inputs, outputs, and timing of operations,
enabling observability and debugging through the Genkit Developer UI and
external telemetry systems.

# Automatic Tracing

Actions and flows defined with Genkit are automatically traced. Each action
execution creates a span with input/output data, timing, and any errors.
Use [core.Run] within flows to create traced sub-steps:

	// In a real scenario, 'r' would be the registry from your Genkit instance.
	var r api.Registry
	flow := core.DefineFlow(r, "myFlow",
		func(ctx context.Context, input string) (string, error) {
			// This creates a traced step named "processData"
			result, err := core.Run(ctx, "processData", func() (string, error) {
				return process(input), nil
			})
			return result, err
		},
	)

# Tracer Access

Access the OpenTelemetry tracer provider for custom instrumentation:

	provider := tracing.TracerProvider()

	// Get a tracer for custom spans
	tracer := tracing.Tracer()

# Telemetry Export

Configure trace export to send telemetry to external systems. For immediate
export (suitable for local storage):

	tracing.WriteTelemetryImmediate(client)

For batched export (more efficient for network calls):

	shutdown := tracing.WriteTelemetryBatch(client)
	defer shutdown(ctx)

# Dev UI Integration

When the GENKIT_ENV environment variable is set to "dev", traces are
automatically sent to the Genkit Developer UI's telemetry server. The Dev UI
provides:

  - Visual trace exploration with timing breakdown
  - Input/output inspection for each action
  - Error highlighting and stack traces
  - Performance analysis across flow executions

Set GENKIT_TELEMETRY_SERVER to configure a custom telemetry endpoint.

# Span Metadata

Create spans with rich metadata for better observability:

	metadata := &tracing.SpanMetadata{
		Name:    "processDocument",
		Type:    "action",
		Subtype: "retriever",
	}

	output, err := tracing.RunInNewSpan(ctx, metadata, input,
		func(ctx context.Context, in Input) (Output, error) {
			// Operation runs within the traced span
			return process(in), nil
		},
	)

# Trace Information

Extract trace context for correlation with external systems:

	info := tracing.GetTraceInfo(ctx)
	if info != nil {
		log.Printf("TraceID: %s, SpanID: %s", info.TraceID, info.SpanID)
	}

This package is primarily intended for Genkit internals and advanced plugin
development. Most application developers will interact with tracing through
the automatic instrumentation provided by the genkit package.

For more information on observability, see https://genkit.dev/docs/observability
*/
package tracing
