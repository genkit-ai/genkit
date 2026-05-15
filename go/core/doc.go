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
Package core implements Genkit's foundational action system and runtime machinery.

This package is primarily intended for plugin developers and Genkit internals.
Application developers should use the genkit package instead, which provides
a higher-level, more convenient API.

# Actions

Actions are the fundamental building blocks of Genkit. Every operation - flows,
model calls, tool invocations, retrieval - is implemented as an action. Actions
provide:

  - Type-safe input/output with JSON schema validation
  - Automatic tracing and observability
  - Consistent error handling
  - Registration in the action registry

Define a non-streaming action:

	action := core.DefineAction(registry, "myAction",
		func(ctx context.Context, input string) (string, error) {
			return "processed: " + input, nil
		},
	)

	result, err := action.Run(context.Background(), "hello")

Define a streaming action that sends chunks during execution:

	streamingAction := core.DefineStreamingAction(registry, "countdown",
		func(ctx context.Context, start int, cb core.StreamCallback[string]) (string, error) {
			for i := start; i > 0; i-- {
				if cb != nil {
					if err := cb(ctx, fmt.Sprintf("T-%d", i)); err != nil {
						return "", err
					}
				}
				time.Sleep(time.Second)
			}
			return "Liftoff!", nil
		},
	)

# Flows

Flows are user-defined actions that orchestrate AI operations. They are the
primary way application developers define business logic in Genkit:

	flow := core.DefineFlow(registry, "myFlow",
		func(ctx context.Context, input string) (string, error) {
			// Use Run to create traced sub-steps
			result, err := core.Run(ctx, "step1", func() (string, error) {
				return process(input), nil
			})
			if err != nil {
				return "", err
			}
			return result, nil
		},
	)

Streaming flows can send intermediate results to callers:

	streamingFlow := core.DefineStreamingFlow(registry, "generateReport",
		func(ctx context.Context, input Input, cb core.StreamCallback[Progress]) (Report, error) {
			for i := 0; i < 100; i += 10 {
				if cb != nil {
					cb(ctx, Progress{Percent: i})
				}
				// ... work ...
			}
			return Report{...}, nil
		},
	)

# Traced Steps with Run

Use [Run] within flows to create traced sub-operations. Each Run call creates
a span in the trace that's visible in the Genkit Developer UI:

	result, err := core.Run(ctx, "fetchData", func() (Data, error) {
		return fetchFromAPI()
	})

	processed, err := core.Run(ctx, "processData", func() (Result, error) {
		return process(result)
	})

# Middleware

Actions support middleware for cross-cutting concerns like logging, metrics,
or authentication:

	loggingMiddleware := func(next core.StreamingFunc[string, string, struct{}]) core.StreamingFunc[string, string, struct{}] {
		return func(ctx context.Context, input string, cb core.StreamCallback[struct{}]) (string, error) {
			log.Printf("Input: %s", input)
			output, err := next(ctx, input, cb)
			log.Printf("Output: %s, Error: %v", output, err)
			return output, err
		}
	}

Chain multiple middleware together:

	combined := core.ChainMiddleware(loggingMiddleware, metricsMiddleware)
	wrappedFn := combined(originalFunc)

# Schema Management

Register JSON schemas for use in prompts and validation:

	// Define a schema from a map
	core.DefineSchema(registry, "Person", map[string]any{
		"type": "object",
		"properties": map[string]any{
			"name": map[string]any{"type": "string"},
			"age":  map[string]any{"type": "integer"},
		},
		"required": []any{"name"},
	})

	// Define a schema from a Go type (recommended)
	core.DefineSchemaFor[Person](registry)

Schemas can be referenced in .prompt files by name.

# Plugin Development

Plugins extend Genkit's functionality by providing models, tools, retrievers,
and other capabilities. Implement the [api.Plugin] interface:

	type MyPlugin struct {
		APIKey string
	}

	func (p *MyPlugin) Name() string {
		return "myplugin"
	}

	func (p *MyPlugin) Init(ctx context.Context) []api.Action {
		// Initialize the plugin and return actions to register
		model := ai.DefineModel(...)
		tool := ai.DefineTool(...)
		return []api.Action{model, tool}
	}

For plugins that resolve actions dynamically (e.g., listing available models
from an API), implement [api.DynamicPlugin]:

	type DynamicModelPlugin struct{}

	func (p *DynamicModelPlugin) ListActions(ctx context.Context) []api.ActionDesc {
		// Return descriptors of available actions
		return []api.ActionDesc{
			{Key: "/model/myplugin/model-a", Name: "model-a"},
			{Key: "/model/myplugin/model-b", Name: "model-b"},
		}
	}

	func (p *DynamicModelPlugin) ResolveAction(atype api.ActionType, name string) api.Action {
		// Create and return the action on demand
		return createModel(name)
	}

# Background Actions

For long-running operations, use background actions that return immediately
with an operation ID that can be polled for completion:

	bgAction := core.DefineBackgroundAction(registry, "longTask",
		func(ctx context.Context, input Input) (Output, error) {
			// Start the operation
			return startLongOperation(input)
		},
		func(ctx context.Context, op *core.Operation[Output]) (*core.Operation[Output], error) {
			// Check operation status
			return checkOperationStatus(op)
		},
	)

# Error Handling

Return user-facing errors with appropriate status codes:

	if err := validate(input); err != nil {
		return nil, core.NewPublicError(core.INVALID_ARGUMENT, "Invalid input", map[string]any{
			"field": "email",
			"error": err.Error(),
		})
	}

For internal errors that should be logged but not exposed to users:

	return nil, core.NewError(core.INTERNAL, "database connection failed: %v", err)

# Context

Access action context for metadata and configuration:

	ctx := core.FromContext(ctx)
	if ctx != nil {
		// Access action-specific context values
	}

Set action context for nested operations:

	ctx = core.WithActionContext(ctx, core.ActionContext{
		"requestId": requestID,
	})

For more information, see https://genkit.dev/docs/plugins
*/
package core
