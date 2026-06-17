# Go Development Guidelines

## Agent Instructions

As an AI developer working on this codebase, you MUST adhere to these operational protocols:

* **Registry-First Thinking**: Genkit is built around a central registry. Always use `genkit.DefineX` (e.g., `DefineFlow`, `DefineTool`) for any component that should be discoverable by tools or the Dev UI.
* **Context is Mandatory**: Every Genkit function and tool execution requires a `context.Context`. Pass it through faithfully; never use `context.Background()` inside an implementation unless it is the entry point.
* **Schema Strictness**: When defining input/output types for Flows or Tools, use clear, JSON-tagged Go structs. The model uses these tags and types to understand the interface.
* **Search Before Implementing**: Many common utilities exist in `internal/base` or `ai/`. Check there before implementing new JSON parsing or string manipulation logic.
* **Idiomatic Concurrency**: Prefer Go's native concurrency (goroutines/channels) but be mindful of the `context` lifecycle within long-running background tasks.

## Architecture & Patterns

* **Package Structure**:
  * `genkit`: The main entry point for application developers. Contains high-level functions for defining flows, prompts, and tools.
  * `ai`: Contains the core AI types and interfaces (Model, Prompt, Tool, Embedder, Retriever, etc.).
  * `core`: Contains the underlying framework primitives (Actions, Flows, Tracing, Registry). Mostly for internal use or plugin development.
  * `plugins`: Contains implementations for specific providers (Google AI, Vertex AI, Ollama, etc.).
  * `internal`: Contains private implementations, shared utilities, and development tools.
    * `base`: Low-level utilities for JSON (extraction, normalization, validation) and context handling.
    * `cmd`: Internal command-line tools for code generation (`jsonschemagen`), file synchronization (`copy`), and documentation (`weave`).
    * `registry`: Implementation of the Genkit action and schema registry.
    * `metrics`: OpenTelemetry instrumentation for framework metrics.
    * `fakeembedder`: Testing utilities for simulating embedding providers.
* **Constructors**:
  * **`DefineX`** (e.g., `DefineFlow`, `DefineTool`): Creates **and registers** a component with the registry. Use this for components that should be discoverable by the reflection API (and thus the Dev UI).
  * **`NewX`** (e.g., `NewTool`): Creates a component **without registering it**. Use this for internal components, dynamic creation, or testing.
* **Options Pattern**:
  * Use functional options (e.g., `ai.WithModel`, `genkit.WithPlugins`) for optional configuration parameters in constructors and method calls. This maintains API stability while allowing for expansion.

## Canonical Coding Patterns

### Defining a Flow
Use `genkit.DefineFlow` for orchestration tasks. Use typed structs for input and output to ensure schema generation.

```go
type OrderInput struct {
	ID int `json:"id"`
}

type OrderOutput struct {
	Status string `json:"status"`
}

// Define the flow
var GetOrderStatusFlow = genkit.DefineFlow(g, "getOrderStatus",
	func(ctx context.Context, input *OrderInput) (*OrderOutput, error) {
		// Use genkit.Run to create a trace span for a specific step
		status, err := genkit.Run(ctx, "lookup-db", func() (string, error) {
			return "Shipped", nil 
		})
		if err != nil {
			return nil, err
		}
		return &OrderOutput{Status: status}, nil
	},
)
```

### Defining a Tool
Tools are used by models. The description is crucial as the model uses it to decide when to call the tool.

```go
type WeatherInput struct {
	Location string `json:"location"`
}

var WeatherTool = genkit.DefineTool(g, "getWeather", "fetches current weather for a location",
	func(ctx *ai.ToolContext, input *WeatherInput) (string, error) {
		// Implementation logic
		return "Sunny", nil
	},
)
```

### Generation with Typed Data
Use `genkit.GenerateData` to get structured output from a model directly into a Go struct.

```go
type Recipe struct {
	Name        string   `json:"name"`
	Ingredients []string `json:"ingredients"`
}

recipe, _, err := genkit.GenerateData[Recipe](ctx, g,
	ai.WithModelName("googleai/gemini-2.0-flash"),
	ai.WithPrompt("Suggest a pancake recipe"),
)
```

## Code Quality & Linting

* **Run Linting**: Always run `go vet ./...` from the `go/` directory for all Go code changes.
* **Format Code**: Run `bin/fmt` (which runs `go fmt`) to ensure code is formatted correctly.
* **Pass All Tests**: Ensure all unit tests pass (`go test ./...`).
* **Production Ready**: The objective is to produce production-grade code.
* **Shift Left**: Employ a "shift left" strategy—catch errors early.
* **Strict Typing**: Go is statically typed. Do not use `interface{}` (or `any`) unless absolutely necessary and documented.
* **No Warning Suppression**: Avoid ignoring linter warnings unless there is a compelling, documented reason.
* Group imports: standard library first, then third-party, then internal. `goimports` handles this automatically.

## Generated Files & Data Model

* **Do Not Edit Generated Files**: Files generated by tools (like protobufs or strict-type generators) should not be modified directly.
* **Do Not Edit gen.go**: `go/ai/gen.go` is an auto-generated file. **DO NOT MODIFY IT DIRECTLY.**
* **Generator/Sanitizer**: Any necessary transformations to the core types must be applied to the generator script or the schema sanitizer.
* **Canonical Parity**: The data model MUST be identical to the JSON schema defined in the JavaScript (canonical) implementation.
* **Regenerating gen.go**: If updates to types in `go/ai/gen.go` are needed:
  1. Modify the Zod schemas in `genkit-tools/common/src/types/` (e.g., `model.ts` for `ToolDefinition`).
  2. Regenerate the JSON schema:

     ```bash
     cd genkit-tools && pnpm run export:schemas
     ```

  3. Regenerate the Go code:

     ```bash
     cd go/core && go run ../internal/cmd/jsonschemagen -outdir .. -config schemas.config ../../genkit-tools/genkit-schema.json ai
     ```

  * **Overwrite Risk**: Do not edit `genkit-tools/genkit-schema.json` directly. It is a generated file and will be overwritten by the `export:schemas` script.
  * **Note**: When adding new Part types, ensure `schemas.config` is updated to add the new Part type on all existing Parts and add "omit" to it accordingly.

## Detailed Coding Guidelines

### Target Environment

* **Go Version**: Target Go 1.24 or newer.
* **Environment**: Use `go mod` for dependency management.

### Typing & Style

* **Syntax**:
  * Use standard Go formatting (`gofmt`).
  * Use idioms like `if err != nil` for error handling.
  * Prefer short variable names for short scopes (e.g., `i` for index, `ctx` for context).
* **Interfaces**: Define interfaces where they are used (consumer-side), not where they are implemented (producer-side).
* **Concurrency**: Use channels and goroutines for concurrency. Avoid shared mutable state where possible.
* **Comments**:
  * Use proper punctuation.
  * Start function comments with the function name.
  * Use `// TODO(issue-id): Fix this later.` format for stubs.
* Ensure that `go vet` passes without errors.

### Documentation

* **Format**: Write comprehensive GoDoc-style comments for exported packages, types, and functions.
* **Content**:
  * **Explain Concepts**: Explain the terminology and concepts used in the code to someone unfamiliar with the code.
  * **Visuals**: Prefer using diagrams if helpful to explain complex flows.
* **Required Sections**:
  * **Overview**: Description of what the package/function does.
  * **Examples**: Provide examples for complex APIs (using `Example` functions in `_test.go` files is best practice).
* **External Packages**: **Use the `go doc` command to understand type and function definitions when working with external packages.**
* **References**:
  * Please use the descriptions from genkit.dev and github.com/genkit-ai/docsite as the source of truth for the API and concepts.
  * When you are not sure about the API or concepts, please refer to the JavaScript implementation for the same.
* Keep examples in documentation and comments simple.
* Add links to relevant documentation on the Web or elsewhere in the relevant places in comments.
* Always update package comments and function comments when updating code.
* Scan documentation for every package you edit and keep it up-to-date.

### Implementation

* Always add unit tests to improve coverage. Use Genkit primitives and helper
functions instead of mocking types.
* When aiming to achieve parity the API and behavior should be identical to the JS canonical implementation.
* Always add/update samples to demonstrate the usage of the API or functionality.
* Use default input values for flows and actions to make them easier to use.
* In the samples, explain the whys, hows, and whats of the sample in the package comment so the user learns more about the feature being demonstrated. Also explain how to test the sample.
* Avoid mentioning sample specific stuff in core framework or plugin code.
* Always check for missing dependencies in `go.mod` for each sample and add them if we're using them.
* When a plugin such as a model provider is updated or changes, please also update relevant documentation and samples.
* Avoid boilerplate comments in the code. Comments should tell why, not what.
* Always update the README.md (if exists) to match the updated code changes.
* Make sure to not leave any dead code or unused imports.

### Formatting

* **Tool**: Format code using `go fmt` (via `bin/fmt` or editor).
* **Line Length**: Go doesn't have a strict line length limit, but keep it reasonable (e.g., 80-100 characters).
* **Strings**: Wrap long lines and strings appropriately.

### Testing

* **Framework**: Use the standard `testing` package. **Do not use external assertion libraries (like `testify`) for new core code.**
  * *Note: Existing plugins may use `testify`, but prefer standard library for consistency.*
* **Assertions**: Use plain `if/else` blocks following the `want`/`got` pattern.
  * Example:

      ```go
      if got := func(); got != want {
          t.Errorf("func() = %v, want %v", got, want)
      }
      ```

  * For complex object comparisons (structs, slices, maps), use `github.com/google/go-cmp/cmp` (and `cmpopts` if needed).

      ```go
      if diff := cmp.Diff(want, got); diff != "" {
          t.Errorf("mismatch (-want +got):\n%s", diff)
      }
      ```

* **Scope**: Write comprehensive unit tests following the fail-fast approach.
* **Execution**: Run via `go test ./...`.
* **Porting**: Maintain 1:1 logic parity accurately if porting tests. Do not invent behavior.
* **Fixes**: Fix underlying code issues rather than special-casing tests.
* **Modernize**: Consider using `modernize` to update code to modern Go idioms (e.g., `slices`, `maps`) when fixing underlying issues.
* **Genkit Testing**:
  * **Test Actions Directly**: Use `flow.Run(ctx, input)` or `tool.Run(ctx, input)` to test the logic of your Genkit components.
  * **Verify Schemas**: Ensure that the `Action` returned by `DefineX` has the expected input and output schemas.
  * **Mock Models**: When testing Flows that call models, use a mock model implementation to ensure deterministic tests.
  * **Trace Inspection**: For complex flows, use tests to verify that `genkit.Run` steps are being executed as expected.

### Logging

* **Library**: Use `log/slog` (available in Go 1.21+) or the internal logger.
* **Format**: Use structured logging keys and values.

### Licensing

Include the Apache 2.0 license header at the top of each file (update year as needed):

```go
// Copyright [year] Google LLC
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
```

## Git commit message guidelines

* Please draft a plain-text commit message after you're done with changes.
* Please do not include absolute file paths as links in commit messages.
* Since lines starting with `#` are treated as comments, please use a simpler format for headings.
* Add a rationale paragraph explaining the why and the what before listing all the changes.
* Please use conventional commits for the format.
* For scope, please refer to release-please configuration if available.
* Keep it short and simple.
