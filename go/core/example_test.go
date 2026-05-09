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

package core_test

import (
	"context"
	"fmt"
	"strings"

	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/internal/registry"
)

// This example demonstrates defining a simple flow.
func ExampleDefineFlow() {
	r := registry.New()

	// Define a flow that processes input
	flow := core.DefineFlow(r, "uppercase",
		func(ctx context.Context, input string) (string, error) {
			return strings.ToUpper(input), nil
		},
	)

	// Run the flow
	result, err := flow.Run(context.Background(), "hello")
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	fmt.Println(result)
	// Output: HELLO
}

// This example demonstrates defining a streaming flow.
func ExampleDefineStreamingFlow() {
	r := registry.New()

	// Define a streaming flow that counts down
	flow := core.DefineStreamingFlow(r, "countdown",
		func(ctx context.Context, start int, cb core.StreamCallback[int]) (string, error) {
			for i := start; i > 0; i-- {
				if cb != nil {
					if err := cb(ctx, i); err != nil {
						return "", err
					}
				}
			}
			return "Done!", nil
		},
	)

	// Use Stream() iterator to receive chunks
	iter := flow.Stream(context.Background(), 3)
	iter(func(val *core.StreamingFlowValue[string, int], err error) bool {
		if err != nil {
			fmt.Println("Error:", err)
			return false
		}
		if val.Done {
			fmt.Println("Result:", val.Output)
		} else {
			fmt.Println("Count:", val.Stream)
		}
		return true
	})
	// Output:
	// Count: 3
	// Count: 2
	// Count: 1
	// Result: Done!
}

// This example demonstrates using Run to create traced sub-steps.
func ExampleRun() {
	r := registry.New()

	// Define a flow that uses Run for traced steps
	flow := core.DefineFlow(r, "pipeline",
		func(ctx context.Context, input string) (string, error) {
			// Each Run creates a traced step visible in the Dev UI
			upper, err := core.Run(ctx, "toUpper", func() (string, error) {
				return strings.ToUpper(input), nil
			})
			if err != nil {
				return "", err
			}

			result, err := core.Run(ctx, "addPrefix", func() (string, error) {
				return "RESULT: " + upper, nil
			})
			return result, err
		},
	)

	result, err := flow.Run(context.Background(), "hello")
	if err != nil {
		fmt.Println("Error:", err)
		return
	}
	fmt.Println(result)
	// Output: RESULT: HELLO
}

// This example demonstrates defining a schema from a Go type.
func ExampleDefineSchemaFor() {
	r := registry.New()

	// Define a struct type
	type Person struct {
		Name string `json:"name"`
		Age  int    `json:"age"`
	}

	// Register the schema
	core.DefineSchemaFor[Person](r)

	// The schema is now registered and can be referenced in .prompt files
	fmt.Println("Schema registered")
	// Output: Schema registered
}

// This example demonstrates defining a schema from a map.
func ExampleDefineSchema() {
	r := registry.New()

	// Define a JSON schema as a map
	core.DefineSchema(r, "Address", map[string]any{
		"type": "object",
		"properties": map[string]any{
			"street": map[string]any{"type": "string"},
			"city":   map[string]any{"type": "string"},
			"zip":    map[string]any{"type": "string"},
		},
		"required": []any{"street", "city"},
	})

	fmt.Println("Schema registered: Address")
	// Output: Schema registered: Address
}

// This example demonstrates using ChainMiddleware to combine middleware.
func ExampleChainMiddleware() {
	// Define a middleware that wraps function calls
	logMiddleware := func(next core.StreamingFunc[string, string, struct{}]) core.StreamingFunc[string, string, struct{}] {
		return func(ctx context.Context, input string, cb core.StreamCallback[struct{}]) (string, error) {
			fmt.Println("Before:", input)
			result, err := next(ctx, input, cb)
			fmt.Println("After:", result)
			return result, err
		}
	}

	// The original function
	originalFn := func(ctx context.Context, input string, cb core.StreamCallback[struct{}]) (string, error) {
		return strings.ToUpper(input), nil
	}

	// Chain and apply middleware
	wrapped := core.ChainMiddleware(logMiddleware)(originalFn)

	result, _ := wrapped(context.Background(), "hello", nil)
	fmt.Println("Final:", result)
	// Output:
	// Before: hello
	// After: HELLO
	// Final: HELLO
}

// This example demonstrates creating user-facing errors.
func ExampleNewPublicError() {
	// Create a user-facing error with details
	err := core.NewPublicError(core.INVALID_ARGUMENT, "Invalid email format", map[string]any{
		"field": "email",
		"value": "not-an-email",
	})

	fmt.Println("Status:", err.Status)
	fmt.Println("Message:", err.Message)
	// Output:
	// Status: INVALID_ARGUMENT
	// Message: Invalid email format
}
