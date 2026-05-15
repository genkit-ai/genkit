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

package genkit_test

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/genkit"
)

// This example shows basic initialization and flow definition.
func Example() {
	ctx := context.Background()

	// Initialize Genkit (without plugins for this example)
	g := genkit.Init(ctx)

	// Define a simple flow
	greetFlow := genkit.DefineFlow(g, "greet",
		func(ctx context.Context, name string) (string, error) {
			return fmt.Sprintf("Hello, %s!", name), nil
		},
	)

	// Run the flow
	greeting, err := greetFlow.Run(ctx, "World")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(greeting)
	// Output: Hello, World!
}

// This example demonstrates defining a simple non-streaming flow.
func ExampleDefineFlow() {
	ctx := context.Background()
	g := genkit.Init(ctx)

	// Define a flow that processes input
	uppercaseFlow := genkit.DefineFlow(g, "uppercase",
		func(ctx context.Context, input string) (string, error) {
			return strings.ToUpper(input), nil
		},
	)

	// Run the flow
	result, err := uppercaseFlow.Run(ctx, "hello")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result)
	// Output: HELLO
}

// This example demonstrates defining a streaming flow that sends
// chunks to the caller as they are produced.
func ExampleDefineStreamingFlow() {
	ctx := context.Background()
	g := genkit.Init(ctx)

	// Define a streaming flow that counts down
	countdownFlow := genkit.DefineStreamingFlow(g, "countdown",
		func(ctx context.Context, start int, sendChunk func(context.Context, int) error) (string, error) {
			for i := start; i > 0; i-- {
				if err := sendChunk(ctx, i); err != nil {
					return "", err
				}
			}
			return "Liftoff!", nil
		},
	)

	// Stream results using the iterator
	iter := countdownFlow.Stream(ctx, 3)
	iter(func(val *core.StreamingFlowValue[string, int], err error) bool {
		if err != nil {
			log.Fatal(err)
		}
		if val.Done {
			fmt.Println("Final:", val.Output)
		} else {
			fmt.Println("Count:", val.Stream)
		}
		return true
	})
	// Output:
	// Count: 3
	// Count: 2
	// Count: 1
	// Final: Liftoff!
}

// This example demonstrates using Run to create traced sub-steps
// within a flow for better observability.
func ExampleRun() {
	ctx := context.Background()
	g := genkit.Init(ctx)

	// Define a flow with traced sub-steps
	pipelineFlow := genkit.DefineFlow(g, "pipeline",
		func(ctx context.Context, input string) (string, error) {
			// Each Run call creates a traced step visible in the Dev UI
			upper, err := genkit.Run(ctx, "uppercase", func() (string, error) {
				return strings.ToUpper(input), nil
			})
			if err != nil {
				return "", err
			}

			result, err := genkit.Run(ctx, "addPrefix", func() (string, error) {
				return "Processed: " + upper, nil
			})
			return result, err
		},
	)

	result, err := pipelineFlow.Run(ctx, "hello")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(result)
	// Output: Processed: HELLO
}

// This example demonstrates defining a tool that models can call
// during generation.
func ExampleDefineTool() {
	ctx := context.Background()
	g := genkit.Init(ctx)

	// Define a tool that adds two numbers
	_ = genkit.DefineTool(g, "add",
		"Adds two numbers together",
		func(ctx *ai.ToolContext, input struct {
			A float64 `json:"a" jsonschema:"description=First number"`
			B float64 `json:"b" jsonschema:"description=Second number"`
		}) (float64, error) {
			return input.A + input.B, nil
		},
	)

	// The tool is now registered and can be used with ai.WithTools()
	// when calling genkit.Generate()
	fmt.Println("Tool registered: add")
	// Output: Tool registered: add
}

// This example demonstrates defining a reusable prompt with a template.
func ExampleDefinePrompt() {
	ctx := context.Background()
	g := genkit.Init(ctx)

	// Define a prompt with Handlebars template syntax
	prompt := genkit.DefinePrompt(g, "greeting",
		ai.WithPrompt("Say hello to {{name}} in a {{style}} way."),
	)

	// Render the prompt (without executing - useful for inspection)
	rendered, err := prompt.Render(ctx, map[string]any{
		"name":  "Alice",
		"style": "friendly",
	})
	if err != nil {
		log.Fatal(err)
	}
	// The rendered prompt contains the messages that would be sent
	fmt.Println(rendered.Messages[0].Content[0].Text)
	// Output: Say hello to Alice in a friendly way.
}

// This example demonstrates registering a Go type as a named schema.
func ExampleDefineSchemaFor() {
	ctx := context.Background()
	g := genkit.Init(ctx)

	// Define a struct type
	type Person struct {
		Name string `json:"name" jsonschema:"description=The person's name"`
		Age  int    `json:"age" jsonschema:"description=The person's age"`
	}

	// Register the schema - this makes it available for .prompt files
	// that reference it by name (e.g., "output: { schema: Person }")
	genkit.DefineSchemaFor[Person](g)

	fmt.Println("Schema registered: Person")
	// Output: Schema registered: Person
}

// This example demonstrates creating an HTTP server that exposes
// all registered flows as endpoints.
func ExampleListFlows_httpServer() {
	ctx := context.Background()
	g := genkit.Init(ctx)

	// Define some flows
	genkit.DefineFlow(g, "echo", func(ctx context.Context, s string) (string, error) {
		return s, nil
	})

	genkit.DefineFlow(g, "reverse", func(ctx context.Context, s string) (string, error) {
		runes := []rune(s)
		for i, j := 0, len(runes)-1; i < j; i, j = i+1, j-1 {
			runes[i], runes[j] = runes[j], runes[i]
		}
		return string(runes), nil
	})

	// Create HTTP handlers for all flows
	mux := http.NewServeMux()
	for _, flow := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+flow.Name(), genkit.Handler(flow))
	}

	// The mux now has:
	// - POST /echo
	// - POST /reverse
	fmt.Printf("Registered %d flow handlers\n", len(genkit.ListFlows(g)))
	// Output: Registered 2 flow handlers
}

// This example demonstrates using Handler to expose a single flow
// as an HTTP endpoint.
func ExampleHandler() {
	ctx := context.Background()
	g := genkit.Init(ctx)

	// Define a flow
	greetFlow := genkit.DefineFlow(g, "greet",
		func(ctx context.Context, name string) (string, error) {
			return fmt.Sprintf("Hello, %s!", name), nil
		},
	)

	// Create an HTTP handler for the flow
	mux := http.NewServeMux()
	mux.HandleFunc("POST /greet", genkit.Handler(greetFlow))

	// The handler accepts JSON: {"data": "World"}
	// and returns JSON: {"result": "Hello, World!"}
	fmt.Println("Handler registered at POST /greet")
	// Output: Handler registered at POST /greet
}

// This example demonstrates using type-safe data prompts with
// strongly-typed input and output.
func ExampleDefineDataPrompt() {
	ctx := context.Background()
	g := genkit.Init(ctx)

	// Define input and output types
	type JokeRequest struct {
		Topic string `json:"topic"`
	}

	type Joke struct {
		Setup     string `json:"setup"`
		Punchline string `json:"punchline"`
	}

	// Define a type-safe prompt
	// Note: In production, you'd also set ai.WithModel(...)
	_ = genkit.DefineDataPrompt[JokeRequest, *Joke](g, "joke",
		ai.WithPrompt("Tell a joke about {{topic}}. Return JSON with setup and punchline."),
	)

	// The prompt can now be executed with:
	// for result, err := range jokePrompt.ExecuteStream(ctx, JokeRequest{Topic: "cats"}) {
	//     if result.Done {
	//         fmt.Println(result.Output.Setup)
	//         fmt.Println(result.Output.Punchline)
	//     }
	// }

	fmt.Println("DataPrompt registered: joke")
	// Output: DataPrompt registered: joke
}

// This example demonstrates looking up a prompt that was loaded
// from a .prompt file.
func ExampleLookupPrompt() {
	ctx := context.Background()

	// In production, you would initialize with a prompt directory:
	// g := genkit.Init(ctx, genkit.WithPromptDir("./prompts"))

	g := genkit.Init(ctx)

	// Define a prompt programmatically (simulating a loaded prompt)
	genkit.DefinePrompt(g, "greeting",
		ai.WithPrompt("Hello {{name}}!"),
	)

	// Look up the prompt by name
	prompt := genkit.LookupPrompt(g, "greeting")
	if prompt == nil {
		log.Fatal("Prompt not found")
	}

	fmt.Println("Found prompt:", prompt.Name())
	// Output: Found prompt: greeting
}
