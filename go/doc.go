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
Package genkit provides a framework for building AI-powered applications in Go.

Genkit is an open-source framework that helps you build, deploy, and monitor
production-ready AI features. It provides a unified interface for working with
large language models (LLMs), managing prompts, defining workflows, and integrating
with various AI service providers.

For comprehensive documentation, tutorials, and examples, visit https://genkit.dev

# Getting Started

Initialize Genkit with a plugin to connect to an AI provider:

	ctx := context.Background()
	g, err := genkit.Init(ctx,
		genkit.WithPlugins(&googlegenai.GoogleAI{}),
	)
	if err != nil {
		log.Fatal(err)
	}

Generate text with a simple prompt:

	text, err := g.GenerateText(ctx,
		ai.WithModelName("googleai/gemini-2.5-flash"),
		ai.WithPrompt("Tell me a joke"),
	)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(text)

# Models

Models represent AI language models that generate content. Use plugins to access
models from providers like Google AI, Vertex AI, Anthropic, or Ollama. Models are
referenced by name and can include provider-specific configuration:

	resp, err := g.Generate(ctx,
		ai.WithModelName("googleai/gemini-2.5-flash"),
		ai.WithPrompt("Explain quantum computing in simple terms"),
	)

You can set a default model during initialization:

	g, err := genkit.Init(ctx,
		genkit.WithPlugins(&googlegenai.GoogleAI{}),
		genkit.WithDefaultModel("googleai/gemini-2.5-flash"),
	)
	if err != nil {
		log.Fatal(err)
	}

# Flows

Flows are reusable, observable functions that orchestrate AI operations. They
provide automatic tracing, can be exposed as HTTP endpoints, and support both
streaming and non-streaming execution.

Define a simple flow:

	jokesFlow := g.DefineFlow("jokesFlow",
		func(ctx context.Context, topic string) (string, error) {
			return g.GenerateText(ctx,
				ai.WithPrompt("Share a joke about %s.", topic),
			)
		},
	)

	joke, err := jokesFlow.Run(ctx, "programming")

Define a streaming flow that sends chunks as they're generated:

	streamingFlow := g.DefineStreamingFlow("streamingJokes",
		func(ctx context.Context, topic string, sendChunk ai.ModelStreamCallback) (string, error) {
			resp, err := g.Generate(ctx,
				ai.WithPrompt("Share a joke about %s.", topic),
				ai.WithStreaming(sendChunk),
			)
			if err != nil {
				return "", err
			}
			return resp.Text(), nil
		},
	)

Use [Run] within flows to create traced sub-steps for observability:

	g.DefineFlow("pipeline",
		func(ctx context.Context, input string) (string, error) {
			result, err := genkit.Run(ctx, "processStep", func() (string, error) {
				return process(input), nil
			})
			return result, err
		},
	)

# Prompts

Prompts can be defined programmatically or loaded from .prompt files (Dotprompt format).
They encapsulate model configuration, input schemas, and template logic for reuse.

Define a prompt in code:

	type JokeRequest struct {
		Topic string `json:"topic"`
	}

	jokePrompt := g.DefinePrompt[JokeRequest]("joke",
		ai.WithModelName("googleai/gemini-flash-latest"),
		ai.WithPrompt("Share a joke about {{topic}}."),
	)

	text, resp, err := jokePrompt.Execute(ctx, JokeRequest{Topic: "cats"})
	if err != nil {
		return err
	}
	fmt.Println(text)

For prompts with structured output, use [Genkit.DefineDataPrompt]:

	type RecipeRequest struct {
		Cuisine     string `json:"cuisine"`
		Dish        string `json:"dish"`
		ServingSize int    `json:"servingSize"`
	}

	type Recipe struct {
		Title        string   `json:"title"`
		Ingredients  []string `json:"ingredients"`
		Instructions []string `json:"instructions"`
	}

	recipePrompt := g.DefineDataPrompt[RecipeRequest, *Recipe]("recipe",
		ai.WithSystem("You are an experienced chef."),
		ai.WithPrompt("Create a {{cuisine}} {{dish}} recipe for {{servingSize}} people."),
	)

	for result, err := range recipePrompt.ExecuteStream(ctx, RecipeRequest{
		Cuisine: "Italian", Dish: "pasta", ServingSize: 4,
	}) {
		// result.Chunk is *Recipe, result.Output is final *Recipe
	}

Load prompts from .prompt files by specifying a prompt directory:

	g, err := genkit.Init(ctx,
		genkit.WithPlugins(&googlegenai.GoogleAI{}),
		genkit.WithPromptDir("./prompts"),
	)
	if err != nil {
		log.Fatal(err)
	}

	// Look up a loaded prompt
	jokePrompt := g.LookupPrompt("joke")

	// Or with type parameters for structured I/O
	recipePrompt := g.LookupDataPrompt[RecipeRequest, *Recipe]("recipe")

When using .prompt files with custom output schemas, register the schema first:

	g.DefineSchemaFor[Recipe]()

# Tools

Tools extend model capabilities by allowing them to call functions during generation.
Define tools that the model can invoke to perform actions or retrieve information:

	weatherTool := g.DefineTool("getWeather",
		"Gets the current weather for a city",
		func(ctx context.Context, city string) (string, error) {
			// Fetch weather data...
			return "Sunny, 72°F", nil
		},
	)

	resp, err := g.Generate(ctx,
		ai.WithPrompt("What's the weather in Paris?"),
		ai.WithTools(weatherTool),
	)

# Structured Output

Generate structured data that conforms to Go types using [GenerateData] or
[GenerateDataStream]. Use jsonschema struct tags to provide descriptions and
constraints that help the model understand the expected output:

	type Joke struct {
		Joke     string `json:"joke" jsonschema:"description=The joke text"`
		Category string `json:"category" jsonschema:"description=The joke category"`
	}

	joke, resp, err := g.GenerateData[*Joke](ctx,
		ai.WithPrompt("Tell me a programming joke"),
	)

For streaming structured output:

	stream := g.GenerateDataStream[*Recipe](ctx,
		ai.WithPrompt("Create a pasta recipe"),
	)
	for result, err := range stream {
		if err != nil {
			return nil, err
		}
		if result.Done {
			return result.Output, nil
		}
		// result.Chunk contains partial Recipe as it streams
		fmt.Printf("Got %d ingredients so far\n", len(result.Chunk.Ingredients))
	}

# Streaming

Genkit supports streaming at multiple levels. Use [GenerateStream] for streaming
model responses:

	stream := g.GenerateStream(ctx,
		ai.WithPrompt("Write a short story"),
	)
	for result, err := range stream {
		if err != nil {
			log.Fatal(err)
		}
		if result.Done {
			fmt.Println("\n--- Complete ---")
		} else {
			fmt.Print(result.Chunk.Text())
		}
	}

Use [DefineStreamingFlow] for flows that stream custom data types:

	g.DefineStreamingFlow("countdown",
		func(ctx context.Context, count int, sendChunk func(context.Context, int) error) (string, error) {
			for i := count; i > 0; i-- {
				if err := sendChunk(ctx, i); err != nil {
					return "", err
				}
				time.Sleep(time.Second)
			}
			return "Liftoff!", nil
		},
	)

# Development Mode and Dev UI

Set GENKIT_ENV=dev to enable development features including the Reflection API
server that powers the Genkit Developer UI:

	$ export GENKIT_ENV=dev
	$ go run main.go

Then run the Dev UI to inspect flows, test prompts, and view traces:

	$ npx genkit start -- go run main.go

The Dev UI provides:
  - Interactive flow testing with input/output inspection
  - Prompt playground for iterating on prompts
  - Trace viewer for debugging and performance analysis
  - Action browser for exploring registered actions

# HTTP Server Integration

Expose flows as HTTP endpoints for production deployment using [Handler]:

	mux := http.NewServeMux()
	for _, flow := range g.ListFlows() {
		mux.HandleFunc("POST /"+flow.Name(), genkit.Handler(flow))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))

Handlers support streaming responses via Server-Sent Events when the client
sends Accept: text/event-stream. For durable streaming that survives reconnects,
use [WithStreamManager]:

	mux.HandleFunc("POST /countdown", genkit.Handler(countdown,
		genkit.WithStreamManager(streaming.NewInMemoryStreamManager(
			streaming.WithTTL(10*time.Minute),
		)),
	))

# Plugins

Genkit's functionality is extended through plugins that provide models, tools,
and other capabilities. Common plugins include:

  - googlegenai: Google AI (Gemini models)
  - vertexai: Google Cloud Vertex AI
  - ollama: Local Ollama models

Initialize plugins during [Init]:

	g, err := genkit.Init(ctx,
		genkit.WithPlugins(
			&googlegenai.GoogleAI{},
			&vertexai.VertexAI{ProjectID: "my-project"},
		),
	)
	if err != nil {
		log.Fatal(err)
	}

# Messages and Parts

Build conversation messages using helper functions from the [ai] package. These
are used with [ai.WithMessages] or when building custom conversation flows:

	// Create messages for a conversation
	messages := []*ai.Message{
		ai.NewSystemTextMessage("You are a helpful assistant."),
		ai.NewUserTextMessage("Hello!"),
		ai.NewModelTextMessage("Hi there! How can I help?"),
	}

	resp, err := g.Generate(ctx,
		ai.WithMessages(messages...),
		ai.WithPrompt("What can you do?"),
	)

For multi-modal content, combine text and media parts:

	userMsg := ai.NewUserMessage(
		ai.NewTextPart("What's in this image?"),
		ai.NewMediaPart("image/png", base64ImageData),
	)

Available message constructors in the [ai] package:

  - [ai.NewUserTextMessage], [ai.NewUserMessage]: User messages
  - [ai.NewModelTextMessage], [ai.NewModelMessage]: Model responses
  - [ai.NewSystemTextMessage], [ai.NewSystemMessage]: System instructions

Available part constructors in the [ai] package:

  - [ai.NewTextPart]: Text content
  - [ai.NewMediaPart]: Images, audio, video (base64-encoded)
  - [ai.NewDataPart]: Raw data strings
  - [ai.NewToolRequestPart], [ai.NewToolResponsePart]: Tool interactions

# Generation Options

Generation functions ([Generate], [GenerateText], [GenerateData], [GenerateStream])
accept options from the [ai] package to control behavior. The most common options:

Model and Configuration:

  - [ai.WithModel]: Specify the model (accepts [ai.ModelRef] or plugin model refs)
  - [ai.WithModelName]: Specify model by name string (e.g., "googleai/gemini-2.5-flash")
  - [ai.WithConfig]: Set generation parameters (temperature, max tokens, etc.)

Prompting:

  - [ai.WithPrompt]: Set the user prompt (supports format strings)
  - [ai.WithSystem]: Set system instructions
  - [ai.WithMessages]: Provide conversation history

Tools and Output:

  - [ai.WithTools]: Enable tools the model can call
  - [ai.WithOutputType]: Request structured output matching a Go type
  - [ai.WithOutputFormat]: Specify output format (json, text, etc.)

Streaming:

  - [ai.WithStreaming]: Enable streaming with a callback function

Example combining multiple options:

	resp, err := g.Generate(ctx,
		ai.WithModelName("googleai/gemini-2.5-flash"),
		ai.WithSystem("You are a helpful coding assistant."),
		ai.WithMessages(conversationHistory...),
		ai.WithPrompt("Explain this code: %s", code),
		ai.WithTools(searchTool, calculatorTool),
		// Config is provider-specific (e.g., genai.GenerateContentConfig for Google AI)
	)

# Unregistered Components

For advanced use cases, the [ai] package provides New* functions to create
components without registering them in Genkit. This is useful for plugins
or when you need to pass components directly:

  - [ai.NewTool]: Create an unregistered tool
  - [ai.NewModel]: Create an unregistered model
  - [ai.NewEmbedder]: Create an unregistered embedder

Use the corresponding Define* functions in this package to create and register
components for use with Genkit's action system, tracing, and Dev UI.

# Additional Resources

  - Documentation: https://genkit.dev
  - Go Getting Started: https://genkit.dev/go/docs/get-started-go
  - Samples: https://github.com/genkit-ai/genkit/tree/main/go/samples
  - GitHub: https://github.com/genkit-ai/genkit
*/
package genkit
