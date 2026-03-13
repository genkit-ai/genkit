<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="../docs/resources/genkit-logo-dark.png">
    <img alt="Genkit logo" src="../docs/resources/genkit-logo.png" width="400">
  </picture>
  <br>
  <strong>Genkit Go</strong>
  <br>
  <em>AI SDK for Go &bull; LLM Framework &bull; AI Agent Toolkit</em>
</p>

<p align="center">
  <a href="https://pkg.go.dev/github.com/firebase/genkit/go"><img src="https://pkg.go.dev/badge/github.com/firebase/genkit/go.svg" alt="Go Reference"></a>
  <a href="https://goreportcard.com/report/github.com/firebase/genkit/go"><img src="https://goreportcard.com/badge/github.com/firebase/genkit/go" alt="Go Report Card"></a>
</p>

<p align="center">
  Build production-ready AI-powered applications in Go with a unified interface for text generation, structured output, tool calling, and agentic workflows.
</p>

<p align="center">
  <a href="https://genkit.dev/docs/overview/?lang=go">Documentation</a> &bull;
  <a href="https://pkg.go.dev/github.com/firebase/genkit/go">API Reference</a> &bull;
  <a href="https://discord.gg/qXt5zzQKpc">Discord</a>
</p>

---

## Installation

```bash
go get github.com/firebase/genkit/go
```

## Quick Start

Get up and running in under a minute:

```go
package main

import (
    "context"
    "fmt"

    "github.com/firebase/genkit/go/ai"
    "github.com/firebase/genkit/go/genkit"
    "github.com/firebase/genkit/go/plugins/googlegenai"
)

func main() {
    ctx := context.Background()
    g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))

    answer, err := genkit.GenerateText(ctx, g,
        ai.WithModelName("googleai/gemini-2.5-flash"),
        ai.WithPrompt("Why is Go a great language for AI applications?"),
    )
    if err != nil {
        fmt.Println("could not generate: %s", err)
    }
    fmt.Println(answer)
}
```

```bash
export GEMINI_API_KEY="your-api-key"
go run main.go
```

---

## Features

Genkit Go gives you everything you need to build AI applications with confidence.

### Generate Text

Call any model with a simple, unified API:

```go
text, _ := genkit.GenerateText(ctx, g,
    ai.WithModelName("googleai/gemini-2.5-flash"),
    ai.WithPrompt("Explain quantum computing in simple terms."),
)
fmt.Println(text)
```

### Generate Structured Data

Get type-safe JSON output that maps directly to your Go structs:

```go
type Recipe struct {
    Title       string   `json:"title"`
    Ingredients []string `json:"ingredients"`
    Steps       []string `json:"steps"`
}

recipe, _ := genkit.GenerateData[Recipe](ctx, g,
    ai.WithModelName("googleai/gemini-2.5-flash"),
    ai.WithPrompt("Create a recipe for chocolate chip cookies."),
)
fmt.Printf("Recipe: %s\n", recipe.Title)
```

[See full example](samples/basic-structured)

### Stream Responses

Stream text as it's generated for responsive user experiences:

```go
stream := genkit.GenerateStream(ctx, g,
    ai.WithModelName("googleai/gemini-2.5-flash"),
    ai.WithPrompt("Write a short story about a robot learning to paint."),
)

for result, err := range stream {
    if err != nil {
        log.Fatal(err)
    }
    if result.Done {
        break
    }
    fmt.Print(result.Chunk.Text())
}
```

[See full example](samples/basic-structured)

### Stream Structured Data

Stream typed JSON objects as they're being generated:

```go
type Ingredient struct {
    Name   string `json:"name"`
    Amount string `json:"amount"`
}

type Recipe struct {
    Title       string        `json:"title"`
    Ingredients []*Ingredient `json:"ingredients"`
}

stream := genkit.GenerateDataStream[*Recipe](ctx, g,
    ai.WithModelName("googleai/gemini-2.5-flash"),
    ai.WithPrompt("Create a recipe for spaghetti carbonara."),
)

for result, err := range stream {
    if err != nil {
        log.Fatal(err)
    }
    if result.Done {
        fmt.Printf("\nComplete recipe: %s\n", result.Output.Title)
        break
    }
    // Access partial data as it streams in
    if result.Chunk != nil && len(result.Chunk.Ingredients) > 0 {
        fmt.Printf("Found ingredient: %s\n", result.Chunk.Ingredients[0].Name)
    }
}
```

[See full example](samples/basic-structured)

### Define Tools

Give models the ability to take actions and access external data:

```go
type WeatherInput struct {
    Location string `json:"location"`
}

weatherTool := genkit.DefineTool(g, "getWeather",
    "Gets the current weather for a location",
    func(ctx *ai.ToolContext, input WeatherInput) (string, error) {
        // Call your weather API here
        return fmt.Sprintf("Weather in %s: 72°F and sunny", input.Location), nil
    },
)

response, _ := genkit.Generate(ctx, g,
    ai.WithModelName("googleai/gemini-2.5-flash"),
    ai.WithPrompt("What's the weather like in San Francisco?"),
    ai.WithTools(weatherTool),
)
fmt.Println(response.Text())
```

[See full example](samples/basic)

### Tool Interrupts

Pause execution for human approval, then resume with modified inputs or direct responses:

```go
type TransferInput struct {
    ToAccount string  `json:"toAccount"`
    Amount    float64 `json:"amount"`
}

type TransferInterrupt struct {
    Reason  string  `json:"reason"`
    Amount  float64 `json:"amount"`
    Balance float64 `json:"balance"`
}

transferTool := genkit.DefineTool(g, "transfer",
    "Transfer money to an account",
    func(ctx *ai.ToolContext, input TransferInput) (string, error) {
        // Confirm large transfers
        if !ctx.IsResumed() && input.Amount > 1000 {
            return "", ai.InterruptWith(ctx, TransferInterrupt{
                Reason:  "confirm_large",
                Amount:  input.Amount,
                Balance: currentBalance,
            })
        }
        return "Transfer completed", nil
    },
)

// Handle interrupts in your flow
resp, _ := genkit.Generate(ctx, g,
    ai.WithModelName("googleai/gemini-2.5-flash"),
    ai.WithPrompt("Transfer $5000 to account ABC123"),
    ai.WithTools(transferTool),
)

if resp.FinishReason == ai.FinishReasonInterrupted {
    for _, interrupt := range resp.Interrupts() {
        meta, _ := ai.InterruptAs[TransferInterrupt](interrupt)

        // Get user confirmation, then resume
        part, _ := transferTool.RestartWith(interrupt)
        resp, _ = genkit.Generate(ctx, g,
            ai.WithMessages(resp.History()...),
            ai.WithTools(transferTool),
            ai.WithToolRestarts(part),
        )
    }
}
```

[See full example](samples/intermediate-interrupts)

### Define Flows

Wrap your AI logic in flows for better observability, testing, and deployment:

```go
jokeFlow := genkit.DefineFlow(g, "tellJoke",
    func(ctx context.Context, topic string) (string, error) {
        return genkit.GenerateText(ctx, g,
            ai.WithModelName("googleai/gemini-2.5-flash"),
            ai.WithPrompt("Tell me a joke about %s", topic),
        )
    },
)

joke, _ := jokeFlow.Run(ctx, "programming")
fmt.Println(joke)
```

[See full example](samples/basic)

### Streaming Flows

Stream data from your flows using Server-Sent Events (SSE):

```go
genkit.DefineStreamingFlow(g, "streamStory",
    func(ctx context.Context, topic string, send core.StreamCallback[string]) (string, error) {
        stream := genkit.GenerateStream(ctx, g,
            ai.WithModelName("googleai/gemini-2.5-flash"),
            ai.WithPrompt("Write a story about %s", topic),
        )

        for result, err := range stream {
            if err != nil {
                return "", err
            }
            if result.Done {
                return result.Response.Text(), nil
            }
            send(ctx, result.Chunk.Text())
        }
        return "", nil
    },
)
```

[See full example](samples/basic)

### Traced Sub-steps

Add observability to complex flows by breaking them into traced operations:

```go
genkit.DefineFlow(g, "processDocument",
    func(ctx context.Context, doc string) (string, error) {
        // Each Run call creates a traced step visible in the Dev UI
        summary, _ := genkit.Run(ctx, "summarize", func() (string, error) {
            return genkit.GenerateText(ctx, g,
                ai.WithModelName("googleai/gemini-2.5-flash"),
                ai.WithPrompt("Summarize: %s", doc),
            )
        })

        keywords, _ := genkit.Run(ctx, "extractKeywords", func() ([]string, error) {
            return genkit.GenerateData[[]string](ctx, g,
                ai.WithModelName("googleai/gemini-2.5-flash"),
                ai.WithPrompt("Extract keywords from: %s", summary),
            )
        })

        return fmt.Sprintf("Summary: %s\nKeywords: %v", summary, keywords), nil
    },
)
```

[See full example](samples/basic)

### Define Prompts

Create reusable prompts with Handlebars templating:

```go
greetingPrompt := genkit.DefinePrompt(g, "greeting",
    ai.WithModelName("googleai/gemini-2.5-flash"),
    ai.WithPrompt("Write a {{style}} greeting for {{name}}."),
)

response, _ := greetingPrompt.Execute(ctx, ai.WithInput(map[string]any{
    "name":  "Alice",
    "style": "formal",
}))
fmt.Println(response.Text())
```

[See full example](samples/basic-prompts)

### Type-Safe Data Prompts

Get compile-time type safety for your prompt inputs and outputs:

```go
type JokeRequest struct {
    Topic string `json:"topic"`
}

type Joke struct {
    Setup     string `json:"setup"`
    Punchline string `json:"punchline"`
}

jokePrompt := genkit.DefineDataPrompt[JokeRequest, *Joke](g, "joke",
    ai.WithModelName("googleai/gemini-2.5-flash"),
    ai.WithPrompt("Tell a joke about {{topic}}."),
)

for result, err := range jokePrompt.ExecuteStream(ctx, JokeRequest{Topic: "cats"}) {
    if err != nil {
        log.Fatal(err)
    }
    if result.Done {
        fmt.Printf("Punchline: %s\n", result.Output.Punchline)
        break
    }
    // Access typed partial data as it streams
    if result.Chunk != nil && result.Chunk.Setup != "" {
        fmt.Printf("Got setup: %s\n", result.Chunk.Setup)
    }
}
```

[See full example](samples/basic-prompts)

### Load Prompts from Files

Keep prompts separate from code using `.prompt` files with YAML frontmatter:

```yaml
# prompts/recipe.prompt
---
model: googleai/gemini-2.5-flash
input:
  schema: RecipeRequest
output:
  format: json
  schema: Recipe
---
{{role "system"}}
You are an experienced chef.

{{role "user"}}
Create a {{cuisine}} {{dish}} recipe for {{servingSize}} people.
{{#if dietaryRestrictions}}
Dietary restrictions: {{#each dietaryRestrictions}}{{this}}{{#unless @last}}, {{/unless}}{{/each}}.
{{/if}}
```

```go
// Register schemas so .prompt files can reference them by name
genkit.DefineSchemaFor[RecipeRequest](g)
genkit.DefineSchemaFor[Recipe](g)

// Look up and execute the prompt
recipePrompt := genkit.LookupDataPrompt[RecipeRequest, *Recipe](g, "recipe")
recipe, _ := recipePrompt.Execute(ctx, RecipeRequest{
    Dish:        "tacos",
    Cuisine:     "Mexican",
    ServingSize: 4,
})
fmt.Printf("%s (%s)\n", recipe.Title, recipe.PrepTime)
```

[See full example](samples/basic-prompts)

### Embed Prompts in Your Binary

Ship a single binary with prompts compiled in using Go's embed package:

```go
//go:embed prompts/*
var promptsFS embed.FS

func main() {
    ctx := context.Background()
    g := genkit.Init(ctx,
        genkit.WithPlugins(&googlegenai.GoogleAI{}),
        genkit.WithPromptFS(promptsFS),
    )

    prompt := genkit.LookupPrompt(g, "greeting")
    response, _ := prompt.Execute(ctx)
    fmt.Println(response.Text())
}
```

[See full example](samples/prompts-embed)

### Expose Flows as HTTP Endpoints

Serve your flows over HTTP with automatic JSON serialization:

```go
mux := http.NewServeMux()
for _, flow := range genkit.ListFlows(g) {
    mux.HandleFunc("POST /"+flow.Name(), genkit.Handler(flow))
}
log.Fatal(http.ListenAndServe(":8080", mux))
```

```bash
curl -X POST http://localhost:8080/tellJoke \
  -H "Content-Type: application/json" \
  -d '{"data": "programming"}'
```

### Works with Any HTTP Framework

`genkit.Handler` returns a standard `http.HandlerFunc`, so it works with any Go HTTP framework:

```go
// net/http (standard library)
mux := http.NewServeMux()
mux.HandleFunc("POST /joke", genkit.Handler(jokeFlow))
log.Fatal(http.ListenAndServe(":8080", mux))

// Gin
r := gin.Default()
r.POST("/joke", gin.WrapF(genkit.Handler(jokeFlow)))
r.Run(":8080")

// Echo
e := echo.New()
e.POST("/joke", echo.WrapHandler(genkit.Handler(jokeFlow)))
e.Start(":8080")

// Chi
r := chi.NewRouter()
r.Post("/joke", genkit.Handler(jokeFlow))
http.ListenAndServe(":8080", r)
```

### Frameworks with Centralized Error Handling

`genkit.HandlerFunc` returns:

```go
func(http.ResponseWriter, *http.Request) error
```

This is useful for frameworks that support centralized error handling and middleware chains, and expect handlers to return an error.

For example, Echo natively supports handlers that return error, so `genkit.HandlerFunc` can be adapted directly:

```go
e := echo.New()
h := genkit.HandlerFunc(jokeFlow)

e.POST("/joke", func(c *echo.Context) error {
	  return h(c.Response(), c.Request())
})

e.Start(":8080")
```

Any error returned by `genkit.HandlerFunc` will be handled by Echo's middleware stack.

---

## Model Providers

Genkit provides a unified interface across all major AI providers. Use whichever model fits your needs:

| Provider | Plugin | Models |
|----------|--------|--------|
| **Google AI** | `googlegenai.GoogleAI` | Gemini 2.5 Flash, Gemini 2.5 Pro, and more |
| **Vertex AI** | `vertexai.VertexAI` | Gemini models via Google Cloud |
| **Anthropic** | `anthropic.Anthropic` | Claude 3.5, Claude 3 Opus, and more |
| **Ollama** | `ollama.Ollama` | Llama, Mistral, and other open models |
| **OpenAI Compatible** | `compat_oai` | Any OpenAI-compatible API |

```go
// Google AI
g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))

// Anthropic
g := genkit.Init(ctx, genkit.WithPlugins(&anthropic.Anthropic{}))

// Ollama (local models)
g := genkit.Init(ctx, genkit.WithPlugins(&ollama.Ollama{
    ServerAddress: "http://localhost:11434",
}))

// Multiple providers at once
g := genkit.Init(ctx, genkit.WithPlugins(
    &googlegenai.GoogleAI{},
    &anthropic.Anthropic{},
))
```

Use `ai.WithModelName` for simple cases, or pair a model with provider-specific config using `ModelRef`:

```go
import "google.golang.org/genai"

// Simple: just the model name
response, _ := genkit.Generate(ctx, g,
    ai.WithModelName("googleai/gemini-2.5-flash"),
    ai.WithPrompt("Hello!"),
)

// Advanced: model name + provider-specific configuration
response, _ := genkit.Generate(ctx, g,
    ai.WithModel(googlegenai.ModelRef("googleai/gemini-2.5-flash", &genai.GenerateContentConfig{
        Temperature:     genai.Ptr(float32(0.7)),
        MaxOutputTokens: genai.Ptr(int32(1000)),
        TopP:            genai.Ptr(float32(0.9)),
    })),
    ai.WithPrompt("Hello!"),
)
```

---

## Development Tools

### Genkit CLI

Use the Genkit CLI to run your app with tracing and a local development UI:

```bash
curl -sL cli.genkit.dev | bash
genkit start -- go run main.go
```

### Developer UI

The local developer UI lets you:

- **Test flows** with different inputs interactively
- **Inspect traces** to debug complex multi-step operations
- **Compare models** by switching providers in real-time
- **Evaluate prompts** against datasets

---

## Experimental

These features are available in `core/x` and may change in future releases.

### Durable Streaming

Allow clients to reconnect to in-progress or completed streams using a stream ID:

```go
import "github.com/firebase/genkit/go/core/x/streaming"

mux.HandleFunc("POST /myFlow", genkit.Handler(myStreamingFlow,
    genkit.WithStreamManager(streaming.NewInMemoryStreamManager(
        streaming.WithTTL(10*time.Minute),
    )),
))
```

Clients receive a stream ID in the `X-Genkit-Stream-Id` header and can reconnect to replay buffered chunks.

[See full example](samples/durable-streaming)

### Sessions

Maintain typed state across multiple requests and throughout generation including tools:

```go
import "github.com/firebase/genkit/go/core/x/session"

type CartState struct {
    Items []string `json:"items"`
}

store := session.NewInMemoryStore[CartState]()

genkit.DefineFlow(g, "manageCart", func(ctx context.Context, input string) (string, error) {
    sess, err := session.Load(ctx, store, "session-id")
    if err != nil {
        sess, _ = session.New(ctx,
            session.WithID[CartState]("session-id"),
            session.WithStore(store),
            session.WithInitialState(CartState{}),
        )
    }
    ctx = session.NewContext(ctx, sess)

    // Tools can now access session state via session.FromContext[CartState](ctx)
    return genkit.GenerateText(ctx, g, ai.WithPrompt(input), ai.WithTools(myTools...))
})
```

[See full example](samples/session)

---

## Samples

Explore working examples to see Genkit in action:

| Sample | Description |
|--------|-------------|
| [basic](samples/basic) | Simple text generation with streaming |
| [basic-structured](samples/basic-structured) | Typed JSON output with `GenerateData` and `GenerateDataStream` |
| [basic-prompts](samples/basic-prompts) | Prompt templates with Handlebars and `.prompt` files |
| [intermediate-interrupts](samples/intermediate-interrupts) | Human-in-the-loop with tool interrupts |
| [prompts-embed](samples/prompts-embed) | Embed prompts in your binary |
| [durable-streaming](samples/durable-streaming) | Reconnectable streams with replay |
| [session](samples/session) | Stateful flows with typed session data |

---

<p align="center">
  Built by Google with contributions from the <a href="https://github.com/firebase/genkit/graphs/contributors">Open Source Community</a>
</p>
