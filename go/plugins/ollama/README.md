# Ollama Plugin

The Ollama plugin provides a unified interface to connect with locally hosted (or remote) models through the [Ollama](https://ollama.com/) API.

The plugin supports a wide range of capabilities, relying on the models you have pulled:

- **Language Models**: Chat and text generation models, including support for tools, vision (multimodal), and reasoning (thinking).
- **Embedding Models**: Text embeddings.

## Setup

### Installation

```bash
go get github.com/firebase/genkit/go/plugins/ollama
```

### Configuration

You can configure the Ollama plugin with your server address. By default, Ollama runs on `http://localhost:11434`.

```go
import (
 "context"

 "github.com/firebase/genkit/go/genkit"
 "github.com/firebase/genkit/go/plugins/ollama"
)

func main() {
 ctx := context.Background()

 // Initialize the Ollama plugin
 o := &ollama.Ollama{
  ServerAddress: "http://localhost:11434", // Required
  Timeout:       60,                       // Optional: Response timeout in seconds (default: 30)
 }

 g := genkit.Init(ctx, genkit.WithPlugins(o))
}
```

## Language Models

Because Ollama models are locally hosted and the available models depend on what the user has pulled (`ollama pull <model>`), the plugin doesn't pre-initialize any default models. You must define them explicitly.

### Defining a Model

To use an Ollama model, you must define it using `DefineModel` after initializing the plugin.

```go
// Define a chat model (e.g., tinyllama)
o.DefineModel(g, ollama.ModelDefinition{
 Name: "tinyllama",
 Type: "chat", // Use "chat" for interactive models, or "" for text-completion only
}, nil)

// You can now retrieve and use it
m := ollama.Model(g, "tinyllama")
```

### Basic Usage

```go
import (
 "context"
 "fmt"
 "log"

 "github.com/firebase/genkit/go/ai"
 "github.com/firebase/genkit/go/genkit"
 "github.com/firebase/genkit/go/plugins/ollama"
)

func main() {
 // ... Init genkit with ollama plugin and define "tinyllama" ...

 m := ollama.Model(g, "tinyllama")
 resp, err := genkit.Generate(ctx, g,
  ai.WithModel(m),
  ai.WithPrompt("Explain how neural networks learn in simple terms."),
 )
 if err != nil {
  log.Fatal(err)
 }

 fmt.Println(resp.Text())
}
```

### Configuration Options

You can pass advanced parameters to Ollama models using `ollama.GenerateContentConfig`.

:::note
The plugin provides an `ollama.Ptr` helper function to easily initialize optional numeric fields (like `Temperature` or `Seed`).
:::

```go
import "github.com/firebase/genkit/go/plugins/ollama"

resp, err := genkit.Generate(ctx, g,
 ai.WithModel(m),
 ai.WithPrompt("Write a poem about the sea."),
 ai.WithConfig(&ollama.GenerateContentConfig{
  Temperature: ollama.Ptr(0.8),
  TopK:        ollama.Ptr(40),
  TopP:        ollama.Ptr(0.9),
  NumPredict:  ollama.Ptr(100), // Max output tokens
  KeepAlive:   "5m",           // Keep model loaded in memory for 5 minutes
  Seed:        ollama.Ptr(42),
 }),
)
```

### Tool Calling

Ollama supports tool calling for specific models. Models like `llama3.1` or `mistral` are excellent for this.

```go
// Define a model that supports tools
o.DefineModel(g, ollama.ModelDefinition{
 Name: "llama3.1",
 Type: "chat",
}, nil)

m := ollama.Model(g, "llama3.1")

// Define a tool
weatherTool := genkit.DefineTool(g, "getWeather", "gets the weather",
 func(ctx *ai.ToolContext, input *WeatherInput) (*WeatherOutput, error) {
  return &WeatherOutput{Temp: 72}, nil
 },
)

resp, err := genkit.Generate(ctx, g,
 ai.WithModel(m),
 ai.WithPrompt("What is the weather in New York?"),
 ai.WithTools(weatherTool),
)
```

:::note
The Ollama plugin supports receiving text content alongside tool calls. If a model returns both a text response and a tool request, both will be preserved in the message content.
:::

### Thinking and Reasoning

The plugin supports models with reasoning capabilities (like `deepseek-r1`).

```go
resp, err := genkit.Generate(ctx, g,
 ai.WithModel(m),
 ai.WithPrompt("What is heavier, one kilo of steel or one kilo of feathers?"),
 ai.WithConfig(&ollama.GenerateContentConfig{
  Think: ollama.ThinkEnabled(true), // Enable thinking mode for supported models
 }),
)

// The model's reasoning process is returned as a specific part
fmt.Println("Reasoning:", resp.Reasoning())
fmt.Println("Final Answer:", resp.Text())
```

#### Note on Streaming Reasoning

For models that output reasoning using `<think>` tags within standard text (instead of Ollama's native `thinking` API field), the tags will be automatically parsed and mapped to Genkit's `ReasoningPart` in **non-streaming** mode.

In **streaming** mode, due to the nature of token-by-token chunks, these tags will be streamed as standard `TextPart` content. Models that natively support Ollama's `thinking` field will stream `ReasoningPart` chunks correctly in both modes.

### Multimodal Input Capabilities

The plugin supports multimodal models (like `llava`) for image analysis.

```go
// Define a multimodal model
o.DefineModel(g, ollama.ModelDefinition{
 Name: "llava",
 Type: "chat",
}, nil)

m := ollama.Model(g, "llava")

// Using inline data (base64)
imagePart := ai.NewMediaPart("image/jpeg", "data:image/jpeg;base64,...")

resp, err := genkit.Generate(ctx, g,
 ai.WithModel(m),
 ai.WithMessages(
  ai.NewUserMessage(
   ai.NewTextPart("Describe this image"),
   imagePart,
  ),
 ),
)
```

## Embedding Models

You can define and use text embedding models hosted on Ollama (e.g., `nomic-embed-text`).

### Defining an Embedder

```go
// Define an embedder model
o.DefineEmbedder(g, "http://localhost:11434", "nomic-embed-text", nil)

embedder := ollama.Embedder(g, "http://localhost:11434")
```

### Usage

```go
res, err := genkit.Embed(ctx, g,
 ai.WithEmbedder(embedder),
 ai.WithTextDocs("Machine learning models process data to make predictions."),
)
if err != nil {
 log.Fatal(err)
}

fmt.Printf("Embedding length: %d\n", len(res.Embeddings[0].Embedding))
```
