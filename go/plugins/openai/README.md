# OpenAI Plugin for Genkit

The OpenAI plugin for Genkit provides integration with the [OpenAI API](https://platform.openai.com/docs/api-reference), allowing you to use models like GPT-4o, GPT-4 Turbo, and embeddings within your Genkit applications.

> [!NOTE]
> This plugin uses the modern OpenAI **Responses API**. For applications requiring the legacy **Chat Completions API**, please use the [`compat_oai`](../compat_oai) plugin.

This plugin uses the official [openai-go](https://github.com/openai/openai-go) SDK.

## Setup

First, ensure you have your OpenAI API key set in your environment variables:

```bash
export OPENAI_API_KEY=your-api-key
```

Then, initialize the plugin in your Genkit application:

```go
import (
 "context"
 "log"

 "github.com/firebase/genkit/go/genkit"
 "github.com/firebase/genkit/go/plugins/openai"
)

func main() {
 ctx := context.Background()

 // Initialize Genkit with the OpenAI plugin
 g := genkit.Init(ctx, genkit.WithPlugins(&openai.OpenAI{}))

 // ...
}
```

## OpenAI-Compatible Providers

You can use this plugin with other providers that support the OpenAI API by configuring the `BaseURL` and `Provider` fields.

### Groq

```go
g := genkit.Init(ctx, genkit.WithPlugins(&openai.OpenAI{
 BaseURL:  "https://api.groq.com/openai/v1",
 APIKey:   os.Getenv("GROQ_API_KEY"),
 Provider: "groq",
}))

model := openai.Model(g, "llama-3.3-70b-versatile")
```

### Mistral

```go
g := genkit.Init(ctx, genkit.WithPlugins(&openai.OpenAI{
 BaseURL:  "https://api.mistral.ai/v1",
 APIKey:   os.Getenv("MISTRAL_API_KEY"),
 Provider: "mistral",
}))

model := openai.Model(g, "mistral-large-latest")
```

### xAI (Grok)

```go
g := genkit.Init(ctx, genkit.WithPlugins(&openai.OpenAI{
 BaseURL:  "https://api.x.ai/v1",
 APIKey:   os.Getenv("XAI_API_KEY"),
 Provider: "xai",
}))

model := openai.Model(g, "grok-4-1-fast-reasoning")
```

### OpenRouter

```go
g := genkit.Init(ctx, genkit.WithPlugins(&openai.OpenAI{
 BaseURL:  "https://openrouter.ai/api/v1",
 APIKey:   os.Getenv("OPENROUTER_API_KEY"),
 Provider: "openrouter",
}))

model := openai.Model(g, "deepseek/deepseek-r1")
```

## Usage

### Generative Models

You can reference OpenAI models using the `openai.Model` helper.

```go
import (
 "context"
 "fmt"
 "log"

 "github.com/firebase/genkit/go/ai"
 "github.com/firebase/genkit/go/genkit"
 "github.com/firebase/genkit/go/plugins/openai"
)

func main() {
 ctx := context.Background()
 g := genkit.Init(ctx, genkit.WithPlugins(&openai.OpenAI{}))

 // Reference a model (e.g., gpt-4o)
 model := openai.Model(g, "gpt-4o")

 // Generate text
 resp, err := genkit.GenerateText(ctx, g,
  ai.WithModel(model),
  ai.WithPrompt("Tell me a joke about Go programming."),
 )
 if err != nil {
  log.Fatal(err)
 }

 fmt.Println(resp)
}
```

### Model Configuration

Pass configuration options like temperature and max output tokens using `ai.WithConfig`:

```go
import (
 "github.com/openai/openai-go/v3"
 "github.com/openai/openai-go/v3/responses"
)

// ...

resp, err := genkit.Generate(ctx, g,
 ai.WithModel(model),
 ai.WithPrompt("Write a creative story."),
 ai.WithConfig(&responses.ResponseNewParams{
  Temperature:     openai.Float(1.2),
  MaxOutputTokens: openai.Int(500),
 }),
)
```

### Streaming

Stream responses for real-time interactions:

```go
import (
 "context"
 "fmt"

 "github.com/firebase/genkit/go/ai"
 "github.com/firebase/genkit/go/genkit"
 "github.com/firebase/genkit/go/plugins/openai"
)

// ...

model := openai.Model(g, "gpt-4o")

_, err := genkit.Generate(ctx, g,
 ai.WithModel(model),
 ai.WithPrompt("Write a long poem about the sea."),
 ai.WithStreaming(func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
  fmt.Print(chunk.Text())
  return nil
 }),
)
```

### Structured Output

Request structured data (JSON) using `WithOutputType`. This leverages OpenAI's structured output capabilities.

```go
type MovieReview struct {
 Title  string `json:"title"`
 Rating int    `json:"rating"`
 Reason string `json:"reason"`
}

// ...

model := openai.Model(g, "gpt-4o")

resp, err := genkit.Generate(ctx, g,
 ai.WithModel(model),
 ai.WithPrompt("Review the movie 'Inception'"),
 ai.WithOutputType(MovieReview{}),
)
if err != nil {
 log.Fatal(err)
}

var review MovieReview
if err := resp.Output(&review); err != nil {
 log.Fatal(err)
}

fmt.Printf("Rating: %d/10\nReason: %s\n", review.Rating, review.Reason)
```

### Tool Calling

Register tools and provide them to the model.

```go
// Define a tool
weatherTool := genkit.DefineTool(g, "weather", "Returns the weather for the given location",
 func(ctx *ai.ToolContext, input struct {
  Location string `json:"location"`
 }) (string, error) {
  // Call actual weather API here
  return fmt.Sprintf("The weather in %s is sunny", input.Location), nil
 },
)

// Use the tool in generation
resp, err := genkit.Generate(ctx, g,
 ai.WithModel(model),
 ai.WithPrompt("What is the weather in San Francisco?"),
 ai.WithTools(weatherTool),
)
```

### Multimodal Input

Pass images to models that support vision (like `gpt-4o`).

```go
import "github.com/firebase/genkit/go/ai"

// ...

resp, err := genkit.Generate(ctx, g,
 ai.WithModel(model),
 ai.WithMessages(
  ai.NewUserMessage(
   ai.NewTextPart("Describe this image"),
   ai.NewMediaPart("image/jpeg", "data:image/jpeg;base64,iVBORw0KGgo..."),
  ),
 ),
)
```

### Embeddings

Use OpenAI embedding models for vector search tasks.

```go
import "github.com/firebase/genkit/go/plugins/openai"

// ...

embedder := openai.Embedder(g, "text-embedding-3-small")

res, err := ai.Embed(ctx, g,
 ai.WithEmbedder(embedder),
 ai.WithTextDocs("Genkit is awesome!"),
)
```
