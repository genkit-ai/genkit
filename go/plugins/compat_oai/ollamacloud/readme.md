# Ollama Cloud Plugin

This plugin provides a simple interface for using Ollama Cloud services through OpenAI-compatible API.

## Supported Models

The plugin supports the following Ollama Cloud models:

### Large Language Models (Text)
- **GPT-OSS 20B** - `gpt-oss:20b`
- **GPT-OSS 120B** - `gpt-oss:120b`
- **Qwen3 Coder 480B** - `qwen3-coder:480b`
- **DeepSeek v3.1 671B** - `deepseek-v3.1:671b`
- **GLM-4.6** - `glm-4.6`
- **MiniMax M2** - `minimax-m2`
- **Kimi K2 1T** - `kimi-k2:1t`
- **Kimi K2 Thinking** - `kimi-k2-thinking`

### Multimodal Models (Vision + Text)
- **Qwen3 VL 235B Instruct** - `qwen3-vl:235b-instruct` - Vision-language model (images + text, tools)
- **Qwen3 VL 235B** - `qwen3-vl:235b` - Vision-language model (images + text, tools)

## Prerequisites

- Go installed on your system
- An Ollama Cloud API key

## Usage

Here's a simple example of how to use the Ollama Cloud plugin:

```go
import (
    "context"
    "os"
    "github.com/firebase/genkit/go/ai"
    "github.com/firebase/genkit/go/genkit"
    "github.com/firebase/genkit/go/plugins/compat_oai/ollamacloud"
    "github.com/openai/openai-go/option"
)

// Initialize the Ollama Cloud plugin with your API key
plugin := &ollamacloud.OllamaCloud{
    APIKey: "your-ollamacloud-api-key", // or use the OLLAMACLOUD_API_KEY environment variable
    Opts: []option.RequestOption{
        option.WithAPIKey("your-ollamacloud-api-key"),
    },
}

// Initialize Genkit with the OllamaCloud plugin
g := genkit.Init(ctx,
    genkit.WithDefaultModel("ollamacloud/gpt-oss:20b"),
    genkit.WithPlugins(plugin))

// Basic text generation
resp, err := genkit.Generate(ctx, g,
    ai.WithPromptText("Explain quantum computing in simple terms."))

// Use a multimodal model (image + text)
resp, err := genkit.Generate(ctx, g,
    ai.WithModelName("ollamacloud/qwen3-vl:235b-instruct"),
    ai.WithMessages(
        ai.NewUserMessage(
            ai.NewMediaPart("image/png", imageData),
            ai.NewTextPart("What do you see in this image?"),
        ),
    ))

// Use with tools
calculator := genkit.DefineTool(g, "calculator", "simple calculator",
    func(ctx *ai.ToolContext, input struct {
        Operation string  `json:"operation"` // "add", "subtract", "multiply", "divide"
        A         float64 `json:"a"`
        B         float64 `json:"b"`
    }) (float64, error) {
        switch input.Operation {
        case "add": return input.A + input.B, nil
        case "subtract": return input.A - input.B, nil
        case "multiply": return input.A * input.B, nil
        case "divide": 
            if input.B == 0 { return 0, fmt.Errorf("division by zero") }
            return input.A / input.B, nil
        }
        return 0, fmt.Errorf("unknown operation")
    })

resp, err := genkit.Generate(ctx, g,
    ai.WithPromptText("What is 15 * 23?"),
    ai.WithTools(calculator))

// Streaming responses
resp, err := genkit.Generate(ctx, g,
    ai.WithPromptText("Write a short story about space exploration."),
    ai.WithStreaming(func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
        for _, content := range chunk.Content {
            fmt.Print(content.Text)
        }
        return nil
    }))
```

## Environment Variables

- `OLLAMACLOUD_API_KEY`: Your Ollama Cloud API key (required)

The base URL defaults to `https://ollama.com/v1`. To override it (for example, when using a proxy),
pass a custom `option.WithBaseURL(...)` value via the plugin's `Opts` field.

## Running Tests

First, set your Ollama Cloud API key as an environment variable:

```bash
export OLLAMACLOUD_API_KEY=<your-api-key>
```

### Running All Tests

To run all tests in the directory:

```bash
go test -v .
```

### Running Tests from Specific Files

To run tests from a specific file:

```bash
# Run only the main plugin tests
go test -run "^TestPlugin"
```

### Running Individual Tests

To run a specific test case:

```bash
# Run only the basic completion test
go test -run "TestPlugin/basic completion"

# Run only the streaming test
go test -run "TestPlugin/streaming"

# Run only the tool usage test
go test -run "TestPlugin/tool usage"

# Run only the multimodal test
go test -run "TestPlugin/media part"
```

### Test Output Verbosity

Add the `-v` flag for verbose output:

```bash
go test -v -run "TestPlugin/streaming"
```

## Features

- ✅ **OpenAI-compatible API**: Uses the standard OpenAI SDK
- ✅ **Streaming responses**: Real-time streaming for better UX
- ✅ **Tool calling**: Function calling support for interactive applications
- ✅ **Multimodal support**: Vision models can process images + text
- ✅ **Multiple models**: Access to various open-source models
- ✅ **Genkit integration**: Seamless integration with Genkit framework

## Troubleshooting

### Common Issues

1. **API Key Error**: Make sure `OLLAMA_API_KEY` is set correctly
2. **Network Issues**: Check your internet connection and firewall settings
3. **Model Not Found**: Verify the model name is supported by Ollama Cloud
4. **Rate Limiting**: Check if you've hit API rate limits

### Error Messages

- `"ollamacloud plugin initialization failed: API key is required"`: Set your API key
- `"unexpected config type: string"`: Use proper OpenAI config types
- Network errors: Check API endpoint and network connectivity

Note: All live tests require the OLLAMA_API_KEY environment variable to be set. Tests will be skipped if the API key is not provided.
