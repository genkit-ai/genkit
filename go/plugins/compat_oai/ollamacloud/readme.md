# Ollama Cloud Plugin

This plugin provides a Genkit interface for Ollama Cloud through Ollama's OpenAI-compatible API.

## Supported Models

The plugin registers the direct API model IDs returned by `https://ollama.com/v1/models`. Do not add the local Ollama `-cloud` suffix when using this plugin.

### Text Models

| Model | ID | Tools |
| --- | --- | --- |
| Cogito 2.1 671B | `cogito-2.1:671b` | No |
| DeepSeek V3.1 671B | `deepseek-v3.1:671b` | Yes |
| DeepSeek V3.2 | `deepseek-v3.2` | Yes |
| DeepSeek V4 Flash | `deepseek-v4-flash` | Yes |
| DeepSeek V4 Pro | `deepseek-v4-pro` | Yes |
| Devstral 2 123B | `devstral-2:123b` | Yes |
| GLM-4.6 | `glm-4.6` | Yes |
| GLM-4.7 | `glm-4.7` | Yes |
| GLM-5 | `glm-5` | Yes |
| GLM-5.1 | `glm-5.1` | Yes |
| GPT-OSS 20B | `gpt-oss:20b` | Yes |
| GPT-OSS 120B | `gpt-oss:120b` | Yes |
| Kimi K2 1T | `kimi-k2:1t` | Yes |
| Kimi K2 Thinking | `kimi-k2-thinking` | Yes |
| MiniMax M2 | `minimax-m2` | Yes |
| MiniMax M2.1 | `minimax-m2.1` | Yes |
| MiniMax M2.5 | `minimax-m2.5` | Yes |
| MiniMax M2.7 | `minimax-m2.7` | Yes |
| Nemotron 3 Nano 30B | `nemotron-3-nano:30b` | Yes |
| Nemotron 3 Super | `nemotron-3-super` | Yes |
| Qwen3 Coder 480B | `qwen3-coder:480b` | Yes |
| Qwen3 Coder Next | `qwen3-coder-next` | Yes |
| Qwen3 Next 80B | `qwen3-next:80b` | Yes |
| RNJ-1 8B | `rnj-1:8b` | Yes |

### Vision Models

| Model | ID | Tools |
| --- | --- | --- |
| Devstral Small 2 24B | `devstral-small-2:24b` | Yes |
| Gemini 3 Flash Preview | `gemini-3-flash-preview` | Yes |
| Gemma 3 4B | `gemma3:4b` | No |
| Gemma 3 12B | `gemma3:12b` | No |
| Gemma 3 27B | `gemma3:27b` | No |
| Gemma 4 31B | `gemma4:31b` | Yes |
| Kimi K2.5 | `kimi-k2.5` | Yes |
| Kimi K2.6 | `kimi-k2.6` | Yes |
| Ministral 3 3B | `ministral-3:3b` | Yes |
| Ministral 3 8B | `ministral-3:8b` | Yes |
| Ministral 3 14B | `ministral-3:14b` | Yes |
| Mistral Large 3 675B | `mistral-large-3:675b` | Yes |
| Qwen3 VL 235B | `qwen3-vl:235b` | Yes |
| Qwen3 VL 235B Instruct | `qwen3-vl:235b-instruct` | Yes |
| Qwen3.5 397B | `qwen3.5:397b` | Yes |

The current compatibility layer maps image media inputs for vision models. It does not advertise audio support in Genkit metadata.

## Prerequisites

- Go installed on your system
- An Ollama Cloud API key

## Usage

Set `OLLAMACLOUD_API_KEY`, then initialize Genkit with the plugin:

```go
package main

import (
	"context"
	"fmt"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/compat_oai/ollamacloud"
)

func main() {
	ctx := context.Background()
	g := genkit.Init(ctx,
		genkit.WithDefaultModel("ollamacloud/gpt-oss:20b"),
		genkit.WithPlugins(&ollamacloud.OllamaCloud{}))

	resp, err := genkit.Generate(ctx, g,
		ai.WithPrompt("Explain quantum computing in simple terms."))
	if err != nil {
		panic(err)
	}
	fmt.Println(resp.Text())
}
```

Use a vision model with image input:

```go
resp, err := genkit.Generate(ctx, g,
	ai.WithModelName("ollamacloud/qwen3-vl:235b-instruct"),
	ai.WithMessages(
		ai.NewUserMessage(
			ai.NewMediaPart("image/png", imageData),
			ai.NewTextPart("What do you see in this image?"),
		),
	))
```

Use a tool-capable model with tools:

```go
calculator := genkit.DefineTool(g, "calculator", "simple calculator",
	func(ctx *ai.ToolContext, input struct {
		Operation string  `json:"operation"`
		A         float64 `json:"a"`
		B         float64 `json:"b"`
	}) (float64, error) {
		switch input.Operation {
		case "add":
			return input.A + input.B, nil
		case "subtract":
			return input.A - input.B, nil
		case "multiply":
			return input.A * input.B, nil
		case "divide":
			if input.B == 0 {
				return 0, fmt.Errorf("division by zero")
			}
			return input.A / input.B, nil
		}
		return 0, fmt.Errorf("unknown operation")
	})

resp, err := genkit.Generate(ctx, g,
	ai.WithModelName("ollamacloud/qwen3-coder:480b"),
	ai.WithPrompt("What is 15 * 23? Use the calculator tool."),
	ai.WithTools(calculator))
```

Stream responses:

```go
resp, err := genkit.Generate(ctx, g,
	ai.WithPrompt("Write a short story about space exploration."),
	ai.WithStreaming(func(ctx context.Context, chunk *ai.ModelResponseChunk) error {
		fmt.Print(chunk.Text())
		return nil
	}))
```

## Environment Variables

- `OLLAMACLOUD_API_KEY`: Your Ollama Cloud API key. Required unless `APIKey` is set on the plugin.

The base URL defaults to `https://ollama.com/v1`. To override it, for example when using a proxy, pass a custom `option.WithBaseURL(...)` value through the plugin's `Opts` field.

## Running Tests

First, set your Ollama Cloud API key as an environment variable:

```bash
export OLLAMACLOUD_API_KEY=<your-api-key>
```

Run all tests in the directory:

```bash
go test -v .
```

Run individual live test cases:

```bash
go test -run "TestPlugin/basic completion"
go test -run "TestPlugin/streaming"
go test -run "TestPlugin/tool usage"
go test -run "TestPlugin/media part"
```

## Features

- OpenAI-compatible API through the OpenAI Go SDK
- Streaming responses
- Tool calling for models tagged with `tools`
- Image input for models tagged with `vision`
- Genkit model registration for the current Ollama Cloud direct API model IDs

## Troubleshooting

### Common Issues

1. **API Key Error**: Make sure `OLLAMACLOUD_API_KEY` is set correctly.
2. **Network Issues**: Check your internet connection and firewall settings.
3. **Model Not Found**: Verify the model name is listed in `https://ollama.com/v1/models`.
4. **Rate Limiting**: Check if you have hit API rate limits.

### Error Messages

- `"ollamacloud plugin initialization failed: API key is required"`: Set `OLLAMACLOUD_API_KEY` or the plugin `APIKey` field.
- `"unexpected config type: string"`: Use proper OpenAI config types.
- Network errors: Check the API endpoint and network connectivity.

Note: All live tests require the `OLLAMACLOUD_API_KEY` environment variable to be set. Tests will be skipped if the API key is not provided.
