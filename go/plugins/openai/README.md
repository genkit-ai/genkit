# OpenAI Plugin

This plugin provides Genkit models backed by OpenAI's Responses API.

## Prerequisites

- Go installed on your system
- An OpenAI API key

Set the API key in your environment:

```bash
export OPENAI_API_KEY=<your-api-key>
```

You can also set `OPENAI_BASE_URL` or pass `BaseURL` and additional OpenAI SDK
request options on the plugin struct.

## Usage

```go
package main

import (
	"context"
	"fmt"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/openai"
)

func main() {
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&openai.OpenAI{}))

	resp, err := genkit.Generate(ctx, g,
		ai.WithModelName("openai/gpt-4.1"),
		ai.WithPrompt("Write a short sentence about artificial intelligence."),
	)
	if err != nil {
		panic(err)
	}
	fmt.Println(resp.Text())
}
```

Use `ai.WithConfig` with `responses.ResponseNewParams` for OpenAI-specific
generation options.

## OpenAI Compatible Plugin

This package targets OpenAI's Responses API. If you need a generic
OpenAI-compatible Chat Completions provider, use `plugins/compat_oai/openai`.

## Running Tests

Unit tests do not require an API key:

```bash
go test ./plugins/openai ./plugins/internal/openai
```

Live tests require `OPENAI_API_KEY`:

```bash
OPENAI_API_KEY=<your-api-key> go test -run TestOpenAILive ./plugins/openai
```
