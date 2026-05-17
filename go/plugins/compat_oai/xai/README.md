# OpenAI Plugin

This plugin provides a simple interface for using xAI's services.

## Supported Models
The plugin supports the following xAI models.

### Grok 4 (grok-4-0709)

Modalities: Text input, text output

Context window: 256 000

Features:
- Function calling
- Structured outputs
- Reasoning

### Grok Code Fast 1 (grok-code-fast-1).

Modalities: Text input, text output

Context window: 256 000

Features:
- Function calling
- Structured outputs
- Reasoning

### Grok 4 Fast (grok-4-fast-reasoning).

Modalities: Text input, image input, text output

Context window: 2 000 000

Features:
- Function calling
- Structured outputs
- Reasoning

### Grok 4 Fast (Non-Reasoning)

Modalities: Text input, image input, text output

Context window: 2 000 000

Features:
- Function calling
- Structured outputs


### Grok 3 Mini (grok-3-mini)

Modalities: Text input, text output

Context window: 131 072

Features:
- Function calling
- Structured outputs
- Reasoning

### Grok 3 (grok-3)

Modalities: Text input, text output

Context window: 131 072

Features:
- Function calling
- Structured outputs

### Grok 2 Vision (grok-2-vision)

Modalities: Text input, image input, text output

Context window: 32 768

Features:
- Function calling
- Structured outputs

### Grok 2 Image Gen

Grok 2 Image Gen

## Prerequisites

- Go installed on your system
- An xAI API key

## Usage

Here's a simple example of how to use the OpenAI plugin:

```go
import (
  "context"
  "github.com/firebase/genkit/go/ai"
  "github.com/firebase/genkit/go/genkit"
  "github.com/openai/openai-go"
  "github.com/openai/openai-go/option"
)
// Initialize the xAI plugin with your API key
x := &XAi{
  Opts: []option.RequestOption{
    option.WithAPIKey(apiKey),
  },
}

// Initialize Genkit with the xAI plugin
g := genkit.Init(ctx,
  genkit.WithDefaultModel("xai/grok-3-mini"),
  genkit.WithPlugins(x),
)

config := &openai.ChatCompletionNewParams{
// define optional config fields
}

resp, err = genkit.Generate(ctx, g,
ai.WithPromptText("Write a short sentence about artificial intelligence."),
ai.WithConfig(config),
)
```

## Running Tests

First, set your xAI API key as an environment variable:

```bash
export XAI_API_KEY=<your-api-key>
```

### Running All Tests
To run all tests in the directory:
```bash
go test -v .
```

### Running Tests from Specific Files
To run tests from a specific file:
```bash
# Run only generate_live_test.go tests
go test -run "^TestGenerator"

# Run only openai_live_test.go tests
go test -run "^TestPlugin"
```

### Running Individual Tests
To run a specific test case:
```bash
# Run only the streaming test from xai_live_test.go
go test -run "TestPlugin/streaming"

# Run only the Complete test from generate_live_test.go
go test -run "TestGenerator_Complete"

# Run only the Stream test from generate_live_test.go
go test -run "TestGenerator_Stream"
```

### Test Output Verbosity
Add the `-v` flag for verbose output:
```bash
go test -v -run "TestPlugin/streaming"
```

Note: All live tests require the XAI_API_KEY environment variable to be set. Tests will be skipped if the API key is not provided.
