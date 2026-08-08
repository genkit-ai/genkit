# OpenAI Plugin

This plugin provides a simple interface for using OpenAI's services.

## Prerequisites

- Go installed on your system
- An OpenAI API key

## Usage

Here's a simple example of how to use the OpenAI plugin:

```go
import (
  // ignoring Genkit imports
  oai "github.com/firebase/genkit/go/plugins/compat_oai/openai"
  "github.com/openai/openai-go"
)
// Initialize the OpenAI plugin with your API key
oai := &oai.OpenAI{APIKey: apiKey}

// Initialize Genkit with the OpenAI plugin
g, err := genkit.Init(ctx,
    genkit.WithDefaultModel("openai/gpt-4o-mini"),
    genkit.WithPlugins(oai),
)
if err != nil {
    // handle errors
}

config := &openai.ChatCompletionNewParams{
    // define optional config fields
}

resp, err = genkit.Generate(ctx, g,
    ai.WithPromptText("Write a short sentence about artificial intelligence."),
    ai.WithConfig(config),
)
```

### Generate an image

The plugin registers `dall-e-3` and `gpt-image-1` as image generation models.
Image responses are returned as Genkit media parts.

```go
model := oai.ImageModelRef("dall-e-3", &oai.ImageGenerationConfig{
    Quality: openai.ImageGenerateParamsQualityHD,
    Size:    openai.ImageGenerateParamsSize1024x1024,
})

resp, err := genkit.Generate(ctx, g,
    ai.WithModel(model),
    ai.WithPrompt("A watercolor painting of a lighthouse at sunrise"),
)
if err != nil {
    // handle error
}
imageURL := resp.Message.Content[0].Text
```

## Running Tests

First, set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY=<your-api-key>
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
# Run only the streaming test from openai_live_test.go
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

Note: All live tests require the OPENAI_API_KEY environment variable to be set. Tests will be skipped if the API key is not provided.
