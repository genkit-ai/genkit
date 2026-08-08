# OpenAI-Compatible Plugin Package

This directory contains a package for building plugins that are compatible with the OpenAI API specification, along with plugins built on top of this package.

## Package Overview

The `compat_oai` package provides a base implementation (`OpenAICompatible`) that handles:
- Model and embedder registration
- Image generation through the OpenAI-compatible Images API
- Message handling
- Tool support
- Configuration management

## Usage Example

Here's how to implement a new OpenAI-compatible plugin:

```go
type MyPlugin struct {
    compat_oai.OpenAICompatible
    // define other plugin-specific fields
}

var (
    supportedModels = map[string]ai.ModelInfo{
        // define supported models
    }
)

// Implement required methods
func (p *MyPlugin) Init(ctx context.Context, g *genkit.Genkit) error {
    // initialize the plugin with the common compatible package
    if err := p.OpenAICompatible.Init(ctx, g); err != nil {
        return err
    }

    // Define plugin-specific models
    for model, info := range supportedModels {
        if _, err := p.DefineModel(g, p.Provider, model, info); err != nil {
            return err
        }
    }

    // Define embedders, if applicable

    return nil
}

func (p *MyPlugin) Name() string {
    return p.Provider
}
```

See the `openai`, `anthropic`, and `dashscope` directories for complete implementations.

## Running Tests

Set your API keys:
```bash
export OPENAI_API_KEY=<your-openai-key>
export ANTHROPIC_API_KEY=<your-anthropic-key>
export DASHSCOPE_API_KEY=<your-dashscope-key>
export ZAI_API_KEY=<your-zai-key>
export KIMI_API_KEY=<your-kimi-key>
```

Run all tests:
```bash
go test -v ./...
```

Run specific plugin tests:
```bash
# OpenAI tests
go test -v ./openai

# Anthropic tests
go test -v ./anthropic

# DashScope tests
go test -v ./dashscope

# Z.ai tests
go test -v ./zai

# Kimi tests
go test -v ./kimi
```

Note: Tests will be skipped if the required API keys are not set.
