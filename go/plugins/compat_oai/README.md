# OpenAI-Compatible Plugin Package

This directory contains a package for building plugins that are compatible with the OpenAI API specification, along with plugins built on top of this package.

## Package Overview

The `compat_oai` package provides a base implementation (`OpenAICompatible`) that handles:
- Model and embedder registration
- Message handling
- Tool support
- Configuration management

## Usage Example

Here's how to implement a new OpenAI-compatible plugin. A plugin defines a
config type for its models that declares every field the provider's API
accepts, embedding `compat_oai.RequestConfig` for the two settings Genkit owns
(the per-request API key and the model version). SDK-modeled fields are written
directly and anything else goes through the request's extra fields. The
framework validates every request against the schema inferred from the config
type, and that schema is what the Dev UI offers, so declare what the provider's
API reference lists and nothing more.

```go
// ChatConfig is the plugin's per-request model config.
type ChatConfig struct {
    compat_oai.RequestConfig

    // Temperature controls the degree of randomness, from 0 to 1.
    Temperature  *float64 `json:"temperature,omitempty"`
    EnableSearch *bool    `json:"enableSearch,omitempty"`
}

func (c ChatConfig) ApplyToChatCompletion(params *openai.ChatCompletionNewParams) {
    c.ApplyVersion(params)
    if c.Temperature != nil {
        params.Temperature = openai.Float(*c.Temperature)
    }
    if c.EnableSearch != nil {
        compat_oai.AddExtraFields(params, map[string]any{"enable_search": *c.EnableSearch})
    }
}

type MyPlugin struct {
    openAICompatible compat_oai.OpenAICompatible
    // define other plugin-specific fields
}

var supportedModels = map[string]ai.ModelOptions{
    // define supported models
}

func (p *MyPlugin) Name() string {
    return "myprovider"
}

func (p *MyPlugin) Init(ctx context.Context) []api.Action {
    // initialize the plugin with the common compatible package
    p.openAICompatible.Provider = p.Name()
    actions := p.openAICompatible.Init(ctx)

    // Define plugin-specific models
    for model, opts := range supportedModels {
        actions = append(actions, compat_oai.NewChatModel[ChatConfig](&p.openAICompatible, model, opts))
    }

    // Define embedders, if applicable

    return actions
}
```

A plugin whose config is the raw OpenAI request (the `openai` plugin, or a
proxy for the real OpenAI API) uses `OpenAICompatible.NewModel` instead,
which takes the SDK's `openai.ChatCompletionNewParams` as the model config.

A typed config can also carry a per-request API key (`RequestConfig.APIKey` /
`EmbeddingConfig.APIKey`) that overrides the plugin's key for that request
alone. The key is a client credential: it never serializes, so it stays out of
the advertised schema, recorded traces, and the request body, and it cannot be
supplied through JSON or map configs.

Plugins declare their generation fields rather than inheriting them because
providers disagree about which ones exist and what they are called: DeepSeek
dropped the frequency and presence penalties, Z.ai caps `temperature` at 1,
Kimi's K-series takes neither, and `maxOutputTokens` is `max_tokens` on some
providers and `max_completion_tokens` on others. Use the same camelCase name
other plugins use for the same setting; `conformance_test.go` enforces that
across the package.

See the `openai`, `anthropic`, `dashscope`, `deepseek`, `kimi`, `xai`, and
`zai` directories for complete implementations.

## Running Tests

Set your API keys:
```bash
export OPENAI_API_KEY=<your-openai-key>
export ANTHROPIC_API_KEY=<your-anthropic-key>
export DASHSCOPE_API_KEY=<your-dashscope-key>
export ZAI_API_KEY=<your-zai-key>
export KIMI_API_KEY=<your-kimi-key>
export XAI_API_KEY=<your-xai-key>
export DEEPSEEK_API_KEY=<your-deepseek-key>
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

# xAI tests
go test -v ./xai

# DeepSeek tests
go test -v ./deepseek
```

Note: Tests will be skipped if the required API keys are not set.
