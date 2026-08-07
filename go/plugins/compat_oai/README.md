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

// Capability sets shared by the entries below.
var (
    textOnly = ai.ModelSupports{
        Multiturn: true, Tools: true, SystemRole: true,
        Media: false, ToolChoice: true,
        Output: []string{"text", "json"},
        Constrained: ai.ConstrainedSupportAll,
    }
    multimodal = ai.ModelSupports{
        Multiturn: true, Tools: true, SystemRole: true,
        Media: true, ToolChoice: true,
        Output: []string{"text", "json"},
        Constrained: ai.ConstrainedSupportAll,
    }
)

// supportedModels curates capabilities for well-known models. It is not the
// set of usable models: any model resolves on demand and takes
// [dynamicModelOptions], so an ID absent here still works.
//
// Catalog: https://myprovider.example/docs/models
var supportedModels = map[string]ai.ModelOptions{
    "my-model":       {Label: "My Model", Supports: &textOnly},
    "my-model-vision": {Label: "My Model Vision", Supports: &multimodal},
}

// dynamicModelOptions is advertised for models that resolve dynamically rather
// than appearing in supportedModels.
var dynamicModelOptions = ai.ModelOptions{
    Supports: &multimodal,
    Versions: []string{},
    Stage:    ai.ModelStageStable,
}

func modelOptions(id string) ai.ModelOptions {
    if opts, ok := supportedModels[id]; ok {
        return opts
    }
    return dynamicModelOptions
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

Every plugin in this directory lays its catalog out the same way, so the shape
above transfers: named capability sets, a documented `supportedModels` map of
one-line entries, a `dynamicModelOptions` fallback, and a `modelOptions` lookup.
Where a provider publishes dated snapshots, fold them into the entry's
`Versions` instead of registering a model per snapshot.

`Constrained` is the one capability worth checking against the provider's docs
rather than copying. Genkit sends `response_format` as `json_schema` whenever
the request carries a schema, but it only skips injecting schema instructions
into the prompt when the model advertises constrained support. Set
`ConstrainedSupportAll` only where the provider documents `response_format`
with `type: json_schema`; a provider offering `json_object` alone (DashScope,
DeepSeek, Z.ai) or ignoring `response_format` outright (Anthropic's compatible
endpoint) must leave it unset, or structured output loses the prompt
instructions that were the only thing enforcing the schema. Use
`ConstrainedSupportNoTools` where the provider supports schemas but not
alongside tools, as xAI does outside the Grok 4 family.

Model IDs are string literals rather than exported constants. An exported
`ModelMyModel` outlives the model it names: the ID churns every few months,
but the constant cannot be removed without a breaking change. The map key is
already the single source of truth that `modelOptions` looks up, and a model on
its way out is marked with `Stage: ai.ModelStageDeprecated`, which is data
rather than API surface.

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
