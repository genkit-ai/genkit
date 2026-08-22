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

Here's how to implement a new OpenAI-compatible plugin. A plugin defines a
config type for its models that declares every field the provider's API
accepts, embedding `compat_oai.RequestConfig` for the settings Genkit owns
(the per-request API key, the model version, and the `extra` passthrough).
SDK-modeled fields are written directly and anything else goes through the
request's extra fields. The framework validates every request against the
schema inferred from the config type, and that schema is what the Dev UI
offers, so declare what the provider's API reference lists and nothing more.
Give every field a `jsonschema_description` for the Dev UI, and put the ranges
and enums the provider documents as hard limits in a `jsonschema` tag, where
validation rejects a violation before it is sent and billed; leave per-model
limits to the descriptions, since the schema is shared by every model the
plugin serves.

A provider field the config does not declare yet is not stranded: callers can
send it through the config's `extra` map, which the framework forwards
verbatim (keys in the provider's wire names) after the declared fields, so a
plugin never declares a passthrough of its own.

```go
// ChatConfig is the plugin's per-request model config.
type ChatConfig struct {
    compat_oai.RequestConfig

    // Temperature controls the degree of randomness, from 0 to 1.
    Temperature  *float64 `json:"temperature,omitempty" jsonschema:"minimum=0,maximum=1" jsonschema_description:"Controls the degree of randomness in token selection, from 0 to 1."`
    EnableSearch *bool    `json:"enableSearch,omitempty" jsonschema_description:"Lets the model consult web search, sent as the API's enable_search."`
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
    // Models overrides what the plugin knows about a model, keyed by model ID,
    // bare or provider-prefixed. Fields left at their zero value keep what the
    // plugin resolves.
    Models map[string]ai.ModelOptions

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

// modelOptions is the one source of model capabilities, shared by Init,
// ListActions and ResolveAction, which is what makes a caller's Models entry
// authoritative no matter which path describes the model first.
func (p *MyPlugin) modelOptions(id string) ai.ModelOptions {
    return compat_oai.ModelOptionsFor("myprovider", id, supportedModels, dynamicModelOptions, p.Models)
}

func (p *MyPlugin) Name() string {
    return "myprovider"
}

func (p *MyPlugin) Init(ctx context.Context) []api.Action {
    // initialize the plugin with the common compatible package
    p.openAICompatible.Provider = p.Name()
    actions := p.openAICompatible.Init(ctx)

    // Define plugin-specific models
    for model := range supportedModels {
        actions = append(actions, compat_oai.NewChatModel[ChatConfig](&p.openAICompatible, model, p.modelOptions(model)))
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
one-line entries, a `dynamicModelOptions` fallback, and a `modelOptions` method
that overlays the caller's `Models` on whichever of the two applies. Where a
provider publishes dated snapshots, fold them into the entry's `Versions`
instead of registering a model per snapshot.

Route every path that describes a model through that one method. `Init`
registers the curated models and cannot be undone afterwards, so a catalog the
plugin got wrong is only correctable if `Init` reads the caller's overrides
too.

`Models` is the whole mechanism, and a plugin exposes no way to register a
model itself. An application never needs one: an ID the plugin does not curate
resolves on demand, and an entry in `Models` describes it. This is why the
plugins here take no `RegisterModel`, matching `googlegenai` and the native
`anthropic` plugin. A registration call would only be able to add IDs that
already work, while the models most likely to need correcting are exactly the
ones `Init` has already registered and nothing can register twice.

Fields Genkit owns are not part of a model's config. `messages`, `tools`,
`tool_choice`, `response_format`, and the deprecated
`functions`/`function_call` pair are built from the Genkit request, so the
SDK-typed models hide them from the advertised schema and reject a config
that sets one, naming the Genkit option to use instead (`ai.WithTools()`,
`ai.WithOutputType()`, and so on); `n` is rejected the same way because the
response carries the first candidate only. A curated config type simply omits
them. The rest of the SDK schema carries descriptions from OpenAI's API
reference, so the Dev UI's config sidebar documents each field.

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

Not every provider is a model vendor. `openrouter` fronts a gateway serving
hundreds of models from dozens of vendors, so it curates no `supportedModels`
map at all: it registers nothing at `Init`, returns no descriptors from
`ListActions` (a descriptor carries full request and response schemas, so
listing that catalog would put megabytes on every reflection poll), and
describes every model it resolves with one permissive capability set. `Models`
is how a caller narrows one. Follow that shape for a gateway, and the curated
shape above for a vendor.

See the `openai`, `anthropic`, `dashscope`, `deepseek`, `kimi`, `openrouter`,
`xai`, and `zai` directories for complete implementations.

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
export OPENROUTER_API_KEY=<your-openrouter-key>
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

# OpenRouter tests
go test -v ./openrouter
```

Note: Tests will be skipped if the required API keys are not set.
