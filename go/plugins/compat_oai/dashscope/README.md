# DashScope Plugin

This plugin provides Genkit support for Qwen models through Alibaba Cloud
Model Studio's OpenAI-compatible DashScope endpoint.

## Setup

Set a Model Studio API key:

```bash
export DASHSCOPE_API_KEY=<your-api-key>
```

By default the plugin points at the international endpoint,
`https://dashscope-intl.aliyuncs.com/compatible-mode/v1`. Mainland-China
accounts use `https://dashscope.aliyuncs.com/compatible-mode/v1` instead, and
Alibaba recommends a workspace-dedicated domain for production; set either
through `DASHSCOPE_BASE_URL`, or pass `option.WithBaseURL` through the
plugin's `Opts`. See https://help.aliyun.com/en/model-studio/base-url for the
endpoint reference.

```go
import (
    "context"

    "github.com/firebase/genkit/go/ai"
    "github.com/firebase/genkit/go/genkit"
    "github.com/firebase/genkit/go/plugins/compat_oai/dashscope"
)

ctx := context.Background()
plugin := &dashscope.DashScope{}
g := genkit.Init(ctx,
    genkit.WithPlugins(plugin),
    genkit.WithDefaultModel("dashscope/qwen-plus"),
)

response, err := genkit.Generate(ctx, g, ai.WithPrompt("Explain mixture-of-experts models."))
```

Qwen's `reasoning_content` output is returned as Genkit reasoning parts and
is available through `response.Reasoning()`.

## Models

The registered catalog spans the commercial Qwen line (`qwen-flash`,
`qwen-plus`, the `qwen3.5` through `qwen3.7` series, `qwen3-max`), vision
(`qwen3-vl-plus`), and coding (`qwen3-coder-plus`) models. The catalog is not a ceiling: any model ID Model Studio serves
resolves on demand, and the `Models` field describes or corrects any model,
curated or not:

```go
plugin := &dashscope.DashScope{Models: map[string]ai.ModelOptions{
    "qwen4-max": {Label: "Qwen4 Max", Supports: &compat_oai.Multimodal},
}}
```

The current model list is at
https://www.alibabacloud.com/help/en/model-studio/models.

## Config

Models take a typed `dashscope.ChatConfig`: the generation fields the
compatible mode accepts plus the DashScope-specific controls (`seed`,
`enableThinking`, `thinkingBudget`, `enableSearch`). `dashscope.ModelRef`
carries the config with the model ID:

```go
response, err := genkit.Generate(ctx, g,
    ai.WithModel(dashscope.ModelRef("qwen-plus", &dashscope.ChatConfig{
        EnableThinking: openai.Ptr(true),
        ThinkingBudget: openai.Ptr(2048),
    })),
    ai.WithPrompt("Explain mixture-of-experts models."),
)
```

Every config also carries the settings Genkit owns: `version` pins the exact
model version a request is served by, `apiKey` (settable only from Go code)
serves one request with a different credential, and `extra` forwards request
body fields the config does not declare, keyed by DashScope's wire names
(for example `search_options`).

## Tool choice

Qwen models support tool calling, but forced tool-choice modes
(`required`/`none`) carry model- and thinking-mode-specific restrictions.
This plugin does not advertise `ToolChoice` support and always uses automatic
tool selection.

## Live tests

Live tests are skipped unless `DASHSCOPE_API_KEY` is set:

```bash
go test -race ./plugins/compat_oai/dashscope -run '^TestPluginLive$' -v -count=1
```
