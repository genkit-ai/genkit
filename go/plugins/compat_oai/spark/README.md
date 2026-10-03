# Spark Plugin

This plugin provides Genkit support for iFLYTEK's Spark (讯飞星火) models
through their OpenAI-compatible HTTP API.

## Setup

Set a Spark API key:

```bash
export SPARK_API_KEY=<your-api-key>
```

The key is the Spark HTTP service **API Password** (the single `APIPassword`
value shown in the [iFLYTEK console](https://console.xfyun.cn/services/bmx1)),
sent as a Bearer token. It is **not** the legacy WebSocket API's
`APPID`/`APIKey`/`APISecret` triple.

The plugin uses `https://spark-api-open.xf-yun.com/v1` by default. Set
`SPARK_BASE_URL`, or pass `option.WithBaseURL` through the plugin's `Opts`, to
use another compatible endpoint. Do not point it at the WebSocket host
`spark-api.xf-yun.com`, which uses a different protocol and authentication.

```go
import (
    "context"

    "github.com/firebase/genkit/go/ai"
    "github.com/firebase/genkit/go/genkit"
    "github.com/firebase/genkit/go/plugins/compat_oai/spark"
)

ctx := context.Background()
plugin := &spark.Spark{}
g := genkit.Init(ctx,
    genkit.WithPlugins(plugin),
    genkit.WithDefaultModel("spark/4.0Ultra"),
)

response, err := genkit.Generate(ctx, g, ai.WithPrompt("介绍一下讯飞星火大模型。"))
```

## Models

`4.0Ultra`, `generalv3.5` (Max), `max-32k` (Max-32K), `generalv3` (Pro),
`pro-128k` (Pro-128K), and `lite` are registered. The catalog is not a ceiling:
any model ID the Spark HTTP endpoint serves resolves on demand, and the `Models`
field describes or corrects any model, curated or not. The current model list is
in the iFLYTEK Spark HTTP API reference at
https://www.xfyun.cn/doc/spark/HTTP%E8%B0%83%E7%94%A8%E6%96%87%E6%A1%A3.html.
