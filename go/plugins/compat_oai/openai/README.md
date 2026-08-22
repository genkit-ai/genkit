# OpenAI Plugin

This plugin provides Genkit support for OpenAI chat models and embedders
through the Chat Completions API.

## Setup

Set an OpenAI API key:

```bash
export OPENAI_API_KEY=<your-api-key>
```

```go
import (
    "context"

    "github.com/firebase/genkit/go/ai"
    "github.com/firebase/genkit/go/genkit"
    oai "github.com/firebase/genkit/go/plugins/compat_oai/openai"
)

ctx := context.Background()
plugin := &oai.OpenAI{}
g := genkit.Init(ctx,
    genkit.WithPlugins(plugin),
    genkit.WithDefaultModel("openai/gpt-5.4"),
)

response, err := genkit.Generate(ctx, g, ai.WithPrompt("Write a haiku about Go."))
```

Client options such as a different base URL or an organization ride the
plugin's `Opts` (`option.WithBaseURL`, `option.WithOrganization`, and the rest
of the SDK's request options).

## Models

The plugin registers a curated catalog spanning the GPT-5 line (`gpt-5.6-sol`,
`gpt-5.6-terra`, `gpt-5.6-luna`, `gpt-5.5`, `gpt-5.4`, and earlier), the GPT-4
line, and the o-series reasoning models. The catalog is not a ceiling: any
model ID the API serves resolves on demand, and the models the endpoint
reports are listed dynamically. Use the `Models` field to describe or correct
any model, most often one released after this plugin:

```go
plugin := &oai.OpenAI{Models: map[string]ai.ModelOptions{
    "gpt-6": {Label: "GPT-6", Supports: &compat_oai.Multimodal},
}}
```

The current model list and capabilities are at
https://developers.openai.com/api/docs/models.

Models served only by the Responses API (the `-pro` variants) are not part of
this plugin, which speaks the Chat Completions API; naming one still resolves
it and fails at request time.

## Config

Models take the OpenAI SDK's own request type,
`openai.ChatCompletionNewParams`, with the SDK's wire names. `oai.ModelRef`
carries a config with the model ID:

```go
import "github.com/openai/openai-go"

response, err := genkit.Generate(ctx, g,
    ai.WithModel(oai.ModelRef("gpt-5.4", &openai.ChatCompletionNewParams{
        Temperature:         openai.Float(0.2),
        MaxCompletionTokens: openai.Int(1024),
    })),
    ai.WithPrompt("Answer concisely."),
)
```

The GPT-5 generation rejects the legacy `max_tokens`, so cap output with
`MaxCompletionTokens`; the older field remains for the models that still
read it.

Set the config's `Model` to a dated snapshot to pin the exact version a
request is served by. The fields Genkit builds from the request (`messages`,
`tools`, and their variants) are not config: a config naming one is rejected.

The advertised schema is reflected from the openai-go version your build
links, so a field OpenAI ships tomorrow becomes usable, and validated, by
bumping `github.com/openai/openai-go` in your own go.mod.

## Text to speech

The plugin registers `tts-1`, `tts-1-hd`, and `gpt-4o-mini-tts`. Speech
responses contain a base64 data URI in a media part. `gpt-4o-mini-tts` also
accepts instructions for controlling delivery.

```go
resp, err := genkit.Generate(ctx, g,
    ai.WithModelName("openai/tts-1"),
    ai.WithPrompt("Hello from Genkit."),
    ai.WithConfig(&compat_oai.SpeechConfig{
        Voice:          openai.AudioSpeechNewParamsVoiceAlloy,
        ResponseFormat: openai.AudioSpeechNewParamsResponseFormatMP3,
    }),
)
```

## Speech to text

The plugin registers `whisper-1`, `gpt-4o-transcribe`, and
`gpt-4o-mini-transcribe`. Supply audio as a data URI media part; remote media
URIs and unsupported audio types are rejected. An optional text part in the
same message is sent as the transcription prompt. Transcription models
advertise text output; GPT transcription uses the provider's JSON wire format
internally. Set `Translate: true` in `WhisperConfig` to translate Whisper
input into English.

```go
resp, err := genkit.Generate(ctx, g,
    ai.WithModelName("openai/whisper-1"),
    ai.WithMessages(ai.NewUserMessage(
        ai.NewTextPart("Use the provided spelling for Genkit."),
        ai.NewMediaPart("audio/wav", "data:audio/wav;base64,..."),
    )),
    ai.WithConfig(&compat_oai.WhisperConfig{
        TranscriptionConfig: compat_oai.TranscriptionConfig{
            ResponseFormat: openai.AudioResponseFormatText,
        },
        Translate: true,
    }),
)
```

## Embedders

`text-embedding-3-large`, `text-embedding-3-small`, and
`text-embedding-ada-002` are registered, and the `Embedders` field overrides
what the plugin knows the way `Models` does. Embedders take a typed
`oai.TextEmbeddingConfig`:

```go
res, err := genkit.Embed(ctx, g,
    ai.WithEmbedder(oai.NewEmbedderRef("text-embedding-3-small", &oai.TextEmbeddingConfig{
        Dimensions: 256,
    })),
    ai.WithTextDocs("Genkit is an AI framework."),
)
```

The embedding config also carries the settings Genkit owns: `apiKey`
(settable only from Go code) serves one request with a different credential,
and `extra` forwards request body fields the config does not declare, keyed
by OpenAI's wire names.

## Live tests

Live tests are skipped unless `OPENAI_API_KEY` is set:

```bash
go test -v ./plugins/compat_oai/openai
```

