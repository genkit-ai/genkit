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

## Text to speech

The plugin registers `tts-1`, `tts-1-hd`, and `gpt-4o-mini-tts` as Genkit
models. Speech responses contain a base64 data URI in a media part.
`gpt-4o-mini-tts` also accepts `Instructions` for controlling delivery; the
legacy TTS models do not expose this option.

```go
resp, err := genkit.Generate(ctx, g,
    ai.WithModelName("openai/tts-1"),
    ai.WithPrompt("Hello from Genkit."),
    ai.WithConfig(&compat_oai.SpeechConfig{
        Voice: openai.AudioSpeechNewParamsVoiceAlloy,
        ResponseFormat: openai.AudioSpeechNewParamsResponseFormatMP3,
    }),
)
```

## Speech to text

The plugin registers `whisper-1`, `gpt-4o-transcribe`, and
`gpt-4o-mini-transcribe`. Supply audio as a data URI media part; remote media
URIs and unsupported audio types are rejected. An optional text part in the
same message is sent as the transcription prompt. GPT transcription models
return JSON, while Whisper supports text and JSON output. Set `Translate: true`
in `WhisperConfig` to translate Whisper input into English instead of
transcribing it in the source language.

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
