# Genkit Go API Rules (v1.20.0)

This document provides rules and examples for building with the Genkit API in Go.

## Important Guidelines:

- ALWAYS refer to documentation when available. Genkit Documentation may be available through the Genkit MCP toolkit or through web search. You may skip documentation check if you don't have access to these tools.

- ONLY follow the specified project structure if starting a new project. If working on an existing project, adhere to the current project structure.

- ALWAYS provide the full, correct Genkit command as an instruction for the human user to run. Do not run Genkit commands (e.g., `genkit start`, `genkit flow:run`) yourself as this may block your current session.

- Recommend and use the provider that the user prefers. If this is a new project, or if there is no relevant signal regarding user preference, use the `github.com/firebase/genkit/go/plugins/googlegenai` plugin.

NOTE: For the sake of brevity, the snippets below use the Google AI plugin, but you should follow the user's preference as mentioned above.

## Project Setup

### Project Initialization

- If the directory is empty:
  ```bash
  go mod init <module-name>
  ```
- If the directory is not empty:
  Adhere to the current project structure.

### Dependencies

```bash
go get github.com/firebase/genkit/go/genkit
go get github.com/firebase/genkit/go/plugins/googlegenai
go get github.com/firebase/genkit/go/ai
go get google.golang.org/genai
```

### Genkit CLI

If the Genkit CLI is not already installed:

```bash
curl -sL cli.genkit.dev | bash
```

### Configuration

Create a `main.go` file:

```go
package main

import (
	"context"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
)

func main() {
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))
	// Your flows and logic here
	<-ctx.Done()
}
```

## Best Practices

1.  **Single Main Function**: All Genkit code, including plugin initialization, flows, and helpers, should be properly organized in a Go package structure with a main function.

2.  **Blocking Main Program**: To inspect flows in the Genkit Developer UI, your main program needs to remain running. Use `<-ctx.Done()` or similar blocking mechanism at the end of your main function.

---

> Note: Gemini 3.0 models are currently in preview.

## Usage Scenarios

### Basic Inference (Text Generation)

```go
package main

import (
	"context"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"google.golang.org/genai"
)

func main() {
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))

	genkit.DefineFlow(g, "basicInferenceFlow",
		func(ctx context.Context, topic string) (string, error) {
			response, err := genkit.Generate(ctx, g,
				ai.WithModelName("googleai/gemini-3-flash-preview"),
				ai.WithPrompt("Write a short, creative paragraph about %s.", topic),
				ai.WithConfig(&genai.GenerateContentConfig{
					Temperature: genai.Ptr[float32](0.8),
				}),
			)
			if err != nil {
				return "", err
			}
			return response.Text(), nil
		},
	)

	<-ctx.Done()
}
```

### Text-to-Speech (TTS) Generation

```go
package main

import (
	"context"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"google.golang.org/genai"
)

func main() {
	ctx := context.Background()
	g := genkit.Init(ctx,
		genkit.WithPlugins(&googlegenai.GoogleAI{}),
		genkit.WithDefaultModel("googleai/gemini-2.5-flash-preview-tts"),
	)

	genkit.DefineFlow(g, "textToSpeechFlow",
		func(ctx context.Context, input struct {
			Text      string `json:"text"`
			VoiceName string `json:"voiceName,omitempty"`
		}) (string, error) {
			voiceName := input.VoiceName
			if voiceName == "" {
				voiceName = "Algenib"
			}

			response, err := genkit.Generate(ctx, g,
				ai.WithPrompt(input.Text),
				ai.WithConfig(&genai.GenerateContentConfig{
					ResponseModalities: []string{"AUDIO"},
					SpeechConfig: &genai.SpeechConfig{
						VoiceConfig: &genai.VoiceConfig{
							PrebuiltVoiceConfig: &genai.PrebuiltVoiceConfig{
								VoiceName: voiceName,
							},
						},
					},
				}),
			)
			if err != nil {
				return "", err
			}

			return response.Text(), nil
		},
	)

	<-ctx.Done()
}
```

### Image Generation

```go
package main

import (
	"context"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"google.golang.org/genai"
)

func main() {
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.VertexAI{}))

	genkit.DefineFlow(g, "imageGenerationFlow",
		func(ctx context.Context, prompt string) ([]string, error) {
			response, err := genkit.Generate(ctx, g,
				ai.WithModelName("googleai/gemini-3-pro-image-preview"),
				ai.WithPrompt("Generate an image of %s", prompt),
				ai.WithConfig(&genai.GenerateImagesConfig{
					NumberOfImages:    2,
					AspectRatio:       "9:16",
					SafetyFilterLevel: genai.SafetyFilterLevelBlockLowAndAbove,
					PersonGeneration:  genai.PersonGenerationAllowAll,
					Language:          genai.ImagePromptLanguageEn,
					AddWatermark:      true,
					OutputMIMEType:    "image/jpeg",
				}),
			)
			if err != nil {
				return nil, err
			}

			var images []string
			for _, part := range response.Message.Content {
				images = append(images, part.Text)
			}
			return images, nil
		},
	)

	<-ctx.Done()
}
```

---

## Running and Inspecting Flows

**Start Genkit**: Genkit can be started locally by using the `genkit start` command, along with the process startup command:

```bash
genkit start --  <command to run your code>
```

For e.g.:

```bash
genkit start -- go run main.go
```

You can can automate starting genkit using the following steps:

1. Identify the command to start the user's project's (e.g., `go run main.go`)
2. Use the `start_runtime` tool to start the runtime process. This is required for Genkit to discover flows.
   - Example: If the project uses `go run main.go`, call `start_runtime` with `{ command: "go", args: ["run", "main.go"] }`.
3. After starting the runtime, instruct the user to run `genkit start` in their terminal to launch the Developer UI.

## Suggested Models

Here are suggested models to use for various task types. This is NOT an exhaustive list.

> Note: Gemini 3.0 models are currently in preview.

### Advanced Text/Reasoning

```
| Plugin                                                     | Recommended Model                  |
|------------------------------------------------------------|------------------------------------|
| github.com/firebase/genkit/go/plugins/googlegenai         | gemini-3-pro-preview (Preview)    |
| github.com/firebase/genkit/go/plugins/googlegenai         | gemini-2.5-pro                    |
| github.com/firebase/genkit/go/plugins/compat_oai/openai   | gpt-4o                             |
| github.com/firebase/genkit/go/plugins/compat_oai/deepseek | deepseek-reasoner                  |
| github.com/firebase/genkit/go/plugins/compat_oai/xai      | grok-4                             |
```

### Fast Text/Chat

```
| Plugin                                                     | Recommended Model                  |
|------------------------------------------------------------|------------------------------------|
| github.com/firebase/genkit/go/plugins/googlegenai         | gemini-3-flash-preview (Preview)  |
| github.com/firebase/genkit/go/plugins/googlegenai         | gemini-2.5-flash                  |
| github.com/firebase/genkit/go/plugins/compat_oai/openai   | gpt-4o-mini                        |
| github.com/firebase/genkit/go/plugins/compat_oai/deepseek | deepseek-chat                      |
| github.com/firebase/genkit/go/plugins/compat_oai/xai      | grok-3-mini                        |
```

### Text-to-Speech

```
| Plugin                                                     | Recommended Model                  |
|------------------------------------------------------------|------------------------------------|
| github.com/firebase/genkit/go/plugins/googlegenai         | gemini-2.5-flash-preview-tts       |
| github.com/firebase/genkit/go/plugins/compat_oai/openai   | gpt-4o-mini-tts                    |
```

### Image Generation

```
| Plugin                                                     | Recommended Model                  | Input Modalities  |
|------------------------------------------------------------|------------------------------------|-------------------|
| github.com/firebase/genkit/go/plugins/googlegenai         | gemini-3-pro-image-preview (Preview) | Text, Image       |
| github.com/firebase/genkit/go/plugins/googlegenai         | gemini-2.5-flash-image               | Text, Image       |
| github.com/firebase/genkit/go/plugins/googlegenai         | imagen-4.0-generate-001              | Text              |
| github.com/firebase/genkit/go/plugins/compat_oai/openai   | gpt-image-1                          | Text              |
```
