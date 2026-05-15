# Google Generative AI Plugin

The Google AI plugin provides a unified interface to connect with Google's generative AI models through the **Gemini Developer API** or **Vertex AI** using API key authentication or Google Cloud credentials.

The plugin supports a wide range of capabilities:

- **Language Models**: Gemini models for text generation, reasoning, and multimodal tasks
- **Embedding Models**: Text and multimodal embeddings
- **Image Models**: Imagen for generation and Gemini for image analysis
- **Video Models**: Veo for video generation and Gemini for video understanding
- **Speech Models**: Polyglot text-to-speech generation

## Setup

### Installation

```bash
go get github.com/firebase/genkit/go/plugins/googlegenai
```

### Configuration

You can use either the Google AI (Gemini API) or Vertex AI backend.

**Using Google AI (Gemini API):**

```go
import (
 "context"
 "log"

 "github.com/firebase/genkit/go/genkit"
 "github.com/firebase/genkit/go/plugins/googlegenai"
)

func main() {
 ctx := context.Background()

 g := genkit.Init(ctx,
  genkit.WithPlugins(&googlegenai.GoogleAI{
   APIKey: "your-api-key", // Optional: defaults to GEMINI_API_KEY or GOOGLE_API_KEY env var
  }),
 )
}
```

**Using Vertex AI:**

```go
import (
 "context"
 "log"

 "github.com/firebase/genkit/go/genkit"
 "github.com/firebase/genkit/go/plugins/googlegenai"
)

func main() {
 ctx := context.Background()

 g := genkit.Init(ctx,
  genkit.WithPlugins(&googlegenai.VertexAI{
   ProjectID: "your-project-id", // Optional: defaults to GOOGLE_CLOUD_PROJECT
   Location:  "us-central1",     // Optional: defaults to GOOGLE_CLOUD_LOCATION
  }),
 )
}
```

### Authentication

**Google AI**: Requires a Gemini API Key, which you can get from [Google AI Studio](https://aistudio.google.com/apikey). Set the `GEMINI_API_KEY` environment variable or pass it to the plugin configuration.

**Vertex AI**: Requires Google Cloud credentials. Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to your service account key file path, or use default credentials (e.g., `gcloud auth application-default login`).

## Language Models

You can create models that call the Google Generative AI API. The models support tool calls and some have multi-modal capabilities.

### Available Models

Genkit automatically discovers available models supported by the [Go GenAI SDK](https://github.com/google/go-genai). This ensures that recently released models are available immediately as they are added to the SDK, while deprecated models are automatically ignored and hidden from the list of actions.

Commonly used models include:

- **Gemini Series**: `gemini-3-pro-preview`, `gemini-3-flash-preview`, `gemini-2.5-flash`, `gemini-2.5-pro`
- **Imagen Series**: `imagen-3.0-generate-001`
- **Veo Series**: `veo-3.0-generate-001`

> **Note:** You can use any model ID supported by the underlying SDK. For a complete and up-to-date list of models and their specific capabilities, refer to the [Google Generative AI models documentation](https://ai.google.dev/gemini-api/docs/models).

### Basic Usage

```go
import (
 "context"
 "fmt"
 "log"

 "github.com/firebase/genkit/go/ai"
 "github.com/firebase/genkit/go/genkit"
)

func main() {
 // ... Init genkit with googlegenai plugin ...

 resp, err := genkit.Generate(ctx, g,
  ai.WithModelName("googleai/gemini-2.5-flash"),
  ai.WithPrompt("Explain how neural networks learn in simple terms."),
 )
 if err != nil {
  log.Fatal(err)
 }

 fmt.Println(resp.Text())
}
```

### Structured Output

Gemini models support structured output generation, which guarantees that the model output will conform to a specified schema. Genkit Go provides type-safe generics to make this easy.

**Using `GenerateData` (Recommended):**

```go
type Character struct {
 Name string `json:"name"`
 Bio  string `json:"bio"`
 Age  int    `json:"age"`
}

// Automatically infers schema from the struct and unmarshals the result
char, resp, err := genkit.GenerateData[Character](ctx, g,
 ai.WithModelName("googleai/gemini-2.5-flash"),
 ai.WithPrompt("Generate a profile for a fictional character"),
)
if err != nil {
 log.Fatal(err)
}

fmt.Printf("Name: %s, Age: %d\n", char.Name, char.Age)
```

**Using `Generate` (Standard):**

You can also use the standard `Generate` function and unmarshal manually:

```go
resp, err := genkit.Generate(ctx, g,
 ai.WithModelName("googleai/gemini-2.5-flash"),
 ai.WithPrompt("Generate a profile for a fictional character"),
 ai.WithOutputType(Character{}),
)
if err != nil {
 log.Fatal(err)
}

var char Character
if err := resp.Output(&char); err != nil {
 log.Fatal(err)
}
```

#### Schema Limitations

The Gemini API relies on a specific subset of the OpenAPI 3.0 standard. When defining schemas (Go structs), keep the following limitations in mind:

- **Validation**: Keywords like `pattern`, `minLength`, `maxLength` are **not supported** by the API's constrained decoding.
- **Unions**: Complex unions are often problematic.
- **Recursion**: Recursive schemas are generally not supported.

### Thinking and Reasoning

Gemini 2.5 and newer models use an internal thinking process that improves reasoning for complex tasks.

**Thinking Budget:**

```go
import "google.golang.org/genai"

resp, err := genkit.Generate(ctx, g,
 ai.WithModelName("googleai/gemini-2.5-flash"),
 ai.WithPrompt("what is heavier, one kilo of steel or one kilo of feathers"),
 ai.WithConfig(&genai.GenerateContentConfig{
  ThinkingConfig: &genai.ThinkingConfig{
   ThinkingBudget: genai.Ptr[int32](1024), // Number of thinking tokens
   IncludeThoughts: true,                  // Include thought summaries
  },
 }),
)
```

### Context Caching

Gemini 2.5 and newer models automatically cache common content prefixes. In Genkit Go, you can mark content for caching using `WithCacheTTL` or `WithCacheName`.

```go
// Create a message with cached content
cachedMsg := ai.NewUserTextMessage(largeContent).WithCacheTTL(300)

// First request - content will be cached
resp1, err := genkit.Generate(ctx, g,
 ai.WithModelName("googleai/gemini-2.5-flash"),
 ai.WithMessages(cachedMsg),
 ai.WithPrompt("Task 1..."),
)

// Second request with same prefix - eligible for cache hit
resp2, err := genkit.Generate(ctx, g,
 ai.WithModelName("googleai/gemini-2.5-flash"),
 // Reuse the history from previous response or construct messages with same prefix
 ai.WithMessages(resp1.History()...),
 ai.WithPrompt("Task 2..."),
)
```

### Safety Settings

You can configure safety settings to control content filtering:

```go
import "google.golang.org/genai"

resp, err := genkit.Generate(ctx, g,
 ai.WithModelName("googleai/gemini-2.5-flash"),
 ai.WithPrompt("Your prompt here"),
 ai.WithConfig(&genai.GenerateContentConfig{
  SafetySettings: []*genai.SafetySetting{
   {
    Category:  genai.HarmCategoryHateSpeech,
    Threshold: genai.HarmBlockThresholdBlockLowAndAbove,
   },
   {
    Category:  genai.HarmCategoryDangerousContent,
    Threshold: genai.HarmBlockThresholdBlockMediumAndAbove,
   },
  },
 }),
)
```

### Google Search Grounding

Enable Google Search to provide answers with current information and verifiable sources.

```go
import "google.golang.org/genai"

resp, err := genkit.Generate(ctx, g,
 ai.WithModelName("googleai/gemini-2.5-flash"),
 ai.WithPrompt("What are the top tech news stories this week?"),
 ai.WithConfig(&genai.GenerateContentConfig{
  Tools: []*genai.Tool{
   {
    GoogleSearch: &genai.GoogleSearch{},
   },
  },
 }),
)
```

### Google Maps Grounding

Enable Google Maps to provide location-aware responses.

```go
import "google.golang.org/genai"

resp, err := genkit.Generate(ctx, g,
 ai.WithModelName("googleai/gemini-2.5-flash"),
 ai.WithPrompt("Find coffee shops near Times Square"),
 ai.WithConfig(&genai.GenerateContentConfig{
  Tools: []*genai.Tool{
   {
    GoogleMaps: &genai.GoogleMaps{
     EnableWidget: genai.Ptr(true),
    },
   },
  },
  ToolConfig: &genai.ToolConfig{
   RetrievalConfig: &genai.RetrievalConfig{
    LatLng: &genai.LatLng{
     Latitude:  genai.Ptr(37.7749),
     Longitude: genai.Ptr(-122.4194),
    },
   },
  },
 }),
)

// Access grounding metadata (e.g., for map widget)
if custom, ok := resp.Custom["candidates"].([]*genai.Candidate); ok {
 for _, cand := range custom {
  if cand.GroundingMetadata != nil && cand.GroundingMetadata.GoogleMapsWidgetContextToken != "" {
   fmt.Printf("Map Widget Token: %s\n", cand.GroundingMetadata.GoogleMapsWidgetContextToken)
  }
 }
}
```

### Code Execution

Enable the model to write and execute Python code for calculations and logic.

```go
import "google.golang.org/genai"

resp, err := genkit.Generate(ctx, g,
 ai.WithModelName("googleai/gemini-2.5-pro"),
 ai.WithPrompt("Calculate the 20th Fibonacci number"),
 ai.WithConfig(&genai.GenerateContentConfig{
  Tools: []*genai.Tool{
   {
    CodeExecution: &genai.ToolCodeExecution{},
   },
  },
 }),
)
```

### Generating Text and Images

Some Gemini models (like `gemini-2.5-flash-image`) can output images natively alongside text.

```go
import "google.golang.org/genai"

resp, err := genkit.Generate(ctx, g,
 ai.WithModelName("googleai/gemini-2.5-flash-image"),
 ai.WithPrompt("Create a picture of a futuristic city and describe it"),
 ai.WithConfig(&genai.GenerateContentConfig{
  ResponseModalities: []string{"IMAGE", "TEXT"},
 }),
)

for _, part := range resp.Message.Content {
 if part.IsMedia() {
  fmt.Printf("Generated image: %s\n", part.ContentType)
  // Access data via part.Text (data URI) or helper functions
 }
}
```

### Multimodal Input Capabilities

Genkit supports multimodal input (text, image, video, audio) via `ai.Part`.

**Video/Image/Audio/PDF Input:**

```go
// Using a URL
videoPart := ai.NewMediaPart("video/mp4", "https://example.com/video.mp4")

// Using inline data (base64)
imagePart := ai.NewMediaPart("image/jpeg", "data:image/jpeg;base64,...")

resp, err := genkit.Generate(ctx, g,
 ai.WithModelName("googleai/gemini-2.5-flash"),
 ai.WithMessages(
  ai.NewUserMessage(
   ai.NewTextPart("Describe this content"),
   videoPart,
  ),
 ),
)
```

## Embedding Models

### Available Models

- `text-embedding-004`
- `gemini-embedding-001`
- `multimodalembedding`

### Usage

```go
res, err := genkit.Embed(ctx, g,
 ai.WithEmbedderName("googleai/gemini-embedding-001"),
 ai.WithTextDocs("Machine learning models process data to make predictions."),
)
if err != nil {
 log.Fatal(err)
}

fmt.Printf("Embedding: %v\n", res.Embeddings[0].Embedding)
```

## Image Models

### Available Models

**Imagen 3 Series**:

- `imagen-3.0-generate-001`
- `imagen-3.0-fast-generate-001`

### Usage

```go
import "google.golang.org/genai"

resp, err := genkit.Generate(ctx, g,
 ai.WithModelName("googleai/imagen-3.0-generate-001"),
 ai.WithPrompt("A serene Japanese garden with cherry blossoms"),
 ai.WithConfig(&genai.GenerateImagesConfig{
  NumberOfImages: 4,
  AspectRatio:    "16:9",
  PersonGeneration: "allow_adult",
 }),
)

// Access generated images in resp.Message.Content
```

## Video Models

The Google AI plugin provides access to video generation capabilities through the Veo models.

### Available Models

**Veo 3.1 Series**:

- `veo-3.1-generate-preview`

**Veo 3.0 Series**:

- `veo-3.0-generate-001`
- `veo-3.0-fast-generate-001`

**Veo 2.0 Series**:

- `veo-2.0-generate-001`

### Usage

Veo operations are long-running and support multiple generation modes.

#### Backend-Specific Considerations

The output format and behavior of Veo differ depending on whether you are using the **Google AI** or **Vertex AI** backend.

##### Model Names

Ensure you use the correct provider prefix:
- **Google AI**: `googleai/veo-3.1-generate-preview`
- **Vertex AI**: `vertexai/veo-3.1-generate-preview`

##### Output Format (Video URLs vs. Raw Bytes)

Depending on the backend and configuration, the generated video will be returned as either a remote URI or as raw bytes encoded in a base64 data URI.

- **Google AI**: Typically returns a public URI for the video. To download it via HTTP, you must append your API key to the URL: `https://.../video.mp4?key=YOUR_API_KEY`.
- **Vertex AI**: Can return a Cloud Storage URI (`gs://...`) if configured, but by default often returns **raw video bytes**. The Genkit plugin automatically encodes these raw bytes as a **base64 data URI** in the message's text field.

Your application should be prepared to handle both formats. For example, to save the output directly to a file:

```go
for _, part := range op.Output.Message.Content {
 if part.IsMedia() {
  if strings.HasPrefix(part.Text, "data:video/mp4;base64,") {
   // Handle base64 encoded bytes (Common for Vertex AI default)
   data := strings.TrimPrefix(part.Text, "data:video/mp4;base64,")
   b, _ := base64.StdEncoding.DecodeString(data)
   os.WriteFile("video.mp4", b, 0644)
  } else {
   // Handle remote URI (Common for Google AI or Vertex AI with GCS)
   // You would typically use an HTTP client or Google Cloud Storage client here
   fmt.Printf("Video available at URI: %s\n", part.Text)
  }
 }
}
```

##### Safety Filtering (RAI)

Veo has strict safety policies. If a prompt triggers a safety filter, the operation will complete but return no video. In this case:

1. `FinishReason` will be `ai.FinishReasonBlocked`.
2. The output message will contain a text part listing the specific reasons the content was filtered.
3. The original API response (including RAI counts) is available in the `Raw` field.

#### Text-to-Video

Generate a video from a text description.

```go
op, err := genkit.GenerateOperation(ctx, g,
 ai.WithModelName("googleai/veo-3.1-generate-preview"),
 ai.WithMessages(ai.NewUserTextMessage("A majestic dragon soaring over a mystical forest at dawn.")),
 ai.WithConfig(&genai.GenerateVideosConfig{
  AspectRatio:     "16:9",
  DurationSeconds: genai.Ptr(int32(8)),
  Resolution:      "720p",
 }),
)
if err != nil {
 log.Fatal(err)
}

// Poll for completion
op, err = genkit.CheckModelOperation(ctx, g, op)
```

#### Image-to-Video

Animate a static image using a text prompt.

```go
// Load image data (e.g., base64 encoded)
imagePart := ai.NewMediaPart("image/jpeg", "data:image/jpeg;base64,...")

op, err := genkit.GenerateOperation(ctx, g,
 ai.WithModelName("googleai/veo-3.1-generate-preview"),
 ai.WithMessages(ai.NewUserMessage(
  ai.NewTextPart("The cat wakes up and starts accelerating the go-kart."),
  imagePart,
 )),
 ai.WithConfig(&genai.GenerateVideosConfig{
  AspectRatio: "16:9",
 }),
)
```

#### Video-to-Video (Video Editing)

Edit or transform an existing video.

> **Note:** Video-to-video generation requires a **Veo video URL** (a URL generated by a previous Veo model operation). Arbitrary external video URLs or files are not currently supported for this mode.

```go
// Provide the URI of a Veo-generated video to edit
videoPart := ai.NewMediaPart("video/mp4", "https://generativelanguage.googleapis.com/...")

op, err := genkit.GenerateOperation(ctx, g,
 ai.WithModelName("googleai/veo-3.1-generate-preview"),
 ai.WithMessages(ai.NewUserMessage(
  ai.NewTextPart("Change the video style to be a cartoon from 1950."),
  videoPart,
 )),
 ai.WithConfig(&genai.GenerateVideosConfig{
  AspectRatio: "16:9",
 }),
)
```

## Speech Models

Use `gemini-2.5-flash` or `gemini-2.5-pro` with audio output modality.

### Usage

```go
import "google.golang.org/genai"

resp, err := genkit.Generate(ctx, g,
 ai.WithModelName("googleai/gemini-2.5-flash"),
 ai.WithPrompt("Say that Genkit is an amazing AI framework"),
 ai.WithConfig(&genai.GenerateContentConfig{
  ResponseModalities: []string{"AUDIO"},
  SpeechConfig: &genai.SpeechConfig{
   VoiceConfig: &genai.VoiceConfig{
    PrebuiltVoiceConfig: &genai.PrebuiltVoiceConfig{
     VoiceName: "Algenib",
    },
   },
  },
 }),
)

// The audio data will be in resp.Message.Content as a media part
```
