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
            ProjectID:  "your-project-id", // Optional: defaults to GOOGLE_CLOUD_PROJECT
            Location:   "us-central1",     // Optional: defaults to GOOGLE_CLOUD_LOCATION. Also accepts multi-region ("us", "eu") or "global".
            APIVersion: "v1",              // Optional: defaults to v1beta1. Can be overridden per-request via config.HTTPOptions.APIVersion.
        }),
    )
}
```

**Using Vertex AI Express Mode (API key, no Google Cloud project):**

```go
g := genkit.Init(ctx,
 genkit.WithPlugins(&googlegenai.VertexAI{
  APIKey: "your-express-mode-api-key", // Optional: defaults to VERTEX_API_KEY, GOOGLE_API_KEY, or GOOGLE_GENAI_API_KEY
 }),
)
```

Express Mode authenticates with an API key alone: no project, location, or
Application Default Credentials are involved, which makes it the fastest way
to try Vertex AI. Get a key from the
[Express Mode overview](https://cloud.google.com/vertex-ai/generative-ai/docs/start/express-mode/overview).

The plugin picks a mode as follows. The precedence rule (explicit
configuration first, and a project or location outranking an ambient API
key) matches the underlying genai SDK; the environment variable names follow
the JS plugin:

1. An explicit `APIKey` selects Express Mode. It is mutually exclusive with
   `ProjectID`, `Location`, and `Credentials`.
2. An explicit `ProjectID`, `Location`, or `Credentials` selects credential
   authentication, and any API key in the environment is ignored.
3. Otherwise, if the environment names neither a project
   (`GOOGLE_CLOUD_PROJECT`) nor a location (`GOOGLE_CLOUD_LOCATION`,
   `GOOGLE_CLOUD_REGION`), an API key in `VERTEX_API_KEY`, `GOOGLE_API_KEY`,
   or `GOOGLE_GENAI_API_KEY` selects Express Mode.
4. A `BaseURL` with nothing else configured selects custom-endpoint mode:
   requests go to that endpoint as-is and the endpoint owns authentication,
   which suits API gateways and proxies.

`GEMINI_API_KEY` is deliberately not consulted for Vertex AI: it names a
Gemini Developer API key, which Vertex AI does not accept. A project or
location in the environment always outranks an ambient API key, so a key
exported for the Google AI plugin cannot move an existing Vertex AI
deployment onto a different authentication path.

### Authentication

**Google AI**: Requires a Gemini API Key, which you can get from [Google AI Studio](https://aistudio.google.com/apikey). Set the `GEMINI_API_KEY` environment variable or pass it to the plugin configuration.

**Vertex AI**: Requires Google Cloud credentials. Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable to your service account key file path, or use default credentials (e.g., `gcloud auth application-default login`). Alternatively use Express Mode (above), which needs only an API key, or supply custom credentials via the `Credentials` field.

### Network and transport options

Both plugins accept optional fields for nonstandard network setups:

```go
g := genkit.Init(ctx,
 genkit.WithPlugins(&googlegenai.GoogleAI{
  APIVersion: "v1alpha",                           // pin the API version ("v1", "v1beta", or "v1alpha"; default v1beta)
  BaseURL: "https://my-gateway.example.com",       // route through a proxy or API gateway
  Headers: http.Header{"X-Team": {"platform"}},    // extra headers on every request
  HTTPClient: myClient,                            // custom *http.Client, used verbatim
 }),
)
```

`VertexAI` additionally accepts `Credentials` (a `*auth.Credentials` from
`cloud.google.com/go/auth`) to override Application Default Credentials. When
you supply `HTTPClient` to `VertexAI`, that client must handle authentication
itself; use `Credentials` instead if you only need a different identity. Note
that Vertex AI names its API versions differently: `VertexAI.APIVersion` takes
`"v1"` or `"v1beta1"` (default `v1beta1`).

### Accessing the underlying client

The plugin's `Client` method returns the `*genai.Client` it uses, so you can
reach SDK features Genkit does not wrap (Files, Caches, Batches, Tunings)
without constructing and authenticating a second client:

```go
plugin := &googlegenai.GoogleAI{}
g := genkit.Init(ctx, genkit.WithPlugins(plugin))

client, err := plugin.Client()
if err != nil {
 log.Fatal(err)
}
file, err := client.Files.UploadFromPath(ctx, "photo.jpg", &genai.UploadFileConfig{
 MIMEType: "image/jpeg",
})
```

See `go/samples/basic-media` for a complete example.

## Language Models

You can create models that call the Google Generative AI API. The models support tool calls and some have multi-modal capabilities.

### Available Models

Genkit automatically discovers available models supported by the [Go GenAI SDK](https://github.com/google/go-genai). This ensures that recently released models are available immediately as they are added to the SDK, while deprecated models are automatically ignored and hidden from the list of actions.

Commonly used models include:

- **Gemini Series**: `gemini-flash-latest`, `gemini-3.7-flash`, `gemini-3.6-flash`, `gemini-3.5-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`
- **Imagen Series**: `imagen-4.0-generate-001`
- **Veo Series**: `veo-3.1-generate-preview`

> **Note:** You can use any model ID supported by the underlying SDK. For a complete and up-to-date list of models and their specific capabilities, refer to the [Google Generative AI models documentation](https://ai.google.dev/gemini-api/docs/models). On Vertex AI, see the [Vertex AI model documentation](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models).

The `Models` field on either backend describes or corrects any model, most
often one released after the SDK version your build links:

```go
g := genkit.Init(ctx,
    genkit.WithPlugins(&googlegenai.GoogleAI{
        Models: map[string]ai.ModelOptions{
            "gemini-4.0-flash": {Label: "Gemini 4.0 Flash"},
        },
    }),
)
```

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
        ai.WithModelName("googleai/gemini-flash-latest"),
        ai.WithPrompt("Explain how neural networks learn in simple terms."),
    )
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(resp.Text())
}
```

`googlegenai.ModelRef` binds a typed config to a model name so `ai.WithModel`
carries both:

```go
import "google.golang.org/genai"

resp, err := genkit.Generate(ctx, g,
    ai.WithModel(googlegenai.ModelRef("googleai/gemini-flash-latest", &genai.GenerateContentConfig{
        Temperature: genai.Ptr[float32](0.2),
    })),
    ai.WithPrompt("Explain how neural networks learn in simple terms."),
)
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
    ai.WithModelName("googleai/gemini-flash-latest"),
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
    ai.WithModelName("googleai/gemini-flash-latest"),
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
    ai.WithModelName("googleai/gemini-flash-latest"),
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

There are two kinds of caching, and they bill differently. Most traffic wants
the first one.

#### Implicit caching (automatic, no code)

Gemini 2.5 and newer models cache repeated prompt prefixes on their own. It is
on by default on both the Gemini API and Vertex AI, there is no storage charge,
and a hit bills the repeated tokens at a fraction of the normal input price.
You do not call anything to get it.

What it asks of you is only that the repeated part stays at the front of the
prompt and stays identical between calls, and that the prompt is large enough
to qualify (roughly a thousand tokens and up, depending on the model). Genkit
sends messages in the order you build them, so putting the stable content in
the first message is enough.

Read the hit off the response:

```go
resp, err := genkit.Generate(ctx, g,
    ai.WithModelName("googleai/gemini-flash-latest"),
    ai.WithMessages(ai.NewUserTextMessage(largeStableContext)),
    ai.WithPrompt("Task 1..."),
)
// Tokens billed at the cached rate, implicit or explicit.
fmt.Println(resp.Usage.CachedContentTokens)
```

See [Gemini API context caching][gemini-caching] and
[Vertex AI context caching][vertex-caching] for the current thresholds.

#### Explicit caching (`WithCacheTTL`)

`WithCacheTTL` uploads the marked message and everything before it to a cache
resource, and later requests point at it instead of resending it. This buys a
guaranteed hit and a lifetime you control. In exchange you pay to store the
content for that lifetime, so it is worth it when you know you will reuse a
large prefix and want the hit guaranteed rather than best-effort. If you are
not sure, start with implicit caching and measure `CachedContentTokens`.

```go
// Cache this message and everything before it. The marker is inclusive, so
// keep the question in a separate message.
cachedMsg := ai.NewUserTextMessage(largeContent).WithCacheTTL(300)

resp1, err := genkit.Generate(ctx, g,
    ai.WithModelName("googleai/gemini-flash-latest"),
    ai.WithMessages(cachedMsg),
    ai.WithPrompt("Task 1..."),
)

// Later requests reuse the same cache. Replaying the history carries the
// cache name back automatically; rebuilding the same prefix by hand also
// finds it, because Genkit looks for a cache holding that content before it
// creates another one.
resp2, err := genkit.Generate(ctx, g,
    ai.WithModelName("googleai/gemini-flash-latest"),
    ai.WithMessages(resp1.History()...),
    ai.WithPrompt("Task 2..."),
)
```

The cached prefix travels by reference only: it is not also sent inline, so it
is billed once. If the cache has expired or no longer matches the request,
Genkit builds a fresh one rather than failing.

`WithCacheName` points a request at a cache you already hold. Use it when you
kept the name yourself instead of replaying the history. If that cache is gone,
the request still goes through, uncached.

Tools and system prompts are not supported with explicit caching.

[gemini-caching]: https://ai.google.dev/gemini-api/docs/caching
[vertex-caching]: https://cloud.google.com/vertex-ai/generative-ai/docs/context-cache/context-cache-overview

### Handling Errors and Blocked Content

API errors carry the status the service reported, so status-aware code
(retries, fallbacks) can classify them with `core/status`. Rate-limit errors
include the delay the service asked you to wait:

```go
resp, err := genkit.Generate(ctx, g, /* ... */)
if err != nil {
 if delay, ok := googlegenai.RetryDelay(err); ok {
  time.Sleep(delay) // or hand the delay to your retry policy
 }
 return err
}
```

Content blocked by safety filters is not an error. A blocked response comes
back with `FinishReason` set to `blocked`, an explanation in `FinishMessage`,
and no content. The raw safety ratings and prompt feedback are available under
`resp.Custom["candidates"]` and `resp.Custom["promptFeedback"]`.

```go
if resp.FinishReason == ai.FinishReasonBlocked {
 log.Printf("response blocked: %s", resp.FinishMessage)
}
```

### Safety Settings

You can configure safety settings to control content filtering:

```go
import "google.golang.org/genai"

resp, err := genkit.Generate(ctx, g,
    ai.WithModelName("googleai/gemini-flash-latest"),
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
    ai.WithModelName("googleai/gemini-flash-latest"),
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
    ai.WithModelName("googleai/gemini-flash-latest"),
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
    ai.WithModelName("googleai/gemini-flash-latest"),
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

Some Gemini models (like `gemini-3.1-flash-image`) can output images natively alongside text.

```go
import "google.golang.org/genai"

resp, err := genkit.Generate(ctx, g,
    ai.WithModelName("googleai/gemini-3.1-flash-image"),
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
    ai.WithModelName("googleai/gemini-flash-latest"),
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

Requests with more inputs than the service accepts per call (100 on the
Gemini API, 250 on Vertex AI's prediction service including
`gemini-embedding-001`, 1 for the Vertex AI models served by the
one-content embedContent API) are split into sequential batches
automatically; the response carries one embedding per input, in input order.

## Image Models

### Available Models

**Imagen 4 Series**:

- `imagen-4.0-generate-001`
- `imagen-4.0-fast-generate-001`
- `imagen-4.0-ultra-generate-001`

### Usage

```go
import "google.golang.org/genai"

resp, err := genkit.Generate(ctx, g,
    ai.WithModelName("googleai/imagen-4.0-generate-001"),
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
- `veo-3.1-fast-generate-preview`

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

Use Gemini TTS models to generate speech. Dedicated TTS models include
`gemini-3.1-flash-tts-preview`.

Gemini TTS responses are returned as media parts. The media data may be raw PCM
audio, commonly `audio/L16;codec=pcm;rate=24000`, rather than a WAV or MP3 file.
Genkit preserves the provider MIME type and bytes as returned. If you need a
browser- or player-friendly file, decode the media data URI and wrap `audio/L16`
PCM bytes in a WAV container before playback. This is the same pattern used by
the JavaScript Gemini TTS samples, which convert the returned PCM bytes with a
`toWav` helper.

### Usage

```go
import "google.golang.org/genai"

resp, err := genkit.Generate(ctx, g,
    ai.WithModelName("googleai/gemini-3.1-flash-tts-preview"),
    ai.WithPrompt("Say that Genkit is an amazing AI framework"),
    ai.WithConfig(&genai.GenerateContentConfig{
        SpeechConfig: &genai.SpeechConfig{
            VoiceConfig: &genai.VoiceConfig{
                PrebuiltVoiceConfig: &genai.PrebuiltVoiceConfig{
                    VoiceName: "Algenib",
                },
            },
        },
    }),
)

// Audio-only TTS responses usually have no text output.
audio := resp.Media()
```

For conversational Gemini models that can produce multiple modalities, set
`ResponseModalities: []string{"AUDIO"}`. For dedicated `*-tts` models, configure
the voice with `SpeechConfig`; the model already produces audio.

The returned `audio` value may look like:

```text
data:audio/L16;codec=pcm;rate=24000;base64,...
```

To write a playable file, strip the `data:` prefix, decode the base64 payload,
and wrap the raw PCM bytes in a WAV container using the sample rate from the
media type, 24000 Hz above.
