# Google GenAI plugin for Genkit

This plugin provides a unified interface to connect with Google's generative AI models, offering access through both the **Gemini API** and the **Gemini Enterprise Agent Platform**. It is a replacement for the previous `googleAI` and `vertexAI` plugins.

> **Building with a coding agent? Install the Genkit JS skill first.**
>
> ```bash
> npx skills add genkit-ai/skills --skill developing-genkit-js
> ```
>
> It teaches your agent the current Genkit JS APIs and common gotchas.
> Source, manual install and skills for other languages:
> [genkit-ai/skills](https://github.com/genkit-ai/skills).

> **Note on Naming:** In April 2026, Google Cloud rebranded Vertex AI to the **Gemini Enterprise Agent Platform**. To maintain backward compatibility with Google Cloud's underlying API infrastructure and existing deployments, this plugin retains the `vertexAI` namespace, export names, and configuration keys.

Official documentation:

- [Genkit + Gemini API](https://genkit.dev/docs/integrations/google-genai/)
- [Genkit + Gemini Enterprise Agent Platform](https://genkit.dev/docs/integrations/vertex-ai/)

## Installation

```bash
npm i --save @genkit-ai/google-genai
```

## Configuration

This unified plugin exports two main initializers:

- `googleAI`: Allows access to models via the Gemini API using API key authentication.
- `vertexAI`: Allows access to models via the Gemini Enterprise Agent Platform (formerly Vertex AI). Authentication can be done via Google Cloud Application Default Credentials (ADC) or a simpler API Key for Express Mode.

You can configure one or both in your Genkit setup depending on your needs.

### Using the Gemini API (`googleAI`)

Ideal for quick prototyping and access to models available in Google AI Studio.

**Authentication:** Requires a Google AI API Key, which you can get from [Google AI Studio](https://aistudio.google.com/apikey). You can provide this key by setting the `GEMINI_API_KEY` or `GOOGLE_API_KEY` environment variables, or by passing it in the plugin configuration.

```typescript
import { genkit } from 'genkit';
import { googleAI } from '@genkit-ai/google-genai';

const ai = genkit({
  plugins: [
    googleAI(),
    // Or with an explicit API key:
    // googleAI({ apiKey: 'your-api-key' }),
  ],
});
```

### Using Gemini Enterprise Agent Platform (`vertexAI`)

Suitable for applications leveraging Google Cloud's AI infrastructure.

**Authentication Methods:**

-   **Application Default Credentials (ADC):** The standard method for most platform use cases, especially in production. It uses the credentials from the environment (e.g., service account on GCP, user credentials from `gcloud auth application-default login` locally). This method requires a Google Cloud Project with billing enabled and the Gemini Enterprise Agent Platform API enabled.
-   **Platform Express Mode:** A streamlined way to try out many features using just an API key, without needing to set up billing or full project configurations. This is ideal for quick experimentation and has generous free tier quotas. [Learn More about Express Mode](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/start/express-mode/overview).

```typescript
import { genkit } from 'genkit';
import { vertexAI } from '@genkit-ai/google-genai';

const ai = genkit({
  plugins: [
    // Using Application Default Credentials (Recommended for full features)
    vertexAI({ location: 'us-central1' }), // Regional endpoint
    // vertexAI({ location: 'global' }),      // Global endpoint

    // OR

    // Using Express Mode (Easy to start, some limitations)
    // Get an API key from the Agent Platform Studio Express Mode setup.
    // vertexAI({ apiKey: process.env.VERTEX_EXPRESS_API_KEY }),
  ],
});
```

*Note: When using Express Mode, you do not provide `projectId` and `location` in the plugin config.*

### Using Both Gemini API (`googleAI`) and the Gemini Enterprise Agent Platform (`vertexAI`)

You can configure both plugins if you need to access models or features from both services.

```typescript
import { genkit } from 'genkit';
import { googleAI, vertexAI } from '@genkit-ai/google-genai';

const ai = genkit({
  plugins: [
    googleAI(),
    vertexAI()
  ],
});
```

## Usage Examples

Access models and embedders through the configured plugin instance (`googleAI` or `vertexAI`).

### Text Generation (Gemini)

**With `googleAI`:**
```typescript
import { genkit } from 'genkit';
import { googleAI } from '@genkit-ai/google-genai';

const ai = genkit({
  plugins: [googleAI()],
});

const response = await ai.generate({
  model: googleAI.model('gemini-flash-latest'),
  prompt: 'Tell me something interesting about Google AI.',
});

console.log(response.text());
```

**With `vertexAI`:**
```typescript
import { genkit } from 'genkit';
import { vertexAI } from '@genkit-ai/google-genai';

const ai = genkit({
  plugins: [vertexAI()],
});

const response = await ai.generate({
  model: vertexAI.model('gemini-3.1-pro-preview'),
  prompt: 'Explain Gemini Enterprise Agent Platform in simple terms.',
});

console.log(response.text());
```

### Text Embedding

**With `googleAI`:**
```typescript
import { genkit } from 'genkit';
import { googleAI } from '@genkit-ai/google-genai';

const ai = genkit({
  plugins: [googleAI()],
});

const embeddings = await ai.embed({
  embedder: googleAI.embedder('gemini-embedding-2'),
  content: 'Embed this text.',
});
```

**With `vertexAI`:**
```typescript
import { genkit } from 'genkit';
import { vertexAI } from '@genkit-ai/google-genai';

const ai = genkit({
  plugins: [vertexAI()],
});

const embeddings = await ai.embed({
  embedder: vertexAI.embedder('text-embedding-005'),
  content: 'Embed this text.',
});
```

### Video Understanding (agentic processing)

With `googleAI`, newer models (e.g. `gemini-4-argon`, `gemini-3.8-flash`) can
explore a video and load only the parts they need, instead of reading frames
at a fixed rate. This works better for long videos. Set it per video with
`metadata.processing` (`'agentic'`, or `'static'`, the default):

```typescript
const response = await ai.generate({
  model: googleAI.model('gemini-4-argon'),
  prompt: [
    {
      media: { url: 'https://www.youtube.com/watch?v=QO30-W3b_vw' },
      metadata: { processing: 'agentic' },
    },
    { text: 'What happens in this video? List the main moments with timestamps.' },
  ],
});
```

The model's loading steps come back as `custom.processingCall` and
`custom.processingResult` parts. Keep them in the history for follow-up
questions. `response.usage.custom.toolUseTokens` counts the tokens loaded.

### Image Generation

**With `googleAI` (Gemini Image):**
```typescript
import { genkit } from 'genkit';
import { googleAI } from '@genkit-ai/google-genai';

const ai = genkit({
  plugins: [googleAI()],
});

const response = await ai.generate({
  model: googleAI.model('gemini-nano-banana-2.1'),
  prompt: 'A beautiful watercolor painting of a castle in the mountains.',
});

const generatedImage = response.media;
```

**With `vertexAI` (Gemini Image):**
```typescript
import { genkit } from 'genkit';
import { vertexAI } from '@genkit-ai/google-genai';

const ai = genkit({
  plugins: [vertexAI()],
});

const response = await ai.generate({
  model: vertexAI.model('gemini-3.1-flash-image-preview'),
  prompt: 'A beautiful watercolor painting of a castle in the mountains.',
});

const generatedImage = response.media;
```

### Video Generation

**With `googleAI` (Gemini Omni):**
```typescript
import { genkit } from 'genkit';
import { googleAI } from '@genkit-ai/google-genai';

const ai = genkit({
  plugins: [googleAI()],
});

const response = await ai.generate({
  model: googleAI.model('gemini-omni-1.1-flash'),
  prompt: 'An origami butterfly flaps its wings and flies out of the french doors into the garden.',
  config: {
    videoFormat: { aspectRatio: '16:9', resolution: '720p' },
  },
});

// An MP4 video (video/mp4), returned inline.
const video = response.media;
```

- Gemini Omni replaces the Veo 3.1 preview models on the Gemini API, which
  shut down on October 22, 2026. Unlike Veo, it isn't a long-running
  operation: `ai.generate()` returns the video directly.
- Images and videos in the prompt are used as inputs (image to video, first
  and last frame, subject references, editing or extending a video).
- `config.videoFormat` sets `aspectRatio` (`16:9`, the default, or `9:16`)
  and `resolution` (`360p`, `720p` (the default), `1080p` or `4k`).
  `delivery: 'uri'` returns a Files API URI instead of inline data, which is
  recommended for large videos (a 10 s 4k video is about 50 MB); it requires
  `store: true`. Download the file with your API key (`x-goog-api-key`
  header). It can take a few seconds after the response to become `ACTIVE`,
  and it expires after 48 hours.
- `config.videoConfig.task` (`text_to_video`, `image_to_video`,
  `reference_to_video`, `edit`, `extend`) tells the model which kind of
  generation to do; without it, the model infers it from the prompt.
- To edit a generated video in a follow-up turn, set `store: true` and pass
  the previous response's messages: the plugin continues the stored
  interaction.
- Tools aren't supported.

**With `vertexAI` (Veo):**
```typescript
import { genkit } from 'genkit';
import { vertexAI } from '@genkit-ai/google-genai';

const ai = genkit({
  plugins: [vertexAI()],
});

let { operation } = await ai.generate({
  model: vertexAI.model('veo-3.1-lite-generate-001'),
  prompt: 'An origami butterfly flaps its wings and flies out of the french doors into the garden.',
});

// The generation is asynchronous and returns an operation.
// You must poll the operation until it completes.
while (!operation?.done) {
  await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
  operation = await ai.checkOperation(operation!);
}

// Once complete, the generated video is available in `operation.output`.
```

### Music Generation (Lyria)

**With `googleAI`:**
```typescript
import { genkit } from 'genkit';
import { googleAI } from '@genkit-ai/google-genai';

const ai = genkit({
  plugins: [googleAI()],
});

const response = await ai.generate({
  model: googleAI.model('lyria-3-clip-preview'),
  prompt: 'A cheerful acoustic folk song with guitar and harmonica.',
});

const generatedAudio = response.media;
```

**With `vertexAI`:**
```typescript
import { genkit } from 'genkit';
import { vertexAI } from '@genkit-ai/google-genai';

const ai = genkit({
  plugins: [vertexAI()],
});

const response = await ai.generate({
  model: vertexAI.model('lyria-3-clip-preview'),
  prompt: 'A cheerful acoustic folk song with guitar and harmonica.',
});

const generatedAudio = response.media;
```

### Text-to-Speech

**With `googleAI`:**
```typescript
import { genkit } from 'genkit';
import { googleAI } from '@genkit-ai/google-genai';

const ai = genkit({
  plugins: [googleAI()],
});

const response = await ai.generate({
  model: googleAI.model('gemini-3.8-flash-tts'),
  prompt: [
    {
      text: 'Have a wonderful day! <laugh>',
      // Delivery goes in metadata: the text itself is read verbatim.
      metadata: { speechMetadata: { style: 'cheerful and friendly' } },
    },
  ],
  config: {
    speechConfig: {
      voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } },
    },
  },
});

// A complete WAV file (audio/wav), ready to play or save.
const audio = response.media;
```

**Audio format and streaming (`gemini-3.8-flash-tts`, `gemini-3.8-flash-lite-tts`):**

- Without `audioFormat`, a non-streamed request returns one `audio/wav` file,
  and a streamed request returns raw PCM: `audio/l16`, 24 kHz, mono, 16-bit,
  with no header.
- Set `config.audioFormat.mimeType` to choose the format: `audio/wav`,
  `audio/l16`, `audio/mulaw` or `audio/alaw`. `audioFormat.sampleRate` is also
  sent, but the models currently return 24 kHz regardless.
- When streaming, audio chunks arrive as the speech is generated, so you can
  start playing before it's finished. The final response joins them into one
  media part in the requested format. Request `audio/wav` when streaming to
  get a playable WAV file at the end (the chunks themselves are still mostly
  raw PCM).

```typescript
const { stream, response } = ai.generateStream({
  model: googleAI.model('gemini-3.8-flash-lite-tts'),
  prompt: 'Have a wonderful day!',
  config: {
    speechConfig: {
      voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } },
    },
    audioFormat: { mimeType: 'audio/wav' },
  },
});

for await (const chunk of stream) {
  // chunk.media: the next piece of audio.
}

// One complete WAV file.
const audio = (await response).media;
```

### Autonomous Agents (Deep Research & Antigravity)

**Deep Research:**
```typescript
import { genkit } from 'genkit';
import { googleAI } from '@genkit-ai/google-genai';

const ai = genkit({ plugins: [googleAI()] });

let { operation } = await ai.generate({
  model: googleAI.model('deep-research-preview-04-2026'),
  prompt: 'Analyze global semiconductor market trends. Include graphics showing market share changes.',
});

// Deep Research runs asynchronously. Poll until complete.
while (!operation?.done) {
  await new Promise((resolve) => setTimeout(resolve, LONG_POLL_INTERVAL_MS));
  operation = await ai.checkOperation(operation!);
}

// Once complete, the research report is available in `operation.output`.
```

**Antigravity:**
```typescript
import { genkit } from 'genkit';
import { googleAI } from '@genkit-ai/google-genai';

const ai = genkit({ plugins: [googleAI()] });

const response = await ai.generate({
  model: googleAI.model('antigravity-preview-09-2026'),
  prompt: 'Read Hacker News, summarize the top 10 stories.',
  // Required: the sandbox the agent works in.
  config: { environment: 'remote' },
});
```

**On Gemini Enterprise Agent Platform (`vertexAI`):** Deep Research is
available as `vertexAI.model('deep-research-preview-04-2026')` and is used the
same way. It runs on the `global` location (used when you don't set one). It
uses the server's default for `store` unless you set it (in the request config
or `vertexAI({ store })`); `store: false` is sent as is, and the agent rejects
it.

## Interactions API Architecture & `response.raw`

For newer Gemini models accessed through the `googleAI()` plugin, requests are routed through Google's modern **Interactions API** (`/v1beta/interactions`) rather than the legacy `generateContent` API.

### Model Routing

Gemini and Gemma models use the Interactions API, including the `-latest`
aliases, agents (Deep Research, Antigravity) and any model not listed below.
Only these older models still use `generateContent`, until they are retired:

- `gemini-3.5-flash`
- `gemini-3.1-flash-lite`
- `gemini-3.1-pro-preview`
- `gemini-3.1-pro-preview-customtools`
- `gemini-3-flash-preview`
- `gemini-2.5-pro`
- `gemini-2.5-flash`
- `gemini-2.5-flash-lite`
- `gemini-3.1-flash-image`
- `gemini-3-pro-image`
- `gemini-2.5-flash-image`
- `gemini-2.5-flash-preview-tts`
- `gemini-2.5-pro-preview-tts`
- `gemini-3.1-flash-tts-preview`
- `gemma-4-26b-a4b-it`
- `gemma-4-31b-it`

### Model Routing (`vertexAI`)

On Gemini Enterprise Agent Platform, newer Gemini models (for example
`gemini-4-argon`, `gemini-3.8-flash`), Gemini Omni video models and any Gemini
model not listed below use the Interactions API (`.../interactions`). These
models still use `generateContent`:

- `gemini-3.8-flash-cyber`
- `gemini-3.5-flash`
- `gemini-3.1-flash-lite`
- `gemini-3.1-pro-preview`, `gemini-3.1-pro-preview-customtools`
- `gemini-3-flash-preview`
- `gemini-2.5-pro`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`
- image models (`gemini-*-image*`, `gemini-nano-banana-*`) and text-to-speech
  models (`gemini-*-tts*`): the platform's Interactions API doesn't support
  them
- tuned model endpoints

Deep Research also uses the Interactions API.

The Interactions API is available on the `global` location (and on the `us`
and `eu` multi-regions with CMEK), and in Express mode. If you don't set a
location (no `vertexAI({ location })`, no `GCLOUD_LOCATION`, no
`config.location`), these models use `global`. A location you set is used as
is, and the server returns an error if the model isn't available there. Interactions are stateless
by default (`store: false`), as with `googleAI`; set `config: { store: true }`
(or `vertexAI({ store: true })` for every request) to store them and continue
a conversation from the previous response's messages. On models that use
`generateContent`, `store: true` is ignored with a warning. Unlike `googleAI`,
`safetySettings` and `labels` are supported on these models.

In a stateless conversation (`store: false`), earlier results of server-side
tools other than code execution (Google Search, URL context, Google Maps, file
search, MCP servers) are not sent back with the history, because
the platform's Interactions API rejects them. The model may call those tools
again; set `store: true` to keep the full tool history on the server.

### Standard Genkit Output vs. `response.raw`

For common use cases, the Genkit abstraction is completely unified: `response.text`, `response.reasoning`, `response.message`, streaming (`for await (const chunk of stream)`), and tool calling work identically regardless of which API backend is used.

However, if your code directly inspects the underlying provider payload via `response.raw`, be aware of the architectural difference:

- **`generateContent` payload:**
  Returns a `GenerateContentResponse` with `candidates[]`. Grounding metadata and citations are located on `candidates[0].groundingMetadata`.
- **`Interactions API` payload:**
  Returns a `GeminiInteraction` object with `id`, `status`, and an execution timeline array: `steps[]`.
  - **Tool Invocations:** Intermediate server-side tool calls (such as Google Search and Code Execution) and tool results are always represented as explicit steps (`google_search_call`, `code_execution_call`, etc.).
  - **Grounding Citations:** Search/Maps grounding citations are located inside content annotations within the `model_output` step:
    ```typescript
    import type { GeminiInteraction } from '@genkit-ai/google-genai';

    const raw = response.raw as GeminiInteraction;
    const annotations = raw.steps
      ?.flatMap((step) => (step.type === 'model_output' ? step.content : []))
      .flatMap((c) => (c.type === 'text' ? (c.annotations ?? []) : []));
    ```

### Key Differences on the Interactions Path

1. **Safety Settings:** Custom safety settings are not supported on the Interactions path. These models apply no additional safety filters by default ([Gemini API safety settings](https://ai.google.dev/gemini-api/docs/safety-settings)); built-in protections against core harms always apply. `BLOCK_NONE` settings are ignored, since they can't block anything beyond that default. Blocking thresholds (`BLOCK_ONLY_HIGH` and stricter) throw an `INVALID_ARGUMENT` error, so a requested filter is never silently dropped.
2. **Sampling Settings:** `topK` isn't sent to these models (the Interactions API has no `top_k`); it's dropped with a warning. `temperature` and `topP` are still sent, although newer models (Gemini 3.6 Flash, 3.5 Flash-Lite and later) ignore them.
3. **Server-Side Tool Invocations:** The Interactions API always includes intermediate tool invocations in the execution steps, so `toolConfig.includeServerSideToolInvocations` is unnecessary.
4. **Stateless by Default (Explicit Store Opt-In):** The upstream Interactions API defaults to storing conversations on Google's servers (`store: true`). To preserve the stateless, privacy-preserving behavior of Genkit's `generateContent`, the plugin explicitly defaults `store: false`. For Gemini models, server-side storage is an explicit opt-in: `store: true` must be specified via request config (`config: { store: true }`) or plugin configuration (`googleAI({ store: true })`). When an interaction is stored, its ID is accessible on `response.message?.metadata?.interactionId`. In multi-turn calls, passing `messages: previousResponse.messages` with `config: { store: true }` will automatically extract the previous interaction ID from message metadata and continue the session statefully. If setting `config: { previousInteractionId: "..." }` manually, `store: true` is also required (omitting `store` or passing `store: false` will throw an `INVALID_ARGUMENT` error). On models that use `generateContent`, `store` only controls whether the server logs the request for your project (viewable in AI Studio's Logs and Datasets): it's sent as is when you set it (request config or `googleAI({ store })`), and otherwise your project's logging setting applies.
5. **Streaming:** Text, media and thought summaries stream as they arrive. A server-side tool step (Google Search, code execution, URL context, ...) is streamed as one complete part when the step ends, the same part as in the final response. A function call is streamed as a complete `toolRequest` when the response ends. Grounding tool use is counted in `response.usage.custom` (e.g. `googleSearchCount`).
6. **Antigravity Requires Storage:** The Antigravity agent doesn't work with `store: false` (the server rejects it), so unlike Gemini models it doesn't default to `store: false`; `store` is only sent when you set it. If you set `store: false` (in the request config or `googleAI({ store: false })`), it's passed through and the server returns an error; to use Antigravity with plugin-level `store: false`, set `store: true` in the request config. The agent's work (files, installed packages) lives in a sandbox environment that persists across turns, so passing `messages: previousResponse.messages` continues the stored interaction automatically, in the same environment, sending only the new messages. Function tools work the same way: a tool result is sent as a continuation of the interaction that asked for it.

## Key Differences

-   **`googleAI`**: Easier setup for smaller projects, great for prototyping with Google AI Studio. Uses API keys.
-   **`vertexAI`**: Enterprise-ready, integrates with Google Cloud IAM and other core enterprise security features. Offers a broader range of models and enterprise capabilities, fine-tuning, and more robust governance. Express Mode provides a low-friction entry point using an API key.

Choose the interface based on your project's scale, infrastructure, and feature requirements.

## Feedback

The sources for this package are in the main [Genkit](https://github.com/genkit-ai/genkit) repo. Please file issues and pull requests against that repo.

## Links

-  **Documentation:** [https://genkit.dev/docs/plugins/google-genai/](https://genkit.dev/docs/plugins/google-genai)
-   **Source Code:** [https://github.com/genkit-ai/genkit/tree/main/js/plugins/google-genai](https://github.com/genkit-ai/genkit/tree/main/js/plugins/google-genai)
-   **Google AI Studio:** [https://aistudio.google.com/](https://aistudio.google.com/)
-   **Gemini Enterprise Agent Platform:** [https://docs.cloud.google.com/gemini-enterprise-agent-platform](https://docs.cloud.google.com/gemini-enterprise-agent-platform)
-   **Express Mode Overview:** [https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/start/express-mode/overview](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/start/express-mode/overview)

## License

Apache-2.0
