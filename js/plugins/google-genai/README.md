# Google GenAI plugin for Genkit

This plugin provides a unified interface to connect with Google's generative AI models, offering access through both the **Gemini API** and the **Gemini Enterprise Agent Platform**. It is a replacement for the previous `googleAI` and `vertexAI` plugins.

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

### Image Generation

**With `googleAI` (Imagen):**
```typescript
import { genkit } from 'genkit';
import { googleAI } from '@genkit-ai/google-genai';

const ai = genkit({
  plugins: [googleAI()],
});

const response = await ai.generate({
  model: googleAI.model('imagen-4.0-generate-001'),
  prompt: 'A beautiful watercolor painting of a castle in the mountains.',
});

const generatedImage = response.media();
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

const generatedImage = response.media();
```

### Video Generation (Veo)

**With `googleAI`:**
```typescript
import { genkit } from 'genkit';
import { googleAI } from '@genkit-ai/google-genai';

const ai = genkit({
  plugins: [googleAI()],
});

let { operation } = await ai.generate({
  model: googleAI.model('veo-3.1-generate-preview'),
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

**With `vertexAI`:**
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

const generatedAudio = response.media();
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

const generatedAudio = response.media();
```

### Autonomous Agents (Deep Research & Antigravity - Gemini API Only)

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
  model: googleAI.model('antigravity-preview-05-2026'),
  prompt: 'Read Hacker News, summarize the top 10 stories.',
});
```

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