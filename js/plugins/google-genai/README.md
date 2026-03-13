# Google GenAI plugin for Genkit

This plugin provides a unified interface to connect with Google's generative AI models, offering access through both the **Gemini Developer API** and **Vertex AI**. It is a replacement for the previous `googleAI` and `vertexAI` plugins.

Official documentation:

- [Genkit + Gemini Developer API](https://genkit.dev/docs/integrations/google-genai/)
- [Genkit + Vertex AI](https://genkit.dev/docs/integrations/vertex-ai/)

## Installation

```bash
npm i --save @genkit-ai/google-genai
```

## Configuration

This unified plugin exports two main initializers:

- `googleAI`: Allows access to models via the Gemini Developer API using API key authentication.
- `vertexAI`: Allows access to models via Google Cloud Vertex AI. Authentication can be done via Google Cloud Application Default Credentials (ADC) or a simpler API Key for Express Mode.

You can configure one or both in your Genkit setup depending on your needs.

### Using the Gemini Developer API (`googleAI`)

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

### Using Vertex AI (`vertexAI`)

Suitable for applications leveraging Google Cloud's AI infrastructure.

**Authentication Methods:**

-   **Application Default Credentials (ADC):** The standard method for most Vertex AI use cases, especially in production. It uses the credentials from the environment (e.g., service account on GCP, user credentials from `gcloud auth application-default login` locally). This method requires a Google Cloud Project with billing enabled and the Vertex AI API enabled.
-   **Vertex AI Express Mode:** A streamlined way to try out many Vertex AI features using just an API key, without needing to set up billing or full project configurations. This is ideal for quick experimentation and has generous free tier quotas. [Learn More about Express Mode](https://cloud.google.com/vertex-ai/generative-ai/docs/start/express-mode/overview).

```typescript
import { genkit } from 'genkit';
import { vertexAI } from '@genkit-ai/google-genai';

const ai = genkit({
  plugins: [
    // Using Application Default Credentials (Recommended for full features)
    vertexAI({ location: 'us-central1' }), // Regional endpoint
    // vertexAI({ location: 'global' }),      // Global endpoint

    // OR

    // Using Vertex AI Express Mode (Easy to start, some limitations)
    // Get an API key from the Vertex AI Studio Express Mode setup.
    // vertexAI({ apiKey: process.env.VERTEX_EXPRESS_API_KEY }),
  ],
});
```

*Note: When using Express Mode, you do not provide `projectId` and `location` in the plugin config.*

### Using Both Google AI and Vertex AI

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
  model: googleAI.model('gemini-2.5-flash'),
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
  model: vertexAI.model('gemini-2.5-pro'),
  prompt: 'Explain Vertex AI in simple terms.',
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
  embedder: googleAI.embedder('gemini-embedding-001'),
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

### Image Generation (Imagen)

**With `googleAI`:**
```typescript
import { genkit } from 'genkit';
import { googleAI } from '@genkit-ai/google-genai';

const ai = genkit({
  plugins: [googleAI()],
});

const response = await ai.generate({
  model: googleAI.model('imagen-3.0-generate-002'),
  prompt: 'A beautiful watercolor painting of a castle in the mountains.',
});

const generatedImage = response.media();
```

**With `vertexAI`:**
```typescript
import { genkit } from 'genkit';
import { vertexAI } from '@genkit-ai/google-genai';

const ai = genkit({
  plugins: [vertexAI()],
});

const response = await ai.generate({
  model: vertexAI.model('imagen-3.0-generate-002'),
  prompt: 'A beautiful watercolor painting of a castle in the mountains.',
});

const generatedImage = response.media();
```

### Music Generation (Lyria - Vertex AI Only)

```typescript
import { genkit } from 'genkit';
import { vertexAI } from '@genkit-ai/google-genai';

const ai = genkit({
  plugins: [vertexAI()],
});

const response = await ai.generate({
  model: vertexAI.model('lyria-002'),
  prompt: 'A relaxing, instrumental piano track.',
});

const generatedAudio = response.media();
```

## Key Differences

-   **`googleAI`**: Easier setup for smaller projects, great for prototyping with Google AI Studio. Uses API keys.
-   **`vertexAI`**: Enterprise-ready, integrates with Google Cloud IAM and other Vertex AI services. Offers a broader range of models and features like Lyria, fine-tuning, and more robust governance. Vertex AI Express Mode provides a low-friction entry point using an API key.

Choose the interface based on your project's scale, infrastructure, and feature requirements.

## Feedback

The sources for this package are in the main [Genkit](https://github.com/firebase/genkit) repo. Please file issues and pull requests against that repo.

## Links

-  **Documentation:** [https://genkit.dev/docs/plugins/google-genai/](https://genkit.dev/docs/plugins/google-genai)
-   **Source Code:** [https://github.com/firebase/genkit/tree/main/js/plugins/google-genai](https://github.com/firebase/genkit/tree/main/js/plugins/google-genai)
-   **Google AI Studio:** [https://aistudio.google.com/](https://aistudio.google.com/)
-   **Vertex AI:** [https://cloud.google.com/vertex-ai](https://cloud.google.com/vertex-ai)
-   **Vertex AI Express Mode:** [https://cloud.google.com/vertex-ai/generative-ai/docs/start/express-mode/overview](https://cloud.google.com/vertex-ai/generative-ai/docs/start/express-mode/overview)

## License

Apache-2.0
