To use a Gemini model with Genkit:

```ts
import { ai, z } from '...'; // path to genkit instance
import { googleAI } from '@genkit-ai/google-genai';

const { text } = await ai.generate({
  model: googleAI.model('gemini-2.5-flash'),
  prompt: '...',
});
```

ALWAYS use `gemini-2.5-*` series models, they are the best and current generation of Gemini models. NEVER use `gemini-2.0-*` or `gemini-1.5-*` models. For general purpose inference, use one of these models:

- `gemini-2.5-flash`: balance of speed/performance, good default
- `gemini-2.5-pro`: most powerful, use for complex prompts
- `gemini-2.5-flash-lite`: very fast, use for simple prompts

All of these models can accept multi-modal input, but for image or audio output see the available documentation for specialized models.
