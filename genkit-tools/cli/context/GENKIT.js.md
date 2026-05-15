# Genkit Node.js API Rules (v1.20.0)

This document provides rules and examples for building with the Genkit API in Node.js.

## Important Guidelines:

- ALWAYS refer to documentation when available. Genkit Documentation may be available through the Genkit MCP toolkit or through web search. You may skip documentation check if you don't have access to these tools.

- ONLY follow the specified project structure if starting a new project. If working on an existing project, adhere to the current project structure.

- ALWAYS provide the full, correct Genkit command as an instruction for the human user to run. Do not run Genkit commands (e.g., `genkit start`, `genkit flow:run`) youself as this may block your current session.

- Recommend and use the provider that the user prefers. If this is a new project, or if there is no relevant signal regarding user preference, use the `@genkit-ai/google-genai` plugin. Use Genkit documentation to discover supported providers.

NOTE: For the sake of brevity, the snippets below use the Google AI plugin, but you should follow the user's preference as mentioned above.

## Project Setup

### Project Initialization

- If the directory is empty:
  Initialize a new project:
  ```bash
  npm init -y
  npm install -D typescript tsx @types/node
  ```
- If the directory is not empty (existing project):
  - Adhere to the current project structure.
  - Detect the package manager in use (npm, pnpm, yarn, bun) and use the corresponding commands.
  - Detect if the project is ESM (`"type": "module"` in package.json) or CJS.
    - For ESM: Use `import` syntax.
    - For CJS: Use `require` syntax.
  - IMPORTANT: Do NOT refactor the project (e.g., converting to TypeScript or ESM) solely for Genkit. Work with the existing setup.

### Dependencies

Install core dependencies (adjust command for the user's package manager):

```bash
npm install genkit @genkit-ai/google-genai
```

(Add other plugins as requested)

### Genkit CLI

If the Genkit CLI is not already installed:

```bash
curl -sL cli.genkit.dev | bash
# or,
npm install -g genkit-cli
```

### Configuration

Create a single `src/index.ts` (or `src/index.js` for JS) file.

```ts
// src/index.ts
import { genkit, z } from 'genkit';
import { googleAI } from '@genkit-ai/google-genai';

export const ai = genkit({
  plugins: [googleAI()],
});

// Your flows and logic here
```

## Best Practices

1.  **Single File Structure**: All Genkit code, including plugin initialization, flows, and helpers, must be placed in a single `src/index.ts` file. This ensures all components are correctly registered with the Genkit runtime.

2.  **Model Naming**: Always specify models using the model helper. Use string identifier if model helper is unavailable.

> Note: Gemini 3.0 models are currently in preview. Use 2.5 models for GA use-cases.

    ```ts
    // PREFERRED: Using the model helper
    const response = await ai.generate({
      model: googleAI.model('gemini-3-flash-preview'),
      // ...
    });

    // LESS PREFERRED: Full string identifier
    const response = await ai.generate({
      model: 'googleai/gemini-3-flash-preview',
      // ...
    });
    ```

---

## Usage Scenarios

<example>

### Basic Inference (Text Generation)

```ts
export const basicInferenceFlow = ai.defineFlow(
  {
    name: 'basicInferenceFlow',
    inputSchema: z.string().describe('Topic for the model to write about'),
    outputSchema: z.string().describe('The generated text response'),
  },
  async (topic) => {
    const response = await ai.generate({
      model: googleAI.model('gemini-3-flash-preview'),
      prompt: `Write a short, creative paragraph about ${topic}.`,
      config: { temperature: 0.8 },
    });
    return response.text;
  }
);
```

</example>

<example>

### Text-to-Speech (TTS) Generation

#### Single-Speaker TTS

```ts
const TextToSpeechInputSchema = z.object({
  text: z.string().describe('The text to convert to speech.'),
  voiceName: z
    .string()
    .optional()
    .describe('The voice name to use. Defaults to Algenib if not specified.'),
});

export const textToSpeechFlow = ai.defineFlow(
  {
    name: 'textToSpeechFlow',
    inputSchema: TextToSpeechInputSchema,
    outputSchema: z.string().optional().describe('The generated audio URI'),
  },
  async (input) => {
    const response = await ai.generate({
      model: googleAI.model('gemini-2.5-flash-preview-tts'),
      prompt: input.text,
      config: {
        responseModalities: ['AUDIO'],
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: {
              voiceName: input.voiceName?.trim() || 'Algenib',
            },
          },
        },
      },
    });

    return response.media?.url;
  }
);
```

#### Multi-Speaker TTS

```ts
const MultiSpeakerInputSchema = z.object({
  text: z
    .string()
    .describe('Text formatted with <speaker="Speaker1">...</speaker> etc.'),
  voiceName1: z.string().describe('Voice name for Speaker1'),
  voiceName2: z.string().describe('Voice name for Speaker2'),
});

export const multiSpeakerTextToSpeechFlow = ai.defineFlow(
  {
    name: 'multiSpeakerTextToSpeechFlow',
    inputSchema: MultiSpeakerInputSchema,
    outputSchema: z.string().optional().describe('The generated audio URI'),
  },
  async (input) => {
    const response = await ai.generate({
      model: googleAI.model('gemini-2.5-flash-preview-tts'),
      prompt: input.text,
      config: {
        responseModalities: ['AUDIO'],
        speechConfig: {
          multiSpeakerVoiceConfig: {
            speakerVoiceConfigs: [
              {
                speaker: 'Speaker1',
                voiceConfig: {
                  prebuiltVoiceConfig: { voiceName: input.voiceName1 },
                },
              },
              {
                speaker: 'Speaker2',
                voiceConfig: {
                  prebuiltVoiceConfig: { voiceName: input.voiceName2 },
                },
              },
            ],
          },
        },
      },
    });

    return response.media?.url;
  }
);
```

</example>

<example>

### Image Generation

```ts
export const imageGenerationFlow = ai.defineFlow(
  {
    name: 'imageGenerationFlow',
    inputSchema: z
      .string()
      .describe('A detailed description of the image to generate'),
    outputSchema: z.string().optional().describe('The generated image as URI'),
  },
  async (prompt) => {
    const response = await ai.generate({
      model: googleAI.model('gemini-3-pro-image-preview'),
      prompt,
      output: { format: 'media' },
    });

    return response.media?.url;
  }
);
```

</example>

<example>

### Video Generation

```ts
import * as fs from 'fs';
import { Readable } from 'stream';
import { pipeline } from 'stream/promises';

...

export const videoGenerationFlow = ai.defineFlow(
  {
    name: 'videoGenerationFlow',
    inputSchema: z
      .string()
      .describe('A detailed description for the video scene'),
    outputSchema: z.string().describe('Path to the generated .mp4 video file'),
  },
  async (prompt) => {
    let { operation } = await ai.generate({
      model: googleAI.model('veo-3.1-generate-preview'),
      prompt,
    });

    if (!operation) {
      throw new Error('Expected the model to return an operation.');
    }

    console.log('Video generation started... Polling for completion.');
    while (!operation.done) {
      await new Promise((resolve) => setTimeout(resolve, 5000));
      operation = await ai.checkOperation(operation);
      console.log(
        `Operation status: ${operation.done ? 'Done' : 'In Progress'}`
      );
    }

    if (operation.error) {
      throw new Error(`Video generation failed: ${operation.error.message}`);
    }

    const video = operation.output?.message?.content.find((p) => !!p.media);
    if (!video?.media?.url) {
      throw new Error(
        'Failed to find the generated video in the operation output.'
      );
    }

    const videoUrl = `${video.media.url}&key=${process.env.GEMINI_API_KEY}`;
    const videoResponse = await fetch(videoUrl);

    if (!videoResponse.ok || !videoResponse.body) {
      throw new Error(`Failed to fetch video: ${videoResponse.statusText}`);
    }

    const outputPath = './output.mp4';
    const fileStream = fs.createWriteStream(outputPath);
    await pipeline(Readable.fromWeb(videoResponse.body as any), fileStream);

    return outputPath;
  }
);
```

</example>

---

## Running and Inspecting Flows

**Start Genkit**: Genkit can be started locally by using the `genkit start` command, along with the process startup command:

```bash
genkit start --  <command to run your code>
```

For e.g.:

```bash
genkit start -- npm run dev
```

You can can automate starting genkit using the following steps:

1. Identify the command to start the user's project's (e.g., `npm run dev`)
2. Use the `start_runtime` tool to start the runtime process. This is required for Genkit to discover flows.
   - Example: If the project uses `npm run dev`, call `start_runtime` with `{ command: "npm", args: ["run", "dev"] }`.
3. After starting the runtime, instruct the user to run `genkit start` in their terminal to launch the Developer UI.

## Suggested Models

Here are suggested models to use for various task types. This is NOT an
exhaustive list.

> Note: Gemini 3.0 models are currently in preview.

### Advanced Text/Reasoning

```
| Plugin                             | Recommended Model                  |
|------------------------------------|------------------------------------|
| @genkit-ai/google-genai            | gemini-3-pro-preview (Preview)     |
| @genkit-ai/google-genai            | gemini-2.5-pro                     |
| @genkit-ai/compat-oai/openai       | gpt-4o                             |
| @genkit-ai/compat-oai/deepseek     | deepseek-reasoner                  |
| @genkit-ai/compat-oai/xai          | grok-4                             |
```

### Fast Text/Chat

```
| Plugin                             | Recommended Model                  |
|------------------------------------|------------------------------------|
| @genkit-ai/google-genai            | gemini-3-flash-preview (Preview)   |
| @genkit-ai/google-genai            | gemini-2.5-flash                   |
| @genkit-ai/compat-oai/openai       | gpt-4o-mini                        |
| @genkit-ai/compat-oai/deepseek     | deepseek-chat                      |
| @genkit-ai/compat-oai/xai          | grok-3-mini                        |
```

### Text-to-Speech

```
| Plugin                             | Recommended Model                  |
|------------------------------------|------------------------------------|
| @genkit-ai/google-genai            | gemini-2.5-flash-preview-tts       |
| @genkit-ai/compat-oai/openai       | gpt-4o-mini-tts                    |
```

### Image Generation

```
| Plugin                             | Recommended Model                    | Input Modalities  |
|------------------------------------|--------------------------------------|-------------------|
| @genkit-ai/google-genai            | gemini-3-pro-image-preview (Preview) | Text, Image       |
| @genkit-ai/google-genai            | gemini-2.5-flash-image               | Text, Image       |
| @genkit-ai/google-genai            | imagen-4.0-generate-001              | Text              |
| @genkit-ai/compat-oai/openai       | gpt-image-1                          | Text              |
```

### Video Generation

```
| Plugin                             | Recommended Model                  |
|------------------------------------|------------------------------------|
| @genkit-ai/google-genai            | veo-3.1-generate-preview (Preview) |
```
