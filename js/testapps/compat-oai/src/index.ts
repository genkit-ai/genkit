/**
 * Copyright 2024 The Fire Company
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import {
  compatOaiModelRef,
  defineCompatOpenAIModel,
  openAICompatible,
} from '@genkit-ai/compat-oai';
import { deepSeek } from '@genkit-ai/compat-oai/deepseek';
import { openAI } from '@genkit-ai/compat-oai/openai';
import { xAI } from '@genkit-ai/compat-oai/xai';
import { startFlowServer } from '@genkit-ai/express';
import dotenv from 'dotenv';
import * as fs from 'fs';
import { genkit, z } from 'genkit';
import wav from 'wav';

dotenv.config();

const DECLARED_MODELS = ['z-ai/glm-4.5-air:free'];

const ai = genkit({
  plugins: [
    openAI(),
    deepSeek(),
    xAI(),
    openAICompatible({
      name: 'openrouter',
      baseURL: 'https://openrouter.ai/api/v1',
      apiKey: process.env['OPENROUTER_API_KEY'],
      async initializer(client) {
        return DECLARED_MODELS.map((model) =>
          defineCompatOpenAIModel({
            name: `openrouter/${model}`,
            client,
            modelRef: compatOaiModelRef({
              name: `openrouter/${model}`,
            }),
          })
        );
      },
    }),
  ],
});

export const jokeFlow = ai.defineFlow(
  {
    name: 'jokeFlow',
    inputSchema: z.string(),
    outputSchema: z.string(),
  },
  async (subject) => {
    const llmResponse = await ai.generate({
      prompt: `tell me a joke about ${subject}`,
      model: compatOaiModelRef({
        name: 'openrouter/z-ai/glm-4.5-air:free',
      }),
    });
    return llmResponse.text;
  }
);

export const modelWrapFlow = ai.defineFlow(
  {
    name: 'modelWrapFlow',
    inputSchema: z.string(),
    outputSchema: z.string(),
    streamSchema: z.string(),
  },
  async (subject, { sendChunk }) => {
    const { stream, response } = ai.generateStream({
      model: deepSeek.model('deepseek-chat'),
      prompt: `tell me a fun fact about ${subject}`,
    });

    for await (const chunk of stream) {
      sendChunk(chunk.text);
    }

    const { text } = await response;

    return text;
  }
);

export const webSearchFlow = ai.defineFlow(
  {
    name: 'webSearchFlow',
    outputSchema: z.string(),
  },
  async () => {
    const llmResponse = await ai.generate({
      prompt: `What was a positive news story from today?`,
      model: openAI.model('gpt-4o-search-preview'),
      config: {
        web_search_options: {},
      },
    });
    return llmResponse.text;
  }
);

export const embedFlow = ai.defineFlow(
  {
    name: 'embedFlow',
    inputSchema: z.string(),
    outputSchema: z.string(),
  },
  async (text) => {
    const embedding = await ai.embed({
      embedder: openAI.embedder('text-embedding-ada-002'),
      content: text,
    });

    return JSON.stringify(embedding);
  }
);

ai.defineFlow('basic-hi', async () => {
  const { text } = await ai.generate({
    model: openAI.model('o4-mini'),
    prompt: 'You are a helpful AI assistant named Walt, say hello',
  });

  return text;
});

// Multimodal input
ai.defineFlow('multimodal-input', async () => {
  const photoBase64 = fs.readFileSync('photo.jpg', { encoding: 'base64' });

  const { text } = await ai.generate({
    model: openAI.model('gpt-4o'),
    prompt: [
      { text: 'describe this photo' },
      {
        media: {
          contentType: 'image/jpeg',
          url: `data:image/jpeg;base64,${photoBase64}`,
        },
      },
    ],
  });

  return text;
});

// Streaming
ai.defineFlow('streaming', async (_, { sendChunk }) => {
  const { stream } = ai.generateStream({
    model: openAI.model('gpt-4o'),
    prompt: 'Write a poem about AI.',
  });

  let poem = '';
  for await (const chunk of stream) {
    poem += chunk.text;
    sendChunk(chunk.text);
  }

  return poem;
});

// Web search
ai.defineFlow('web-search', async () => {
  const response = await ai.generate({
    model: openAI.model('gpt-4o-search-preview'),
    prompt: 'Who is Albert Einstein?',
    config: {
      web_search_options: {},
    },
  });

  return {
    text: response.text,
    annotations: (response.raw as any)?.choices?.[0].message.annotations,
  };
});

const getWeather = ai.defineTool(
  {
    name: 'getWeather',
    inputSchema: z.object({
      location: z
        .string()
        .describe(
          'Location for which to get the weather, ex: San-Francisco, CA'
        ),
    }),
    description: 'can be used to calculate gablorken value',
  },
  async (input) => {
    // pretend we call an actual API
    return {
      location: input.location,
      temperature_celcius: 21.5,
      conditions: 'cloudy',
    };
  }
);

// Tool calling
ai.defineFlow(
  {
    name: 'tool-calling',
    inputSchema: z.string().default('Paris, France'),
    outputSchema: z.string(),
    streamSchema: z.any(),
  },
  async (location, { sendChunk }) => {
    const { response, stream } = ai.generateStream({
      model: openAI.model('gpt-4o'),
      config: {
        temperature: 1,
      },
      tools: [getWeather],
      prompt: `tell what's the weather in ${location} (in Fahrenheit)`,
    });

    for await (const chunk of stream) {
      sendChunk(chunk);
    }

    return (await response).text;
  }
);

const baseCategorySchema = z.object({
  name: z.string(),
});

type Category = z.infer<typeof baseCategorySchema> & {
  subcategories?: Category[];
};

const categorySchema: z.ZodType<Category> = baseCategorySchema.extend({
  subcategories: z
    .lazy(() => categorySchema.array())
    .describe('make sure there are 2-3 levels of subcategories')
    .optional(),
});

const WeaponSchema = z.object({
  name: z.string(),
  damage: z.number(),
  category: categorySchema,
});

const RpgCharacterSchema = z.object({
  name: z.string().describe('name of the character'),
  backstory: z.string().describe("character's backstory, about a paragraph"),
  weapons: z.array(WeaponSchema),
  class: z.enum(['RANGER', 'WIZZARD', 'TANK', 'HEALER', 'ENGINEER']),
});

// A simple example of structured output.
ai.defineFlow(
  {
    name: 'structured-output',
    inputSchema: z.string().default('Glorb'),
    outputSchema: RpgCharacterSchema,
  },
  async (name, { sendChunk }) => {
    const { response, stream } = ai.generateStream({
      model: openAI.model('gpt-4o'),
      config: {
        temperature: 1, // we want creativity
      },
      output: { schema: RpgCharacterSchema },
      prompt: `Generate an RPC character called ${name}`,
    });

    for await (const chunk of stream) {
      sendChunk(chunk.output);
    }

    return (await response).output!;
  }
);

// Image generation.
ai.defineFlow('dall-e-image-generation', async (_, { sendChunk }) => {
  const { media } = await ai.generate({
    model: openAI.model('dall-e-3'),
    prompt: `generate an image of a banana riding bicycle`,
  });

  return media;
});

// TTS sample
ai.defineFlow(
  {
    name: 'tts',
    inputSchema: z.string().default('Genkit is an amazing Gen AI library'),
    outputSchema: z.object({ media: z.string() }),
  },
  async (query) => {
    const { media } = await ai.generate({
      model: openAI.model('gpt-4o-mini-tts'),
      config: {
        voice: 'alloy',
      },
      prompt: query,
    });
    if (!media) {
      throw new Error('no media returned');
    }
    const audioBuffer = Buffer.from(
      media.url.substring(media.url.indexOf(',') + 1),
      'base64'
    );
    return {
      media: 'data:audio/wav;base64,' + (await toWav(audioBuffer)),
    };
  }
);

async function toWav(
  pcmData: Buffer,
  channels = 1,
  rate = 24000,
  sampleWidth = 2
): Promise<string> {
  return new Promise((resolve, reject) => {
    // This code depends on `wav` npm library.
    const writer = new wav.Writer({
      channels,
      sampleRate: rate,
      bitDepth: sampleWidth * 8,
    });

    let bufs = [] as any[];
    writer.on('error', reject);
    writer.on('data', function (d) {
      bufs.push(d);
    });
    writer.on('end', function () {
      resolve(Buffer.concat(bufs).toString('base64'));
    });

    writer.write(pcmData);
    writer.end();
  });
}

// STT sample
ai.defineFlow('transcribe', async () => {
  const audioFile = fs.readFileSync('audio.mp3');

  const { text } = await ai.generate({
    model: openAI.model('whisper-1'),
    prompt: [
      {
        media: {
          contentType: 'audio/mp3',
          url: `data:audio/mp3;base64,${audioFile.toString('base64')}`,
        },
      },
    ],
  });

  return text;
});

// PDF file input example
ai.defineFlow(
  {
    name: 'pdf',
    inputSchema: z.string().default(''),
    outputSchema: z.string(),
  },
  async (pdfPath) => {
    // Use a provided PDF path or create a minimal test PDF
    let pdfBase64: string;

    if (pdfPath && fs.existsSync(pdfPath)) {
      pdfBase64 = fs.readFileSync(pdfPath, { encoding: 'base64' });
    } else {
      // Minimal valid PDF for testing (just contains "Hello World")
      // This is a real PDF that can be parsed
      pdfBase64 =
        'JVBERi0xLjQKJeLjz9MKMSAwIG9iago8PC9UeXBlL0NhdGFsb2cvUGFnZXMgMiAwIFI+PgplbmRvYmoKMiAwIG9iago8PC9UeXBlL1BhZ2VzL0NvdW50IDEvS2lkc1szIDAgUl0+PgplbmRvYmoKMyAwIG9iago8PC9UeXBlL1BhZ2UvTWVkaWFCb3hbMCAwIDYxMiA3OTJdL1BhcmVudCAyIDAgUi9SZXNvdXJjZXM8PC9Gb250PDwvRjE8PC9UeXBlL0ZvbnQvU3VidHlwZS9UeXBlMS9CYXNlRm9udC9IZWx2ZXRpY2E+Pj4+Pj4vQ29udGVudHMgNCAwIFI+PgplbmRvYmoKNCAwIG9iago8PC9MZW5ndGggNDQ+PgpzdHJlYW0KQlQKL0YxIDI0IFRmCjEwMCA3MDAgVGQKKEhlbGxvIFdvcmxkKSBUagpFVAplbmRzdHJlYW0KZW5kb2JqCnhyZWYKMCA1CjAwMDAwMDAwMDAgNjU1MzUgZiAKMDAwMDAwMDAxNSAwMDAwMCBuIAowMDAwMDAwMDY0IDAwMDAwIG4gCjAwMDAwMDAxMjEgMDAwMDAgbiAKMDAwMDAwMDI2MCAwMDAwMCBuIAp0cmFpbGVyCjw8L1NpemUgNS9Sb290IDEgMCBSPj4Kc3RhcnR4cmVmCjM1MgolJUVPRgo=';
    }

    const { text } = await ai.generate({
      model: openAI.model('gpt-4o'),
      prompt: [
        {
          media: {
            contentType: 'application/pdf',
            url: `data:application/pdf;base64,${pdfBase64}`,
          },
        },
        {
          text: 'What text is in this PDF document? Please extract and return all the text you can read.',
        },
      ],
    });

    return text;
  }
);

startFlowServer({
  flows: [jokeFlow, embedFlow],
});
