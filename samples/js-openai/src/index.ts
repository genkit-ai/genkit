/**
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

import openAI from '@genkit-ai/compat-oai/openai';
import * as fs from 'fs';
import { genkit, z } from 'genkit';
import wav from 'wav';

const ai = genkit({
  plugins: [
    // Provide the key via the OPENAI_API_KEY environment variable
    openAI(),
  ],
});

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

const RpgCharacterSchema = z.object({
  name: z.string().describe('name of the character'),
  backstory: z.string().describe("character's backstory, about a paragraph"),
  weapons: z.array(z.string()),
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
        voice: 'sage',
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
