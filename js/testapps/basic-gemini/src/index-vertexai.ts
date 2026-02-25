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

import { vertexAI } from '@genkit-ai/google-genai';
import * as fs from 'fs';
import { genkit, Operation, Part, StreamingCallback, z } from 'genkit';
import wav from 'wav';
import { RpgCharacterSchema } from './types';

const ai = genkit({
  plugins: [
    // Make sure your Application Default Credentials are set
    vertexAI({ experimental_debugTraces: true, location: 'global' }),
  ],
});

// Basic Hi
ai.defineFlow('basic-hi', async () => {
  const { text } = await ai.generate({
    model: vertexAI.model('gemini-2.5-flash'),
    prompt: 'You are a helpful AI assistant named Walt, say hello',
  });

  return text;
});

// Gemini 3.0 thinkingLevel config
ai.defineFlow(
  {
    name: 'thinking-level-pro',
    inputSchema: z.enum(['LOW', 'HIGH']),
    outputSchema: z.any(),
  },
  async (level) => {
    const { text } = await ai.generate({
      model: vertexAI.model('gemini-3-pro-preview'),
      prompt: [
        'Alice, Bob, and Carol each live in a different house on the ',
        'same street: red, green, and blue. The person who lives in the red house ',
        'owns a cat. Bob does not live in the green house. Carol owns a dog. The ',
        'green house is to the left of the red house. Alice does not own a cat. ',
        'The person in the blue house owns a fish. ',
        'Who lives in each house, and what pet do they own? Provide your ',
        'step-by-step reasoning.',
      ].join(''),
      config: {
        location: 'global',
        thinkingConfig: {
          thinkingLevel: level,
        },
      },
    });
    return text;
  }
);

// Gemini 3.1 thinkingLevel config
ai.defineFlow(
  {
    name: 'thinking-level-3.1-pro',
    inputSchema: z.enum(['LOW', 'MEDIUM', 'HIGH']),
    outputSchema: z.any(),
  },
  async (level) => {
    const { text } = await ai.generate({
      model: vertexAI.model('gemini-3.1-pro-preview'),
      prompt: [
        'Alice, Bob, and Carol each live in a different house on the ',
        'same street: red, green, and blue. The person who lives in the red house ',
        'owns a cat. Bob does not live in the green house. Carol owns a dog. The ',
        'green house is to the left of the red house. Alice does not own a cat. ',
        'The person in the blue house owns a fish. ',
        'Who lives in each house, and what pet do they own? Provide your ',
        'step-by-step reasoning.',
      ].join(''),
      config: {
        location: 'global',
        thinkingConfig: {
          thinkingLevel: level,
        },
      },
    });
    return text;
  }
);

ai.defineFlow(
  {
    name: 'thinking-level-flash',
    inputSchema: z.enum(['MINIMAL', 'LOW', 'MEDIUM', 'HIGH']),
    outputSchema: z.any(),
  },
  async (level) => {
    const { text } = await ai.generate({
      model: vertexAI.model('gemini-3-flash-preview'),
      prompt: [
        'Alice, Bob, and Carol each live in a different house on the ',
        'same street: red, green, and blue. The person who lives in the red house ',
        'owns a cat. Bob does not live in the green house. Carol owns a dog. The ',
        'green house is to the left of the red house. Alice does not own a cat. ',
        'The person in the blue house owns a fish. ',
        'Who lives in each house, and what pet do they own? Provide your ',
        'step-by-step reasoning.',
      ].join(''),
      config: {
        location: 'global',
        thinkingConfig: {
          thinkingLevel: level,
        },
      },
    });
    return text;
  }
);

// Multimodal input
ai.defineFlow('multimodal-input', async () => {
  const photoBase64 = fs.readFileSync('photo.jpg', { encoding: 'base64' });

  const { text } = await ai.generate({
    model: vertexAI.model('gemini-2.5-flash'),
    config: {
      location: 'global',
    },
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

// YouTube videos
ai.defineFlow('youtube-videos', async (_, { sendChunk }) => {
  const { text } = await ai.generate({
    model: vertexAI.model('gemini-2.5-flash'),
    prompt: [
      {
        text: 'transcribe this video',
      },
      {
        media: {
          url: 'https://www.youtube.com/watch?v=3p1P5grjXIQ',
          contentType: 'video/mp4',
        },
        // Metadata is optional. You can leave it out if you
        // want the whole video at default fps.
        metadata: {
          videoMetadata: {
            fps: 0.5,
            startOffset: '3.5s',
            endOffset: '10.2s',
          },
        },
      },
    ],
  });

  return text;
});

export const videoUnderstanding = ai.defineFlow(
  {
    name: 'video-understanding-metadata',
    inputSchema: z.void(),
    outputSchema: z.any(),
  },
  async () => {
    const llmResponse = await ai.generate({
      model: vertexAI.model('gemini-2.5-flash'),
      prompt: [
        {
          media: {
            url: 'gs://cloud-samples-data/video/animals.mp4',
            contentType: 'video/mp4',
          },
          metadata: {
            videoMetadata: {
              fps: 0.5,
              startOffset: '3.5s',
              endOffset: '10.2s',
            },
          },
        },
        {
          text: 'describe this video',
        },
      ],
    });
    return llmResponse.text;
  }
);

// streaming
ai.defineFlow('streaming', async (_, { sendChunk }) => {
  const { stream } = ai.generateStream({
    model: vertexAI.model('gemini-2.5-flash'),
    prompt: 'Write a poem about AI.',
  });

  let poem = '';
  for await (const chunk of stream) {
    poem += chunk.text;
    sendChunk(chunk.text);
  }

  return poem;
});

// Google maps grounding
ai.defineFlow('maps-grounding', async () => {
  const { text, raw } = await ai.generate({
    model: vertexAI.model('gemini-2.5-flash'),
    prompt: 'Describe some sights near me',
    config: {
      tools: [
        {
          googleMaps: {
            enableWidget: true,
          },
        },
      ],
      retrievalConfig: {
        latLng: {
          latitude: 43.0896,
          longitude: -79.0849,
        },
      },
    },
  });

  return {
    text,
    groundingMetadata: (raw as any)?.candidates[0]?.groundingMetadata,
  };
});

// Search grounding
ai.defineFlow('search-grounding', async () => {
  const { text, raw } = await ai.generate({
    model: vertexAI.model('gemini-2.5-flash'),
    prompt: 'Who is Albert Einstein?',
    config: {
      tools: [{ googleSearch: {} }],
    },
  });

  return {
    text,
    groundingMetadata: (raw as any)?.candidates[0]?.groundingMetadata,
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
    description: 'used to get current weather for a location',
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

const celsiusToFahrenheit = ai.defineTool(
  {
    name: 'celsiusToFahrenheit',
    inputSchema: z.object({
      celsius: z.number().describe('Temperature in Celsius'),
    }),
    description: 'Converts Celsius to Fahrenheit',
  },
  async ({ celsius }) => {
    return (celsius * 9) / 5 + 32;
  }
);

// Tool calling with Gemini
ai.defineFlow(
  {
    name: 'toolCalling',
    inputSchema: z.string().default('Paris, France'),
    outputSchema: z.string(),
    streamSchema: z.any(),
  },
  async (location, { sendChunk }) => {
    const { response, stream } = ai.generateStream({
      model: vertexAI.model('gemini-2.5-flash'),
      config: {
        temperature: 1,
      },
      tools: [getWeather, celsiusToFahrenheit],
      prompt: `What's the weather in ${location}? Convert the temperature to Fahrenheit.`,
    });

    for await (const chunk of stream) {
      sendChunk(chunk);
    }

    return (await response).text;
  }
);

ai.defineFlow(
  {
    name: 'streamingToolCalling',
    inputSchema: z.string().default('Paris, France'),
    outputSchema: z.string(),
    streamSchema: z.any(),
  },
  async (location, { sendChunk }) => {
    const { response, stream } = ai.generateStream({
      model: vertexAI.model('gemini-3-pro-preview'),
      config: {
        temperature: 1,
        functionCallingConfig: {
          streamFunctionCallArguments: true,
        },
      },
      tools: [getWeather, celsiusToFahrenheit],
      prompt: `What's the weather in ${location}? Convert the temperature to Fahrenheit.`,
    });

    for await (const chunk of stream) {
      sendChunk(chunk);
    }

    return (await response).text;
  }
);

// Tool calling with structured output
ai.defineFlow(
  {
    name: 'structured-tool-calling',
    inputSchema: z.string().default('Paris, France'),
    outputSchema: z
      .object({
        temp: z.number(),
        unit: z.enum(['F', 'C']),
      })
      .nullable(),
    streamSchema: z.any(),
  },
  async (location, { sendChunk }) => {
    const { response, stream } = ai.generateStream({
      model: vertexAI.model('gemini-2.5-flash'),
      config: {
        temperature: 1,
      },
      output: {
        schema: z.object({
          temp: z.number(),
          unit: z.enum(['F', 'C']),
        }),
      },
      tools: [getWeather, celsiusToFahrenheit],
      prompt: `What's the weather in ${location}? Convert the temperature to Fahrenheit.`,
    });

    for await (const chunk of stream) {
      sendChunk(chunk.output);
    }

    return (await response).output;
  }
);

// A simple example of structured output.
ai.defineFlow(
  {
    name: 'structured-output',
    inputSchema: z.string().default('Glorb'),
    outputSchema: RpgCharacterSchema,
  },
  async (name, { sendChunk }) => {
    const { response, stream } = ai.generateStream({
      model: vertexAI.model('gemini-2.5-flash'),
      config: {
        temperature: 2, // we want creativity
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

// Gemini reasoning example.
ai.defineFlow('reasoning', async (_, { sendChunk }) => {
  const { message } = await ai.generate({
    prompt: 'what is heavier, one kilo of steel or one kilo of feathers',
    model: vertexAI.model('gemini-2.5-pro'),
    config: {
      thinkingConfig: {
        thinkingBudget: 1024,
        includeThoughts: true,
      },
    },
    onChunk: sendChunk,
  });

  return message;
});

// Media resolution
ai.defineFlow('gemini-media-resolution', async (_) => {
  const plant = fs.readFileSync('palm_tree.png', { encoding: 'base64' });
  const { text } = await ai.generate({
    model: vertexAI.model('gemini-3-pro-preview'),
    prompt: [
      { text: 'What is in this picture?' },
      {
        media: { url: `data:image/png;base64,${plant}` },
        metadata: {
          mediaResolution: {
            // Or MEDIA_RESOLUTION_LOW Or MEDIA_RESOLUTION_MEDIUM
            level: 'MEDIA_RESOLUTION_HIGH',
          },
        },
      },
    ],
  });

  return text;
});

// Image editing with Gemini.
ai.defineFlow('gemini-image-editing', async (_) => {
  const plant = fs.readFileSync('palm_tree.png', { encoding: 'base64' });
  const room = fs.readFileSync('my_room.png', { encoding: 'base64' });

  const { media } = await ai.generate({
    model: vertexAI.model('gemini-2.5-flash-image'),
    prompt: [
      { text: 'add the plant to my room' },
      { media: { url: `data:image/png;base64,${plant}` } },
      { media: { url: `data:image/png;base64,${room}` } },
    ],
    config: {
      responseModalities: ['TEXT', 'IMAGE'],
    },
  });

  return media;
});

// Nano banana pro config
ai.defineFlow('nano-banana-pro', async (_) => {
  const { media } = await ai.generate({
    model: vertexAI.model('gemini-3-pro-image-preview'),
    prompt: 'Generate a picture of a sunset in the mountains by a lake',
    config: {
      responseModalities: ['TEXT', 'IMAGE'],
      imageConfig: {
        aspectRatio: '21:9',
        imageSize: '4K',
      },
    },
  });

  return media;
});

// A simple example of image generation with Gemini.
ai.defineFlow('imagen-image-generation', async (_) => {
  const { media } = await ai.generate({
    model: vertexAI.model('imagen-3.0-generate-002'),
    prompt: `generate an image of a banana riding a bicycle`,
  });

  return media;
});

async function waitForOperation(
  operation?: Operation,
  sendChunk?: StreamingCallback<any>
) {
  if (!operation) {
    throw new Error('Expected the model to return an operation');
  }

  while (!operation.done) {
    sendChunk?.('check status of operation ' + operation.id);
    operation = await ai.checkOperation(operation);
    await new Promise((resolve) => setTimeout(resolve, 5000));
  }

  if (operation.error) {
    sendChunk?.('Error: ' + operation.error.message);
    throw new Error('failed to generate video: ' + operation.error.message);
  }

  return operation;
}

// Imagen Try-on
ai.defineFlow('imagen-try-on', async (_) => {
  const person = await fs.promises.readFile('woman.png', {
    encoding: 'base64',
  });
  const product = await fs.promises.readFile('coat.png', {
    encoding: 'base64',
  });

  const { media } = await ai.generate({
    model: vertexAI.model('virtual-try-on-001'),
    prompt: [
      {
        media: {
          url: `data:image/png;base64,${person}`,
          contentType: 'image/png',
        },
        metadata: { type: 'personImage' },
      },
      {
        media: {
          url: `data:image/png;base64,${product}`,
          contentType: 'image/png',
        },
        metadata: { type: 'productImage' },
      },
    ],
  });
  return media;
});

ai.defineFlow('veo-text-prompt', async (_, { sendChunk }) => {
  let { operation } = await ai.generate({
    model: vertexAI.model('veo-3.0-generate-001'),
    prompt: [
      {
        text: 'slowly flying over a meadow in full bloom',
      },
    ],
    config: {
      durationSeconds: 8,
      aspectRatio: '16:9',
      personGeneration: 'allow_adult',
    },
  });

  const doneOp = await waitForOperation(operation, sendChunk);

  const mediaPart = doneOp.output?.message?.content.find(
    (p: Part) => !!p.media
  );
  if (!mediaPart) {
    throw new Error('Failed to find the generated video');
  }

  // Download for now until we have DevUI support for video
  const videoBuffer = Buffer.from(mediaPart.media.url.split(',')[1], 'base64');
  fs.writeFileSync('veo-output.mp4', videoBuffer);

  return mediaPart.media;
});

// An example of using Ver 3 model to make a static photo move.
ai.defineFlow('veo-photo-move', async (_, { sendChunk }) => {
  const startingImage = fs.readFileSync('photo.jpg', { encoding: 'base64' });

  let { operation } = await ai.generate({
    model: vertexAI.model('veo-3.0-generate-001'),
    prompt: [
      {
        text: 'make the subject in the photo move',
      },
      {
        media: {
          contentType: 'image/jpeg',
          url: `data:image/jpeg;base64,${startingImage}`,
        },
      },
    ],
    config: {
      durationSeconds: 8,
      aspectRatio: '9:16',
      personGeneration: 'allow_adult',
    },
  });

  const doneOp = await waitForOperation(operation, sendChunk);

  const mediaPart = doneOp.output?.message?.content.find(
    (p: Part) => !!p.media
  );
  if (!mediaPart) {
    throw new Error('Failed to find the generated video');
  }

  // Download for now until we have DevUI support for video
  const videoBuffer = Buffer.from(mediaPart.media.url.split(',')[1], 'base64');
  fs.writeFileSync('veo-output.mp4', videoBuffer);

  return mediaPart.media;
});

ai.defineFlow('veo-reference-images', async (_, { sendChunk }) => {
  const roomImage = fs.readFileSync('my_room.png', { encoding: 'base64' });
  const palmImage = fs.readFileSync('palm_tree.png', { encoding: 'base64' });

  let { operation } = await ai.generate({
    model: vertexAI.model('veo-3.1-generate-001'),
    config: { location: 'us-central1' },
    prompt: [
      {
        text: 'Give the plant legs and friendly cartoon eyes and have it bounce into the room from the left',
      },
      {
        media: {
          contentType: 'image/png',
          url: `data:image/png;base64,${roomImage}`,
        },
        metadata: {
          type: 'referenceImages',
          referenceType: 'asset',
        },
      },
      {
        media: {
          contentType: 'image/png',
          url: `data:image/png;base64,${palmImage}`,
        },
        metadata: {
          type: 'referenceImages',
          referenceType: 'asset',
        },
      },
    ],
  });

  const doneOp = await waitForOperation(operation, sendChunk);

  const mediaPart = doneOp.output?.message?.content.find(
    (p: Part) => !!p.media
  );
  if (!mediaPart) {
    throw new Error('Failed to find the generated video');
  }

  // Download for now until we have DevUI support for video
  const videoBuffer = Buffer.from(mediaPart.media.url.split(',')[1], 'base64');
  fs.writeFileSync('veo-output.mp4', videoBuffer);

  return mediaPart.media;
});

// Music generation with Lyria
ai.defineFlow('lyria-music-generation', async (_) => {
  const { media } = await ai.generate({
    model: vertexAI.model('lyria-002'),
    config: {
      location: 'global',
    },
    prompt: 'generate a relaxing song with piano and violin',
  });

  if (!media) {
    throw new Error('no media returned');
  }
  const audioBuffer = Buffer.from(
    media.url.substring(media.url.indexOf(',') + 1),
    'base64'
  );
  return {
    media: 'data:audio/wav;base64,' + (await toWav(audioBuffer, 2, 48000)),
  };
});

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

function bytesBase64EncodedReplacer(key: string, value: unknown): unknown {
  const startLength = 200;
  const endLength = 10;
  const totalLength = startLength + endLength;
  if (typeof value === 'string' && value.length > totalLength) {
    const start = value.substring(0, startLength);
    const end = value.substring(value.length - endLength);
    return `${start}...[TRUNCATED]...${end}`;
  }
  return value; // Return the original value for other keys or non-string values
}
