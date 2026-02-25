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

import { googleAI } from '@genkit-ai/google-genai';
import * as fs from 'fs';
import {
  genkit,
  z,
  type MediaPart,
  type Operation,
  type Part,
  type StreamingCallback,
} from 'genkit';
import { fallback, retry } from 'genkit/model/middleware';
import { Readable } from 'stream';
import wav from 'wav';
import {
  createFileSearchStore,
  deleteFileSearchStore,
  uploadBlobToFileSearchStore,
} from './helper.js';
import { RpgCharacterSchema } from './types.js';

const ai = genkit({
  plugins: [
    // Provide the key via the GOOGLE_GENAI_API_KEY environment variable or arg { apiKey: 'yourkey'}
    googleAI({ experimental_debugTraces: true }),
  ],
});

ai.defineFlow('basic-hi', async () => {
  const { text } = await ai.generate({
    model: googleAI.model('gemini-2.5-flash'),
    prompt: 'You are a helpful AI assistant named Walt, say hello',
  });

  return text;
});

ai.defineFlow('basic-hi-with-retry', async () => {
  const { text } = await ai.generate({
    model: googleAI.model('gemini-2.5-pro'),
    prompt: 'You are a helpful AI assistant named Walt, say hello',
    use: [
      retry({
        maxRetries: 2,
        onError: (e, attempt) => console.log('--- oops ', attempt, e),
      }),
    ],
  });

  return text;
});

ai.defineFlow('basic-hi-with-fallback', async () => {
  const { text } = await ai.generate({
    model: googleAI.model('gemini-2.5-something-that-does-not-exist'),
    prompt: 'You are a helpful AI assistant named Walt, say hello',
    use: [
      fallback(ai, {
        models: [googleAI.model('gemini-2.5-flash')],
        statuses: ['UNKNOWN'],
      }),
    ],
  });

  return text;
});

// Gemini 3.0 thinkingLevel config. Pro can have Low or High
ai.defineFlow(
  {
    name: 'thinking-level-pro',
    inputSchema: z.enum(['LOW', 'HIGH']),
  },
  async (level) => {
    const { text } = await ai.generate({
      model: googleAI.model('gemini-3-pro-preview'),
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
        thinkingConfig: {
          thinkingLevel: level,
        },
      },
    });
    return text;
  }
);

// Gemini 3 Flash can have minimal and medium thinking levels too.
ai.defineFlow(
  {
    name: 'thinking-level-flash',
    inputSchema: z.enum(['MINIMAL', 'LOW', 'MEDIUM', 'HIGH']),
  },
  async (level) => {
    const { text } = await ai.generate({
      model: googleAI.model('gemini-3-flash-preview'),
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
    model: googleAI.model('gemini-2.5-flash'),
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
    model: googleAI.model('gemini-2.5-flash'),
    prompt: [
      {
        text: 'transcribe this video',
      },
      {
        media: {
          url: 'https://www.youtube.com/watch?v=3p1P5grjXIQ',
          contentType: 'video/mp4',
        },
      },
    ],
  });

  return text;
});

// streaming
ai.defineFlow('streaming', async (_, { sendChunk }) => {
  const { stream } = ai.generateStream({
    model: googleAI.model('gemini-2.5-flash'),
    prompt: 'Write a poem about AI.',
  });

  let poem = '';
  for await (const chunk of stream) {
    poem += chunk.text;
    sendChunk(chunk.text);
  }

  return poem;
});

// Search grounding
ai.defineFlow('search-grounding', async () => {
  const { text, raw } = await ai.generate({
    model: googleAI.model('gemini-2.5-flash'),
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

// Url context
ai.defineFlow('url-context', async () => {
  const { text, raw } = await ai.generate({
    model: googleAI.model('gemini-2.5-flash'),
    prompt:
      'Compare the ingredients and cooking times from the recipes at ' +
      'https://www.foodnetwork.com/recipes/ina-garten/perfect-roast-chicken-recipe-1940592 ' +
      'and https://www.allrecipes.com/recipe/70679/simple-whole-roasted-chicken/',
    config: {
      urlContext: {},
    },
  });

  return {
    text,
    groundingMetadata: (raw as any)?.candidates[0]?.groundingMetadata,
  };
});

// File Search
ai.defineFlow('file-search', async () => {
  // Use the google/genai SDK to upload the story BLOB to a new
  // file search store
  const storeName = await createFileSearchStore();
  await uploadBlobToFileSearchStore(storeName);

  // Use the file search store in your generate request
  const { text, raw } = await ai.generate({
    model: googleAI.model('gemini-2.5-flash'),
    prompt: "What is the character's name in the story?",
    config: {
      fileSearch: {
        fileSearchStoreNames: [storeName],
        metadataFilter: 'author=foo',
      },
    },
  });

  // Clean up the file search store again
  await deleteFileSearchStore(storeName);

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

const screenshot = ai.defineTool(
  {
    name: 'screenshot',
    multipart: true,
    description: 'takes a screenshot',
  },
  async () => {
    // pretend we call an actual API
    const picture = fs.readFileSync('my_room.png', { encoding: 'base64' });
    return {
      output: 'success',
      content: [{ media: { url: `data:image/png;base64,${picture}` } }],
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
      model: googleAI.model('gemini-2.5-flash'),
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

// Multipart tool calling
ai.defineFlow(
  {
    name: 'multipart-tool-calling',
    outputSchema: z.string(),
    streamSchema: z.any(),
  },
  async (_, { sendChunk }) => {
    const { response, stream } = ai.generateStream({
      model: googleAI.model('gemini-3-pro-preview'),
      config: {
        temperature: 1,
      },
      tools: [screenshot],
      prompt: `Tell me what I'm seeing on the screen.`,
    });

    for await (const chunk of stream) {
      sendChunk(chunk.output);
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
      model: googleAI.model('gemini-2.5-flash'),
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
      model: googleAI.model('gemini-2.5-flash'),
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
    model: googleAI.model('gemini-2.5-pro'),
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
    model: googleAI.model('gemini-3-pro-preview'),
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
    config: {
      // MediaResolution is currently only supported in v1alpha for googleAI
      apiVersion: 'v1alpha',
    },
  });
  return text;
});

// Image editing with Gemini.
ai.defineFlow('gemini-image-editing', async (_) => {
  const plant = fs.readFileSync('palm_tree.png', { encoding: 'base64' });
  const room = fs.readFileSync('my_room.png', { encoding: 'base64' });

  const { media } = await ai.generate({
    model: googleAI.model('gemini-2.5-flash-image'),
    prompt: [
      { text: 'add the plant to my room' },
      { media: { url: `data:image/png;base64,${plant}` } },
      { media: { url: `data:image/png;base64,${room}` } },
    ],
    config: {
      responseModalities: ['TEXT', 'IMAGE'],
      imageConfig: {
        aspectRatio: '1:1',
      },
    },
  });

  return media;
});

// Nano banana pro config
ai.defineFlow('nano-banana-pro', async (_) => {
  const { media } = await ai.generate({
    model: googleAI.model('gemini-3-pro-image-preview'),
    prompt: 'Generate a picture of a sunset in the mountains by a lake',
    config: {
      responseModalities: ['TEXT', 'IMAGE'],
      imageConfig: {
        aspectRatio: '3:4',
        imageSize: '1K',
      },
    },
  });

  return media;
});

// A simple example of image generation with Gemini.
ai.defineFlow('imagen-image-generation', async (_) => {
  const { media } = await ai.generate({
    model: googleAI.model('imagen-3.0-generate-002'),
    prompt: `generate an image of a banana riding a bicycle`,
  });

  return media;
});

// TTS sample
ai.defineFlow(
  {
    name: 'tts',
    inputSchema: z
      .string()
      .default(
        'Gemini is amazing. Can say things like: glorg, blub-blub, and ayeeeeee!!!'
      ),
    outputSchema: z.object({ media: z.string() }),
  },
  async (prompt) => {
    const { media } = await ai.generate({
      model: googleAI.model('gemini-2.5-flash-preview-tts'),
      config: {
        responseModalities: ['AUDIO'],
        // For all available options see https://ai.google.dev/gemini-api/docs/speech-generation#javascript
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: { voiceName: 'Algenib' },
          },
        },
      },
      prompt,
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

// An example of using Ver 3 model to make a static photo move.
ai.defineFlow('photo-move-veo', async (_, { sendChunk }) => {
  const startingImage = fs.readFileSync('photo.jpg', { encoding: 'base64' });

  let { operation } = await ai.generate({
    model: googleAI.model('veo-3.1-fast-generate-preview'),
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
      resolution: '4k',
      durationSeconds: 8,
      aspectRatio: '9:16',
      personGeneration: 'allow_adult',
      seed: 42,
    },
  });

  if (!operation) {
    throw new Error('Expected the model to return an operation');
  }

  while (!operation.done) {
    sendChunk('check status of operation ' + operation.id);
    operation = await ai.checkOperation(operation);
    await new Promise((resolve) => setTimeout(resolve, 5000));
  }

  if (operation.error) {
    sendChunk('Error: ' + operation.error.message);
    throw new Error('failed to generate video: ' + operation.error.message);
  }

  // operation done, download generated video to disk
  const video = operation.output?.message?.content.find((p) => !!p.media);
  if (!video) {
    throw new Error('Failed to find the generated video');
  }
  sendChunk('Writing results to photo.mp4');
  await downloadVideo(video, 'photo.mp4');
  sendChunk('Done!');

  return operation;
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

ai.defineFlow('veo-extend-video', async (_, { sendChunk }) => {
  sendChunk('Beginning generation of original video');
  // Veo can only extend videos originally generated by veo.
  // Generate an original video
  let { operation } = await ai.generate({
    model: googleAI.model('veo-3.1-generate-preview'),
    prompt: [
      {
        text: 'An origami butterfly flaps its wings and flies out of the french doors into the garden.',
      },
    ],
  });

  const doneOp = await waitForOperation(operation, sendChunk);
  const video1 = doneOp.output?.message?.content.find((p: Part) => !!p.media);
  if (!video1) {
    throw new Error(
      'failed to find video in operation response: ' +
        JSON.stringify(doneOp, null, 2)
    );
  }

  sendChunk('Writing results of initial video to videoOriginal.mp4');
  await downloadVideo(video1, 'videoOriginal.mp4');
  sendChunk('Downloaded!');

  sendChunk('Beginning Extension of Video');

  ({ operation } = await ai.generate({
    model: googleAI.model('veo-3.1-generate-preview'),
    prompt: [
      {
        text: 'Track the butterfly into the garden as it lands on an orange origami flower. A fluffy white puppy runs up and gently pats the flower.',
      },
      {
        media: {
          contentType: 'video/mp4',
          url: video1.media.url,
        },
      },
    ],
    config: {
      durationSeconds: 8,
      aspectRatio: '16:9', // Must match the original
    },
  }));

  if (!operation) {
    throw new Error('Expected the model to return an operation');
  }

  while (!operation.done) {
    sendChunk('check status of operation ' + operation.id);
    operation = await ai.checkOperation(operation);
    await new Promise((resolve) => setTimeout(resolve, 5000));
  }

  if (operation.error) {
    sendChunk('Error: ' + operation.error.message);
    throw new Error('failed to generate video: ' + operation.error.message);
  }

  // operation done, download generated video to disk
  const video = operation.output?.message?.content.find((p) => !!p.media);
  if (!video) {
    throw new Error('Failed to find the generated video');
  }
  sendChunk('Writing results to videoExtended.mp4');
  await downloadVideo(video, 'videoExtended.mp4');
  sendChunk('Done!');

  return operation;
});

function getApiKeyFromEnvVar(): string | undefined {
  return (
    process.env.GEMINI_API_KEY ||
    process.env.GOOGLE_API_KEY ||
    process.env.GOOGLE_GENAI_API_KEY
  );
}

async function downloadVideo(video: MediaPart, path: string) {
  const fetch = (await import('node-fetch')).default;
  const videoDownloadResponse = await fetch(
    `${video.media!.url}&key=${getApiKeyFromEnvVar()}`
  );
  if (
    !videoDownloadResponse ||
    videoDownloadResponse.status !== 200 ||
    !videoDownloadResponse.body
  ) {
    throw new Error('Failed to fetch video');
  }

  Readable.from(videoDownloadResponse.body).pipe(fs.createWriteStream(path));
}

// Test external URL with Gemini 3.0 (should pass as fileUri)
ai.defineFlow('external-url-gemini-3.0', async () => {
  const { text } = await ai.generate({
    model: googleAI.model('gemini-3-flash-preview'),
    prompt: [
      { text: 'Describe this image.' },
      {
        media: {
          url: 'https://storage.googleapis.com/generativeai-downloads/images/scones.jpg',
          contentType: 'image/jpeg',
        },
      },
    ],
  });
  return text;
});

// Gemini 3.1 thinkingLevel config
ai.defineFlow(
  {
    name: 'thinking-level-3.1-pro',
    inputSchema: z.enum(['LOW', 'MEDIUM', 'HIGH']),
    outputSchema: z.any(),
  },
  async (level) => {
    const { text } = await ai.generate({
      model: googleAI.model('gemini-3.1-pro-preview'),
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
        thinkingConfig: {
          thinkingLevel: level,
          includeThoughts: true,
        },
      },
    });
    return text;
  }
);
