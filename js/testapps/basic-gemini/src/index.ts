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
  Document,
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

ai.defineFlow('deep-research-visualization', async (_, { sendChunk }) => {
  let { operation } = await ai.generate({
    model: googleAI.model('deep-research-preview-04-2026'),
    prompt:
      'Analyze global semiconductor market trends. Include graphics showing market share changes.',
    config: {
      visualization: 'AUTO',
    },
  });

  if (!operation) throw new Error('No operation returned');

  while (!operation.done) {
    sendChunk('check status of operation ' + operation.id);
    operation = await ai.checkOperation(operation);
    await new Promise((resolve) => setTimeout(resolve, 30000));
  }
  return operation.output?.message?.content;
});

ai.defineFlow('deep-research-code-execution', async (_, { sendChunk }) => {
  let { operation } = await ai.generate({
    model: googleAI.model('deep-research-preview-04-2026'),
    prompt:
      'Start with 1. Add 3, then divide by 2. Take that answer, and continue adding 3 and dividing by 2 to each successive answer and tell me the first 20 terms in the sequence',
    config: {
      codeExecution: true,
    },
  });

  if (!operation) throw new Error('No operation returned');

  while (!operation.done) {
    sendChunk('check status of operation ' + operation.id);
    operation = await ai.checkOperation(operation);
    await new Promise((resolve) => setTimeout(resolve, 30000));
  }
  return operation.output?.message?.content;
});

ai.defineFlow('maps-grounding', async () => {
  const { text, raw } = await ai.generate({
    model: googleAI.model('gemini-3.1-pro-preview'),
    prompt: 'Describe some sights near me',
    config: {
      tools: [
        {
          googleMaps: { enableWidget: true },
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

ai.defineFlow('combine tools and builtins', async () => {
  const { text } = await ai.generate({
    model: googleAI.model('gemini-3-flash-preview'),
    prompt:
      'What is the southernmost city in Canada? What is the weather like there today? Use the getWeather tool.',
    config: {
      tools: [{ googleSearch: {} }],
      toolConfig: {
        includeServerSideToolInvocations: true,
      },
    },
    tools: [getWeather],
  });

  return text;
});

ai.defineFlow('basic-hi', async () => {
  const { text } = await ai.generate({
    model: googleAI.model('gemini-flash-lite-latest'),
    prompt: 'You are a helpful AI assistant named Walt, say hello',
  });

  return text;
});

ai.defineFlow('basic-hi-flex-tier', async () => {
  const { text } = await ai.generate({
    model: googleAI.model('gemini-flash-lite-latest'),
    prompt: 'You are a helpful AI assistant named Walt, say hello',
    config: {
      serviceTier: 'flex', // or 'standard' or 'priority'
    },
  });

  return text;
});

ai.defineFlow('basic-hi-with-retry', async () => {
  const { text } = await ai.generate({
    model: googleAI.model('gemini-pro-latest'),
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
        models: [googleAI.model('gemini-flash-latest')],
        statuses: ['UNKNOWN'],
      }),
    ],
  });

  return text;
});

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
          includeThoughts: true,
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
    model: googleAI.model('gemini-flash-latest'),
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
    model: googleAI.model('gemini-flash-latest'),
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
    model: googleAI.model('gemini-flash-latest'),
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
    model: googleAI.model('gemini-flash-latest'),
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
    model: googleAI.model('gemini-flash-latest'),
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
    model: googleAI.model('gemini-flash-latest'),
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
      model: googleAI.model('gemini-flash-latest'),
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
      model: googleAI.model('gemini-pro-latest'),
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
      model: googleAI.model('gemini-flash-latest'),
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
      model: googleAI.model('gemini-flash-latest'),
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

// Gemini reasoning example (legacy thinkingBudget)
ai.defineFlow('reasoning - thinkingBudget', async (_, { sendChunk }) => {
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
    model: googleAI.model('gemini-3.1-pro-preview'),
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
      imageConfig: {
        aspectRatio: '3:4',
        imageSize: '1K',
      },
    },
  });

  return media;
});

// webSearch and imageSearch with Nano Banana 2
ai.defineFlow('nano-banana-2', async (_) => {
  const { media } = await ai.generate({
    model: googleAI.model('gemini-3.1-flash-image-preview'),
    prompt:
      'Generate an accurate image of the CN Tower. Use webSearch to determine the date, weather and current time in Toronto. The weather and time should be reflected in the image (day, night, rainy, sunny, snowy etc). Also use words to show the date, time and weather on the image.',
    config: {
      imageConfig: {
        aspectRatio: '1:4',
        imageSize: '512P',
      },
      google_search: {
        searchTypes: { webSearch: {}, imageSearch: {} },
      },
      thinkingConfig: {
        // Optional
        thinkingLevel: 'HIGH',
        includeThoughts: true,
      },
    },
  });

  return media;
});

// A simple example of image generation with Gemini.
ai.defineFlow('imagen-image-generation', async (_) => {
  const { media } = await ai.generate({
    model: googleAI.model('imagen-4.0-generate-001'),
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

ai.defineFlow(
  {
    name: 'tts-audio-tags',
    inputSchema: z.string().default(
      `DIRECTOR'S NOTES
Style:
* The "Vocal Smile": You must hear the grin in the audio. The soft palate is
always raised to keep the tone bright, sunny, and explicitly inviting.
* Dynamics: High projection without shouting. Punchy consonants and elongated
vowels on excitement words (e.g., "Beauuutiful morning").

Pace: Speaks at an energetic pace, keeping up with the fast music.  Speaks
with A "bouncing" cadence. High-speed delivery with fluid transitions — no dead
air, no gaps.

Accent: Jaz is from Brixton, London

SAMPLE CONTEXT
Jaz is the industry standard for Top 40 radio, high-octane event promos, or any
script that requires a charismatic Estuary accent and 11/10 infectious energy.

TRANSCRIPT
[excitedly] Yes, massive vibes in the studio! You are locked in and it is
absolutely popping off in London right now. If you're stuck on the tube, or
just sat there pretending to work... stop it. Seriously, I see you.
[shouting] Turn this up! We've got the project roadmap landing in three,
two... let's go!`
    ),
    outputSchema: z.object({ media: z.string() }),
  },
  async (prompt: string) => {
    const { media } = await ai.generate({
      model: googleAI.model('gemini-3.1-flash-tts-preview'),
      config: {
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
  const startingImage = fs.readFileSync('woman.png', { encoding: 'base64' });

  let { operation } = await ai.generate({
    model: googleAI.model('veo-3.1-lite-generate-preview'),
    prompt: [
      {
        text: 'make the subject in the photo move',
      },
      {
        media: {
          contentType: 'image/png',
          url: `data:image/png;base64,${startingImage}`,
        },
      },
    ],
    config: {
      resolution: '1080p',
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
    sendChunk(operation);
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

// Embed text
ai.defineFlow('embed-text', async () => {
  const embeddings = await ai.embed({
    embedder: googleAI.embedder('gemini-embedding-001'),
    content: 'Albert Einstein was a German-born theoretical physicist.',
    options: {
      outputDimensionality: 256,
      taskType: 'RETRIEVAL_DOCUMENT',
      title: 'Albert Einstein', // Valid when taskType is RETRIEVAL_DOCUMENT
    },
  });

  return embeddings;
});

// Embed multimodal content
ai.defineFlow('embed-multimodal', async () => {
  const photoBase64 = fs.readFileSync('photo.jpg', { encoding: 'base64' });

  const embeddings = await ai.embed({
    embedder: googleAI.embedder('gemini-embedding-2-preview'),
    content: Document.fromParts([
      { text: 'A picture of Albert Einstein.' },
      {
        media: {
          contentType: 'image/jpeg',
          url: `data:image/jpeg;base64,${photoBase64}`,
        },
      },
    ]),
    options: {
      outputDimensionality: 256,
    },
  });

  return embeddings;
});

ai.defineFlow('embed-multimodal-gemini-embedding-2', async () => {
  const photoBase64 = fs.readFileSync('photo.jpg', { encoding: 'base64' });

  const embeddings = await ai.embed({
    embedder: googleAI.embedder('gemini-embedding-2'),
    content: Document.fromParts([
      { text: 'A picture of Albert Einstein.' },
      {
        media: {
          contentType: 'image/jpeg',
          url: `data:image/jpeg;base64,${photoBase64}`,
        },
      },
    ]),
    options: {
      outputDimensionality: 256,
    },
  });

  return embeddings;
});

// Deep research example
ai.defineFlow('deep-research', async (_, { sendChunk }) => {
  let { operation } = await ai.generate({
    model: googleAI.model('deep-research-pro-preview-12-2025'),
    prompt:
      'Compare the differences between TCP and UDP protocols. Provide the answer in a markdown table focusing on reliability, connection type, and speed.',
  });

  if (!operation) {
    throw new Error('Expected the model to return an operation');
  }

  while (!operation.done) {
    sendChunk('check status of operation ' + operation.id);
    operation = await ai.checkOperation(operation);
    await new Promise((resolve) => setTimeout(resolve, 30000));
  }

  if (operation.error) {
    sendChunk('Error: ' + operation.error.message);
    throw new Error('failed to deep research: ' + operation.error.message);
  }

  return operation.output?.message?.content.find((p) => !!p.text)?.text;
});

ai.defineFlow('deep-research-multi-turn', async (_, { sendChunk }) => {
  // 1. First turn: Initial comparison with specific requirements
  sendChunk('--- Turn 1: Initial Research ---');
  let { operation } = await ai.generate({
    model: googleAI.model('deep-research-pro-preview-12-2025'),
    messages: [
      {
        role: 'system',
        content: [{ text: 'You are a technical research assistant.' }],
      },
      {
        role: 'user',
        content: [
          {
            text: 'Compare TCP vs UDP.',
          },
        ],
      },
    ],
    config: {
      thinkingSummaries: 'AUTO',
      responseModalities: ['TEXT'],
      store: true,
    },
  });

  if (!operation) throw new Error('No operation returned');

  while (!operation.done) {
    sendChunk('Turn 1 status: ' + operation.id);
    operation = await ai.checkOperation(operation);
    await new Promise((resolve) => setTimeout(resolve, 30000));
  }

  if (operation.error) {
    throw new Error('Turn 1 failed: ' + operation.error.message);
  }

  const response1 = operation.output?.message?.content.find(
    (p) => !!p.text
  )?.text;
  sendChunk('Turn 1 Response: ' + response1);

  // 2. Second turn: Follow up using the previous interaction ID
  sendChunk('\n--- Turn 2: Follow up ---');
  const interactionId = operation.id;

  let { operation: op2 } = await ai.generate({
    model: googleAI.model('deep-research-pro-preview-12-2025'),
    messages: [
      {
        role: 'user',
        content: [
          { text: 'Which one is better for video streaming? Explain why.' },
        ],
      },
    ],
    config: {
      thinkingSummaries: 'AUTO',
      responseModalities: ['TEXT'],
      previousInteractionId: interactionId,
    },
  });

  if (!op2) throw new Error('No operation returned for turn 2');

  while (!op2.done) {
    sendChunk('Turn 2 status: ' + op2.id);
    op2 = await ai.checkOperation(op2);
    await new Promise((resolve) => setTimeout(resolve, 30000));
  }

  if (op2.error) {
    throw new Error('Turn 2 failed: ' + op2.error.message);
  }

  return op2.output?.message?.content.find((p) => !!p.text)?.text;
});

ai.defineFlow('deep-research-preview', async (_, { sendChunk }) => {
  const storeName = await createFileSearchStore();
  await uploadBlobToFileSearchStore(storeName);

  let { operation } = await ai.generate({
    model: googleAI.model('deep-research-preview-04-2026'),
    prompt:
      'Analyze the differences between the character in the provided document and modern quantum computing principles. Create a chart to visualize the comparison.',
    config: {
      visualization: 'AUTO',
      googleSearch: true,
      codeExecution: true,
      fileSearch: {
        fileSearchStoreNames: [storeName],
      },
    },
  });

  if (!operation) {
    throw new Error('Expected the model to return an operation');
  }

  while (!operation.done) {
    sendChunk('check status of operation ' + operation.id);
    operation = await ai.checkOperation(operation);
    await new Promise((resolve) => setTimeout(resolve, 30000));
  }

  await deleteFileSearchStore(storeName);

  if (operation.error) {
    sendChunk('Error: ' + operation.error.message);
    throw new Error('failed to deep research: ' + operation.error.message);
  }

  return operation.output?.message?.content;
});

ai.defineFlow('deep-research-collaboration', async (_, { sendChunk }) => {
  sendChunk('--- Turn 1: Requesting a Plan ---');
  let { operation } = await ai.generate({
    model: googleAI.model('deep-research-preview-04-2026'),
    prompt: 'I want to research the history of artificial intelligence.',
    config: {
      collaborativePlanning: true,
      thinkingSummaries: 'AUTO',
      store: true,
    },
  });

  if (!operation) throw new Error('No operation returned');

  while (!operation.done) {
    sendChunk('Turn 1 status: ' + operation.id);
    operation = await ai.checkOperation(operation);
    await new Promise((resolve) => setTimeout(resolve, 30000));
  }

  if (operation.error) {
    throw new Error('Turn 1 failed: ' + operation.error.message);
  }

  const response1 = operation.output?.message?.content.find(
    (p) => !!p.text
  )?.text;
  sendChunk('Proposed Plan: ' + response1);

  sendChunk('\n--- Turn 2: Approving and Executing Plan ---');
  const interactionId = operation.id;

  let { operation: op2 } = await ai.generate({
    model: googleAI.model('deep-research-max-preview-04-2026'),
    prompt: 'Looks great! Proceed with the research.',
    config: {
      collaborativePlanning: false,
      thinkingSummaries: 'AUTO',
      previousInteractionId: interactionId,
    },
  });

  if (!op2) throw new Error('No operation returned for turn 2');

  while (!op2.done) {
    sendChunk('Turn 2 status: ' + op2.id);
    op2 = await ai.checkOperation(op2);
    await new Promise((resolve) => setTimeout(resolve, 30000));
  }

  if (op2.error) {
    throw new Error('Turn 2 failed: ' + op2.error.message);
  }

  return op2.output?.message?.content;
});

// Deep research cancel example
ai.defineFlow('deep-research-cancel', async (_, { sendChunk }) => {
  let { operation } = await ai.generate({
    model: googleAI.model('deep-research-pro-preview-12-2025'),
    prompt:
      'Compare the differences between TCP and UDP protocols. Provide the answer in a markdown table focusing on reliability, connection type, and speed.',
  });

  if (!operation) {
    throw new Error('Expected the model to return an operation');
  }

  sendChunk('Started operation: ' + operation.id);
  // Wait a bit before cancelling
  await new Promise((resolve) => setTimeout(resolve, 5000));

  sendChunk('Cancelling operation: ' + operation.id);

  const canceledOp = await ai.cancelOperation(operation);
  sendChunk('Operation cancelled');

  return JSON.stringify(canceledOp, null, 2);
});

// Lyria music generation
ai.defineFlow('lyria-instrumental-clip', async () => {
  const response = await ai.generate({
    model: googleAI.model('lyria-3-clip-preview'),
    prompt:
      'A bright chiptune melody in C Major, retro 8-bit video game style. Instrumental only, no vocals.',
  });

  return response;
});

ai.defineFlow('lyria-music-generation', async () => {
  const { media, text } = await ai.generate({
    model: googleAI.model('lyria-3-clip-preview'),
    prompt:
      'Create a 30-second cheerful acoustic folk song with guitar and harmonica.',
  });

  // if (media) {
  //   const audioBuffer = Buffer.from(
  //     media.url.substring(media.url.indexOf(',') + 1),
  //     'base64'
  //   );
  //   fs.writeFileSync('clip.mp3', audioBuffer);
  // }

  return { text, media };
});

ai.defineFlow('lyria-full-length-song', async () => {
  const response = await ai.generate({
    model: googleAI.model('lyria-3-pro-preview'),
    prompt:
      'An epic cinematic orchestral piece about a journey home. Starts with a solo piano intro, builds through sweeping strings, and climaxes with a massive wall of sound.',
  });

  return response;
});

ai.defineFlow('lyria-from-image', async () => {
  const photoBase64 = fs.readFileSync('photo.jpg', { encoding: 'base64' });

  const response = await ai.generate({
    model: googleAI.model('lyria-3-pro-preview'),
    prompt: [
      {
        text: 'An atmospheric ambient track inspired by the mood and colors in this image.',
      },
      {
        media: {
          contentType: 'image/jpeg',
          url: `data:image/jpeg;base64,${photoBase64}`,
        },
      },
    ],
  });

  return response;
});

ai.defineFlow('lyria-custom-lyrics', async () => {
  const prompt = `
Create a dreamy indie pop song with the following lyrics:

[Verse 1]
Walking through the neon glow,
city lights reflect below,
every shadow tells a story,
every corner, fading glory.

[Chorus]
We are the echoes in the night,
burning brighter than the light,
hold on tight, don't let me go,
we are the echoes down below.

[Verse 2]
Footsteps lost on empty streets,
rhythms sync to heartbeats,
whispers carried by the breeze,
dancing through the autumn leaves.
`;

  const response = await ai.generate({
    model: googleAI.model('lyria-3-pro-preview'),
    prompt,
  });

  return response;
});

ai.defineFlow('lyria-custom-timing', async () => {
  const prompt = `
[0:00 - 0:10] Intro: Begin with a soft lo-fi beat and muffled vinyl crackle.
[0:10 - 0:30] Verse 1: Add a warm Fender Rhodes piano melody and gentle vocals singing about a rainy morning.
[0:30 - 0:50] Chorus: Full band with upbeat drums and soaring synth leads. The lyrics are hopeful and uplifting.
[0:50 - 1:00] Outro: Fade out with the piano melody alone.
`;

  const response = await ai.generate({
    model: googleAI.model('lyria-3-pro-preview'),
    prompt,
  });

  return response;
});

ai.defineFlow('lyria-foreign-language', async () => {
  const response = await ai.generate({
    model: googleAI.model('lyria-3-pro-preview'),
    prompt:
      'Crée une chanson pop romantique en français sur un coucher de soleil à Paris. Utilise du piano et de la guitare acoustique.',
  });

  return response;
});

// Gemma 3
ai.defineFlow('gemma-3', async () => {
  const { text } = await ai.generate({
    model: googleAI.model('gemma-3-27b-it'),
    prompt: 'Tell me a short joke about a programmer.',
  });
  return text;
});

// Gemma 4 with thinkingConfig
ai.defineFlow('gemma-4', async (_, { sendChunk }) => {
  const { text, message } = await ai.generate({
    model: googleAI.model('gemma-4-31b-it'),
    system: 'You are a physics tutor who loves cats. Use cat analogies.',
    prompt: 'Explain relativity to a 10 year old',
    config: {
      thinkingConfig: {
        thinkingLevel: 'HIGH', // The only possibility is 'HIGH'
      },
    },
    onChunk: sendChunk,
  });
  return {
    text,
    reasoning: message?.content.find((p) => !!p.reasoning)?.reasoning,
  };
});

// Gemma 4 multi-turn
ai.defineFlow('gemma-4-multi-turn', async (_, { sendChunk }) => {
  // First turn
  sendChunk('--- Turn 1 ---');
  const response1 = await ai.generate({
    model: googleAI.model('gemma-4-26b-a4b-it'),
    messages: [
      {
        role: 'user',
        content: [
          {
            text: 'Think of a number between 1 and 10. Explain your reasoning, then tell me the number.',
          },
        ],
      },
    ],
  });

  sendChunk(
    'Reasoning 1: ' +
      response1.message?.content.find((p) => !!p.reasoning)?.reasoning
  );
  sendChunk('Text 1: ' + response1.text);

  // Second turn - passes the previous model response which includes reasoning,
  // but the plugin should strip the reasoning before sending to the API.
  sendChunk('\n--- Turn 2 ---');
  const response2 = await ai.generate({
    model: googleAI.model('gemma-4-26b-a4b-it'),
    messages: [
      {
        role: 'user',
        content: [
          {
            text: 'Think of a number between 1 and 10. Explain your reasoning, then tell me the number.',
          },
        ],
      },
      response1.message!,
      {
        role: 'user',
        content: [
          {
            text: 'Now multiply that number by 5. Again, explain your reasoning.',
          },
        ],
      },
    ],
  });

  sendChunk(
    'Reasoning 2: ' +
      response2.message?.content.find((p) => !!p.reasoning)?.reasoning
  );
  sendChunk('Text 2: ' + response2.text);

  return {
    turn1: response1.text,
    turn2: response2.text,
  };
});
