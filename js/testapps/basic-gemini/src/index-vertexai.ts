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
import {
  Document,
  genkit,
  MessageData,
  Operation,
  Part,
  StreamingCallback,
  z,
} from 'genkit';
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
    model: vertexAI.model('gemini-3.1-flash-lite'),
    prompt: 'You are a helpful AI assistant named Walt, say hello',
  });

  return text;
});

// Newer Gemini models (and any model not listed in the plugin) use the
// Interactions API. It's available on the global location.
ai.defineFlow('argon-basic', async () => {
  const { text } = await ai.generate({
    model: vertexAI.model('gemini-4-argon'),
    prompt: 'You are a helpful AI assistant named Walt, say hello.',
  });
  return text;
});

// Server-side conversation state (Interactions API models only). With
// `store: true`, each response's message carries `metadata.interactionId`;
// passing the previous messages back continues that interaction, and only
// the new message is sent.
ai.defineFlow('interactions-multi-turn-store', async () => {
  const first = await ai.generate({
    model: vertexAI.model('gemini-3.8-flash'),
    prompt:
      'Remember this: my cat is called Biscuit and my favorite color is green. Say hi.',
    config: { store: true },
  });

  const second = await ai.generate({
    model: vertexAI.model('gemini-3.8-flash'),
    messages: first.messages,
    prompt: 'What is my cat called and what is my favorite color?',
    config: { store: true },
  });

  return {
    firstInteractionId: first.message?.metadata?.interactionId,
    secondInteractionId: second.message?.metadata?.interactionId,
    answer: second.text,
  };
});

// Text-to-speech (generateContent only; available on the global location).
// The text is read verbatim, so delivery goes in speechMetadata.style.
ai.defineFlow(
  {
    name: 'tts-voice-and-style',
    inputSchema: z.string().default('Have a wonderful day!'),
    outputSchema: z.object({ media: z.string() }),
  },
  async (text) => {
    const { media } = await ai.generate({
      model: vertexAI.model('gemini-3.8-flash-lite-tts'),
      prompt: [
        {
          text,
          metadata: { speechMetadata: { style: 'cheerful and friendly' } },
        },
      ],
      config: { speechConfig: { voiceConfig: { voice: 'Kore' } } },
    });
    if (!media) {
      throw new Error('no media returned');
    }
    return { media: media.url };
  }
);

// TTS with the googleAI voice shape (prebuiltVoiceConfig.voiceName), which
// the platform also accepts, so TTS code can move between the two plugins.
ai.defineFlow(
  {
    name: 'tts-googleai-voice-shape',
    outputSchema: z.object({ media: z.string() }),
  },
  async () => {
    const { media } = await ai.generate({
      model: vertexAI.model('gemini-3.8-flash-lite-tts'),
      prompt: 'This voice was chosen with the googleAI config shape.',
      config: {
        speechConfig: {
          voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Puck' } },
        },
      },
    });
    if (!media) {
      throw new Error('no media returned');
    }
    return { media: media.url };
  }
);

// Multi-speaker TTS: exactly two speakers, each turn a separate text part
// naming its speaker.
ai.defineFlow(
  {
    name: 'tts-multi-speaker',
    outputSchema: z.object({ media: z.string() }),
  },
  async () => {
    const { media } = await ai.generate({
      model: vertexAI.model('gemini-3.8-flash-tts'),
      prompt: [
        {
          text: "How's it going today, Jane?",
          metadata: { speechMetadata: { speaker: 'Joe', style: 'cheerful' } },
        },
        {
          text: 'Not too bad, how about you? <laugh>',
          metadata: { speechMetadata: { speaker: 'Jane', style: 'relaxed' } },
        },
      ],
      config: {
        speechConfig: {
          multiSpeakerVoiceConfig: {
            speakerVoiceConfigs: [
              { speaker: 'Joe', voiceConfig: { voice: 'Puck' } },
              { speaker: 'Jane', voiceConfig: { voice: 'Kore' } },
            ],
          },
        },
      },
    });
    if (!media) {
      throw new Error('no media returned');
    }
    return { media: media.url };
  }
);

// Streamed TTS. Audio chunks arrive while the speech is being generated, so a
// client can start playing right away; the final response joins them into one
// part. Streaming returns raw PCM (audio/l16, 24 kHz, mono) by default.
ai.defineFlow(
  {
    name: 'tts-streaming',
    outputSchema: z.object({
      chunkCount: z.number(),
      contentType: z.string().optional(),
      bytes: z.number(),
    }),
  },
  async (_, { sendChunk }) => {
    const { response, stream } = ai.generateStream({
      model: vertexAI.model('gemini-3.8-flash-lite-tts'),
      prompt:
        'Streaming lets an app start playing speech before the whole ' +
        'sentence has been generated.',
      config: { speechConfig: { voiceConfig: { voice: 'Kore' } } },
    });

    let chunkCount = 0;
    for await (const chunk of stream) {
      // A real client would play chunk.media here. Report progress only, to
      // keep the stream small.
      chunkCount++;
      sendChunk(`chunk ${chunkCount}: ${chunk.media?.contentType}`);
    }

    const { media } = await response;
    if (!media) {
      throw new Error('no media returned');
    }
    // Raw PCM can't be played in the Dev UI, so return its size instead.
    return {
      chunkCount,
      contentType: media.contentType,
      bytes: Buffer.from(media.url.split(',')[1] ?? '', 'base64').length,
    };
  }
);

// TTS output formats. audio/wav (default) is a complete WAV file; audio/l16 is
// raw PCM (24 kHz); audio/mulaw and audio/alaw are telephony formats (8 kHz).
// Only audio/wav plays in the Dev UI, so the others return their size.
ai.defineFlow(
  {
    name: 'tts-audio-format',
    inputSchema: z
      .enum(['audio/wav', 'audio/l16', 'audio/mulaw', 'audio/alaw'])
      .default('audio/mulaw'),
    outputSchema: z.object({
      contentType: z.string().optional(),
      bytes: z.number(),
      media: z.string().optional(),
    }),
  },
  async (mimeType) => {
    const { media } = await ai.generate({
      model: vertexAI.model('gemini-3.8-flash-lite-tts'),
      prompt: 'Thank you for calling. Please hold.',
      config: {
        speechConfig: { voiceConfig: { voice: 'Kore' } },
        audioFormat: { mimeType },
      },
    });
    if (!media) {
      throw new Error('no media returned');
    }
    return {
      contentType: media.contentType,
      bytes: Buffer.from(media.url.split(',')[1] ?? '', 'base64').length,
      ...(mimeType === 'audio/wav' ? { media: media.url } : {}),
    };
  }
);

// Video generation with Gemini Omni (Interactions API, global location).
// Unlike Veo, it isn't a background model: ai.generate() returns the video
// directly, as an inline MP4 (a data URL), which plays in the Dev UI. The
// flows use 360p to keep them fast; see videoFormat for other sizes.
ai.defineFlow('omni-text-to-video', async () => {
  const { media } = await ai.generate({
    model: vertexAI.model('gemini-omni-1.1-flash-preview'),
    prompt:
      'A paper boat floating down a rain-filled gutter, continuous shot, with the sound of rain.',
    config: {
      videoFormat: { aspectRatio: '9:16', resolution: '360p' },
    },
  });
  if (!media) {
    throw new Error('no video returned');
  }
  return media;
});

// Brings a still image to life: the image is used as the starting frame.
ai.defineFlow('omni-image-to-video', async () => {
  // A living room with a grey sofa in front of sheer white curtains.
  const room = fs.readFileSync('my_room.png', { encoding: 'base64' });
  const { media } = await ai.generate({
    model: vertexAI.model('gemini-omni-1.1-flash-preview'),
    prompt: [
      {
        media: {
          url: `data:image/png;base64,${room}`,
          contentType: 'image/png',
        },
      },
      {
        text: 'Use this image as the starting frame. A ginger cat walks in, jumps up onto the sofa and curls up on a cushion. Single continuous shot, with soft room sound.',
      },
    ],
    config: {
      // Optional: the model infers the task from the prompt if it isn't set.
      videoConfig: { task: 'image_to_video' },
      videoFormat: { resolution: '360p' },
    },
  });
  if (!media) {
    throw new Error('no video returned');
  }
  return media;
});

// Generates a video, then edits it in a second turn. Editing continues the
// stored interaction, so both turns need store: true; the plugin finds the
// previous interaction in the messages and sends only the new message.
ai.defineFlow('omni-edit-video', async () => {
  const config = { store: true, videoFormat: { resolution: '360p' } };
  const original = await ai.generate({
    model: vertexAI.model('gemini-omni-1.1-flash-preview'),
    prompt: 'A woman playing violin outdoors.',
    config,
  });
  if (!original.media) {
    throw new Error('no video returned');
  }

  const edited = await ai.generate({
    model: vertexAI.model('gemini-omni-1.1-flash-preview'),
    messages: [
      ...original.messages,
      {
        role: 'user',
        content: [
          { text: 'Make the violin invisible. Keep everything else the same.' },
        ],
      },
    ],
    config,
  });
  if (!edited.media) {
    throw new Error('no edited video returned');
  }
  // Both videos play in the Dev UI, to compare before and after.
  return { original: original.media, edited: edited.media };
});

// The other Omni model: fast, conversational video generation. It only
// supports 720p (the default); other resolutions return a 400.
ai.defineFlow('omni-flash', async () => {
  const { media } = await ai.generate({
    model: vertexAI.model('gemini-omni-flash-preview'),
    prompt: 'A hot air balloon rising over green hills at sunrise.',
  });
  if (!media) {
    throw new Error('no video returned');
  }
  return media;
});

// Multimodal embedding with gemini-embedding-2 (the `:embedContent` method):
// text and media in one document give one embedding in a shared space.
ai.defineFlow('embed-multimodal', async () => {
  // AI generated portrait.
  const woman = fs.readFileSync('woman.png', { encoding: 'base64' });
  const embeddings = await ai.embed({
    embedder: vertexAI.embedder('gemini-embedding-2'),
    content: Document.fromParts([
      { text: 'A portrait of a woman.' },
      {
        media: {
          contentType: 'image/png',
          url: `data:image/png;base64,${woman}`,
        },
      },
    ]),
    options: { outputDimensionality: 256 },
  });
  return {
    count: embeddings.length,
    dimensions: embeddings[0]?.embedding.length,
  };
});

// Text embedding with gemini-embedding-001 (the `:predict` method), which
// supports taskType and title.
ai.defineFlow('embed-text', async () => {
  const embeddings = await ai.embed({
    embedder: vertexAI.embedder('gemini-embedding-001'),
    content: 'The CN Tower is a communications tower in Toronto.',
    options: {
      outputDimensionality: 256,
      taskType: 'RETRIEVAL_DOCUMENT',
      title: 'CN Tower', // Used with taskType RETRIEVAL_DOCUMENT
    },
  });
  return {
    count: embeddings.length,
    dimensions: embeddings[0]?.embedding.length,
  };
});

// Image generation with Nano Banana 2.1 (generateContent). It doesn't support
// seed, topK, logprobs, temperature or topP.
ai.defineFlow('image-generation', async () => {
  const { media } = await ai.generate({
    model: vertexAI.model('gemini-nano-banana-2.1'),
    prompt: 'A cozy reading nook with a window seat on a rainy afternoon.',
  });
  if (!media) {
    throw new Error('no image returned');
  }
  return media;
});

// The low-cost image model (Nano Banana 2 Lite).
ai.defineFlow('image-generation-lite', async () => {
  const { media } = await ai.generate({
    model: vertexAI.model('gemini-3.1-flash-lite-image'),
    prompt: 'A simple flat icon of a coffee cup.',
  });
  if (!media) {
    throw new Error('no image returned');
  }
  return media;
});

// Agentic video understanding: instead of reading frames at a fixed rate, the
// model explores the video and loads only what it needs (better for long
// videos). Set per video with `metadata.processing` ('agentic' or 'static',
// the default). usage.custom.toolUseTokens counts what was loaded.
ai.defineFlow('agentic-video-understanding', async () => {
  const { text, usage } = await ai.generate({
    model: vertexAI.model('gemini-4-argon'),
    prompt: [
      {
        media: { url: 'https://www.youtube.com/watch?v=QO30-W3b_vw' },
        metadata: { processing: 'agentic' },
      },
      {
        text: 'What happens in this video? List the main moments with timestamps (MM:SS).',
      },
    ],
  });
  return { text, toolUseTokens: usage.custom?.toolUseTokens };
});

// Safety settings are supported on the platform (unlike the Gemini API's
// Interactions API). The plugin sends them in the Interactions format.
ai.defineFlow('safety-settings', async () => {
  const { text, finishReason } = await ai.generate({
    model: vertexAI.model('gemini-3.8-flash'),
    prompt: 'Write a friendly two-line poem about autumn leaves.',
    config: {
      safetySettings: [
        {
          category: 'HARM_CATEGORY_HARASSMENT',
          threshold: 'BLOCK_LOW_AND_ABOVE',
        },
        {
          category: 'HARM_CATEGORY_DANGEROUS_CONTENT',
          threshold: 'BLOCK_MEDIUM_AND_ABOVE',
        },
      ],
    },
  });
  return { text, finishReason };
});

ai.defineFlow('basic-hi-lite', async () => {
  const { text } = await ai.generate({
    model: vertexAI.model('gemini-3.5-flash-lite'),
    prompt: 'You are a helpful AI assistant named Walt, say hello',
  });

  return text;
});

// Flex/Priority PayGo
ai.defineFlow('paygo', async () => {
  const response = await ai.generate({
    model: vertexAI.model('gemini-3.1-flash-lite'),
    prompt: 'You are a helpful AI assistant named Walt, say hello.',
    config: {
      payGo: 'priority', // or priority-only, flex, flex-only.
    },
  });

  return response;
});

// thinkingLevel config
ai.defineFlow(
  {
    name: 'thinking-level',
    inputSchema: z.enum(['LOW', 'MEDIUM', 'HIGH']).default('LOW'),
    outputSchema: z.any(),
  },
  async (level) => {
    const { text } = await ai.generate({
      model: vertexAI.model('gemini-3.8-flash'),
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

ai.defineFlow('multi-regional', async () => {
  const { text } = await ai.generate({
    model: vertexAI.model('gemini-3.5-flash'),
    prompt: 'You are a helpful AI assistant named Walt, say hello.',
    config: {
      location: 'us', // OR 'eu'
      apiVersion: 'v1',
    },
  });
  return text;
});

// Multimodal input
ai.defineFlow('multimodal-input', async () => {
  const photoBase64 = fs.readFileSync('photo.jpg', { encoding: 'base64' });

  const { text } = await ai.generate({
    model: vertexAI.model('gemini-3.8-flash'),
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
    model: vertexAI.model('gemini-3.8-flash'),
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
      model: vertexAI.model('gemini-3.8-flash'),
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
    model: vertexAI.model('gemini-3.8-flash'),
    prompt: 'Write a poem about AI.',
  });

  let poem = '';
  for await (const chunk of stream) {
    poem += chunk.text;
    sendChunk(chunk.text);
  }

  return poem;
});

/**
 * Collects the grounding citations from a response. On the Interactions API
 * they're annotations on the output text, which the plugin copies to each
 * text part's `metadata.annotations` (no need to read `response.raw`).
 */
function citations(message?: MessageData) {
  return message?.content.flatMap((part) => {
    const annotations = part.metadata?.annotations;
    return Array.isArray(annotations) ? annotations : [];
  });
}

// Google maps grounding (Interactions API: the location goes on the
// googleMaps tool; retrievalConfig.latLng is copied there).
ai.defineFlow('maps-grounding', async () => {
  const { text, message } = await ai.generate({
    model: vertexAI.model('gemini-3.8-flash'),
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

  return { text, citations: citations(message) };
});

// Search grounding
ai.defineFlow('search-grounding', async () => {
  const { text, message } = await ai.generate({
    model: vertexAI.model('gemini-3.8-flash'),
    prompt: 'What is the weather like today in Toronto?',
    config: {
      tools: [{ googleSearch: {} }],
    },
  });

  return { text, citations: citations(message) };
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
    name: 'tool-calling',
    inputSchema: z.string().default('Paris, France'),
    outputSchema: z.string(),
    streamSchema: z.any(),
  },
  async (location, { sendChunk }) => {
    const { response, stream } = ai.generateStream({
      model: vertexAI.model('gemini-3.8-flash'),
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
    name: 'tool-calling-streamed-arguments',
    inputSchema: z.string().default('Paris, France'),
    outputSchema: z.string(),
    streamSchema: z.any(),
  },
  async (location, { sendChunk }) => {
    const { response, stream } = ai.generateStream({
      model: vertexAI.model('gemini-3.1-pro-preview'),
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
      model: vertexAI.model('gemini-3.8-flash'),
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
      model: vertexAI.model('gemini-3.8-flash'),
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

// Thought summaries (includeThoughts) come back as reasoning parts.
ai.defineFlow('thinking-summaries', async (_, { sendChunk }) => {
  const { message } = await ai.generate({
    prompt: 'what is heavier, one kilo of steel or one kilo of feathers',
    model: vertexAI.model('gemini-3.8-flash'),
    config: {
      thinkingConfig: {
        thinkingLevel: 'MEDIUM',
        includeThoughts: true,
      },
    },
    onChunk: sendChunk,
  });

  return message;
});

// Media resolution
ai.defineFlow('media-resolution', async (_) => {
  const plant = fs.readFileSync('palm_tree.png', { encoding: 'base64' });
  const { text } = await ai.generate({
    model: vertexAI.model('gemini-3.1-pro-preview'),
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
ai.defineFlow('image-editing', async (_) => {
  const plant = fs.readFileSync('palm_tree.png', { encoding: 'base64' });
  const room = fs.readFileSync('my_room.png', { encoding: 'base64' });

  const { media } = await ai.generate({
    model: vertexAI.model('gemini-nano-banana-2.1'),
    prompt: [
      { text: 'add the plant to my room' },
      { media: { url: `data:image/png;base64,${plant}` } },
      { media: { url: `data:image/png;base64,${room}` } },
    ],
  });

  return media;
});

// Image size and aspect ratio (imageConfig)
ai.defineFlow('image-generation-aspect-ratio-size', async (_) => {
  const { media } = await ai.generate({
    model: vertexAI.model('gemini-3-pro-image'),
    prompt: 'Generate a picture of a sunset in the mountains by a lake',
    config: {
      imageConfig: {
        aspectRatio: '21:9',
        imageSize: '4K',
      },
    },
  });

  return media;
});

// Text rendered in a generated image
ai.defineFlow('image-generation-text-rendering', async (_) => {
  const { media } = await ai.generate({
    model: vertexAI.model('gemini-3.1-flash-image'),
    prompt:
      'Generate an image of the CN Tower. Use words to show the current date, time and weather on the image.',
    config: {
      imageConfig: {
        aspectRatio: '3:4',
        imageSize: '512',
      },
    },
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
    model: vertexAI.model('veo-3.1-lite-generate-001'),
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

// An example of using a Veo model to make a static photo move. The photo is
// AI generated (Veo's safety filter blocks photos of real people).
ai.defineFlow('veo-photo-move', async (_, { sendChunk }) => {
  const startingImage = fs.readFileSync('woman.png', { encoding: 'base64' });

  let { operation } = await ai.generate({
    model: vertexAI.model('veo-3.1-generate-001'),
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

// An example of overriding the region used by a Veo background model at
// call time via config overrides, rather than relying on the
// plugin-configured location ('global', from the plugin init above). This
// is useful when per-request routing (e.g. picking a region closer to the
// caller, or a region where a specific model is available) should override
// the plugin-wide default. Because Veo is a background model, the same
// context must be supplied on both the initial ai.generate() call and every
// subsequent ai.checkOperation() call.
//
// Note: apiKey overrides can't be demonstrated with Veo on Vertex AI
// because (a) standard (non-Express) Vertex AI mode authenticates via
// Application Default Credentials, not an apiKey (apiKey there is only a
// supplementary header, never a replacement for the Bearer token), and
// (b) Vertex AI Express mode -- which *does* use an apiKey for auth --
// does not support Veo's predictLongRunning/fetchPredictOperation methods
// at all. Overriding location is a good stand-in to demonstrate the same
// context-override mechanism working end-to-end.
ai.defineFlow('veo-photo-move-location-override', async (_, { sendChunk }) => {
  const startingImage = fs.readFileSync('woman.png', { encoding: 'base64' });

  let { operation } = await ai.generate({
    model: vertexAI.model('veo-3.1-generate-001'),
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
      durationSeconds: 8,
      aspectRatio: '9:16',
      personGeneration: 'allow_adult',
    },
    // The plugin is initialized with location: 'global' above; override it
    // to 'us-central1' here to prove the override takes effect. On
    // generate(), config overrides ride in context.config.
    context: {
      config: {
        location: 'us-central1',
      },
    },
  });

  if (!operation) {
    throw new Error('Expected the model to return an operation');
  }

  while (!operation.done) {
    sendChunk('check status of operation ' + operation.id);
    // On checkOperation(), config overrides go in the top-level `config`.
    operation = await ai.checkOperation(operation, {
      config: { location: 'us-central1' },
    });
    await new Promise((resolve) => setTimeout(resolve, 5000));
  }

  if (operation.error) {
    sendChunk('Error: ' + operation.error.message);
    throw new Error('failed to generate video: ' + operation.error.message);
  }

  const mediaPart = operation.output?.message?.content.find(
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

ai.defineFlow('lyria-3-music-generation', async () => {
  const response = await ai.generate({
    model: vertexAI.model('lyria-3-clip-preview'),
    prompt:
      'Create a 30-second cheerful acoustic folk song with guitar and harmonica.',
  });

  return response;
});

// Polls a background operation (Deep Research) until it's done. Research
// takes several minutes, so this polls every 20 s.
async function waitForResearch(
  operation?: Operation,
  sendChunk?: StreamingCallback<any>
) {
  if (!operation) {
    throw new Error('Expected the model to return an operation');
  }
  while (!operation.done) {
    sendChunk?.('check status of operation ' + operation.id);
    await new Promise((resolve) => setTimeout(resolve, 20000));
    operation = await ai.checkOperation(operation);
  }
  if (operation.error) {
    throw new Error('Deep Research failed: ' + operation.error.message);
  }
  return operation;
}

// Deep Research is a background model: ai.generate() returns an operation to
// poll with ai.checkOperation(). It runs on the global location.
ai.defineFlow('deep-research', async (_, { sendChunk }) => {
  const { operation } = await ai.generate({
    model: vertexAI.model('deep-research-preview-04-2026'),
    prompt:
      'Compare TCP and UDP in a short markdown table (reliability, connection type, speed).',
    config: { thinkingSummaries: 'AUTO' },
  });
  const done = await waitForResearch(operation, sendChunk);
  return done.output?.message?.content;
});

// Enterprise Web Search instead of Google Search (web data with additional
// compliance controls). Any tool given replaces the default tools.
ai.defineFlow('deep-research-enterprise-search', async (_, { sendChunk }) => {
  const { operation } = await ai.generate({
    model: vertexAI.model('deep-research-preview-04-2026'),
    prompt:
      "Summarize this week's most important AI announcements in 5 bullets.",
    config: {
      tools: [{ googleSearch: { searchTypes: ['enterpriseWebSearch'] } }],
    },
  });
  const done = await waitForResearch(operation, sendChunk);
  return done.output?.message?.content;
});

// Cancelling a running Deep Research task with ai.cancelOperation(). The
// returned operation is done, with finishReason 'aborted'.
ai.defineFlow('deep-research-cancel', async (_, { sendChunk }) => {
  const { operation } = await ai.generate({
    model: vertexAI.model('deep-research-preview-04-2026'),
    prompt: 'Write a detailed history of the TCP protocol.',
  });
  if (!operation) {
    throw new Error('Expected the model to return an operation');
  }
  sendChunk('Started operation: ' + operation.id);
  // Give the task a moment to start before cancelling it.
  await new Promise((resolve) => setTimeout(resolve, 5000));

  const cancelled = await ai.cancelOperation(operation);
  return {
    id: cancelled.id,
    done: cancelled.done,
    finishReason: cancelled.output?.finishReason,
    error: cancelled.error,
  };
});

// Tuned model. Replace the 12345 with your ENDPOINT ID
// from Google Cloud console -> Vertex AI -> Deploy and Use -> Endpoints
ai.defineFlow(
  {
    name: 'tuned-model',
    inputSchema: z.string().default('endpoints/12345'),
    outputSchema: z.string(),
  },
  async (endpoint) => {
    const { text } = await ai.generate({
      model: vertexAI.model(endpoint),
      config: { location: 'us-central1' },
      prompt: 'hello tuned model',
    });
    return text;
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
