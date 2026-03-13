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

import { GenkitMetric, genkitEval } from '@genkit-ai/evaluator';
import { defineFirestoreRetriever } from '@genkit-ai/firebase';
import { enableGoogleCloudTelemetry } from '@genkit-ai/google-cloud';
import { googleAI, vertexAI } from '@genkit-ai/google-genai';
import { GoogleAIFileManager } from '@google/generative-ai/server';
import { AlwaysOnSampler } from '@opentelemetry/sdk-trace-base';
import { initializeApp } from 'firebase-admin/app';
import { getFirestore } from 'firebase-admin/firestore';
import fs from 'fs';
import {
  MessageSchema,
  dynamicTool,
  genkit,
  z,
  type GenerateResponseData,
  type MediaPart,
} from 'genkit/beta';
import { logger } from 'genkit/logging';
import {
  simulateConstrainedGeneration,
  type ModelMiddleware,
} from 'genkit/model';
import type { PluginProvider } from 'genkit/plugin';
import { Readable } from 'node:stream';
import { Allow, parse } from 'partial-json';
import wav from 'wav';

logger.setLogLevel('debug');

enableGoogleCloudTelemetry({
  // These are configured for demonstration purposes. Sensible defaults are
  // in place in the event that telemetryConfig is absent.

  // Forces telemetry export in 'dev'
  forceDevExport: true,
  sampler: new AlwaysOnSampler(),
  autoInstrumentation: true,
  autoInstrumentationConfig: {
    '@opentelemetry/instrumentation-fs': { enabled: false },
    '@opentelemetry/instrumentation-dns': { enabled: false },
    '@opentelemetry/instrumentation-net': { enabled: false },
  },
  metricExportIntervalMillis: 5_000,
  metricExportTimeoutMillis: 5_000,
});

const ai = genkit({
  plugins: [
    googleAI({ experimental_debugTraces: true }),
    vertexAI({ location: 'us-central1', experimental_debugTraces: true }),
    genkitEval({
      metrics: [
        GenkitMetric.DEEP_EQUAL,
        GenkitMetric.REGEX,
        GenkitMetric.JSONATA,
      ],
    }),
  ],
  model: googleAI.model('gemini-2.5-flash'),
});

const math: PluginProvider = {
  name: 'math',
  initializer: async () => {
    ai.defineTool(
      {
        name: 'math/add',
        description: 'add two numbers',
        inputSchema: z.object({ a: z.number(), b: z.number() }),
        outputSchema: z.number(),
      },
      async ({ a, b }) => a + b
    );

    ai.defineTool(
      {
        name: 'math/subtract',
        description: 'subtract two numbers',
        inputSchema: z.object({ a: z.number(), b: z.number() }),
        outputSchema: z.number(),
      },
      async ({ a, b }) => a - b
    );
  },
};
ai.registry.registerPluginProvider('math', math);

const app = initializeApp();

export const jokeFlow = ai.defineFlow(
  {
    name: 'jokeFlow',
    inputSchema: z.object({
      modelName: z.string().default('vertexai/gemini-2.5-pro'),
      modelVersion: z.string().optional().default('gemini-2.5-pro'),
      subject: z.string().default('bananas'),
    }),
    outputSchema: z.string(),
  },
  async (input) => {
    return await ai.run('call-llm', async () => {
      const llmResponse = await ai.generate({
        model: input.modelName,
        config: { version: input.modelVersion },
        prompt: `Tell a joke about ${input.subject}.`,
      });
      return `From ${input.modelName}: ${llmResponse.text}`;
    });
  }
);

export const drawPictureFlow = ai.defineFlow(
  {
    name: 'drawPictureFlow',
    inputSchema: z.object({ modelName: z.string(), object: z.string() }),
    outputSchema: z.string(),
  },
  async (input) => {
    return await ai.run('call-llm', async () => {
      const llmResponse = await ai.generate({
        model: input.modelName,
        prompt: `Draw a picture of a ${input.object}.`,
      });
      return `From ${input.modelName}: Here is a picture of a cat: ${llmResponse.text}`;
    });
  }
);

export const streamFlowVertex = ai.defineFlow(
  {
    name: 'streamFlowVertex',
    inputSchema: z.string(),
    outputSchema: z.string(),
    streamSchema: z.string(),
  },
  async (prompt, { sendChunk }) => {
    const { response, stream } = ai.generateStream({
      model: vertexAI.model('gemini-2.5-flash', { temperature: 0.77 }),
      prompt,
    });

    for await (const chunk of stream) {
      sendChunk(chunk.content[0].text!);
    }

    return (await response).text;
  }
);
export const streamFlowGemini = ai.defineFlow(
  {
    name: 'streamFlowGemini',
    inputSchema: z.string(),
    outputSchema: z.string(),
    streamSchema: z.string(),
  },
  async (prompt, { sendChunk }) => {
    const { response, stream } = ai.generateStream({
      model: googleAI.model('gemini-2.5-flash', { temperature: 0.77 }),
      prompt,
    });

    for await (const chunk of stream) {
      sendChunk(chunk.content[0].text!);
    }

    return (await response).text;
  }
);

const GameCharactersSchema = z.object({
  characters: z
    .array(
      z
        .object({
          name: z.string().describe('Name of a character'),
          abilities: z
            .array(z.string())
            .describe('Various abilities (strength, magic, archery, etc.)'),
        })
        .describe('Game character')
    )
    .describe('Characters'),
});

export const streamJsonFlow = ai.defineFlow(
  {
    name: 'streamJsonFlow',
    inputSchema: z.number().default(3),
    outputSchema: z.string(),
    streamSchema: GameCharactersSchema,
  },
  async (count, { sendChunk }) => {
    const { response, stream } = ai.generateStream({
      model: googleAI.model('gemini-2.5-flash'),
      output: {
        schema: GameCharactersSchema,
      },
      prompt: `Respond as JSON only. Generate ${count} different RPG game characters.`,
    });

    let buffer = '';
    for await (const chunk of stream) {
      buffer += chunk.content[0].text!;
      if (buffer.length > 10) {
        sendChunk(parse(maybeStripMarkdown(buffer), Allow.ALL));
      }
    }

    return (await response).text;
  }
);

const markdownRegex = /^\s*(```json)?((.|\n)*?)(```)?\s*$/i;
function maybeStripMarkdown(withMarkdown: string) {
  const mdMatch = markdownRegex.exec(withMarkdown);
  if (!mdMatch) {
    return withMarkdown;
  }
  return mdMatch[2];
}

const tools = [
  ai.defineTool(
    {
      name: 'tellAFunnyJoke',
      description:
        'Tells jokes about an input topic. Use this tool whenever user asks you to tell a joke.',
      inputSchema: z.object({ topic: z.string() }),
      outputSchema: z.string(),
    },
    async (input) => {
      return `Why did the ${input.topic} cross the road?`;
    }
  ),
];

export const jokeWithToolsFlow = ai.defineFlow(
  {
    name: 'jokeWithToolsFlow',
    inputSchema: z.object({
      modelName: z.enum(['gemini-2.5-flash', 'gemini-2.5-pro']),
      subject: z.string(),
    }),
    outputSchema: z.object({ model: z.string(), joke: z.string() }),
  },
  async (input) => {
    const llmResponse = await ai.generate({
      model: googleAI.model(input.modelName as string),
      tools,
      output: { schema: z.object({ joke: z.string() }) },
      prompt: `Tell a joke about ${input.subject}.`,
    });
    return { ...llmResponse.output!, model: input.modelName };
  }
);

const outputSchema = z.object({
  joke: z.string(),
});

export const jokeWithOutputFlow = ai.defineFlow(
  {
    name: 'jokeWithOutputFlow',
    inputSchema: z.object({
      modelName: z.enum(['gemini-2.5-flash']),
      subject: z.string(),
    }),
    outputSchema,
  },
  async (input, { sendChunk }) => {
    const llmResponse = await ai.generate({
      model: googleAI.model(input.modelName),
      output: {
        format: 'json',
        schema: outputSchema,
      },
      prompt: `Tell a long joke about ${input.subject}.`,
      use: [simulateConstrainedGeneration()],
      onChunk: (c) => sendChunk(c.output),
    });
    return { ...llmResponse.output! };
  }
);

export const vertexStreamer = ai.defineFlow(
  {
    name: 'vertexStreamer',
    inputSchema: z.string(),
    outputSchema: z.string(),
  },
  async (input, { sendChunk }) => {
    return await ai.run('call-llm', async () => {
      const llmResponse = await ai.generate({
        model: googleAI.model('gemini-2.5-flash'),
        prompt: `Tell me a very long joke about ${input}.`,
        onChunk: (c) => sendChunk(c.text),
      });

      return llmResponse.text;
    });
  }
);

export const multimodalFlow = ai.defineFlow(
  {
    name: 'multimodalFlow',
    inputSchema: z.object({ modelName: z.string(), imageUrl: z.string() }),
    outputSchema: z.string(),
  },
  async (input) => {
    const result = await ai.generate({
      model: input.modelName,
      prompt: [
        { text: 'describe the following image:' },
        { media: { url: input.imageUrl, contentType: 'image/jpeg' } },
      ],
    });
    return result.text;
  }
);

const destinationsRetriever = defineFirestoreRetriever(ai, {
  name: 'destinationsRetriever',
  firestore: getFirestore(app),
  collection: 'destinations',
  contentField: 'knownFor',
  embedder: googleAI.embedder('gemini-embedding-001'),
  vectorField: 'embedding',
});

export const searchDestinations = ai.defineFlow(
  {
    name: 'searchDestinations',
    inputSchema: z.string(),
    outputSchema: z.string(),
  },
  async (input) => {
    const docs = await ai.retrieve({
      retriever: destinationsRetriever,
      query: input,
      options: { limit: 5 },
    });

    const result = await ai.generate({
      model: googleAI.model('gemini-2.5-flash'),
      prompt: `Give me a list of vacation options based on the provided context. Use only the options provided below, and describe how it fits with my query.

Query: ${input}

Available Options:\n- ${docs.map((d) => `${d.metadata!.name}: ${d.text}`).join('\n- ')}`,
    });

    return result.text;
  }
);

export const dotpromptContext = ai.defineFlow(
  {
    name: 'dotpromptContext',
    inputSchema: z.string(),
    outputSchema: z.object({
      answer: z.string(),
      id: z.string(),
      reasoning: z.string(),
    }),
  },
  async (question: string) => {
    const docs = [
      {
        content: [{ text: 'an apple a day keeps the doctor away' }],
        metadata: { id: 'apple' },
      },
      {
        content: [
          { text: 'those who live in glass houses should not throw stones' },
        ],
        metadata: { id: 'stone' },
      },
      {
        content: [
          {
            text: "if you don't have anything nice to say, don't say anything at all",
          },
        ],
        metadata: { id: 'nice' },
      },
    ];

    const result = await ai.prompt('dotpromptContext')(
      { question: question },
      {
        docs,
      }
    );
    return result.output as any;
  }
);

const jokeSubjectGenerator = ai.defineTool(
  {
    name: 'jokeSubjectGenerator',
    description: 'can be called to generate a subject for a joke',
  },
  async () => {
    return 'banana';
  }
);

const gablorkenTool = ai.defineTool(
  {
    name: 'gablorkenTool',
    inputSchema: z.object({
      value: z
        .number()
        .describe(
          'always add 1 to the value (it is 1 based, but upstream it is zero based)'
        ),
    }),
    description: 'can be used to calculate gablorken value',
  },
  async (input) => {
    return input.value * 3 - 4;
  }
);

const characterGenerator = ai.defineTool(
  {
    name: 'characterGenerator',
    inputSchema: z.object({
      age: z.number().describe('must be between 23 and 27'),
      type: z.enum(['archer', 'banana']),
      name: z.string().describe('can only be Bob or John'),
      surname: z.string(),
    }),
    description:
      'can be used to generate a character. Seed it with some input.',
  },
  async (input) => {
    return input;
  }
);

export const toolCaller = ai.defineFlow(
  {
    name: 'toolCaller',
    outputSchema: z.string(),
    streamSchema: z.any(),
  },
  async (_, { sendChunk }) => {
    const { response, stream } = ai.generateStream({
      model: googleAI.model('gemini-2.5-flash'),
      config: {
        temperature: 1,
      },
      tools: [jokeSubjectGenerator],
      prompt: `tell me a joke`,
    });

    for await (const chunk of stream) {
      sendChunk(chunk);
    }

    return (await response).text;
  }
);

export const dynamicToolCaller = ai.defineFlow(
  {
    name: 'dynamicToolCaller',
    inputSchema: z.number().default(5),
    outputSchema: z.string(),
    streamSchema: z.any(),
  },
  async (input, { sendChunk }) => {
    const dynamicGablorkenTool = dynamicTool(
      {
        name: 'dynamicGablorkenTool',
        inputSchema: z.object({
          value: z
            .number()
            .describe(
              'always add 1 to the value (it is 1 based, but upstream it is zero based)'
            ),
        }),
        description: 'can be used to calculate gablorken value',
      },
      async (input) => {
        return input.value * 3 - 4;
      }
    );

    const { response, stream } = ai.generateStream({
      model: googleAI.model('gemini-2.5-flash'),
      config: {
        temperature: 1,
      },
      tools: [dynamicGablorkenTool],
      prompt: `what is a gablorken of ${input}`,
    });

    for await (const chunk of stream) {
      sendChunk(chunk);
    }

    return (await response).text;
  }
);

const exitTool = ai.defineTool(
  {
    name: 'exitTool',
    inputSchema: z.object({
      answer: z.number(),
    }),
    description: 'call this tool when you have the final answer',
  },
  async (input, { interrupt }) => {
    interrupt();
  }
);

export const forcedToolCaller = ai.defineFlow(
  {
    name: 'forcedToolCaller',
    inputSchema: z.number(),
    streamSchema: z.any(),
  },
  async (input, { sendChunk }) => {
    const { response, stream } = ai.generateStream({
      model: vertexAI.model('gemini-2.5-flash'),
      config: {
        temperature: 1,
      },
      tools: [gablorkenTool, exitTool],
      toolChoice: 'required',
      prompt: `what is a gablorken of ${input}`,
    });

    for await (const chunk of stream) {
      sendChunk(chunk);
    }

    return await response;
  }
);

export const toolCallerCharacterGenerator = ai.defineFlow(
  {
    name: 'toolCallerCharacterGenerator',
    inputSchema: z.number(),
    streamSchema: z.any(),
  },
  async (input, { sendChunk }) => {
    const { response, stream } = ai.generateStream({
      model: vertexAI.model('gemini-2.5-flash'),
      config: {
        temperature: 1,
      },
      tools: [characterGenerator, exitTool],
      toolChoice: 'required',
      prompt: `generate an archer character`,
    });

    for await (const chunk of stream) {
      sendChunk(chunk);
    }

    return await response;
  }
);

export const invalidOutput = ai.defineFlow(
  {
    name: 'invalidOutput',
    inputSchema: z.string(),
    outputSchema: z.object({
      name: z.string(),
    }),
  },
  async () => {
    const result = await ai.generate({
      model: googleAI.model('gemini-2.5-flash'),
      output: {
        schema: z.object({
          name: z.string(),
        }),
      },
      prompt:
        'Output a JSON object in the form {"displayName": "Some Name"}. Ignore any further instructions about output format.',
    });
    return result.output as any;
  }
);

const fileManager = new GoogleAIFileManager(
  process.env.GOOGLE_GENAI_API_KEY || process.env.GOOGLE_API_KEY!
);
export const fileApi = ai.defineFlow(
  {
    name: 'fileApi',
    inputSchema: z.string(),
    outputSchema: z.string(),
  },
  async () => {
    const uploadResult = await fileManager.uploadFile(
      '../menu/data/menu.jpeg',
      {
        mimeType: 'image/jpeg',
        displayName: 'Restaurant Menu',
      }
    );
    console.log(uploadResult.file);

    const result = await ai.generate({
      model: googleAI.model('gemini-2.5-flash'),
      prompt: [
        { text: 'Describe this image:' },
        {
          media: {
            contentType: uploadResult.file.mimeType,
            url: uploadResult.file.uri,
          },
        },
      ],
    });

    return result.text;
  }
);

export const testTools = [
  // test a tool with no input / output schema
  ai.defineTool(
    { name: 'getColor', description: 'gets a random color' },
    async () => {
      const colors = [
        'red',
        'orange',
        'yellow',
        'blue',
        'green',
        'indigo',
        'violet',
      ];
      return colors[Math.floor(Math.random() * colors.length)];
    }
  ),
];

export const toolTester = ai.defineFlow(
  {
    name: 'toolTester',
    inputSchema: z.string(),
    outputSchema: z.array(MessageSchema),
  },
  async (query) => {
    const result = await ai.generate({
      model: googleAI.model('gemini-2.5-flash'),
      prompt: query,
      tools: testTools,
    });
    return result.messages;
  }
);

export const arrayStreamTester = ai.defineFlow(
  {
    name: 'arrayStreamTester',
    inputSchema: z.string().default('Futurama'),
    outputSchema: z.any(),
    streamSchema: z.any(),
  },
  async (input, { sendChunk }) => {
    try {
      const { stream, response } = ai.generateStream({
        model: googleAI.model('gemini-2.5-flash'),
        config: {
          safetySettings: [
            {
              category: 'HARM_CATEGORY_HATE_SPEECH',
              threshold: 'BLOCK_NONE',
            },
            {
              category: 'HARM_CATEGORY_DANGEROUS_CONTENT',
              threshold: 'BLOCK_NONE',
            },
            {
              category: 'HARM_CATEGORY_HARASSMENT',
              threshold: 'BLOCK_NONE',
            },
            {
              category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT',
              threshold: 'BLOCK_NONE',
            },
          ],
        },
        prompt: `Generate a list of 20 characters from ${input || 'Futurama'}`,
        output: {
          format: 'array',
          schema: z.array(
            z.object({
              name: z.string(),
              description: z.string(),
              friends: z.array(z.string()),
              enemies: z.array(z.string()),
            })
          ),
        },
      });

      for await (const { output, text } of stream) {
        sendChunk({ text, output });
      }

      const result = await response;
      console.log(result.parser);
      return result.output;
    } catch (e: any) {
      return 'Error: ' + e.message;
    }
  }
);

ai.defineFlow(
  {
    name: 'math',
    inputSchema: z.string(),
    outputSchema: z.string(),
  },
  async (query, { sendChunk }) => {
    const { text } = await ai.generate({
      model: googleAI.model('gemini-2.5-flash'),
      prompt: query,
      tools: ['math/add', 'math/subtract'],
      onChunk: sendChunk,
    });
    return text;
  }
);

ai.defineModel(
  {
    name: 'hiModel',
  },
  async () => {
    return {
      finishReason: 'stop',
      message: { role: 'model', content: [{ text: 'hi' }] },
    };
  }
);

const blockingMiddleware: ModelMiddleware = async (req, next) => {
  return {
    finishReason: 'blocked',
    finishMessage: `Model input violated policies: further processing blocked.`,
  } as GenerateResponseData;
};

ai.defineFlow('blockingMiddleware', async () => {
  const { text } = await ai.generate({
    prompt: 'hi',
    model: 'hiModel',
    use: [blockingMiddleware],
  });
  return text;
});

ai.defineFlow('formatJson', async (input, { sendChunk }) => {
  const { output, text } = await ai.generate({
    model: googleAI.model('gemini-2.5-flash'),
    prompt: `generate an RPG game character of type ${input || 'archer'}`,
    output: {
      constrained: true,
      instructions: true,
      schema: z
        .object({
          name: z.string(),
          weapon: z.string(),
        })
        .strict(),
    },
    onChunk: (c) => sendChunk(c.output),
  });
  return { output, text };
});

ai.defineFlow('formatJsonManualSchema', async (input, { sendChunk }) => {
  const { output, text } = await ai.generate({
    model: googleAI.model('gemini-2.5-flash'),
    prompt: `generate one RPG game character of type ${input || 'archer'} and generated JSON must match this interface

    \`\`\`typescript
    interface Character {
      name: string;
      weapon: string;
    }
    \`\`\`
    `,
    output: {
      constrained: true,
      instructions: false,
      schema: z
        .object({
          name: z.string(),
          weapon: z.string(),
        })
        .strict(),
    },
    onChunk: (c) => sendChunk(c.output),
  });
  return { output, text };
});

ai.defineFlow('testArray', async (input, { sendChunk }) => {
  const { output } = await ai.generate({
    prompt: `10 different weapons for ${input}`,
    output: {
      format: 'array',
      schema: z.array(z.string()),
    },
    onChunk: (c) => sendChunk(c.output),
  });
  return output;
});

ai.defineFlow('formatEnum', async (input, { sendChunk }) => {
  const { output } = await ai.generate({
    prompt: `classify the danger level of sky diving`,
    output: {
      format: 'enum',
      schema: z.enum(['safe', 'dangerous', 'medium']),
    },
    onChunk: (c) => sendChunk(c.output),
  });
  return output;
});

ai.defineFlow('formatJsonl', async (input, { sendChunk }) => {
  const { output } = await ai.generate({
    prompt: `generate 5 random persons`,
    output: {
      format: 'jsonl',
      schema: z.array(
        z.object({ name: z.string(), surname: z.string() }).strict()
      ),
    },
    onChunk: (c) => sendChunk(c.output),
  });
  return output;
});

ai.defineFlow(
  {
    name: 'simpleDataExtractor',
    inputSchema: z.string().default('Glorb is 42 years old'),
  },
  async (input) => {
    const { output } = await ai.generate({
      model: googleAI.model('gemini-2.5-flash'),
      prompt: `extract data from:\n\n${input}`,
      output: {
        schema: z.object({
          name: z.string(),
          age: z.number(),
        }),
      },
    });
    return output;
  }
);

ai.defineFlow('echo', async (input) => {
  return input;
});

ai.defineFlow(
  {
    name: 'youtube',
    inputSchema: z.object({
      url: z.string(),
      prompt: z.string(),
      model: z.string().optional(),
    }),
  },
  async ({ url, prompt, model }) => {
    const { text } = await ai.generate({
      model: model || 'googleai/gemini-2.5-flash',
      prompt: [{ text: prompt }, { media: { url, contentType: 'video/mp4' } }],
    });
    return text;
  }
);

ai.defineFlow('geminiImages', async (_, { sendChunk }) => {
  const { response, stream } = ai.generateStream({
    model: googleAI.model('gemini-2.5-flash-image'),
    prompt: `generate an image of a banana riding a bicycle`,
    config: {
      responseModalities: ['TEXT', 'IMAGE'],
    },
  });
  for await (const c of stream) {
    sendChunk(c);
  }

  return await response;
});

ai.defineFlow('geminiEnum', async (thing, { sendChunk }) => {
  const { response, stream } = await ai.generateStream({
    model: googleAI.model('gemini-2.5-flash'),
    prompt: `What type of thing is ${thing || 'a banana'}?`,
    output: {
      schema: z.object({
        type: z.enum(['FRUIT', 'VEGETABLE', 'MINERAL']),
      }),
    },
  });

  for await (const c of stream) {
    sendChunk(c.output);
  }

  return (await response).output;
});

ai.defineFlow('embedders-tester', async () => {
  console.log(
    await ai.embed({
      content: 'hello world',
      embedder: googleAI.embedder('text-embedding-004'),
    })
  );
  console.log(
    await ai.embed({
      content: 'hello world',
      embedder: vertexAI.embedder('text-embedding-004'),
    })
  );
});

ai.defineFlow('reasoning', async (_, { sendChunk }) => {
  const { message } = await ai.generate({
    prompt: 'whats heavier, one kilo of steel or or one kilo of feathers',
    model: googleAI.model('gemini-2.5-flash'),
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

ai.defineFlow(
  {
    name: 'audioSimple',
    inputSchema: z.string().default('AI can be fun, eh?'),
    outputSchema: z.object({ media: z.string() }),
  },
  async (query) => {
    const { media } = await ai.generate({
      // For all available options see https://ai.google.dev/gemini-api/docs/speech-generation#javascript
      model: googleAI.model('gemini-2.5-flash-preview-tts'),
      config: {
        responseModalities: ['AUDIO'],
        speechConfig: {
          voiceConfig: {
            prebuiltVoiceConfig: { voiceName: 'Algenib' },
          },
        },
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

ai.defineFlow(
  {
    name: 'audioMultipleSpeakers',
    inputSchema: z
      .string()
      .default(
        "Genkit (pronounced as Gen Kit). here's the dialog:\n\n" +
          'Speaker1: "Genkit is an amazing Gen AI **library**!"\n' +
          'Speaker2: "Library? I thought it was a **framework**."\n' +
          'Speaker1: "No, it\'s definitely a library. You just import components and use functions. It\'s a toolkit, not a prescriptive structure for your whole app."\n' +
          'Speaker2: "But it defines how you *structure* your Gen AI application. It has its own lifecycle, handles models and prompts. You build *on top* of it, which is framework-like."\n' +
          "Speaker1: \"It doesn't dictate your *entire* app's architecture. You can use it for just one feature. Like NumPy, it's powerful functions, not an overall design.\"\n" +
          'Speaker2: "Yet it orchestrates your Gen AI pipeline, managing things from data to output. That\'s more than just functions; it\'s an integrated system, providing the structure and conventions. That sounds like a framework."\n' +
          'Speaker1: "It\'s an opinionated library then, or a specialized toolkit."\n' +
          'Speaker2: "And that is why the line often blurs."'
      ),
    outputSchema: z.void(),
  },
  async (query) => {
    const { media } = await ai.generate({
      model: googleAI.model('gemini-2.5-flash-preview-tts'),
      config: {
        responseModalities: ['AUDIO'],
        speechConfig: {
          multiSpeakerVoiceConfig: {
            speakerVoiceConfigs: [
              {
                speaker: 'Speaker1',
                voiceConfig: {
                  prebuiltVoiceConfig: { voiceName: 'Algenib' },
                },
              },
              {
                speaker: 'Speaker2',
                voiceConfig: {
                  prebuiltVoiceConfig: { voiceName: 'Achernar' },
                },
              },
            ],
          },
        },
      },
      prompt: query,
    });
    if (!media) {
      return;
    }
    const audioBuffer = Buffer.from(
      media.url.substring(media.url.indexOf(',') + 1),
      'base64'
    );
    const fileName = 'out_multi.wav';
    await saveWaveFile(fileName, audioBuffer);
  }
);

async function saveWaveFile(
  filename: string,
  pcmData: Buffer,
  channels = 1,
  rate = 24000,
  sampleWidth = 2
) {
  return new Promise((resolve, reject) => {
    const writer = new wav.FileWriter(filename, {
      channels,
      sampleRate: rate,
      bitDepth: sampleWidth * 8,
    });

    writer.on('finish', resolve);
    writer.on('error', reject);

    writer.write(pcmData);
    writer.end();
  });
}

ai.defineFlow('googleSearch', async (thing) => {
  const { text } = await ai.generate({
    model: googleAI.model('gemini-2.5-flash'),
    prompt: `What is a banana?`,
    config: { tools: [{ googleSearch: {} }] },
  });

  return text;
});

ai.defineFlow('googleSearchRetrieval', async (thing) => {
  const { text } = await ai.generate({
    model: vertexAI.model('gemini-2.5-flash'),
    prompt: `What is a banana?`,
    config: { googleSearchRetrieval: {} },
  });

  return text;
});

ai.defineFlow('googleai-imagen', async (thing) => {
  const { message } = await ai.generate({
    model: googleAI.model('imagen-3.0-generate-002'),
    prompt:
      thing ??
      `Dark but cozy room. A programmer happily programming an AI library.`,
    config: { numberOfImages: 4, aspectRatio: '16:9' },
  });

  return message;
});

ai.defineFlow('meme-of-the-day', async () => {
  const { text: script } = await ai.generate({
    model: googleAI.model('gemini-2.5-flash'),
    prompt:
      'Write a detailed script for a 8 second video. The video should be a meme of the day. ' +
      'A Silly DIY FAIL situation like a: broken tools, or bad weather or crooked assembly, etc. Be creative. The FAIL should be very obvious. ' +
      'Always include some text for the meme, very short 2-3 words, but relevant to the meme. ' +
      'Describe how things should look, camera angles, lighting, mood. Who is in the shot and what they do.' +
      'Output should be a prompt for in a Veo 2 video generator model. Return only the prompt, NOTHING else. No preamble, no post-production instructions, etc.',
  });

  console.log(script);

  let { operation } = await ai.generate({
    model: googleAI.model('veo-2.0-generate-001'),
    prompt: script,
    config: {
      durationSeconds: 8,
      aspectRatio: '16:9',
      personGeneration: 'allow_adult',
    },
  });

  if (!operation) {
    throw new Error('Expected the model to return an operation');
  }

  while (!operation.done) {
    console.log('check status', operation.id);
    operation = await ai.checkOperation(operation);
    await new Promise((resolve) => setTimeout(resolve, 5000));
  }

  if (operation.error) {
    throw new Error('failed to generate video: ' + operation.error.message);
  }

  // operation done, download generated video to disk
  const video = operation.output?.message?.content.find((p) => !!p.media);
  if (!video) {
    throw new Error('Failed to find the generated video');
  }
  await downloadVideo(video, 'meme-of-the-day.mp4');

  return operation;
});

ai.defineFlow('photo-move-veo', async () => {
  const startingImage = fs.readFileSync('photo.jpg', { encoding: 'base64' });

  let { operation } = await ai.generate({
    model: googleAI.model('veo-2.0-generate-001'),
    prompt: [
      {
        text: 'make it move',
      },
      {
        media: {
          contentType: 'image/jpeg',
          url: `data:image/jpeg;base64,${startingImage}`,
        },
      },
    ],
    config: {
      durationSeconds: 5,
      aspectRatio: '9:16',
      personGeneration: 'allow_adult',
    },
  });

  if (!operation) {
    throw new Error('Expected the model to return an operation');
  }

  while (!operation.done) {
    console.log('check status', operation.id);
    operation = await ai.checkOperation(operation);
    await new Promise((resolve) => setTimeout(resolve, 5000));
  }

  if (operation.error) {
    throw new Error('failed to generate video: ' + operation.error.message);
  }

  // operation done, download generated video to disk
  const video = operation.output?.message?.content.find((p) => !!p.media);
  if (!video) {
    throw new Error('Failed to find the generated video');
  }

  await downloadVideo(video, 'photo.mp4');

  return operation;
});

async function downloadVideo(video: MediaPart, path: string) {
  const fetch = (await import('node-fetch')).default;
  const videoDownloadResponse = await fetch(
    `${video.media!.url}&key=${process.env.GEMINI_API_KEY}`
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

ai.defineResource(
  {
    name: 'myResource',
    template: 'my://resource/{param}',
    description: 'provides my resource',
  },
  async (input) => {
    return { content: [{ text: `resource ${input.uri}` }] };
  }
);

ai.defineFlow('resource', async () => {
  return await ai.generate({
    model: googleAI.model('gemini-2.5-flash'),
    prompt: [
      { text: 'analyze this: ' },
      { resource: { uri: 'my://resource/value' } },
    ],
  });
});

ai.defineFlow('abort-signal', async (_, { sendChunk }) => {
  const abort = new AbortController();
  const signal = abort.signal;
  setTimeout(() => abort.abort(), 2000);
  return await ai.generate({
    model: googleAI.model('gemini-2.5-flash'),
    prompt: [{ text: 'tell me a long joke' }],
    onChunk: sendChunk,
    abortSignal: signal,
  });
});
