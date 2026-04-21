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
import { vertexModelGarden } from '@genkit-ai/vertexai/modelgarden';
// Import the Genkit core libraries and plugins.
import { genkit, z } from 'genkit';

// Import models from the Vertex AI plugin. The Vertex AI API provides access to
// several generative models. Here, we import Gemini 1.5 Flash.

const ai = genkit({
  plugins: [
    // Load the Vertex AI plugin. You can optionally specify your project ID
    // by passing in a config object; if you don't, the Vertex AI plugin uses
    // the value from the GCLOUD_PROJECT environment variable.
    vertexAI({
      location: 'us-central1',
    }),
    vertexModelGarden({
      location: 'us-central1',
    }),
  ],
});

export const anthropicSonnet4Model = ai.defineFlow(
  {
    name: 'claude-sonnet-4 - toolCallingFlow',
    inputSchema: z.string().default('Paris, France'),
    outputSchema: z.string(),
    streamSchema: z.any(),
  },
  async (location, { sendChunk }) => {
    const { response, stream } = ai.generateStream({
      model: vertexModelGarden.model('claude-sonnet-4@20250514'),
      config: {
        temperature: 1,
        location: 'us-east5',
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

export const anthropicSonnet46Model = ai.defineFlow(
  {
    name: 'claude-sonnet-4-6 - basic',
    outputSchema: z.string(),
    streamSchema: z.any(),
  },
  async (_input, { sendChunk }) => {
    const { response, stream } = ai.generateStream({
      model: vertexModelGarden.model('claude-sonnet-4-6'),
      config: {
        temperature: 1,
        location: 'global',
      },
      prompt: `You are a helpful assistant named Walt. Say hello`,
    });

    for await (const chunk of stream) {
      sendChunk(chunk);
    }

    return (await response).text;
  }
);

export const anthropicOpus46Model = ai.defineFlow(
  {
    name: 'claude-opus-4-6 - basic',
    outputSchema: z.string(),
    streamSchema: z.any(),
  },
  async (_input, { sendChunk }) => {
    const { response, stream } = ai.generateStream({
      model: vertexModelGarden.model('claude-opus-4-6'),
      config: {
        temperature: 1,
        location: 'global',
      },
      prompt: `You are a helpful assistant named Walt. Say hello`,
    });

    for await (const chunk of stream) {
      sendChunk(chunk);
    }

    return (await response).text;
  }
);

export const llamaModel = ai.defineFlow(
  {
    name: 'llama4 - basicFlow',
    outputSchema: z.string(),
    streamSchema: z.any(),
  },
  async (_input, { sendChunk }) => {
    const { response, stream } = ai.generateStream({
      model: vertexModelGarden.model(
        'meta/llama-4-maverick-17b-128e-instruct-maas'
      ),
      config: {
        temperature: 1,
        location: 'us-east5',
      },
      prompt: `You are a helpful assistant named Walt. Say hello`,
    });

    for await (const chunk of stream) {
      sendChunk(chunk);
    }

    return (await response).text;
  }
);

// Mistral Large for detailed explanations
export const mistralExplainConcept = ai.defineFlow(
  {
    name: 'mistral-medium - explainConcept',
    inputSchema: z.object({
      concept: z.string().default('concurrency'),
    }),
    outputSchema: z.object({
      explanation: z.string(),
      examples: z.array(z.string()),
    }),
  },
  async ({ concept }) => {
    const explanation = await ai.generate({
      model: vertexModelGarden.model('mistral-medium-3'),
      prompt: `Explain ${concept} in programming. Include practical examples.`,
      config: {
        temperature: 0.7,
      },
      output: {
        schema: z.object({
          explanation: z.string(),
          examples: z.array(z.string()),
        }),
      },
    });

    return explanation.output || { explanation: '', examples: [] };
  }
);

// Mistral small for quick validation and analysis
export const analyzeCode = ai.defineFlow(
  {
    name: 'mistral-small - analyzeCode',
    inputSchema: z.object({
      code: z.string().default("console.log('hello world');"),
    }),
    outputSchema: z.string(),
  },
  async ({ code }) => {
    const analysis = await ai.generate({
      model: vertexModelGarden.model('mistral-small-2503'),
      prompt: `Analyze this code for potential issues and suggest improvements:
              ${code}`,
    });

    return analysis.text;
  }
);

// Codestral for code generation
export const generateFunction = ai.defineFlow(
  {
    name: 'codestral - generateFunction',
    inputSchema: z.object({
      description: z.string().default('greets me and asks my favourite colour'),
    }),
    outputSchema: z.string(),
  },
  async ({ description }) => {
    const result = await ai.generate({
      model: vertexModelGarden.model('codestral-2'),
      prompt: `Create a TypeScript function that ${description}. Include error handling and types.`,
    });

    return result.text;
  }
);

// No naming collisions
export const geminiModel = ai.defineFlow(
  {
    name: 'gemini-2.5-flash - tool flow',
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
