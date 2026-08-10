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

import { enableFirebaseTelemetry } from '@genkit-ai/firebase';
import {
  FirebaseUserEngagementSchema,
  collectUserEngagement,
} from '@genkit-ai/firebase/user_engagement';
import { vertexAI } from '@genkit-ai/google-genai';
import { onCallGenkit, onRequest } from 'firebase-functions/https';
import { genkit, z } from 'genkit';

enableFirebaseTelemetry();

const ai = genkit({
  plugins: [vertexAI()],
});

export const simpleFlow = ai.defineFlow(
  {
    name: 'simpleFlow',
    inputSchema: z.string(),
    outputSchema: z.string(),
  },
  async (subject) => {
    return 'hello world!';
  }
);

// No access to secrets (e.g. only works because simpleFlow does not call
// ai.generate). No authPolicy or App Check integration;
export const simple = onCallGenkit(simpleFlow);

const jokeFlow = ai.defineFlow(
  {
    name: 'jokeFlow',
    inputSchema: z.string(),
    outputSchema: z.string(),
  },
  async (subject) => {
    const prompt = `Tell me a joke about ${subject}`;

    return await ai.run('call-llm', async () => {
      const llmResponse = await ai.generate({
        model: vertexAI.model('gemini-2.5-flash'),
        prompt: prompt,
      });

      return llmResponse.text;
    });
  }
);

export const joke = onCallGenkit(
  {
    secrets: ['apiKey'],
    authPolicy: (auth) => auth?.token?.email_verified && auth?.token?.admin,
  },
  jokeFlow
);

const authFlow = ai.defineFlow(
  {
    name: 'authFlow',
    inputSchema: z.object({ uid: z.string(), input: z.string() }),
    outputSchema: z.string(),
  },
  async ({ input }) => input.toUpperCase()
);

export const auth = onCallGenkit(
  {
    authPolicy: (auth, input) => !!auth && auth.uid === input.uid,
  },
  authFlow
);

const streamingFlow = ai.defineFlow(
  {
    name: 'streamer',
    inputSchema: z.number(),
    outputSchema: z.string(),
    streamSchema: z.object({ count: z.number() }),
  },
  async (count, { sendChunk }) => {
    let i = 0;
    for (; i < count; i++) {
      await new Promise((r) => setTimeout(r, 1000));
      sendChunk({ count: i });
    }
    return `done: ${count}, streamed: ${i} times`;
  }
);

// onCallGenkit automatically handles streaming when passed
// a flow that streams. This example sets invoker to "private",
// which means that IAM enforces access.
export const streamer = onCallGenkit(
  {
    invoker: 'private',
  },
  streamingFlow
);

export const streamConsumerFlow = ai.defineFlow(
  {
    name: 'streamConsumerFlow',
  },
  async () => {
    const { output, stream } = streamingFlow.stream(5);

    for await (const chunk of stream) {
      console.log(`chunk ${chunk}`);
    }

    console.log(`Streaming consumer done ${await output}`);
  }
);

export const streamConsumer = onCallGenkit(
  {
    invoker: 'private',
  },
  streamConsumerFlow
);

export const triggerJokeFlow = onRequest(
  {
    invoker: 'private',
  },
  async (req, res) => {
    const { subject } = req.query;
    console.log('req.query', req.query);
    const op = await jokeFlow.run(String(subject), {
      context: {
        auth: { admin: true },
      },
    });
    console.log('operation', op);
    res.send(op);
  }
);

/** Example of user engagement collection using Firebase Functions. */
export const collectEngagement = onRequest(
  {
    memory: '512MiB',
  },
  async (req, res) => {
    await collectUserEngagement(FirebaseUserEngagementSchema.parse(req.body));
    res.send({});
  }
);
