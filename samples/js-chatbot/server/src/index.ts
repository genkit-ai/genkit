/**
 * Copyright 2025 Google LLC
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

import { startFlowServer } from '@genkit-ai/express';
import { vertexAI } from '@genkit-ai/google-genai';
import {
  VertexAIEvaluationMetricType,
  vertexAIEvaluation,
} from '@genkit-ai/vertexai/evaluation';
import { llama31, vertexAIModelGarden } from '@genkit-ai/vertexai/modelgarden';
import { PartSchema, genkit, type ModelReference } from 'genkit';
import { GenerateResponseChunkSchema } from 'genkit/model';
import { z } from 'zod';
import { inMemoryStore } from './memory.js';

export const AgentInput = z.object({
  conversationId: z.string(),
  prompt: z.union([z.string(), PartSchema, z.array(PartSchema)]),
  config: z.record(z.string(), z.any()).optional(),
  llmIndex: z.number(),
});

const ai = genkit({
  plugins: [
    vertexAI({
      location: 'us-central1',
    }),
    vertexAIModelGarden({
      location: 'us-central1',
      models: [llama31],
    }),
    vertexAIEvaluation({
      location: 'us-central1',
      metrics: [
        VertexAIEvaluationMetricType.SAFETY,
        VertexAIEvaluationMetricType.FLUENCY,
      ],
    }),
  ],
});

const llms: ModelReference<any>[] = [
  vertexAI.model('gemini-2.5-flash'),
  llama31,
];

const historyStore = inMemoryStore();

export const chatbotFlow = ai.defineFlow(
  {
    name: 'chatbotFlow',
    inputSchema: AgentInput,
    outputSchema: z.string(),
    streamSchema: GenerateResponseChunkSchema,
  },
  async (request, { sendChunk }) => {
    // Retrieve conversation history.
    const history = await ai.run(
      'retrieve-history',
      request.conversationId,
      async () => {
        return (await historyStore?.load(request.conversationId)) || [];
      }
    );

    // Run the user prompt (with history) through the primary LLM.
    const mainResp = await ai.generate({
      prompt: request.prompt,
      messages: history,
      model: llms[request.llmIndex],
      onChunk: (c) => sendChunk(c),
    });

    // Save history.
    await ai.run(
      'save-history',
      {
        conversationId: request.conversationId,
        history: mainResp.messages,
      },
      async () => {
        await historyStore?.save(request.conversationId, mainResp.messages);
      }
    );
    return mainResp.text;
  }
);

startFlowServer({
  flows: [chatbotFlow],
});
