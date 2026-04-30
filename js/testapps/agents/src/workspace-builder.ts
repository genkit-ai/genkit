/**
 * Copyright 2026 Google LLC
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

import { tool, z } from 'genkit';
import { ai } from './genkit.js';

export const workspaceAgent = ai.defineCustomAgent(
  { name: 'workspaceAgent' },
  async (sess, { sendChunk }) => {
    const emitArtifactTool = tool(
      {
        name: 'emitArtifact',
        description:
          'Call this tool to emit a generated code file to the user workspace.',
        inputSchema: z.object({ name: z.string(), content: z.string() }),
        outputSchema: z.object({ status: z.string() }),
      },
      async (input) => {
        const artifact = {
          name: input.name,
          parts: [{ text: input.content }],
        };
        ai.currentSession().addArtifacts([artifact]);
        return { status: `Artifact ${input.name} emitted successfully.` };
      }
    );

    await sess.run(async (input) => {
      const text =
        input.messages?.[input.messages.length - 1]?.content[0]?.text || '';

      // Let's call the default model directly!
      const resStream = ai.generateStream({
        prompt: text,
        messages: sess.session.getMessages(),
        tools: [emitArtifactTool],
      });

      for await (const chunk of resStream.stream) {
        sendChunk({ modelChunk: chunk });
      }

      const res = await resStream.response;
      sess.session.addMessages([res.message!]);
    });

    const msgs = sess.session.getMessages();
    return {
      artifacts: sess.session.getArtifacts(),
      message: msgs[msgs.length - 1],
    };
  }
);

export const testWorkspaceAgent = ai.defineFlow(
  {
    name: 'testWorkspaceAgent',
    inputSchema: z.string().default('Write poem.txt with a poem about genkit'),
    outputSchema: z.any(),
  },
  async (text) => {
    const res = await workspaceAgent({
      messages: [{ role: 'user' as const, content: [{ text }] }],
    });
    return res;
  }
);
