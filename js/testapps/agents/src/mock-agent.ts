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

import { artifacts } from '@genkit-ai/middleware';
import { z } from 'genkit';
import { FileSessionStore } from 'genkit/beta';
import { ai } from './genkit.js';

// Define a file store so that snapshot state is stored for restore/checkpointing
const store = new FileSessionStore<{}>('./.snapshots-mock');

// Define a mock tool
export const mockTool = ai.defineTool(
  {
    name: 'mockTool',
    description: 'A mock tool that echoes back input.',
    inputSchema: z.object({ query: z.string() }),
    outputSchema: z.object({ response: z.string() }),
  },
  async (input) => {
    return { response: `Tool received query: "${input.query}"` };
  }
);

// Define a tool to generate rich/multimodal artifacts
export const generateMockArtifact = ai.defineTool(
  {
    name: 'generateMockArtifact',
    description: 'Generates a mock artifact of the specified type.',
    inputSchema: z.object({
      type: z.enum(['image', 'code', 'multi', 'text']),
    }),
    outputSchema: z.object({
      status: z.string(),
    }),
  },
  async (input) => {
    const session = ai.currentSession();
    if (!session) {
      return { status: 'Error: no active session found' };
    }

    if (input.type === 'image') {
      session.addArtifacts([
        {
          name: 'mock_image.png',
          parts: [
            {
              media: {
                contentType: 'image/png',
                url: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAIAAAD/gAIDAAABFUlEQVR4nO3OUQkAIABEsetfWiv4Nx4IC7Cd7XvkByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIReeLesrH9s1agAAAABJRU5ErkJggg==',
              },
            },
          ],
        },
      ]);
      return { status: 'Image artifact mock_image.png generated.' };
    }

    if (input.type === 'code') {
      session.addArtifacts([
        {
          name: 'mock_code.ts',
          parts: [
            {
              text: 'export function helloWorld() {\n  console.log("Hello from mock agent!");\n}\n',
            },
          ],
        },
      ]);
      return { status: 'Code artifact mock_code.ts generated.' };
    }

    if (input.type === 'multi') {
      session.addArtifacts([
        {
          name: 'multi_part_artifact.md',
          parts: [
            {
              text: '# Multi-Part Artifact\nHere is the first markdown text part.\n',
            },
            {
              media: {
                contentType: 'image/png',
                url: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAIAAAD/gAIDAAABFUlEQVR4nO3OUQkAIABEsetfWiv4Nx4IC7Cd7XvkByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIReeLesrH9s1agAAAABJRU5ErkJggg==',
              },
            },
            {
              text: '\nHere is the second markdown text part after the image.',
            },
          ],
        },
      ]);
      return {
        status: 'Multi-part artifact multi_part_artifact.md generated.',
      };
    }

    session.addArtifacts([
      {
        name: 'mock_text.txt',
        parts: [{ text: 'This is a simple text artifact.' }],
      },
    ]);
    return { status: 'Text artifact mock_text.txt generated.' };
  }
);

// Define the custom model to mock responses and finish reasons
export const mockModel = ai.defineModel(
  {
    name: 'mockModel',
  },
  async (request) => {
    const lastUserMsg = request.messages
      ?.filter((m) => m.role === 'user')
      .slice(-1)[0];
    const userText = lastUserMsg?.content?.map((p) => p.text).join(' ') || '';

    const lastUserIndex = request.messages
      ? request.messages.map((m) => m.role).lastIndexOf('user')
      : -1;
    const hasToolResponse =
      lastUserIndex !== -1
        ? request.messages
            .slice(lastUserIndex + 1)
            .some((m) => m.role === 'tool')
        : false;

    // Trigger #finishBlocked
    if (userText.includes('#finishBlocked')) {
      return {
        message: {
          role: 'model' as const,
          content: [
            { text: 'Generating response...' },
            { text: 'WARNING: Content generation blocked by safety filters.' },
          ],
        },
        finishReason: 'blocked',
      };
    }

    // Trigger #finishLength
    if (userText.includes('#finishLength')) {
      return {
        message: {
          role: 'model' as const,
          content: [
            { text: 'Maximum token length exceeded while generating.' },
          ],
        },
        finishReason: 'length',
      };
    }

    // Trigger #finishFailed
    if (userText.includes('#finishFailed')) {
      throw new Error('Simulation of model execution failure');
    }

    // Trigger #finishInterrupted
    if (userText.includes('#finishInterrupted')) {
      return {
        message: {
          role: 'model' as const,
          content: [{ text: 'Generation was interrupted.' }],
        },
        finishReason: 'interrupted',
      };
    }

    // Trigger #finishAborted
    if (userText.includes('#finishAborted')) {
      return {
        message: {
          role: 'model' as const,
          content: [{ text: 'Generation was aborted.' }],
        },
        finishReason: 'aborted',
      };
    }

    // Trigger #multimodal
    if (userText.includes('#multimodal')) {
      return {
        message: {
          role: 'model' as const,
          content: [
            {
              text: 'Here is some text content followed by a red square image:',
            },
            {
              media: {
                contentType: 'image/png',
                url: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAIAAAD/gAIDAAABFUlEQVR4nO3OUQkAIABEsetfWiv4Nx4IC7Cd7XvkByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIX4Q4gchfhDiByF+EOIHIReeLesrH9s1agAAAABJRU5ErkJggg==',
              },
            },
          ],
        },
      };
    }

    // Trigger #toolCall
    if (userText.includes('#toolCall')) {
      if (!hasToolResponse) {
        return {
          message: {
            role: 'model' as const,
            content: [
              {
                toolRequest: {
                  name: 'mockTool',
                  input: { query: 'test query' },
                },
              },
            ],
          },
        };
      } else {
        const toolMsg = request.messages?.find((m) => m.role === 'tool');
        const toolOutput = toolMsg?.content?.find((p) => p.toolResponse)
          ?.toolResponse?.output as any;
        return {
          message: {
            role: 'model' as const,
            content: [
              {
                text: `Mock tool completed successfully! Output: ${JSON.stringify(
                  toolOutput
                )}`,
              },
            ],
          },
        };
      }
    }

    // Trigger #imageArtifact
    if (userText.includes('#imageArtifact')) {
      if (!hasToolResponse) {
        return {
          message: {
            role: 'model' as const,
            content: [
              {
                toolRequest: {
                  name: 'generateMockArtifact',
                  input: { type: 'image' },
                },
              },
            ],
          },
        };
      } else {
        return {
          message: {
            role: 'model' as const,
            content: [
              {
                text: 'Mock image artifact has been successfully written to session state!',
              },
            ],
          },
        };
      }
    }

    // Trigger #codeArtifact
    if (userText.includes('#codeArtifact')) {
      if (!hasToolResponse) {
        return {
          message: {
            role: 'model' as const,
            content: [
              {
                toolRequest: {
                  name: 'generateMockArtifact',
                  input: { type: 'code' },
                },
              },
            ],
          },
        };
      } else {
        return {
          message: {
            role: 'model' as const,
            content: [
              {
                text: 'Mock code artifact has been successfully written to session state!',
              },
            ],
          },
        };
      }
    }

    // Trigger #multiArtifact
    if (userText.includes('#multiArtifact')) {
      if (!hasToolResponse) {
        return {
          message: {
            role: 'model' as const,
            content: [
              {
                toolRequest: {
                  name: 'generateMockArtifact',
                  input: { type: 'multi' },
                },
              },
            ],
          },
        };
      } else {
        return {
          message: {
            role: 'model' as const,
            content: [
              {
                text: 'Mock multi-part artifact has been successfully written to session state!',
              },
            ],
          },
        };
      }
    }

    return {
      message: {
        role: 'model' as const,
        content: [
          {
            text:
              'Hello! I am a Mock Agent. Trigger different behaviors with these keywords in your prompt:\n\n' +
              '- `#finishBlocked` to test Safety Filter/Blocked state\n' +
              '- `#finishLength` to test Maximum Token Length state\n' +
              '- `#finishFailed` to test Failure state\n' +
              '- `#finishInterrupted` to test Interrupted state\n' +
              '- `#finishAborted` to test Aborted state\n' +
              '- `#multimodal` to receive text and media/images\n' +
              '- `#toolCall` to run a mock tool\n' +
              '- `#imageArtifact` to generate an image artifact\n' +
              '- `#codeArtifact` to generate a code artifact\n' +
              '- `#multiArtifact` to generate a multi-part artifact',
          },
        ],
      },
    };
  }
);

// Define the server-managed mock agent
export const mockAgent = ai.defineAgent({
  name: 'mockAgent',
  description: 'A server-managed mock agent to test Dev UI edge cases.',
  system: 'You are a mock agent assisting with UI validation.',
  model: mockModel,
  tools: [mockTool, generateMockArtifact],
  use: [artifacts()],
  store,
});

// Define the client-managed (stateless) mock agent
export const mockAgentStateless = ai.defineAgent({
  name: 'mockAgentStateless',
  description:
    'A client-managed (stateless) mock agent to test Dev UI edge cases.',
  system: 'You are a mock agent assisting with UI validation.',
  model: mockModel,
  tools: [mockTool, generateMockArtifact],
  use: [artifacts()],
});

// Test flow for mockAgent
export const testMockAgent = ai.defineFlow(
  {
    name: 'testMockAgent',
    inputSchema: z.string().default('Hello'),
    outputSchema: z.any(),
  },
  async (text, { sendChunk }) => {
    const res = await mockAgent.run(
      {
        messages: [{ role: 'user', content: [{ text }] }],
      },
      {
        onChunk: sendChunk,
      }
    );
    return res.result;
  }
);
