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

import {
  genkit,
  type AgentStreamChunk,
  type GenkitBeta,
} from 'genkit/beta';
import * as assert from 'assert';
import { describe, it, beforeEach } from 'node:test';

import {
  defineA2AAgent,
  type A2AClientFactory,
  type A2AClientLike,
} from '../src/define-a2a-agent.js';

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/**
 * Creates a mock A2A client factory that returns a client with a configurable
 * `sendMessage` implementation.
 */
function createMockClientFactory(
  sendMessageFn: A2AClientLike['sendMessage']
): A2AClientFactory {
  const client: A2AClientLike = { sendMessage: sendMessageFn };
  return {
    createFromUrl: async (_url: string) => client,
  };
}

/**
 * Runs a single turn against an agent, collecting streamed chunks and returning
 * the final output.
 */
async function runSingleTurn(
  agent: ReturnType<typeof defineA2AAgent>,
  messages: Array<{ role: string; content: Array<Record<string, any>> }>,
  init?: Record<string, any>
) {
  const chunks: AgentStreamChunk[] = [];

  const result = await agent.run(
    { messages: messages as any },
    {
      init: init || {},
      onChunk: (chunk: AgentStreamChunk) => {
        chunks.push(chunk);
      },
    }
  );

  return { result, chunks };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('defineA2AAgent', () => {
  let ai: GenkitBeta;

  beforeEach(() => {
    ai = genkit({});
  });

  it('creates an agent that can be looked up in the registry', () => {
    const agent = defineA2AAgent(ai, {
      name: 'test-remote',
      agentUrl: 'https://fake-agent.example.com',
      clientFactory: createMockClientFactory(async () => ({
        kind: 'message',
        messageId: 'msg-1',
        role: 'agent',
        parts: [{ kind: 'text', text: 'hi' }],
      })),
    });

    assert.ok(agent, 'Agent should be defined');
    assert.strictEqual(typeof agent.run, 'function');
  });

  it('sends a user message and receives a text response (message response)', async () => {
    const factory = createMockClientFactory(async (params) => {
      // Verify the outgoing message structure
      const msg = params.message;
      assert.strictEqual(msg.role, 'user');
      assert.strictEqual(msg.parts.length, 1);
      assert.strictEqual((msg.parts[0] as any).kind, 'text');
      assert.strictEqual((msg.parts[0] as any).text, 'Hello remote agent');

      return {
        kind: 'message',
        messageId: 'resp-1',
        role: 'agent',
        parts: [{ kind: 'text', text: 'Hello from A2A!' }],
      } as any;
    });

    const agent = defineA2AAgent(ai, {
      name: 'msg-response-agent-' + Math.random(),
      agentUrl: 'https://fake.example.com',
      clientFactory: factory,
    });

    const { result } = await runSingleTurn(agent, [
      { role: 'user', content: [{ text: 'Hello remote agent' }] },
    ]);

    assert.ok(result.result.message, 'Should have a response message');
    const content = result.result.message!.content;
    assert.ok(content.some((p: any) => p.text === 'Hello from A2A!'));
  });

  it('sends a user message and receives a task response with history', async () => {
    const factory = createMockClientFactory(async (params) => {
      return {
        kind: 'task',
        id: 'task-123',
        contextId: params.message.contextId || 'ctx-1',
        status: { state: 'completed', timestamp: new Date().toISOString() },
        history: [
          {
            kind: 'message',
            messageId: 'h1',
            role: 'user',
            parts: [{ kind: 'text', text: 'What is 2+2?' }],
          },
          {
            kind: 'message',
            messageId: 'h2',
            role: 'agent',
            parts: [{ kind: 'text', text: '2+2 is 4.' }],
          },
        ],
        artifacts: [],
      } as any;
    });

    const agent = defineA2AAgent(ai, {
      name: 'task-response-agent-' + Math.random(),
      agentUrl: 'https://fake.example.com',
      clientFactory: factory,
    });

    const { result } = await runSingleTurn(agent, [
      { role: 'user', content: [{ text: 'What is 2+2?' }] },
    ]);

    assert.ok(result.result.message);
    const content = result.result.message!.content;
    assert.ok(content.some((p: any) => p.text === '2+2 is 4.'));
  });

  it('extracts artifacts from the task response', async () => {
    const factory = createMockClientFactory(async () => {
      return {
        kind: 'task',
        id: 'task-art',
        contextId: 'ctx-art',
        status: { state: 'completed', timestamp: new Date().toISOString() },
        history: [
          {
            kind: 'message',
            messageId: 'h1',
            role: 'agent',
            parts: [{ kind: 'text', text: 'Here is the report.' }],
          },
        ],
        artifacts: [
          {
            artifactId: 'report-1',
            name: 'Monthly Report',
            parts: [{ kind: 'text', text: 'Report content here...' }],
          },
        ],
      } as any;
    });

    const agent = defineA2AAgent(ai, {
      name: 'artifact-agent-' + Math.random(),
      agentUrl: 'https://fake.example.com',
      clientFactory: factory,
    });

    const { result } = await runSingleTurn(agent, [
      { role: 'user', content: [{ text: 'Generate a report' }] },
    ]);

    assert.ok(result.result.artifacts);
    assert.ok(result.result.artifacts!.length > 0);
    const artifact = result.result.artifacts![0];
    assert.strictEqual(artifact.name, 'report-1');
    assert.ok(artifact.parts.some((p: any) => p.text === 'Report content here...'));
  });

  it('passes contextId across turns for multi-turn continuity', async () => {
    const receivedContextIds: (string | undefined)[] = [];

    const factory = createMockClientFactory(async (params) => {
      receivedContextIds.push(params.message.contextId);
      return {
        kind: 'task',
        id: 'task-' + receivedContextIds.length,
        contextId: params.message.contextId || 'ctx-new',
        status: { state: 'completed', timestamp: new Date().toISOString() },
        history: [
          {
            kind: 'message',
            messageId: 'resp',
            role: 'agent',
            parts: [{ kind: 'text', text: 'Response ' + receivedContextIds.length }],
          },
        ],
        artifacts: [],
      } as any;
    });

    const agent = defineA2AAgent(ai, {
      name: 'multi-turn-agent-' + Math.random(),
      agentUrl: 'https://fake.example.com',
      clientFactory: factory,
    });

    // First turn
    await runSingleTurn(agent, [
      { role: 'user', content: [{ text: 'Turn 1' }] },
    ]);

    // Should have received a contextId
    assert.ok(receivedContextIds[0], 'First turn should have a contextId');
  });

  it('caches the A2A client across invocations', async () => {
    let createCount = 0;
    const factory: A2AClientFactory = {
      createFromUrl: async (_url: string) => {
        createCount++;
        return {
          sendMessage: async () =>
            ({
              kind: 'message',
              messageId: 'r',
              role: 'agent',
              parts: [{ kind: 'text', text: 'ok' }],
            }) as any,
        };
      },
    };

    const agent = defineA2AAgent(ai, {
      name: 'cache-agent-' + Math.random(),
      agentUrl: 'https://fake.example.com',
      clientFactory: factory,
    });

    await runSingleTurn(agent, [
      { role: 'user', content: [{ text: 'First' }] },
    ]);
    await runSingleTurn(agent, [
      { role: 'user', content: [{ text: 'Second' }] },
    ]);

    assert.strictEqual(createCount, 1, 'Client should be created only once');
  });

  it('maps tool response parts with role detection', async () => {
    let receivedRole: string | undefined;

    const factory = createMockClientFactory(async (params) => {
      receivedRole = params.message.role;
      return {
        kind: 'message',
        messageId: 'r',
        role: 'agent',
        parts: [{ kind: 'text', text: 'processed' }],
      } as any;
    });

    const agent = defineA2AAgent(ai, {
      name: 'tool-resp-agent-' + Math.random(),
      agentUrl: 'https://fake.example.com',
      clientFactory: factory,
    });

    // Send a message that contains a toolResponse → should be detected as "agent" role
    await runSingleTurn(agent, [
      {
        role: 'tool',
        content: [
          {
            toolResponse: {
              ref: 'c1',
              name: 'myTool',
              output: { result: 42 },
            },
          },
        ],
      },
    ]);

    assert.strictEqual(receivedRole, 'agent', 'Tool response should map to agent role');
  });

  it('handles task response with no history gracefully', async () => {
    const factory = createMockClientFactory(async () => {
      return {
        kind: 'task',
        id: 'task-empty',
        contextId: 'ctx-empty',
        status: { state: 'completed', timestamp: new Date().toISOString() },
        // No history, no artifacts
      } as any;
    });

    const agent = defineA2AAgent(ai, {
      name: 'empty-history-agent-' + Math.random(),
      agentUrl: 'https://fake.example.com',
      clientFactory: factory,
    });

    // Should not throw
    const { result } = await runSingleTurn(agent, [
      { role: 'user', content: [{ text: 'Hi' }] },
    ]);

    // Output should still be valid, just possibly empty
    assert.ok(result.result);
  });

  it('skips intermediate tool call messages and picks the final text message', async () => {
    const factory = createMockClientFactory(async () => {
      return {
        kind: 'task',
        id: 'task-tools',
        contextId: 'ctx-tools',
        status: { state: 'completed', timestamp: new Date().toISOString() },
        history: [
          {
            kind: 'message',
            messageId: 'h1',
            role: 'user',
            parts: [{ kind: 'text', text: 'Get weather' }],
          },
          {
            kind: 'message',
            messageId: 'h2',
            role: 'agent',
            parts: [
              {
                kind: 'data',
                data: { id: 'c1', name: 'get_weather', args: { city: 'London' } },
                metadata: { genkit_type: 'function_call' },
              },
            ],
          },
          {
            kind: 'message',
            messageId: 'h3',
            role: 'agent',
            parts: [
              {
                kind: 'data',
                data: { id: 'c1', name: 'get_weather', response: { temp: 20 } },
                metadata: { genkit_type: 'function_response' },
              },
            ],
          },
          {
            kind: 'message',
            messageId: 'h4',
            role: 'agent',
            parts: [{ kind: 'text', text: 'The weather in London is 20°C.' }],
          },
        ],
        artifacts: [],
      } as any;
    });

    const agent = defineA2AAgent(ai, {
      name: 'tools-agent-' + Math.random(),
      agentUrl: 'https://fake.example.com',
      clientFactory: factory,
    });

    const { result } = await runSingleTurn(agent, [
      { role: 'user', content: [{ text: 'Get weather' }] },
    ]);

    assert.ok(result.result.message);
    const content = result.result.message!.content;
    // Should have the final text answer, not the tool calls
    assert.ok(
      content.some((p: any) => p.text === 'The weather in London is 20°C.'),
      'Should extract the final text message'
    );
  });
});
