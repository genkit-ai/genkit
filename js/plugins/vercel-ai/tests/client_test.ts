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

import { expressHandler } from '@genkit-ai/express';
import type { UIMessage } from 'ai';
import * as assert from 'assert';
import express from 'express';
import { z } from 'genkit';
import { genkit, InMemorySessionStore } from 'genkit/beta';
import type {
  GenerateRequest,
  GenerateResponseChunkData,
  GenerateResponseData,
} from 'genkit/model';
import getPort from 'get-port';
import type * as http from 'http';
import { afterEach, beforeEach, describe, it } from 'node:test';
import {
  GenkitChatTransport,
  restartInterrupt,
  type UIMessageChunk,
} from '../src/client.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Collect all chunks from a ReadableStream into an array. */
async function collectChunks(
  stream: ReadableStream<UIMessageChunk>
): Promise<UIMessageChunk[]> {
  const chunks: UIMessageChunk[] = [];
  const reader = stream.getReader();
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
  }
  return chunks;
}

/** Create a UIMessage with the given role and text. */
function makeUIMessage(
  role: UIMessage['role'],
  text: string,
  parts?: UIMessage['parts']
): UIMessage {
  return {
    id: `msg-${crypto.randomUUID()}`,
    role,
    parts: parts ?? [{ type: 'text', text }],
  };
}

/** A fresh, valid session id (the useChat `id` must be a bare UUID). */
function newChatId(): string {
  return crypto.randomUUID();
}

/** Filter chunks by type. */
function chunksOfType<T extends UIMessageChunk['type']>(
  chunks: UIMessageChunk[],
  type: T
): Extract<UIMessageChunk, { type: T }>[] {
  return chunks.filter((c) => c.type === type) as any;
}

/** Extract the ordered list of chunk types. */
function chunkTypes(chunks: UIMessageChunk[]): string[] {
  return chunks.map((c) => c.type);
}

// ---------------------------------------------------------------------------
// Test suite
// ---------------------------------------------------------------------------

describe('GenkitChatTransport e2e', () => {
  let server: http.Server;
  let port: number;

  // Programmable model response handler — tests set this before each call.
  let modelHandler: (
    req: GenerateRequest,
    sendChunk: (chunk: GenerateResponseChunkData) => void
  ) => Promise<GenerateResponseData>;

  // Captures the HTTP headers of the most recent request to the agent
  // endpoint so tests can assert that the transport forwarded them.
  let lastRequestHeaders: http.IncomingHttpHeaders = {};

  beforeEach(async () => {
    lastRequestHeaders = {};

    const ai = genkit({});

    // ── Programmable model ──────────────────────────────────────────────
    ai.defineModel(
      { name: 'programmableModel', apiVersion: 'v2' },
      async (request, { sendChunk }) => {
        return modelHandler(request, sendChunk);
      }
    );

    // ── Interrupt tool ─────────────────────────────────────────────────
    const confirmAction = ai.defineInterrupt({
      name: 'confirmAction',
      description: 'Ask the user for confirmation',
      inputSchema: z.object({ action: z.string() }),
      outputSchema: z.object({ confirmed: z.boolean() }),
    });

    // ── Regular tool ───────────────────────────────────────────────────
    ai.defineTool(
      {
        name: 'getWeather',
        description: 'Get current weather',
        inputSchema: z.object({ city: z.string() }),
        outputSchema: z.object({ temp: z.number(), condition: z.string() }),
      },
      async (input) => ({ temp: 72, condition: 'sunny' })
    );

    // ── Restartable tool ───────────────────────────────────────────────
    // Interrupts on its first call (asking the user to proceed); once the
    // user opts to *restart* it, the tool re-runs and — because it is now
    // `resumed` — returns a real result instead of interrupting again. The
    // `resumed` metadata supplied by the client is echoed back so tests can
    // assert it was threaded through.
    ai.defineTool(
      {
        name: 'riskyAction',
        description: 'Performs an action that must be confirmed via restart',
        inputSchema: z.object({ target: z.string() }),
        outputSchema: z.object({
          done: z.boolean(),
          resumedWith: z.any().optional(),
        }),
      },
      async (input, { interrupt, resumed }) => {
        if (!resumed) {
          interrupt();
        }
        return { done: true, resumedWith: resumed };
      }
    );

    // ── Agent (server-managed with store) ──────────────────────────────
    const store = new InMemorySessionStore();
    const agent = ai.defineAgent({
      name: 'testAgent',
      model: 'programmableModel',
      tools: [confirmAction, 'getWeather', 'riskyAction'],
      store,
    });

    // ── Express server ─────────────────────────────────────────────────
    const app = express();
    app.use(express.json());
    // Capture request headers so tests can assert the transport forwards them.
    app.use((req, _res, next) => {
      lastRequestHeaders = req.headers;
      next();
    });
    port = await getPort();
    app.post('/testAgent', expressHandler(agent as any));
    server = app.listen(port);
  });

  afterEach(() => {
    server.close();
  });

  // ── Test: Basic text streaming ──────────────────────────────────────

  it('should stream text from agent and emit correct chunk sequence', async () => {
    modelHandler = async (_req, sendChunk) => {
      sendChunk({ content: [{ text: 'Hello' }] });
      sendChunk({ content: [{ text: ' world' }] });
      return {
        message: { role: 'model', content: [{ text: 'Hello world' }] },
        finishReason: 'stop',
      };
    };

    const transport = new GenkitChatTransport({
      url: `http://localhost:${port}/testAgent`,
    });

    const stream = await transport.sendMessages({
      trigger: 'submit-message',
      chatId: newChatId(),
      messageId: undefined,
      messages: [makeUIMessage('user', 'Hi')],
      abortSignal: undefined,
    });

    const chunks = await collectChunks(stream);
    const types = chunkTypes(chunks);

    // Must start with 'start' and end with 'finish'
    assert.strictEqual(types[0], 'start');
    assert.strictEqual(types[types.length - 1], 'finish');

    // No error chunks
    assert.strictEqual(chunksOfType(chunks, 'error').length, 0);

    // Exactly one text block: text-start, text-delta(s), text-end
    const textStarts = chunksOfType(chunks, 'text-start');
    const textEnds = chunksOfType(chunks, 'text-end');
    assert.strictEqual(textStarts.length, 1, 'Exactly one text-start');
    assert.strictEqual(textEnds.length, 1, 'Exactly one text-end');

    // text-start and text-end must share the same block id
    assert.strictEqual(textStarts[0].id, textEnds[0].id);

    // Text deltas carry the actual streamed content
    const textDeltas = chunksOfType(chunks, 'text-delta');
    assert.strictEqual(textDeltas.length, 2, 'Two text-delta chunks expected');
    assert.strictEqual(textDeltas[0].delta, 'Hello');
    assert.strictEqual(textDeltas[1].delta, ' world');

    // All text-delta ids must match the text-start id
    for (const td of textDeltas) {
      assert.strictEqual(td.id, textStarts[0].id);
    }

    // Chunk ordering: start → start-step → text-start → text-delta → … → text-end → finish-step → finish
    const startIdx = types.indexOf('start');
    const startStepIdx = types.indexOf('start-step');
    const textStartIdx = types.indexOf('text-start');
    const firstDeltaIdx = types.indexOf('text-delta');
    const textEndIdx = types.indexOf('text-end');
    const finishStepIdx = types.indexOf('finish-step');
    const finishIdx = types.indexOf('finish');

    assert.ok(startIdx < startStepIdx, 'start before start-step');
    assert.ok(startStepIdx < textStartIdx, 'start-step before text-start');
    assert.ok(textStartIdx < firstDeltaIdx, 'text-start before text-delta');
    assert.ok(firstDeltaIdx < textEndIdx, 'text-delta before text-end');
    assert.ok(textEndIdx < finishStepIdx, 'text-end before finish-step');
    assert.ok(finishStepIdx < finishIdx, 'finish-step before finish');
  });

  // ── Test: Reasoning streaming ───────────────────────────────────────

  it('should stream reasoning parts as reasoning chunks', async () => {
    modelHandler = async (_req, sendChunk) => {
      // Model emits reasoning ("thinking") before the visible answer.
      sendChunk({
        content: [{ reasoning: 'First, I consider the options. ' }],
      });
      sendChunk({ content: [{ reasoning: 'Then I decide.' }] });
      sendChunk({ content: [{ text: 'The answer is 42.' }] });
      return {
        message: { role: 'model', content: [{ text: 'The answer is 42.' }] },
        finishReason: 'stop',
      };
    };

    const transport = new GenkitChatTransport({
      url: `http://localhost:${port}/testAgent`,
    });

    const stream = await transport.sendMessages({
      trigger: 'submit-message',
      chatId: newChatId(),
      messageId: undefined,
      messages: [makeUIMessage('user', 'What is the answer?')],
      abortSignal: undefined,
    });

    const chunks = await collectChunks(stream);
    const types = chunkTypes(chunks);

    // No errors
    assert.strictEqual(chunksOfType(chunks, 'error').length, 0);

    // Exactly one reasoning block: reasoning-start, reasoning-delta(s), reasoning-end
    const reasoningStarts = chunksOfType(chunks, 'reasoning-start');
    const reasoningEnds = chunksOfType(chunks, 'reasoning-end');
    assert.strictEqual(
      reasoningStarts.length,
      1,
      'Exactly one reasoning-start'
    );
    assert.strictEqual(reasoningEnds.length, 1, 'Exactly one reasoning-end');
    assert.strictEqual(reasoningStarts[0].id, reasoningEnds[0].id);

    // Reasoning deltas carry the streamed thinking content
    const reasoningDeltas = chunksOfType(chunks, 'reasoning-delta');
    assert.strictEqual(reasoningDeltas.length, 2, 'Two reasoning-delta chunks');
    assert.strictEqual(
      reasoningDeltas[0].delta,
      'First, I consider the options. '
    );
    assert.strictEqual(reasoningDeltas[1].delta, 'Then I decide.');
    for (const rd of reasoningDeltas) {
      assert.strictEqual(rd.id, reasoningStarts[0].id);
    }

    // Visible answer is still streamed as text
    const textDeltas = chunksOfType(chunks, 'text-delta');
    assert.strictEqual(textDeltas.length, 1);
    assert.strictEqual(textDeltas[0].delta, 'The answer is 42.');

    // Ordering: reasoning block fully closes before the text block opens
    const reasoningEndIdx = types.indexOf('reasoning-end');
    const textStartIdx = types.indexOf('text-start');
    assert.ok(
      reasoningEndIdx < textStartIdx,
      'reasoning-end before text-start'
    );
  });

  // ── Test: Multi-turn conversation (server-managed by sessionId) ──────

  it('should maintain conversation across turns via sessionId', async () => {
    let callCount = 0;
    const capturedRequests: GenerateRequest[] = [];

    modelHandler = async (req, sendChunk) => {
      callCount++;
      capturedRequests.push(JSON.parse(JSON.stringify(req)));
      const userMsgs = req.messages.filter((m) => m.role === 'user');
      const lastUserText =
        userMsgs[userMsgs.length - 1]?.content[0]?.text ?? '';

      sendChunk({ content: [{ text: `Reply ${callCount}` }] });
      return {
        message: {
          role: 'model',
          content: [{ text: `Reply to: ${lastUserText}` }],
        },
        finishReason: 'stop',
      };
    };

    const transport = new GenkitChatTransport({
      url: `http://localhost:${port}/testAgent`,
    });

    // The same chat id is reused across turns; the agent keeps per-session
    // state keyed by that id.
    const chatId = newChatId();

    // Turn 1
    const stream1 = await transport.sendMessages({
      trigger: 'submit-message',
      chatId,
      messageId: undefined,
      messages: [makeUIMessage('user', 'Hello')],
      abortSignal: undefined,
    });
    const chunks1 = await collectChunks(stream1);

    // Turn 1 must produce text
    assert.strictEqual(chunksOfType(chunks1, 'error').length, 0);
    const deltas1 = chunksOfType(chunks1, 'text-delta');
    assert.strictEqual(deltas1.length, 1);
    assert.strictEqual(deltas1[0].delta, 'Reply 1');

    // Turn 2 — same sessionId resumes the server-side state automatically.
    const stream2 = await transport.sendMessages({
      trigger: 'submit-message',
      chatId,
      messageId: undefined,
      messages: [
        makeUIMessage('user', 'Hello'),
        makeUIMessage('assistant', 'Reply to: Hello'),
        makeUIMessage('user', 'How are you?'),
      ],
      abortSignal: undefined,
    });
    const chunks2 = await collectChunks(stream2);

    assert.strictEqual(chunksOfType(chunks2, 'error').length, 0);
    const deltas2 = chunksOfType(chunks2, 'text-delta');
    assert.strictEqual(deltas2.length, 1);
    assert.strictEqual(deltas2[0].delta, 'Reply 2');

    // The model was called twice total
    assert.strictEqual(callCount, 2);

    // Turn 2's model request should contain the accumulated history from
    // the server-side session (not just the latest message). The agent
    // adds the user message from turn 1 and the model reply, plus the
    // new user message — so there should be at least 2 user messages.
    const turn2Req = capturedRequests[1];
    const turn2UserMsgs = turn2Req.messages.filter((m) => m.role === 'user');
    assert.ok(
      turn2UserMsgs.length >= 2,
      `Model should see at least 2 user messages in turn 2, got ${turn2UserMsgs.length}`
    );
  });

  // ── Test: Tool call (auto-executed by agent) ────────────────────────

  it('should stream tool call chunks when agent auto-executes a tool', async () => {
    let callCount = 0;
    const capturedRequests: GenerateRequest[] = [];

    modelHandler = async (req, sendChunk) => {
      callCount++;
      capturedRequests.push(JSON.parse(JSON.stringify(req)));

      if (callCount === 1) {
        // First call: model requests the getWeather tool.
        // Stream the toolRequest chunk so the transport can observe it.
        const toolContent = [
          {
            toolRequest: {
              name: 'getWeather',
              input: { city: 'London' },
              ref: 'tool-call-1',
            },
          },
        ];
        sendChunk({ content: toolContent });
        return {
          message: { role: 'model', content: toolContent },
          finishReason: 'stop',
        };
      }

      // Second call: after tool execution, model returns final text
      sendChunk({ content: [{ text: 'The weather is sunny!' }] });
      return {
        message: {
          role: 'model',
          content: [{ text: 'The weather in London is 72°F and sunny.' }],
        },
        finishReason: 'stop',
      };
    };

    const transport = new GenkitChatTransport({
      url: `http://localhost:${port}/testAgent`,
    });

    const stream = await transport.sendMessages({
      trigger: 'submit-message',
      chatId: newChatId(),
      messageId: undefined,
      messages: [makeUIMessage('user', "What's the weather in London?")],
      abortSignal: undefined,
    });

    const chunks = await collectChunks(stream);
    const types = chunkTypes(chunks);

    // No errors
    assert.strictEqual(chunksOfType(chunks, 'error').length, 0);

    // ── Tool input assertions ──────────────────────────────────────────
    const toolInputStarts = chunksOfType(chunks, 'tool-input-start');
    assert.strictEqual(
      toolInputStarts.length,
      1,
      'Exactly one tool-input-start'
    );
    assert.strictEqual(toolInputStarts[0].toolName, 'getWeather');
    assert.strictEqual(toolInputStarts[0].toolCallId, 'tool-call-1');

    const toolInputAvails = chunksOfType(chunks, 'tool-input-available');
    assert.strictEqual(
      toolInputAvails.length,
      1,
      'Exactly one tool-input-available'
    );
    assert.strictEqual(toolInputAvails[0].toolName, 'getWeather');
    assert.strictEqual(toolInputAvails[0].toolCallId, 'tool-call-1');
    assert.deepStrictEqual(toolInputAvails[0].input, { city: 'London' });

    // ── Tool output assertions ─────────────────────────────────────────
    const toolOutputs = chunksOfType(chunks, 'tool-output-available');
    assert.strictEqual(
      toolOutputs.length,
      1,
      'Exactly one tool-output-available'
    );
    assert.strictEqual(toolOutputs[0].toolCallId, 'tool-call-1');
    assert.deepStrictEqual(toolOutputs[0].output, {
      temp: 72,
      condition: 'sunny',
    });

    // ── Text assertions ────────────────────────────────────────────────
    const textDeltas = chunksOfType(chunks, 'text-delta');
    assert.strictEqual(textDeltas.length, 1, 'Exactly one text-delta');
    assert.strictEqual(textDeltas[0].delta, 'The weather is sunny!');

    // ── Ordering: tool chunks come before text chunks ──────────────────
    const toolInputIdx = types.indexOf('tool-input-start');
    const toolOutputIdx = types.indexOf('tool-output-available');
    const textDeltaIdx = types.indexOf('text-delta');
    assert.ok(toolInputIdx < toolOutputIdx, 'tool-input before tool-output');
    assert.ok(toolOutputIdx < textDeltaIdx, 'tool-output before text-delta');

    // Model was called twice (tool request → tool exec → model reply)
    assert.strictEqual(callCount, 2);

    // Second model request should contain the tool response in its messages
    const req2 = capturedRequests[1];
    const toolMsgs = req2.messages.filter((m: any) => m.role === 'tool');
    assert.ok(
      toolMsgs.length >= 1,
      'Second model call should see tool response'
    );
  });

  // ── Test: Interrupt and resume ──────────────────────────────────────

  it('should handle interrupt and resume flow', async () => {
    let callCount = 0;
    const capturedRequests: GenerateRequest[] = [];

    modelHandler = async (req, sendChunk) => {
      callCount++;
      capturedRequests.push(JSON.parse(JSON.stringify(req)));

      if (callCount === 1) {
        // First call: model requests the confirmAction interrupt tool
        return {
          message: {
            role: 'model',
            content: [
              {
                toolRequest: {
                  name: 'confirmAction',
                  input: { action: 'delete account' },
                  ref: 'interrupt-ref-1',
                },
              },
            ],
          },
          finishReason: 'stop',
        };
      }

      // Second call (after resume): model receives tool response and replies
      sendChunk({ content: [{ text: 'Done!' }] });
      return {
        message: {
          role: 'model',
          content: [{ text: 'Account deletion confirmed and processed.' }],
        },
        finishReason: 'stop',
      };
    };

    const transport = new GenkitChatTransport({
      url: `http://localhost:${port}/testAgent`,
    });

    const chatId = newChatId();

    // ── Phase 1: Initial message triggers interrupt ──────────────────
    const stream1 = await transport.sendMessages({
      trigger: 'submit-message',
      chatId,
      messageId: undefined,
      messages: [makeUIMessage('user', 'Delete my account')],
      abortSignal: undefined,
    });
    const chunks1 = await collectChunks(stream1);

    // No errors
    assert.strictEqual(chunksOfType(chunks1, 'error').length, 0);

    // Must have tool-input-start with correct tool details
    const toolInputStarts1 = chunksOfType(chunks1, 'tool-input-start');
    assert.strictEqual(toolInputStarts1.length, 1);
    assert.strictEqual(toolInputStarts1[0].toolName, 'confirmAction');
    assert.strictEqual(toolInputStarts1[0].toolCallId, 'interrupt-ref-1');

    // Must have tool-input-available with correct input payload
    const toolInputAvails1 = chunksOfType(chunks1, 'tool-input-available');
    assert.strictEqual(toolInputAvails1.length, 1);
    assert.strictEqual(toolInputAvails1[0].toolName, 'confirmAction');
    assert.strictEqual(toolInputAvails1[0].toolCallId, 'interrupt-ref-1');
    assert.deepStrictEqual(toolInputAvails1[0].input, {
      action: 'delete account',
    });

    // Must NOT have any tool-output-available (interrupt = no auto-execution)
    assert.strictEqual(
      chunksOfType(chunks1, 'tool-output-available').length,
      0,
      'Interrupt should not produce tool outputs'
    );

    // Must NOT have any text-delta (model did not produce text)
    assert.strictEqual(
      chunksOfType(chunks1, 'text-delta').length,
      0,
      'Interrupt should not produce text deltas'
    );

    // Stream envelope must be complete
    assert.strictEqual(chunkTypes(chunks1)[0], 'start');
    assert.strictEqual(chunkTypes(chunks1).at(-1), 'finish');

    // ── Phase 2: Resume with resolved tool results (v6 format) ───────
    const messagesForResume: UIMessage[] = [
      makeUIMessage('user', 'Delete my account'),
      {
        id: 'assistant-msg-1',
        role: 'assistant',
        parts: [
          {
            type: 'tool-confirmAction',
            toolCallId: 'interrupt-ref-1',
            state: 'output-available',
            input: { action: 'delete account' },
            output: { confirmed: true },
          } as UIMessage['parts'][number],
        ],
      },
    ];

    const stream2 = await transport.sendMessages({
      trigger: 'submit-message',
      chatId,
      messageId: undefined,
      messages: messagesForResume,
      abortSignal: undefined,
    });
    const chunks2 = await collectChunks(stream2);

    // No errors
    assert.strictEqual(chunksOfType(chunks2, 'error').length, 0);

    // Must have text content from the resumed response
    const textDeltas2 = chunksOfType(chunks2, 'text-delta');
    assert.strictEqual(textDeltas2.length, 1, 'Resume produces one text-delta');
    assert.strictEqual(textDeltas2[0].delta, 'Done!');

    // Stream envelope complete
    assert.strictEqual(chunkTypes(chunks2)[0], 'start');
    assert.strictEqual(chunkTypes(chunks2).at(-1), 'finish');

    // Model was called exactly twice total (once per phase)
    assert.strictEqual(callCount, 2);

    // The second model call (resume) should contain the tool response
    const resumeReq = capturedRequests[1];
    const toolResponseMsgs = resumeReq.messages.filter(
      (m: any) => m.role === 'tool'
    );
    assert.ok(
      toolResponseMsgs.length >= 1,
      'Resume model request should contain tool response message'
    );
    const toolResponsePart = toolResponseMsgs[0].content.find(
      (p: any) => p.toolResponse
    );
    assert.ok(toolResponsePart, 'Should have a toolResponse part');
    assert.deepStrictEqual(toolResponsePart.toolResponse?.output, {
      confirmed: true,
    });
  });

  // ── Test: Interrupt resume with v6 per-tool format ──────────────────
  //
  // SDK v6 represents tool parts as `{ type: 'tool-<name>', toolCallId,
  // state, output }` — without a separate `toolName` property. The
  // transport must derive the tool name from the `type` field.

  it('should handle interrupt resume with v6 per-tool format (no toolName property)', async () => {
    let callCount = 0;
    const capturedRequests: GenerateRequest[] = [];

    modelHandler = async (req, sendChunk) => {
      callCount++;
      capturedRequests.push(JSON.parse(JSON.stringify(req)));

      if (callCount === 1) {
        // First call: model requests the confirmAction interrupt tool
        return {
          message: {
            role: 'model',
            content: [
              {
                toolRequest: {
                  name: 'confirmAction',
                  input: { action: 'transfer funds' },
                  ref: 'v6-interrupt-ref',
                },
              },
            ],
          },
          finishReason: 'stop',
        };
      }

      // Second call (after resume): model receives tool response and replies
      sendChunk({ content: [{ text: 'Transfer complete.' }] });
      return {
        message: {
          role: 'model',
          content: [{ text: 'Transfer complete.' }],
        },
        finishReason: 'stop',
      };
    };

    const transport = new GenkitChatTransport({
      url: `http://localhost:${port}/testAgent`,
    });

    const chatId = newChatId();

    // ── Phase 1: Initial message triggers interrupt ──────────────────
    const stream1 = await transport.sendMessages({
      trigger: 'submit-message',
      chatId,
      messageId: undefined,
      messages: [makeUIMessage('user', 'Transfer 500 to savings')],
      abortSignal: undefined,
    });
    const chunks1 = await collectChunks(stream1);

    assert.strictEqual(chunksOfType(chunks1, 'error').length, 0);
    assert.strictEqual(
      chunksOfType(chunks1, 'tool-input-start').length,
      1,
      'Phase 1 should emit tool-input-start'
    );

    // ── Phase 2: Resume with v6 per-tool format parts ─────────────────
    // This mirrors what SDK v6's `useChat` sends after setMessages:
    // - type: 'tool-confirmAction' (name encoded in type)
    // - NO `toolName` property (SDK v6 ToolUIPart doesn't have one)
    // - state: 'output-available'
    // - output: { confirmed: true }
    const messagesForResume: UIMessage[] = [
      makeUIMessage('user', 'Transfer 500 to savings'),
      {
        id: 'assistant-msg-v6',
        role: 'assistant',
        parts: [
          // v6 per-tool format: no `toolInvocation` wrapper, no `toolName`
          {
            type: 'tool-confirmAction',
            toolCallId: 'v6-interrupt-ref',
            // toolName is NOT present — SDK v6 ToolUIPart doesn't have it
            state: 'output-available',
            input: { action: 'transfer funds' },
            output: { confirmed: true },
          } as UIMessage['parts'][number],
        ],
      },
      makeUIMessage('user', ''), // phantom empty message from sendMessage({ text: '' })
    ];

    const stream2 = await transport.sendMessages({
      trigger: 'submit-message',
      chatId,
      messageId: undefined,
      messages: messagesForResume,
      abortSignal: undefined,
    });
    const chunks2 = await collectChunks(stream2);

    // No errors
    assert.strictEqual(
      chunksOfType(chunks2, 'error').length,
      0,
      'Resume should not produce errors'
    );

    // Must have text content from the resumed response
    const textDeltas2 = chunksOfType(chunks2, 'text-delta');
    assert.strictEqual(
      textDeltas2.length,
      1,
      'Resume should produce text-delta'
    );
    assert.strictEqual(textDeltas2[0].delta, 'Transfer complete.');

    // Model was called exactly twice total (once per phase)
    assert.strictEqual(callCount, 2);

    // The second model call (resume) should contain the tool response
    const resumeReq = capturedRequests[1];
    const toolResponseMsgs = resumeReq.messages.filter(
      (m: any) => m.role === 'tool'
    );
    assert.ok(
      toolResponseMsgs.length >= 1,
      'Resume model request should contain tool response message'
    );
    const toolResponsePart = toolResponseMsgs[0].content.find(
      (p: any) => p.toolResponse
    );
    assert.ok(toolResponsePart, 'Should have a toolResponse part');
    assert.strictEqual(
      toolResponsePart.toolResponse?.name,
      'confirmAction',
      'Tool name should be derived from type field'
    );
    assert.deepStrictEqual(toolResponsePart.toolResponse?.output, {
      confirmed: true,
    });
  });

  // ── Test: reconnectToStream returns null ─────────────────────────────

  it('should return null for reconnectToStream', async () => {
    const transport = new GenkitChatTransport({
      url: `http://localhost:${port}/testAgent`,
    });

    const result = await transport.reconnectToStream();
    assert.strictEqual(result, null);
  });

  // ── Test: Invalid (non-UUID) chatId is rejected ──────────────────────

  it('should return an error chunk when chatId is not a UUID', async () => {
    const transport = new GenkitChatTransport({
      url: `http://localhost:${port}/testAgent`,
    });

    const stream = await transport.sendMessages({
      trigger: 'submit-message',
      chatId: 'not-a-uuid',
      messageId: undefined,
      messages: [makeUIMessage('user', 'Hi')],
      abortSignal: undefined,
    });

    const chunks = await collectChunks(stream);
    const types = chunkTypes(chunks);

    // Error stream uses the standard envelope: start → error → finish
    assert.strictEqual(types[0], 'start');
    assert.strictEqual(types[types.length - 1], 'finish');

    const errorChunks = chunksOfType(chunks, 'error');
    assert.strictEqual(errorChunks.length, 1, 'Exactly one error chunk');
    assert.ok(
      errorChunks[0].errorText.includes('UUID'),
      `Error text should mention UUID, got: ${errorChunks[0].errorText}`
    );

    // No content chunks should be emitted.
    assert.strictEqual(chunksOfType(chunks, 'text-delta').length, 0);
    assert.strictEqual(chunksOfType(chunks, 'start-step').length, 0);
  });

  // ── Test: Error handling ────────────────────────────────────────────

  it('should return error chunk when no user message is found', async () => {
    const transport = new GenkitChatTransport({
      url: `http://localhost:${port}/testAgent`,
    });

    const stream = await transport.sendMessages({
      trigger: 'submit-message',
      chatId: newChatId(),
      messageId: undefined,
      messages: [makeUIMessage('assistant', 'I am assistant')],
      abortSignal: undefined,
    });

    const chunks = await collectChunks(stream);
    const types = chunkTypes(chunks);

    // Error stream uses the standard envelope: start → error → finish
    assert.strictEqual(types[0], 'start', 'Error stream starts with start');
    assert.strictEqual(
      types[types.length - 1],
      'finish',
      'Error stream ends with finish'
    );
    assert.strictEqual(
      chunks.length,
      3,
      'Error stream emits start + error + finish'
    );

    // Should have exactly one error chunk
    const errorChunks = chunksOfType(chunks, 'error');
    assert.strictEqual(errorChunks.length, 1, 'Exactly one error chunk');
    assert.ok(
      errorChunks[0].errorText.includes('No user message'),
      `Error text should mention "No user message", got: ${errorChunks[0].errorText}`
    );

    // No text, tool, or step chunks should be emitted
    assert.strictEqual(chunksOfType(chunks, 'text-delta').length, 0);
    assert.strictEqual(chunksOfType(chunks, 'tool-input-start').length, 0);
    assert.strictEqual(chunksOfType(chunks, 'start-step').length, 0);
  });

  // ── Test: Custom headers ────────────────────────────────────────────

  it('should support static and dynamic headers', async () => {
    modelHandler = async (_req, sendChunk) => {
      sendChunk({ content: [{ text: 'header-ok' }] });
      return {
        message: { role: 'model', content: [{ text: 'header-ok' }] },
        finishReason: 'stop',
      };
    };

    // Static headers
    const transport1 = new GenkitChatTransport({
      url: `http://localhost:${port}/testAgent`,
      headers: { 'X-Custom': 'static' },
    });

    const stream1 = await transport1.sendMessages({
      trigger: 'submit-message',
      chatId: newChatId(),
      messageId: undefined,
      messages: [makeUIMessage('user', 'hi')],
      abortSignal: undefined,
    });
    const chunks1 = await collectChunks(stream1);

    // Must complete with actual content
    assert.strictEqual(chunksOfType(chunks1, 'error').length, 0);
    const deltas1 = chunksOfType(chunks1, 'text-delta');
    assert.strictEqual(deltas1.length, 1);
    assert.strictEqual(deltas1[0].delta, 'header-ok');

    // The server must have actually received the static header.
    assert.strictEqual(
      lastRequestHeaders['x-custom'],
      'static',
      'Server should receive the static X-Custom header'
    );

    // Dynamic headers (function)
    const transport2 = new GenkitChatTransport({
      url: `http://localhost:${port}/testAgent`,
      headers: () => ({ 'X-Dynamic': 'token-123' }),
    });

    const stream2 = await transport2.sendMessages({
      trigger: 'submit-message',
      chatId: newChatId(),
      messageId: undefined,
      messages: [makeUIMessage('user', 'hi')],
      abortSignal: undefined,
    });
    const chunks2 = await collectChunks(stream2);

    assert.strictEqual(chunksOfType(chunks2, 'error').length, 0);
    const deltas2 = chunksOfType(chunks2, 'text-delta');
    assert.strictEqual(deltas2.length, 1);
    assert.strictEqual(deltas2[0].delta, 'header-ok');

    // The server must have actually received the dynamic header.
    assert.strictEqual(
      lastRequestHeaders['x-dynamic'],
      'token-123',
      'Server should receive the dynamic X-Dynamic header'
    );
  });

  // ── Test: Different chatIds get independent sessions ─────────────────

  it('should maintain independent sessions per chatId', async () => {
    let callCount = 0;
    const capturedRequests: GenerateRequest[] = [];

    modelHandler = async (req, sendChunk) => {
      callCount++;
      capturedRequests.push(JSON.parse(JSON.stringify(req)));
      const userMsgs = req.messages.filter((m) => m.role === 'user');
      const lastText = userMsgs[userMsgs.length - 1]?.content[0]?.text ?? '';

      sendChunk({ content: [{ text: `echo: ${lastText}` }] });
      return {
        message: {
          role: 'model',
          content: [{ text: `echo: ${lastText}` }],
        },
        finishReason: 'stop',
      };
    };

    const transport = new GenkitChatTransport({
      url: `http://localhost:${port}/testAgent`,
    });

    const chatA = newChatId();
    const chatB = newChatId();

    // Chat A — turn 1
    const streamA1 = await transport.sendMessages({
      trigger: 'submit-message',
      chatId: chatA,
      messageId: undefined,
      messages: [makeUIMessage('user', 'Hello A')],
      abortSignal: undefined,
    });
    const chunksA1 = await collectChunks(streamA1);
    assert.strictEqual(chunksOfType(chunksA1, 'error').length, 0);
    assert.strictEqual(
      chunksOfType(chunksA1, 'text-delta')[0].delta,
      'echo: Hello A'
    );

    // Chat B — turn 1
    const streamB1 = await transport.sendMessages({
      trigger: 'submit-message',
      chatId: chatB,
      messageId: undefined,
      messages: [makeUIMessage('user', 'Hello B')],
      abortSignal: undefined,
    });
    const chunksB1 = await collectChunks(streamB1);
    assert.strictEqual(chunksOfType(chunksB1, 'error').length, 0);
    assert.strictEqual(
      chunksOfType(chunksB1, 'text-delta')[0].delta,
      'echo: Hello B'
    );

    // Chat A — turn 2 (should use A's session, not B's)
    const streamA2 = await transport.sendMessages({
      trigger: 'submit-message',
      chatId: chatA,
      messageId: undefined,
      messages: [
        makeUIMessage('user', 'Hello A'),
        makeUIMessage('assistant', 'echo: Hello A'),
        makeUIMessage('user', 'Second A'),
      ],
      abortSignal: undefined,
    });
    const chunksA2 = await collectChunks(streamA2);
    assert.strictEqual(chunksOfType(chunksA2, 'error').length, 0);
    assert.strictEqual(
      chunksOfType(chunksA2, 'text-delta')[0].delta,
      'echo: Second A'
    );

    assert.strictEqual(callCount, 3);

    // Chat A turn 2 model request should see history from A turn 1
    // (not B's history) — at least the original user message + model reply
    const reqA2 = capturedRequests[2];
    const a2UserMsgs = reqA2.messages.filter((m: any) => m.role === 'user');
    assert.ok(
      a2UserMsgs.length >= 2,
      `Chat A turn 2 model should see at least 2 user messages, got ${a2UserMsgs.length}`
    );

    // The first user message in the history should be from chat A's turn 1
    const firstUserText = a2UserMsgs[0].content[0]?.text;
    assert.strictEqual(
      firstUserText,
      'Hello A',
      'First user message in A-turn-2 should be "Hello A" from A-turn-1'
    );
  });

  // ── Test: Abort cancellation ──────────────────────────────────────────

  it('should handle abort signal gracefully', async () => {
    modelHandler = async (_req, sendChunk) => {
      // Simulate a slow model: send one chunk, then wait indefinitely
      sendChunk({ content: [{ text: 'Starting...' }] });
      await new Promise((resolve) => setTimeout(resolve, 5000));
      return {
        message: { role: 'model', content: [{ text: 'Done' }] },
        finishReason: 'stop',
      };
    };

    const transport = new GenkitChatTransport({
      url: `http://localhost:${port}/testAgent`,
    });

    const abortController = new AbortController();

    const stream = await transport.sendMessages({
      trigger: 'submit-message',
      chatId: newChatId(),
      messageId: undefined,
      messages: [makeUIMessage('user', 'slow request')],
      abortSignal: abortController.signal,
    });

    // Start reading, abort after a short delay
    const reader = stream.getReader();
    const chunks: UIMessageChunk[] = [];

    // Read first chunk (should be 'start')
    const first = await reader.read();
    if (!first.done) chunks.push(first.value);

    // Abort the request
    abortController.abort();

    // Drain remaining chunks
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
    }

    const types = chunkTypes(chunks);

    // Stream should have started
    assert.strictEqual(types[0], 'start');

    // Stream should end with 'finish' (always emitted in finally block)
    assert.strictEqual(types[types.length - 1], 'finish');

    // Should not have any error chunks (abort is not an error)
    assert.strictEqual(
      chunksOfType(chunks, 'error').length,
      0,
      'Abort should not produce error chunks'
    );
  });

  // ── Test: Media/file parts in user messages ───────────────────────────

  it('should send file parts as media in Genkit messages', async () => {
    const capturedRequests: GenerateRequest[] = [];

    modelHandler = async (req, sendChunk) => {
      capturedRequests.push(JSON.parse(JSON.stringify(req)));
      sendChunk({ content: [{ text: 'I see the image!' }] });
      return {
        message: { role: 'model', content: [{ text: 'I see the image!' }] },
        finishReason: 'stop',
      };
    };

    const transport = new GenkitChatTransport({
      url: `http://localhost:${port}/testAgent`,
    });

    // User message with both text and a file part
    const stream = await transport.sendMessages({
      trigger: 'submit-message',
      chatId: newChatId(),
      messageId: undefined,
      messages: [
        makeUIMessage('user', 'Describe this image', [
          { type: 'text', text: 'Describe this image' },
          {
            type: 'file',
            url: 'data:image/png;base64,iVBORw0KGgo=',
            mediaType: 'image/png',
          },
        ]),
      ],
      abortSignal: undefined,
    });

    const chunks = await collectChunks(stream);

    // No errors
    assert.strictEqual(chunksOfType(chunks, 'error').length, 0);

    // Text response received
    const textDeltas = chunksOfType(chunks, 'text-delta');
    assert.strictEqual(textDeltas.length, 1);
    assert.strictEqual(textDeltas[0].delta, 'I see the image!');

    // Verify the model received both text and media parts
    assert.strictEqual(capturedRequests.length, 1);
    const userMsg = capturedRequests[0].messages.find((m) => m.role === 'user');
    assert.ok(userMsg, 'Model should receive a user message');

    // Should have a text part
    const textParts = userMsg!.content.filter((p: any) => p.text);
    assert.ok(textParts.length >= 1, 'Should have at least one text part');

    // Should have a media part
    const mediaParts = userMsg!.content.filter((p: any) => p.media);
    assert.ok(mediaParts.length >= 1, 'Should have at least one media part');
    assert.strictEqual((mediaParts[0] as any).media.contentType, 'image/png');
  });

  // ── Test: Regenerate re-runs the last user turn ──────────────────────
  //
  // With server-managed state, regeneration is treated as a fresh turn from
  // the current session state (there is no client-side snapshot pointer to
  // rewind to). The transport sends the last *user* message again.

  it('should regenerate the last turn as a fresh run', async () => {
    let callCount = 0;
    const capturedRequests: GenerateRequest[] = [];

    modelHandler = async (req, sendChunk) => {
      callCount++;
      capturedRequests.push(JSON.parse(JSON.stringify(req)));
      const userMsgs = req.messages.filter((m) => m.role === 'user');
      const lastText = userMsgs[userMsgs.length - 1]?.content[0]?.text ?? '';

      sendChunk({ content: [{ text: `answer-${callCount}` }] });
      return {
        message: {
          role: 'model',
          content: [{ text: `answer to "${lastText}" (#${callCount})` }],
        },
        finishReason: 'stop',
      };
    };

    const transport = new GenkitChatTransport({
      url: `http://localhost:${port}/testAgent`,
    });

    const chatId = newChatId();

    // Turn 1
    await collectChunks(
      await transport.sendMessages({
        trigger: 'submit-message',
        chatId,
        messageId: undefined,
        messages: [makeUIMessage('user', 'first')],
        abortSignal: undefined,
      })
    );

    // Turn 2 — a real assistant answer we will later regenerate.
    await collectChunks(
      await transport.sendMessages({
        trigger: 'submit-message',
        chatId,
        messageId: undefined,
        messages: [
          makeUIMessage('user', 'first'),
          makeUIMessage('assistant', 'answer to "first" (#1)'),
          makeUIMessage('user', 'second'),
        ],
        abortSignal: undefined,
      })
    );
    assert.strictEqual(callCount, 2);

    // Regenerate turn 2: re-runs the last user turn against the current
    // session state. It must NOT be treated as an interrupt resume.
    const regenStream = await transport.sendMessages({
      trigger: 'regenerate-message',
      chatId,
      messageId: undefined,
      messages: [
        makeUIMessage('user', 'first'),
        makeUIMessage('assistant', 'answer to "first" (#1)'),
        makeUIMessage('user', 'second'),
      ],
      abortSignal: undefined,
    });
    const regenChunks = await collectChunks(regenStream);

    assert.strictEqual(chunksOfType(regenChunks, 'error').length, 0);
    const regenDeltas = chunksOfType(regenChunks, 'text-delta');
    assert.strictEqual(regenDeltas.length, 1);
    assert.strictEqual(regenDeltas[0].delta, 'answer-3');
    assert.strictEqual(callCount, 3);

    // The regeneration request should see the last user message ("second").
    const regenReq = capturedRequests[2];
    const regenUserMsgs = regenReq.messages.filter((m) => m.role === 'user');
    assert.strictEqual(
      regenUserMsgs[regenUserMsgs.length - 1]?.content[0]?.text,
      'second'
    );
  });

  // ── Test: Regenerate on the very first turn ──────────────────────────

  it('should regenerate the first turn from scratch', async () => {
    let callCount = 0;
    const capturedRequests: GenerateRequest[] = [];

    modelHandler = async (req, sendChunk) => {
      callCount++;
      capturedRequests.push(JSON.parse(JSON.stringify(req)));
      const userMsgs = req.messages.filter((m) => m.role === 'user');
      const lastText = userMsgs[userMsgs.length - 1]?.content[0]?.text ?? '';

      sendChunk({ content: [{ text: `answer-${callCount}` }] });
      return {
        message: {
          role: 'model',
          content: [{ text: `answer to "${lastText}" (#${callCount})` }],
        },
        finishReason: 'stop',
      };
    };

    const transport = new GenkitChatTransport({
      url: `http://localhost:${port}/testAgent`,
    });

    // Regenerate as the *first* interaction for this chat: the session has
    // no prior state, so the transport falls back to a fresh run from the
    // last user message rather than crashing.
    const regenStream = await transport.sendMessages({
      trigger: 'regenerate-message',
      chatId: newChatId(),
      messageId: undefined,
      messages: [makeUIMessage('user', 'only message')],
      abortSignal: undefined,
    });
    const regenChunks = await collectChunks(regenStream);

    // No errors, and a normal streamed answer.
    assert.strictEqual(chunksOfType(regenChunks, 'error').length, 0);
    const deltas = chunksOfType(regenChunks, 'text-delta');
    assert.strictEqual(deltas.length, 1);
    assert.strictEqual(deltas[0].delta, 'answer-1');
    assert.strictEqual(callCount, 1);

    // The model should have seen the last user message (fresh run).
    const req = capturedRequests[0];
    const userMsgs = req.messages.filter((m) => m.role === 'user');
    assert.strictEqual(
      userMsgs[userMsgs.length - 1]?.content[0]?.text,
      'only message'
    );

    // Stream envelope complete.
    assert.strictEqual(chunkTypes(regenChunks)[0], 'start');
    assert.strictEqual(chunkTypes(regenChunks).at(-1), 'finish');
  });

  // ── Test: Interrupt → restart re-runs the tool server-side ───────────
  //
  // Unlike a plain interrupt response (where the user *supplies* the tool
  // output), a restart instructs the agent to re-execute the interrupted
  // tool. The transport detects the `restartInterrupt()` marker and emits a
  // `resume.restart` entry carrying the original input + metadata.

  it('should restart an interrupted tool via restartInterrupt()', async () => {
    let callCount = 0;
    const capturedRequests: GenerateRequest[] = [];

    modelHandler = async (req, sendChunk) => {
      callCount++;
      capturedRequests.push(JSON.parse(JSON.stringify(req)));

      if (callCount === 1) {
        // Model asks to run the restartable tool, which interrupts.
        return {
          message: {
            role: 'model',
            content: [
              {
                toolRequest: {
                  name: 'riskyAction',
                  input: { target: 'prod-db' },
                  ref: 'risky-ref-1',
                },
              },
            ],
          },
          finishReason: 'stop',
        };
      }

      // After the tool re-runs (restart) the model produces final text.
      sendChunk({ content: [{ text: 'Action completed.' }] });
      return {
        message: { role: 'model', content: [{ text: 'Action completed.' }] },
        finishReason: 'stop',
      };
    };

    const transport = new GenkitChatTransport({
      url: `http://localhost:${port}/testAgent`,
    });

    const chatId = newChatId();

    // ── Phase 1: triggers the interrupt ──────────────────────────────
    const chunks1 = await collectChunks(
      await transport.sendMessages({
        trigger: 'submit-message',
        chatId,
        messageId: undefined,
        messages: [makeUIMessage('user', 'Run the risky action')],
        abortSignal: undefined,
      })
    );
    assert.strictEqual(chunksOfType(chunks1, 'error').length, 0);
    assert.strictEqual(chunksOfType(chunks1, 'tool-input-start').length, 1);
    assert.strictEqual(
      chunksOfType(chunks1, 'tool-output-available').length,
      0,
      'Interrupt should not produce a tool output'
    );

    // ── Phase 2: resolve via restart ─────────────────────────────────
    // The user opts to restart the tool with some metadata. The output
    // carries the restart marker via restartInterrupt().
    const messagesForRestart: UIMessage[] = [
      makeUIMessage('user', 'Run the risky action'),
      {
        id: 'assistant-restart-1',
        role: 'assistant',
        parts: [
          {
            type: 'tool-riskyAction',
            toolCallId: 'risky-ref-1',
            state: 'output-available',
            input: { target: 'prod-db' },
            output: restartInterrupt({ confirmedBy: 'user' }),
          } as UIMessage['parts'][number],
        ],
      },
    ];

    const chunks2 = await collectChunks(
      await transport.sendMessages({
        trigger: 'submit-message',
        chatId,
        messageId: undefined,
        messages: messagesForRestart,
        abortSignal: undefined,
      })
    );

    assert.strictEqual(chunksOfType(chunks2, 'error').length, 0);

    // The restarted tool re-runs server-side and produces a real output.
    const toolOutputs = chunksOfType(chunks2, 'tool-output-available');
    assert.strictEqual(
      toolOutputs.length,
      1,
      'Restarted tool should produce a real output'
    );
    assert.strictEqual((toolOutputs[0].output as any).done, true);
    // The metadata supplied to restartInterrupt() is threaded to `resumed`.
    assert.deepStrictEqual((toolOutputs[0].output as any).resumedWith, {
      confirmedBy: 'user',
    });

    // The model then produced its final text.
    const textDeltas2 = chunksOfType(chunks2, 'text-delta');
    assert.strictEqual(textDeltas2.length, 1);
    assert.strictEqual(textDeltas2[0].delta, 'Action completed.');

    assert.strictEqual(callCount, 2);
  });
});
