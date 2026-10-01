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

import { genkit, modelRef, z, type GenerateRequest } from 'genkit';
import assert from 'node:assert';
import { describe, it } from 'node:test';
import {
  ContextCompressionOptionsSchema,
  DeduplicateToolResponsesOptionsSchema,
  contextCompression,
} from '../src/context-compression.js';

describe('contextCompression middleware', () => {
  it('skips compression when token count is below maxInputTokens', async () => {
    const ai = genkit({});
    let capturedRequest: GenerateRequest | undefined;

    const pm = ai.defineModel({ name: 'echoModel' }, async (req) => {
      capturedRequest = req;
      return {
        message: { role: 'model', content: [{ text: 'response' }] },
        usage: { inputTokens: 50 },
      };
    });

    const response = await ai.generate({
      model: pm,
      prompt: 'short prompt',
      use: [
        contextCompression({
          maxInputTokens: 1000,
        }),
      ],
    });

    assert.strictEqual(response.text, 'response');
    assert.strictEqual(capturedRequest?.messages.length, 1);
    assert.strictEqual(
      (response as any).custom?.contextCompression?.triggered,
      undefined
    );
  });

  it('triggers compression on initial call when estimated tokens exceed maxInputTokens', async () => {
    const ai = genkit({});
    let capturedRequest: GenerateRequest | undefined;

    const pm = ai.defineModel({ name: 'echoModel' }, async (req) => {
      capturedRequest = req;
      return {
        message: { role: 'model', content: [{ text: 'response' }] },
        usage: { inputTokens: 50 },
      };
    });

    const response = (await ai.generate({
      model: pm,
      messages: [
        {
          role: 'tool',
          content: [
            {
              toolResponse: {
                name: 'search',
                ref: '1',
                output: 'X'.repeat(500),
              },
            },
          ],
        },
        { role: 'user', content: [{ text: 'summarize' }] },
      ],
      use: [
        contextCompression({
          maxInputTokens: 50,
          toolResponses: { maxChars: 100, preserveRecent: 0 },
        }),
      ],
    })) as any;

    assert.strictEqual(response.text, 'response');
    assert.strictEqual(response.custom?.contextCompression?.triggered, true);
    assert.strictEqual(
      response.custom?.contextCompression?.toolResponsesTruncated,
      1
    );
    const toolMsg = capturedRequest?.messages.find((m) => m.role === 'tool');
    assert.match(
      String(toolMsg?.content[0].toolResponse?.output),
      /\[Truncated \d+ characters\]/
    );
  });

  it('counts reasoning and data parts in estimated tokens on initial turn', async () => {
    const ai = genkit({});
    let capturedRequest: GenerateRequest | undefined;

    const pm = ai.defineModel({ name: 'echoModel' }, async (req) => {
      capturedRequest = req;
      return {
        message: { role: 'model', content: [{ text: 'response' }] },
        usage: { inputTokens: 50 },
      };
    });

    const response = (await ai.generate({
      model: pm,
      messages: [
        {
          role: 'model',
          content: [
            { reasoning: 'R'.repeat(400) } as any,
            { data: { payload: 'D'.repeat(400) } } as any,
          ],
        },
        {
          role: 'tool',
          content: [
            {
              toolResponse: {
                name: 'search',
                ref: '1',
                output: 'X'.repeat(500),
              },
            },
          ],
        },
        { role: 'user', content: [{ text: 'summarize' }] },
      ],
      use: [
        contextCompression({
          maxInputTokens: 50,
          toolResponses: { maxChars: 100, preserveRecent: 0 },
        }),
      ],
    })) as any;

    assert.strictEqual(response.text, 'response');
    assert.strictEqual(response.custom?.contextCompression?.triggered, true);
    assert.strictEqual(
      response.custom?.contextCompression?.toolResponsesTruncated,
      1
    );
  });

  it('truncates tool responses exceeding maxChars while preserving recent responses', async () => {
    const ai = genkit({});
    let turn = 0;
    const capturedRequests: GenerateRequest[] = [];

    const heavyTool = ai.defineTool(
      {
        name: 'heavyTool',
        description: 'returns large data',
        inputSchema: z.object({ query: z.string() }),
        outputSchema: z.string(),
      },
      async (input) => `Result for ${input.query}: ${'X'.repeat(300)}`
    );

    const pm = ai.defineModel({ name: 'toolLoopModel' }, async (req) => {
      capturedRequests.push(req);
      turn++;
      if (turn === 1) {
        return {
          message: {
            role: 'model',
            content: [
              { toolRequest: { name: 'heavyTool', input: { query: 'call1' } } },
            ],
          },
          usage: { inputTokens: 200 },
        };
      }
      if (turn === 2) {
        return {
          message: {
            role: 'model',
            content: [
              { toolRequest: { name: 'heavyTool', input: { query: 'call2' } } },
            ],
          },
          usage: { inputTokens: 500 },
        };
      }
      return {
        message: { role: 'model', content: [{ text: 'finished' }] },
        usage: { inputTokens: 100 },
      };
    });

    const result = await ai.generate({
      model: pm,
      prompt: 'Run tool calls',
      tools: [heavyTool],
      use: [
        contextCompression({
          maxInputTokens: 150,
          toolResponses: { maxChars: 50, preserveRecent: 1 },
        }),
      ],
    });

    assert.strictEqual(result.text, 'finished');
    assert.strictEqual(capturedRequests.length, 3);

    // On turn 3, call 1 should be truncated, and call 2 should be preserved
    const turn3Messages = capturedRequests[2].messages;
    const toolMessages = turn3Messages.filter((m) => m.role === 'tool');
    assert.strictEqual(toolMessages.length, 2);

    const firstToolOutput = toolMessages[0].content[0].toolResponse?.output;
    const secondToolOutput = toolMessages[1].content[0].toolResponse?.output;

    assert.match(String(firstToolOutput), /\[Truncated \d+ characters\]/);
    assert.strictEqual(String(secondToolOutput).includes('[Truncated'), false);
  });

  it('applies safety cap to oversized tool responses', async () => {
    const ai = genkit({});
    let turn = 0;
    const capturedRequests: GenerateRequest[] = [];

    const hugeTool = ai.defineTool(
      {
        name: 'hugeTool',
        description: 'returns large data',
        inputSchema: z.object({}),
        outputSchema: z.string(),
      },
      async () => 'x'.repeat(1000)
    );

    const pm = ai.defineModel({ name: 'hugeToolModel' }, async (req) => {
      capturedRequests.push(req);
      turn++;
      if (turn === 1) {
        return {
          message: {
            role: 'model',
            content: [{ toolRequest: { name: 'hugeTool', input: {} } }],
          },
          usage: { inputTokens: 200 },
        };
      }
      return {
        message: { role: 'model', content: [{ text: 'done' }] },
        usage: { inputTokens: 100 },
      };
    });

    const result = await ai.generate({
      model: pm,
      prompt: 'Call huge tool',
      tools: [hugeTool],
      use: [
        contextCompression({
          maxInputTokens: 100,
          maxToolResponseChars: 100,
        }),
      ],
    });

    assert.strictEqual(result.text, 'done');
    const turn2Messages = capturedRequests[1].messages;
    const toolMsg = turn2Messages.find((m) => m.role === 'tool');
    const output = String(toolMsg?.content[0].toolResponse?.output);
    assert.ok(output.includes('[TRUNCATED: Response was 1000 chars'));
  });

  it('caps message count and inserts truncation notice', async () => {
    const ai = genkit({});
    let capturedRequest: GenerateRequest | undefined;

    const pm = ai.defineModel({ name: 'capModel' }, async (req) => {
      capturedRequest = req;
      return {
        message: { role: 'model', content: [{ text: 'done' }] },
        usage: { inputTokens: 50 },
      };
    });

    const response = await ai.generate({
      model: pm,
      messages: [
        { role: 'user', content: [{ text: 'msg 1' }] },
        { role: 'model', content: [{ text: 'msg 2' }] },
        { role: 'user', content: [{ text: 'msg 3' }] },
        { role: 'model', content: [{ text: 'msg 4' }] },
        { role: 'user', content: [{ text: 'msg 5' }] },
      ],
      use: [
        contextCompression({
          maxMessages: 4,
          insertTruncationNotice: true,
        }),
      ],
    });

    assert.strictEqual(response.text, 'done');
    const msgs = capturedRequest!.messages;
    // Notice message (system) + 3 kept messages (starting with user) = 4 total messages
    assert.strictEqual(msgs.length, 4);
    assert.match(msgs[0].content[0].text!, /\[NOTE\] Some earlier messages/);
    assert.strictEqual(msgs[1].content[0].text, 'msg 3');
    assert.strictEqual(msgs[2].content[0].text, 'msg 4');
    assert.strictEqual(msgs[3].content[0].text, 'msg 5');
  });

  it('respects custom truncation notice text', async () => {
    const ai = genkit({});
    let capturedRequest: GenerateRequest | undefined;

    const pm = ai.defineModel({ name: 'customNoticeModel' }, async (req) => {
      capturedRequest = req;
      return {
        message: { role: 'model', content: [{ text: 'ok' }] },
        usage: { inputTokens: 50 },
      };
    });

    await ai.generate({
      model: pm,
      messages: [
        { role: 'user', content: [{ text: '1' }] },
        { role: 'model', content: [{ text: '2' }] },
        { role: 'user', content: [{ text: '3' }] },
      ],
      use: [
        contextCompression({
          maxMessages: 2,
          truncationNotice: 'Custom drop notice',
        }),
      ],
    });

    const msgs = capturedRequest!.messages;
    assert.strictEqual(msgs[0].content[0].text, 'Custom drop notice');
    assert.strictEqual(msgs[1].content[0].text, '3');
  });

  it('preserves system messages during message truncation', async () => {
    const ai = genkit({});
    let capturedRequest: GenerateRequest | undefined;

    const pm = ai.defineModel({ name: 'systemModel' }, async (req) => {
      capturedRequest = req;
      return {
        message: { role: 'model', content: [{ text: 'ok' }] },
        usage: { inputTokens: 50 },
      };
    });

    await ai.generate({
      model: pm,
      messages: [
        { role: 'system', content: [{ text: 'System Instructions' }] },
        { role: 'user', content: [{ text: 'msg 1' }] },
        { role: 'model', content: [{ text: 'msg 2' }] },
        { role: 'user', content: [{ text: 'msg 3' }] },
      ],
      use: [
        contextCompression({
          maxMessages: 2,
          preserveSystem: true,
          insertTruncationNotice: false,
        }),
      ],
    });

    const msgs = capturedRequest!.messages;
    assert.strictEqual(msgs.length, 2);
    assert.strictEqual(msgs[0].role, 'system');
    assert.strictEqual(msgs[0].content[0].text, 'System Instructions');
    assert.strictEqual(msgs[1].role, 'user');
    assert.strictEqual(msgs[1].content[0].text, 'msg 3');
  });

  it('attaches compression metadata to custom property on response when turn compresses', async () => {
    const ai = genkit({});
    const pm = ai.defineModel({ name: 'metaModel' }, async () => {
      return {
        message: { role: 'model', content: [{ text: 'done' }] },
        usage: { inputTokens: 50 },
      };
    });

    const response = (await ai.generate({
      model: pm,
      messages: [
        { role: 'user', content: [{ text: 'hello' }] },
        { role: 'model', content: [{ text: 'response 1' }] },
        { role: 'user', content: [{ text: 'world' }] },
      ],
      use: [
        contextCompression({
          maxMessages: 2,
          insertTruncationNotice: false,
        }),
      ],
    })) as any;

    assert.strictEqual(response.text, 'done');
    assert.strictEqual(response.text, 'done');
    assert.strictEqual(response.custom?.contextCompression?.triggered, true);
    assert.strictEqual(response.custom.contextCompression.messagesOriginal, 3);
    assert.strictEqual(response.custom.contextCompression.messagesAfter, 1);
  });

  it('does not leak compression metadata to outer turns when only child turn compresses', async () => {
    const ai = genkit({});
    let turn = 0;

    const dummyTool = ai.defineTool(
      {
        name: 'step',
        description: 'step',
        inputSchema: z.object({}),
        outputSchema: z.string(),
      },
      async () => 'tool result'
    );

    const pm = ai.defineModel({ name: 'isolationModel' }, async () => {
      turn++;
      if (turn === 1) {
        return {
          message: {
            role: 'model',
            content: [{ toolRequest: { name: 'step', input: {} } }],
          },
          usage: { inputTokens: 500 },
        };
      }
      return {
        message: { role: 'model', content: [{ text: 'done' }] },
        usage: { inputTokens: 100 },
      };
    });

    const response = (await ai.generate({
      model: pm,
      prompt: 'test metadata',
      tools: [dummyTool],
      use: [
        contextCompression({
          maxInputTokens: 200,
          toolResponses: { maxChars: 5, preserveRecent: 0 },
        }),
      ],
    })) as any;

    assert.strictEqual(response.text, 'done');
    assert.strictEqual(response.custom?.contextCompression?.triggered, true);
    assert.strictEqual(
      response.custom?.contextCompression?.toolResponsesTruncated,
      1
    );
  });

  it('isolates state across concurrent generate requests using the same middleware instance', async () => {
    const ai = genkit({});

    const pm = ai.defineModel({ name: 'concurrencyModel' }, async () => {
      await new Promise((resolve) => setTimeout(resolve, 15));
      return {
        message: { role: 'model', content: [{ text: 'done' }] },
        usage: { inputTokens: 50 },
      };
    });

    const sharedCC = contextCompression({
      maxInputTokens: 100,
      maxMessages: 2,
      insertTruncationNotice: false,
    });

    const [resp1, resp2] = (await Promise.all([
      ai.generate({
        model: pm,
        messages: [
          { role: 'user', content: [{ text: 'Req 1 message 1' }] },
          { role: 'model', content: [{ text: 'Req 1 message 2' }] },
          { role: 'user', content: [{ text: 'Req 1 message 3' }] },
        ],
        use: [sharedCC],
      }),
      ai.generate({
        model: pm,
        messages: [
          { role: 'user', content: [{ text: 'Req 2 single message' }] },
        ],
        use: [sharedCC],
      }),
    ])) as any[];

    assert.strictEqual(resp1.custom?.contextCompression?.triggered, true);
    assert.strictEqual(resp1.custom.contextCompression.messagesOriginal, 3);
    assert.strictEqual(resp1.custom.contextCompression.messagesAfter, 1);
    assert.strictEqual(resp2.custom?.contextCompression, undefined);
  });

  it('prevents orphaned tool messages during message truncation', async () => {
    const ai = genkit({});
    let capturedRequest: GenerateRequest | undefined;

    const pm = ai.defineModel({ name: 'orphanedModel' }, async (req) => {
      capturedRequest = req;
      return {
        message: { role: 'model', content: [{ text: 'ok' }] },
        usage: { inputTokens: 50 },
      };
    });

    await ai.generate({
      model: pm,
      messages: [
        { role: 'user', content: [{ text: 'hello' }] },
        {
          role: 'model',
          content: [{ toolRequest: { name: 'tool', input: {} } }],
        },
        {
          role: 'tool',
          content: [{ toolResponse: { name: 'tool', output: 'result' } }],
        },
        { role: 'model', content: [{ text: 'result is ok' }] },
        { role: 'user', content: [{ text: 'next' }] },
      ],
      use: [
        contextCompression({
          maxMessages: 3,
          insertTruncationNotice: false,
        }),
      ],
    });

    const msgs = capturedRequest!.messages;
    assert.strictEqual(msgs.length, 1);
    assert.strictEqual(msgs[0].role, 'user');
    assert.strictEqual(msgs[0].content[0].text, 'next');
  });

  it('discards leading tool messages during message truncation to prevent dangling tool responses', async () => {
    const ai = genkit({});
    let capturedRequest: GenerateRequest | undefined;

    const pm = ai.defineModel({ name: 'danglingModel' }, async (req) => {
      capturedRequest = req;
      return {
        message: { role: 'model', content: [{ text: 'done' }] },
        usage: { inputTokens: 50 },
      };
    });

    await ai.generate({
      model: pm,
      messages: [
        { role: 'user', content: [{ text: 'msg 1' }] },
        {
          role: 'model',
          content: [{ toolRequest: { name: 'myTool', input: {} } }],
        },
        {
          role: 'tool',
          content: [{ toolResponse: { name: 'myTool', output: 'result' } }],
        },
        { role: 'user', content: [{ text: 'msg 2' }] },
      ],
      use: [
        contextCompression({
          maxMessages: 2,
          insertTruncationNotice: false,
        }),
      ],
    });

    const msgs = capturedRequest!.messages;
    assert.strictEqual(msgs.length, 1);
    assert.strictEqual(msgs[0].content[0].text, 'msg 2');
  });

  it('handles keepCount === 0 without retaining all messages (slice(-0) guard)', async () => {
    const ai = genkit({});
    let capturedRequest: GenerateRequest | undefined;

    const pm = ai.defineModel({ name: 'zeroKeepModel' }, async (req) => {
      capturedRequest = req;
      return {
        message: { role: 'model', content: [{ text: 'done' }] },
        usage: { inputTokens: 50 },
      };
    });

    await ai.generate({
      model: pm,
      messages: [
        { role: 'user', content: [{ text: 'msg 1' }] },
        { role: 'model', content: [{ text: 'msg 2' }] },
      ],
      use: [
        contextCompression({
          maxMessages: 1,
          insertTruncationNotice: true,
        }),
      ],
    });

    const msgs = capturedRequest!.messages;
    assert.strictEqual(msgs.length, 1);
    assert.strictEqual(msgs[0].role, 'system');
    assert.match(msgs[0].content[0].text!, /\[NOTE\] Some earlier messages/);
  });

  it('does not re-truncate already truncated tool responses across turns (idempotency)', async () => {
    const ai = genkit({});
    let capturedRequest: GenerateRequest | undefined;

    const pm = ai.defineModel({ name: 'idempotentModel' }, async (req) => {
      capturedRequest = req;
      return {
        message: { role: 'model', content: [{ text: 'ok' }] },
        usage: { inputTokens: 500 },
      };
    });

    const alreadyTruncatedOutput = '12345\n\n[Truncated 95 characters]';

    const response = (await ai.generate({
      model: pm,
      messages: [
        { role: 'user', content: [{ text: 'run tool' }] },
        {
          role: 'model',
          content: [{ toolRequest: { name: 'myTool', input: {} } }],
        },
        {
          role: 'tool',
          content: [
            {
              metadata: { contextCompression: { truncated: true } },
              toolResponse: {
                name: 'myTool',
                output: alreadyTruncatedOutput,
              },
            },
          ],
        },
        { role: 'user', content: [{ text: 'next question' }] },
      ],
      use: [
        contextCompression({
          maxInputTokens: 100,
          toolResponses: { maxChars: 5, preserveRecent: 0 },
        }),
      ],
    })) as any;

    const toolMsg = capturedRequest!.messages.find((m) => m.role === 'tool');
    assert.strictEqual(
      toolMsg?.content[0].toolResponse?.output,
      alreadyTruncatedOutput
    );
    assert.strictEqual(response.custom?.contextCompression, undefined);
  });

  it('does not under-count message slots when system message merges notice', async () => {
    const ai = genkit({});
    let capturedRequest: GenerateRequest | undefined;

    const pm = ai.defineModel({ name: 'slotModel' }, async (req) => {
      capturedRequest = req;
      return {
        message: { role: 'model', content: [{ text: 'done' }] },
        usage: { inputTokens: 50 },
      };
    });

    await ai.generate({
      model: pm,
      messages: [
        { role: 'system', content: [{ text: 'System prompt' }] },
        { role: 'user', content: [{ text: 'user 1' }] },
        { role: 'model', content: [{ text: 'model 1' }] },
        { role: 'user', content: [{ text: 'user 2' }] },
        { role: 'model', content: [{ text: 'model 2' }] },
        { role: 'user', content: [{ text: 'user 3' }] },
      ],
      use: [
        contextCompression({
          maxMessages: 4,
          insertTruncationNotice: true,
        }),
      ],
    });

    const msgs = capturedRequest!.messages;
    // 1 system message (with merged notice) + 3 non-system messages = 4 total messages
    assert.strictEqual(msgs.length, 4);
    assert.strictEqual(msgs[0].role, 'system');
    assert.match(msgs[0].content[1].text!, /\[NOTE\] Some earlier messages/);
    assert.strictEqual(
      (msgs[0].metadata?.contextCompression as any)?.notice,
      true
    );
    assert.strictEqual(msgs[1].role, 'user');
    assert.strictEqual(msgs[1].content[0].text, 'user 2');
    assert.strictEqual(msgs[2].role, 'model');
    assert.strictEqual(msgs[2].content[0].text, 'model 2');
    assert.strictEqual(msgs[3].role, 'user');
    assert.strictEqual(msgs[3].content[0].text, 'user 3');
  });

  it('does not duplicate truncation notice on system message across turns', async () => {
    const ai = genkit({});
    let capturedRequest: GenerateRequest | undefined;

    const pm = ai.defineModel({ name: 'dedupNoticeModel' }, async (req) => {
      capturedRequest = req;
      return {
        message: { role: 'model', content: [{ text: 'done' }] },
        usage: { inputTokens: 50 },
      };
    });

    const noticeText = '[NOTE] Custom drop notice';
    await ai.generate({
      model: pm,
      messages: [
        {
          role: 'system',
          metadata: { contextCompression: { notice: true } },
          content: [{ text: 'System prompt' }, { text: `\n\n${noticeText}` }],
        },
        { role: 'user', content: [{ text: 'user 1' }] },
        { role: 'model', content: [{ text: 'model 1' }] },
        { role: 'user', content: [{ text: 'user 2' }] },
      ],
      use: [
        contextCompression({
          maxMessages: 2,
          insertTruncationNotice: true,
          truncationNotice: noticeText,
        }),
      ],
    });

    const msgs = capturedRequest!.messages;
    assert.strictEqual(msgs[0].role, 'system');
    assert.strictEqual(msgs[0].content.length, 2);
  });

  it('appends truncation notice to the first system message when multiple exist', async () => {
    const ai = genkit({});
    let capturedRequest: GenerateRequest | undefined;

    const pm = ai.defineModel({ name: 'multiSysModel' }, async (req) => {
      capturedRequest = req;
      return {
        message: { role: 'model', content: [{ text: 'done' }] },
        usage: { inputTokens: 50 },
      };
    });

    await ai.generate({
      model: pm,
      messages: [
        { role: 'system', content: [{ text: 'System 1' }] },
        { role: 'system', content: [{ text: 'System 2' }] },
        { role: 'user', content: [{ text: 'user 1' }] },
        { role: 'model', content: [{ text: 'model 1' }] },
        { role: 'user', content: [{ text: 'user 2' }] },
      ],
      use: [
        contextCompression({
          maxMessages: 3,
          insertTruncationNotice: true,
        }),
      ],
    });

    const msgs = capturedRequest!.messages;
    assert.strictEqual(msgs.length, 3);
    assert.strictEqual(msgs[0].role, 'system');
    assert.strictEqual(msgs[0].content[0].text, 'System 1');
    assert.match(msgs[0].content[1].text!, /\[NOTE\] Some earlier messages/);
    assert.strictEqual(
      (msgs[0].metadata?.contextCompression as any)?.notice,
      true
    );
    assert.strictEqual(msgs[1].role, 'system');
    assert.strictEqual(msgs[1].content[0].text, 'System 2');
    assert.strictEqual(msgs[2].role, 'user');
    assert.strictEqual(msgs[2].content[0].text, 'user 2');
  });

  it('stamps part metadata and respects metadata flags across toolResponse parts', async () => {
    const ai = genkit({});
    let capturedRequest: GenerateRequest | undefined;

    const pm = ai.defineModel({ name: 'metaFlagModel' }, async (req) => {
      capturedRequest = req;
      return {
        message: { role: 'model', content: [{ text: 'done' }] },
        usage: { inputTokens: 50 },
      };
    });

    // First generate: verify that truncation stamps part metadata
    await ai.generate({
      model: pm,
      messages: [
        { role: 'user', content: [{ text: 'query' }] },
        {
          role: 'model',
          content: [{ toolRequest: { name: 'toolA', input: {} } }],
        },
        {
          role: 'tool',
          content: [
            {
              toolResponse: {
                name: 'toolA',
                output: 'A'.repeat(500),
              },
            },
          ],
        },
        { role: 'user', content: [{ text: 'next' }] },
      ],
      use: [
        contextCompression({
          maxInputTokens: 10,
          toolResponses: { maxChars: 50, preserveRecent: 0 },
        }),
      ],
    });

    const toolMsg = capturedRequest!.messages.find((m) => m.role === 'tool');
    assert.strictEqual(
      (toolMsg!.content[0].metadata?.contextCompression as any)?.truncated,
      true
    );
    assert.strictEqual(
      (toolMsg!.metadata?.contextCompression as any)?.truncated,
      undefined
    );

    // Second generate: pass a part that only has the metadata flag and no string notice
    let secondCaptured: GenerateRequest | undefined;
    const pm2 = ai.defineModel({ name: 'metaCheckModel' }, async (req) => {
      secondCaptured = req;
      return {
        message: { role: 'model', content: [{ text: 'ok' }] },
        usage: { inputTokens: 500 },
      };
    });

    const cleanOutput = 'Custom already-truncated text without marker';
    await ai.generate({
      model: pm2,
      messages: [
        { role: 'user', content: [{ text: 'query' }] },
        {
          role: 'model',
          content: [{ toolRequest: { name: 'toolA', input: {} } }],
        },
        {
          role: 'tool',
          content: [
            {
              metadata: { contextCompression: { truncated: true } },
              toolResponse: {
                name: 'toolA',
                output: cleanOutput,
              },
            },
          ],
        },
        { role: 'user', content: [{ text: 'next' }] },
      ],
      use: [
        contextCompression({
          maxInputTokens: 10,
          toolResponses: { maxChars: 10, preserveRecent: 0 },
        }),
      ],
    });

    const secondToolMsg = secondCaptured!.messages.find(
      (m) => m.role === 'tool'
    );
    assert.strictEqual(
      secondToolMsg!.content[0].toolResponse?.output,
      cleanOutput
    );
  });

  it('stamps capped in part metadata for safety cap and allows subsequent truncation', async () => {
    const ai = genkit({});
    let capturedRequest: GenerateRequest | undefined;

    const pm = ai.defineModel({ name: 'capMetaModel' }, async (req) => {
      capturedRequest = req;
      return {
        message: { role: 'model', content: [{ text: 'ok' }] },
        usage: { inputTokens: 50 },
      };
    });

    // 1. Tool response is preserved from toolMaxChars, but exceeds maxToolResponseChars (safety cap)
    await ai.generate({
      model: pm,
      messages: [
        { role: 'user', content: [{ text: 'q' }] },
        {
          role: 'model',
          content: [{ toolRequest: { name: 'toolA', input: {} } }],
        },
        {
          role: 'tool',
          content: [
            {
              toolResponse: {
                name: 'toolA',
                output: 'X'.repeat(500),
              },
            },
          ],
        },
        { role: 'user', content: [{ text: 'q2' }] },
      ],
      use: [
        contextCompression({
          maxInputTokens: 10,
          maxToolResponseChars: 200,
          toolResponses: { maxChars: 50, preserveRecent: 1 },
        }),
      ],
    });

    const toolMsg1 = capturedRequest!.messages.find((m) => m.role === 'tool');
    assert.strictEqual(
      (toolMsg1!.content[0].metadata?.contextCompression as any)?.capped,
      true
    );
    assert.strictEqual(
      (toolMsg1!.content[0].metadata?.contextCompression as any)?.truncated,
      undefined
    );

    // 2. In next turn, toolA is no longer the most recent tool response, so it becomes truncatable
    let capturedRequest2: GenerateRequest | undefined;
    const pm2 = ai.defineModel({ name: 'capMetaModel2' }, async (req) => {
      capturedRequest2 = req;
      return {
        message: { role: 'model', content: [{ text: 'ok' }] },
        usage: { inputTokens: 50 },
      };
    });

    await ai.generate({
      model: pm2,
      messages: [
        { role: 'user', content: [{ text: 'q' }] },
        {
          role: 'model',
          content: [{ toolRequest: { name: 'toolA', input: {} } }],
        },
        toolMsg1!,
        { role: 'user', content: [{ text: 'q2' }] },
        {
          role: 'model',
          content: [{ toolRequest: { name: 'toolB', input: {} } }],
        },
        {
          role: 'tool',
          content: [
            {
              toolResponse: {
                name: 'toolB',
                output: 'recent tool output',
              },
            },
          ],
        },
        { role: 'user', content: [{ text: 'q3' }] },
      ],
      use: [
        contextCompression({
          maxInputTokens: 10,
          maxToolResponseChars: 200,
          toolResponses: { maxChars: 50, preserveRecent: 1 },
        }),
      ],
    });

    const toolMsgA = capturedRequest2!.messages[2];
    assert.strictEqual(
      (toolMsgA.content[0].metadata?.contextCompression as any)?.truncated,
      true
    );
    assert.strictEqual(
      (toolMsgA.content[0].metadata?.contextCompression as any)?.capped,
      true
    );
    assert.ok(
      (toolMsgA.content[0].toolResponse?.output as string).startsWith(
        'X'.repeat(50)
      )
    );
  });

  it('applies maxToolResponseChars safety cap even when maxInputTokens is not set', async () => {
    const ai = genkit({});
    let capturedRequest: GenerateRequest | undefined;

    const pm = ai.defineModel({ name: 'defaultSafetyModel' }, async (req) => {
      capturedRequest = req;
      return {
        message: { role: 'model', content: [{ text: 'ok' }] },
        usage: { inputTokens: 50 },
      };
    });

    const response = (await ai.generate({
      model: pm,
      messages: [
        { role: 'user', content: [{ text: 'run' }] },
        {
          role: 'model',
          content: [{ toolRequest: { name: 'bigTool', input: {} } }],
        },
        {
          role: 'tool',
          content: [
            {
              toolResponse: {
                name: 'bigTool',
                output: 'Z'.repeat(1000),
              },
            },
          ],
        },
      ],
      use: [
        contextCompression({
          maxToolResponseChars: 200,
        }),
      ],
    })) as any;

    assert.strictEqual(
      response.custom?.contextCompression?.toolResponsesSafetyCapped,
      1
    );
    const toolMsg = capturedRequest!.messages.find((m) => m.role === 'tool');
    assert.match(
      String(toolMsg!.content[0].toolResponse?.output),
      /\[TRUNCATED: Response was 1000 chars/
    );
  });

  it('preserves initiating user message and recent tool pairs in a single-prompt tool loop with maxMessages', async () => {
    const ai = genkit({});
    let capturedRequest: GenerateRequest | undefined;

    const pm = ai.defineModel(
      { name: 'singlePromptLoopModel' },
      async (req) => {
        capturedRequest = req;
        return {
          message: { role: 'model', content: [{ text: 'done' }] },
          usage: { inputTokens: 50 },
        };
      }
    );

    await ai.generate({
      model: pm,
      messages: [
        { role: 'system', content: [{ text: 'Sys' }] },
        { role: 'user', content: [{ text: 'Investigate Project Alpha' }] },
        {
          role: 'model',
          content: [{ toolRequest: { name: 't', input: { step: 1 } } }],
        },
        {
          role: 'tool',
          content: [{ toolResponse: { name: 't', output: 'r1' } }],
        },
        {
          role: 'model',
          content: [{ toolRequest: { name: 't', input: { step: 2 } } }],
        },
        {
          role: 'tool',
          content: [{ toolResponse: { name: 't', output: 'r2' } }],
        },
        {
          role: 'model',
          content: [{ toolRequest: { name: 't', input: { step: 3 } } }],
        },
        {
          role: 'tool',
          content: [{ toolResponse: { name: 't', output: 'r3' } }],
        },
      ],
      use: [
        contextCompression({
          maxMessages: 6,
          insertTruncationNotice: true,
        }),
      ],
    });

    const msgs = capturedRequest!.messages;
    // Sys + anchorUser ('Investigate Project Alpha') + [m2, t2, m3, t3] = 6 messages
    assert.strictEqual(msgs.length, 6);
    assert.strictEqual(msgs[0].role, 'system');
    assert.strictEqual(msgs[1].role, 'user');
    assert.strictEqual(msgs[1].content[0].text, 'Investigate Project Alpha');
    assert.strictEqual(msgs[2].role, 'model');
    assert.strictEqual(msgs[3].role, 'tool');
    assert.strictEqual(msgs[4].role, 'model');
    assert.strictEqual(msgs[5].role, 'tool');
  });

  it('counts preserveRecent by tool messages so parallel tool responses in the newest turn are preserved', async () => {
    const ai = genkit({});
    let capturedRequest: GenerateRequest | undefined;

    const pm = ai.defineModel({ name: 'parallelToolModel' }, async (req) => {
      capturedRequest = req;
      return {
        message: { role: 'model', content: [{ text: 'done' }] },
        usage: { inputTokens: 50 },
      };
    });

    await ai.generate({
      model: pm,
      messages: [
        { role: 'user', content: [{ text: 'fetch reports' }] },
        {
          role: 'model',
          content: [
            { toolRequest: { name: 'fetch', input: { id: '1' } } },
            { toolRequest: { name: 'fetch', input: { id: '2' } } },
          ],
        },
        {
          role: 'tool',
          content: [
            {
              toolResponse: {
                name: 'fetch',
                output: 'OldA-' + 'X'.repeat(200),
              },
            },
            {
              toolResponse: {
                name: 'fetch',
                output: 'OldB-' + 'X'.repeat(200),
              },
            },
          ],
        },
        {
          role: 'model',
          content: [
            { toolRequest: { name: 'fetch', input: { id: '3' } } },
            { toolRequest: { name: 'fetch', input: { id: '4' } } },
            { toolRequest: { name: 'fetch', input: { id: '5' } } },
          ],
        },
        {
          role: 'tool',
          content: [
            {
              toolResponse: {
                name: 'fetch',
                output: 'New1-' + 'Y'.repeat(200),
              },
            },
            {
              toolResponse: {
                name: 'fetch',
                output: 'New2-' + 'Y'.repeat(200),
              },
            },
            {
              toolResponse: {
                name: 'fetch',
                output: 'New3-' + 'Y'.repeat(200),
              },
            },
          ],
        },
      ],
      use: [
        contextCompression({
          maxInputTokens: 50,
          toolResponses: { maxChars: 20, preserveRecent: 1 },
        }),
      ],
    });

    const toolMsgs = capturedRequest!.messages.filter((m) => m.role === 'tool');
    assert.strictEqual(toolMsgs.length, 2);

    // Older tool message: both parallel parts truncated
    assert.match(
      String(toolMsgs[0].content[0].toolResponse?.output),
      /\[Truncated \d+ characters\]/
    );
    assert.match(
      String(toolMsgs[0].content[1].toolResponse?.output),
      /\[Truncated \d+ characters\]/
    );

    // Newest tool message (within preserveRecent: 1): all 3 parallel parts untouched
    for (const part of toolMsgs[1].content) {
      assert.strictEqual(
        String(part.toolResponse?.output).includes('[Truncated'),
        false
      );
    }
  });

  it('persists inputTokens stamp on model messages to trigger compression across separate generate calls', async () => {
    const ai = genkit({});
    let capturedRequest: GenerateRequest | undefined;

    const pm = ai.defineModel({ name: 'stampedTokenModel' }, async (req) => {
      capturedRequest = req;
      return {
        message: { role: 'model', content: [{ text: 'reply' }] },
        usage: { inputTokens: 800 },
      };
    });

    const mw = contextCompression({
      maxInputTokens: 500,
      toolResponses: { maxChars: 20, preserveRecent: 0 },
    });

    // Call 1: short messages (estimate < 500), but model reports usage.inputTokens = 800
    const res1 = await ai.generate({
      model: pm,
      messages: [{ role: 'user', content: [{ text: 'hi' }] }],
      use: [mw],
    });

    const stampedModelMsg = res1.message!.toJSON();
    assert.strictEqual(
      (stampedModelMsg.metadata?.contextCompression as any)?.inputTokens,
      800
    );

    // Call 2: pass history including stampedModelMsg; should trigger on turn 0 via stamp
    const res2 = (await ai.generate({
      model: pm,
      messages: [
        { role: 'user', content: [{ text: 'hi' }] },
        {
          role: 'tool',
          content: [
            {
              toolResponse: {
                name: 'search',
                output: 'Short text exceeding 20 chars for truncation test',
              },
            },
          ],
        },
        stampedModelMsg,
        { role: 'user', content: [{ text: 'follow up' }] },
      ],
      use: [mw],
    })) as any;

    assert.strictEqual(res2.custom?.contextCompression?.triggered, true);
    assert.strictEqual(res2.custom?.contextCompression?.inputTokensBefore, 800);
    const toolMsg = capturedRequest!.messages.find((m) => m.role === 'tool');
    assert.match(
      String(toolMsg!.content[0].toolResponse?.output),
      /\[Truncated \d+ characters\]/
    );
  });

  it('does not split UTF-16 surrogate pairs when slicing at maxChars boundary', async () => {
    const ai = genkit({});
    let capturedRequest: GenerateRequest | undefined;

    const pm = ai.defineModel({ name: 'surrogateModel' }, async (req) => {
      capturedRequest = req;
      return {
        message: { role: 'model', content: [{ text: 'ok' }] },
        usage: { inputTokens: 50 },
      };
    });

    // 'abcd' (4 code units) + '😀' (2 code units: index 4 is high surrogate, index 5 is low surrogate)
    const emojiPayload = 'abcd😀' + 'Z'.repeat(200);

    await ai.generate({
      model: pm,
      messages: [
        { role: 'user', content: [{ text: 'q' }] },
        {
          role: 'tool',
          content: [
            {
              toolResponse: {
                name: 'emojiTool',
                output: emojiPayload,
              },
            },
          ],
        },
      ],
      use: [
        contextCompression({
          maxInputTokens: 10,
          toolResponses: { maxChars: 5, preserveRecent: 0 },
        }),
      ],
    });

    const toolMsg = capturedRequest!.messages.find((m) => m.role === 'tool');
    const output = String(toolMsg!.content[0].toolResponse?.output);
    // Limit 5 falls after high surrogate of '😀', so slice backs up to 4 ('abcd')
    assert.ok(output.startsWith('abcd\n\n[Truncated '));
  });

  it('compresses tool responses with deduplication and truncation', async () => {
    const ai = genkit({});
    let turn = 0;
    const capturedRequests: GenerateRequest[] = [];

    const searchTool = ai.defineTool(
      {
        name: 'search',
        description: 'search tool',
        inputSchema: z.object({ query: z.string() }),
        outputSchema: z.string(),
      },
      async (input) => `Result for ${input.query}: ${'A'.repeat(500)}`
    );

    const pm = ai.defineModel({ name: 'loopModel' }, async (req) => {
      capturedRequests.push(req);
      turn++;
      if (turn === 1) {
        return {
          message: {
            role: 'model',
            content: [
              { toolRequest: { name: 'search', input: { query: 'test' } } },
            ],
          },
          usage: { inputTokens: 200 },
        };
      }
      if (turn === 2) {
        return {
          message: {
            role: 'model',
            content: [
              { toolRequest: { name: 'search', input: { query: 'test' } } },
            ],
          },
          usage: { inputTokens: 500 },
        };
      }
      return {
        message: { role: 'model', content: [{ text: 'finished' }] },
        usage: { inputTokens: 100 },
      };
    });

    const result = await ai.generate({
      model: pm,
      prompt: 'Search multiple times',
      tools: [searchTool],
      use: [
        contextCompression({
          maxInputTokens: 150,
          deduplicateToolResponses: { matchBy: 'name-and-input' },
          toolResponses: { maxChars: 50, preserveRecent: 0 },
        }),
      ],
    });

    assert.strictEqual(result.text, 'finished');
    assert.strictEqual(capturedRequests.length, 3);

    const turn3Messages = capturedRequests[2].messages;
    const toolMessages = turn3Messages.filter((m) => m.role === 'tool');
    assert.ok(toolMessages.length >= 2);
    const firstToolOutput = toolMessages[0].content[0].toolResponse?.output;
    assert.match(String(firstToolOutput), /Deduplicated/);
  });

  it('deduplicates tool responses by correlating tool call IDs (ref) to tool inputs', async () => {
    const ai = genkit({});
    let turn = 0;
    const capturedRequests: GenerateRequest[] = [];

    const searchTool = ai.defineTool(
      {
        name: 'search',
        description: 'search tool',
        inputSchema: z.object({ query: z.string() }),
        outputSchema: z.string(),
      },
      async (input) => `Result for ${input.query}: ${'A'.repeat(200)}`
    );

    const pm = ai.defineModel({ name: 'refDedupModel' }, async (req) => {
      capturedRequests.push(req);
      turn++;
      if (turn === 1) {
        return {
          message: {
            role: 'model',
            content: [
              {
                toolRequest: {
                  name: 'search',
                  ref: 'call_unique_1',
                  input: { query: 'same-query' },
                },
              },
            ],
          },
          usage: { inputTokens: 200 },
        };
      }
      if (turn === 2) {
        return {
          message: {
            role: 'model',
            content: [
              {
                toolRequest: {
                  name: 'search',
                  ref: 'call_unique_2', // Distinct call id, but identical tool input
                  input: { query: 'same-query' },
                },
              },
            ],
          },
          usage: { inputTokens: 500 },
        };
      }
      return {
        message: { role: 'model', content: [{ text: 'finished' }] },
        usage: { inputTokens: 100 },
      };
    });

    const result = await ai.generate({
      model: pm,
      prompt: 'Search multiple times with refs',
      tools: [searchTool],
      use: [
        contextCompression({
          maxInputTokens: 150,
          deduplicateToolResponses: { matchBy: 'name-and-input' },
          toolResponses: { maxChars: 50, preserveRecent: 0 },
        }),
      ],
    });

    assert.strictEqual(result.text, 'finished');
    assert.strictEqual(capturedRequests.length, 3);

    const turn3Messages = capturedRequests[2].messages;
    const toolMessages = turn3Messages.filter((m) => m.role === 'tool');
    assert.ok(toolMessages.length >= 2);
    const firstToolOutput = toolMessages[0].content[0].toolResponse?.output;
    assert.match(String(firstToolOutput), /Deduplicated/);
  });

  it('preserves distinct calls with different arguments when matchBy is name-and-input', async () => {
    const ai = genkit({});
    let turn = 0;
    const capturedRequests: GenerateRequest[] = [];

    const searchTool = ai.defineTool(
      {
        name: 'search',
        description: 'search tool',
        inputSchema: z.object({ query: z.string() }),
        outputSchema: z.string(),
      },
      async (input) => `Result for ${input.query}: ${'B'.repeat(50)}`
    );

    const pm = ai.defineModel({ name: 'distinctArgsModel' }, async (req) => {
      capturedRequests.push(req);
      turn++;
      if (turn === 1) {
        return {
          message: {
            role: 'model',
            content: [
              {
                toolRequest: {
                  name: 'search',
                  input: { query: 'first-query' },
                },
              },
            ],
          },
          usage: { inputTokens: 200 },
        };
      }
      if (turn === 2) {
        return {
          message: {
            role: 'model',
            content: [
              {
                toolRequest: {
                  name: 'search',
                  input: { query: 'second-query' },
                },
              },
            ],
          },
          usage: { inputTokens: 500 },
        };
      }
      return {
        message: { role: 'model', content: [{ text: 'finished' }] },
        usage: { inputTokens: 100 },
      };
    });

    await ai.generate({
      model: pm,
      prompt: 'Search with distinct queries',
      tools: [searchTool],
      use: [
        contextCompression({
          maxInputTokens: 150,
          deduplicateToolResponses: { matchBy: 'name-and-input' },
        }),
      ],
    });

    const turn3Messages = capturedRequests[2].messages;
    const toolMessages = turn3Messages.filter((m) => m.role === 'tool');
    assert.strictEqual(toolMessages.length, 2);
    // Neither should be deduplicated because arguments differ
    assert.strictEqual(
      String(toolMessages[0].content[0].toolResponse?.output).includes(
        'Deduplicated'
      ),
      false
    );
    assert.strictEqual(
      String(toolMessages[1].content[0].toolResponse?.output).includes(
        'Deduplicated'
      ),
      false
    );
  });

  it('deduplicates only duplicate parts in parallel tool calls without replacing unique sibling parts', async () => {
    const ai = genkit({});
    let capturedRequest: GenerateRequest | undefined;

    const pm = ai.defineModel({ name: 'parallelDedupModel' }, async (req) => {
      capturedRequest = req;
      return {
        message: { role: 'model', content: [{ text: 'done' }] },
        usage: { inputTokens: 50 },
      };
    });

    await ai.generate({
      model: pm,
      messages: [
        { role: 'user', content: [{ text: 'run parallel tools' }] },
        {
          role: 'model',
          content: [
            {
              toolRequest: {
                name: 'fetch',
                ref: 'call_1',
                input: { id: 'shared' },
              },
            },
            {
              toolRequest: {
                name: 'fetch',
                ref: 'call_2',
                input: { id: 'unique' },
              },
            },
          ],
        },
        {
          role: 'tool',
          content: [
            {
              toolResponse: {
                name: 'fetch',
                ref: 'call_1',
                output: 'Shared output 1 ' + 'X'.repeat(200),
              },
            },
            {
              toolResponse: {
                name: 'fetch',
                ref: 'call_2',
                output: 'Unique output 2 ' + 'Y'.repeat(200),
              },
            },
          ],
        },
        {
          role: 'model',
          content: [
            {
              toolRequest: {
                name: 'fetch',
                ref: 'call_3',
                input: { id: 'shared' },
              },
            },
          ],
        },
        {
          role: 'tool',
          content: [
            {
              toolResponse: {
                name: 'fetch',
                ref: 'call_3',
                output: 'Shared output 3 ' + 'Z'.repeat(200),
              },
            },
          ],
        },
      ],
      use: [
        contextCompression({
          maxInputTokens: 50,
          deduplicateToolResponses: { matchBy: 'name-and-input' },
        }),
      ],
    });

    const toolMessages = capturedRequest!.messages.filter(
      (m) => m.role === 'tool'
    );
    assert.strictEqual(toolMessages.length, 2);
    // First tool message: part 0 ('shared') is deduplicated, part 1 ('unique') is preserved intact
    assert.match(
      String(toolMessages[0].content[0].toolResponse?.output),
      /Deduplicated/
    );
    assert.ok(
      String(toolMessages[0].content[1].toolResponse?.output).startsWith(
        'Unique output 2 '
      )
    );
    // Second tool message: newest occurrence of 'shared' is preserved intact
    assert.ok(
      String(toolMessages[1].content[0].toolResponse?.output).startsWith(
        'Shared output 3 '
      )
    );
  });

  it('summarizes older messages using summary model', async () => {
    const ai = genkit({});
    let turn = 0;
    const capturedRequests: GenerateRequest[] = [];

    const summaryModel = ai.defineModel({ name: 'summaryModel' }, async () => ({
      message: {
        role: 'model',
        content: [{ text: 'Summary of past events: steps were executed.' }],
      },
    }));

    const dummyTool = ai.defineTool(
      {
        name: 'step',
        description: 'step',
        inputSchema: z.object({ step: z.number() }),
        outputSchema: z.string(),
      },
      async (input) => `output ${input.step}`
    );

    const pm = ai.defineModel({ name: 'multiTurnModel' }, async (req) => {
      capturedRequests.push(req);
      turn++;
      if (turn <= 3) {
        return {
          message: {
            role: 'model',
            content: [{ toolRequest: { name: 'step', input: { step: turn } } }],
          },
          usage: { inputTokens: 500 },
        };
      }
      return {
        message: { role: 'model', content: [{ text: 'all done' }] },
        usage: { inputTokens: 200 },
      };
    });

    const result = await ai.generate({
      model: pm,
      system: 'System instructions',
      prompt: 'Run multi turn steps',
      tools: [dummyTool],
      use: [
        contextCompression({
          maxInputTokens: 100,
          summarize: {
            model: { name: 'summaryModel' },
            preserveRecent: 2,
          },
        }),
      ],
    });

    assert.strictEqual(result.text, 'all done');

    const lastReq = capturedRequests[capturedRequests.length - 1];
    const summaryMsg = lastReq.messages.find((m) =>
      m.content.some((p) => p.text?.includes('Summary of past events'))
    );
    assert.ok(summaryMsg);
  });

  it('respects custom summarization prompt', async () => {
    const ai = genkit({});
    let capturedPrompt = '';

    const summaryModel = ai.defineModel(
      { name: 'customPromptModel' },
      async (req) => {
        capturedPrompt = req.messages[0]?.content[0]?.text ?? '';
        return {
          message: {
            role: 'model',
            content: [{ text: 'Custom summary result' }],
          },
        };
      }
    );

    const pm = ai.defineModel({ name: 'testModel' }, async () => ({
      message: { role: 'model', content: [{ text: 'done' }] },
      usage: { inputTokens: 50 },
    }));

    await ai.generate({
      model: pm,
      messages: [
        { role: 'user', content: [{ text: 'A long discussion part 1' }] },
        { role: 'model', content: [{ text: 'A long discussion response 1' }] },
        { role: 'user', content: [{ text: 'A long discussion part 2' }] },
        { role: 'model', content: [{ text: 'A long discussion response 2' }] },
      ],
      use: [
        contextCompression({
          maxInputTokens: 10,
          summarize: {
            model: summaryModel,
            preserveRecent: 1,
            prompt: 'TLDR THIS: {conversation}\nEND TLDR',
          },
        }),
      ],
    });

    assert.match(capturedPrompt, /^TLDR THIS:/);
    assert.match(capturedPrompt, /A long discussion part 1/);
    assert.match(capturedPrompt, /END TLDR$/);
  });

  it('skips summarization when cheap strategies achieve skipSummarizationThreshold', async () => {
    const ai = genkit({});
    let summaryCalled = false;

    const summaryModel = ai.defineModel(
      { name: 'trackedSummaryModel' },
      async () => {
        summaryCalled = true;
        return {
          message: { role: 'model', content: [{ text: 'Summary' }] },
        };
      }
    );

    const pm = ai.defineModel({ name: 'skipModel' }, async () => ({
      message: { role: 'model', content: [{ text: 'done' }] },
      usage: { inputTokens: 50 },
    }));

    const response = (await ai.generate({
      model: pm,
      messages: [
        {
          role: 'tool',
          content: [
            {
              toolResponse: {
                name: 'huge',
                ref: '1',
                output: 'X'.repeat(2000),
              },
            },
          ],
        },
        { role: 'user', content: [{ text: 'msg 2' }] },
        { role: 'model', content: [{ text: 'msg 3' }] },
        { role: 'user', content: [{ text: 'msg 4' }] },
      ],
      use: [
        contextCompression({
          maxInputTokens: 100,
          toolResponses: { maxChars: 100, preserveRecent: 0 },
          summarize: {
            model: summaryModel,
            preserveRecent: 1,
          },
          skipSummarizationThreshold: 0.25, // 2000 chars reduced to ~100 is > 90% savings
        }),
      ],
    })) as any;

    assert.strictEqual(summaryCalled, false);
    assert.strictEqual(
      response.custom?.contextCompression?.summarizationSkipped,
      true
    );
  });

  it('dynamically adjusts preserveRecent window when token usage overshoots budget', async () => {
    const ai = genkit({});
    let capturedRequest: GenerateRequest | undefined;

    const pm = ai.defineModel({ name: 'overshootModel' }, async (req) => {
      capturedRequest = req;
      return {
        message: { role: 'model', content: [{ text: 'done' }] },
        usage: { inputTokens: 50 },
      };
    });

    // 10 messages, maxMessages = 6, basePreserveRecent = 4
    // overshootRatio = 3000 estimated chars / 3.5 / 50 tokens = ~17x overshoot (>= 2.0)
    // adjustForOvershoot caps preserveRecent to min(4, 2) = 2
    // effectiveMaxMessages = min(maxMessages: 6, adjustedPreserveRecent: 2) = 2
    await ai.generate({
      model: pm,
      messages: Array.from({ length: 10 }, (_, i) => ({
        role: i % 2 === 0 ? ('user' as const) : ('model' as const),
        content: [{ text: `Message number ${i}: ${'X'.repeat(300)}` }],
      })),
      use: [
        contextCompression({
          maxInputTokens: 50,
          maxMessages: 6,
          preserveRecent: 4,
          insertTruncationNotice: false,
        }),
      ],
    });

    // Truncation should have clamped to 2 messages (last user/model pair) due to >= 2x overshoot
    assert.strictEqual(capturedRequest?.messages.length, 2);
  });

  it('preserves the latest [model, tool] turn when maxMessages is 3 (keepCount: 2) in a tool loop', async () => {
    const ai = genkit({});
    let capturedRequest: GenerateRequest | undefined;

    const pm = ai.defineModel(
      { name: 'smallKeepCountLoopModel' },
      async (req) => {
        capturedRequest = req;
        return {
          message: { role: 'model', content: [{ text: 'done' }] },
          usage: { inputTokens: 50 },
        };
      }
    );

    await ai.generate({
      model: pm,
      messages: [
        { role: 'system', content: [{ text: 'Sys' }] },
        { role: 'user', content: [{ text: 'Original task' }] },
        {
          role: 'model',
          content: [{ toolRequest: { name: 't', input: { step: 1 } } }],
        },
        {
          role: 'tool',
          content: [{ toolResponse: { name: 't', output: 'r1' } }],
        },
        {
          role: 'model',
          content: [{ toolRequest: { name: 't', input: { step: 2 } } }],
        },
        {
          role: 'tool',
          content: [{ toolResponse: { name: 't', output: 'r2' } }],
        },
      ],
      use: [
        contextCompression({
          maxMessages: 3,
          insertTruncationNotice: true,
        }),
      ],
    });

    const msgs = capturedRequest!.messages;
    assert.strictEqual(msgs.length, 4);
    assert.strictEqual(msgs[0].role, 'system');
    assert.strictEqual(msgs[1].role, 'user');
    assert.strictEqual(msgs[1].content[0].text, 'Original task');
    assert.strictEqual(msgs[2].role, 'model');
    assert.deepStrictEqual((msgs[2].content[0].toolRequest as any)?.input, {
      step: 2,
    });
    assert.strictEqual(msgs[3].role, 'tool');
    assert.strictEqual(msgs[3].content[0].toolResponse?.output, 'r2');
  });

  it('reconciles a saved standalone notice into a newly prepended system message without producing two system messages', async () => {
    const ai = genkit({});
    let capturedRequest: GenerateRequest | undefined;

    const pm = ai.defineModel({ name: 'reconcileNoticeModel' }, async (req) => {
      capturedRequest = req;
      return {
        message: { role: 'model', content: [{ text: 'ok' }] },
        usage: { inputTokens: 20 },
      };
    });

    const mw = contextCompression({
      maxMessages: 4,
      insertTruncationNotice: true,
    });

    // Turn 1: no system message -> standalone system notice is inserted at index 0
    await ai.generate({
      model: pm,
      messages: [
        { role: 'user', content: [{ text: 'u1' }] },
        { role: 'model', content: [{ text: 'm1' }] },
        { role: 'user', content: [{ text: 'u2' }] },
        { role: 'model', content: [{ text: 'm2' }] },
        { role: 'user', content: [{ text: 'u3' }] },
      ],
      use: [mw],
    });

    const turn1Messages = capturedRequest!.messages;
    assert.strictEqual(
      turn1Messages.filter((m) => m.role === 'system').length,
      1
    );

    // Turn 2: caller saves turn1Messages (including the standalone notice) and
    // prepends a real system prompt on a non-compressing turn (<= maxMessages: 4)
    await ai.generate({
      model: pm,
      messages: [
        { role: 'system', content: [{ text: 'You are an assistant.' }] },
        ...turn1Messages,
      ],
      use: [mw],
    });

    const turn2SystemMsgs = capturedRequest!.messages.filter(
      (m) => m.role === 'system'
    );
    assert.strictEqual(turn2SystemMsgs.length, 1);
    const sysText = turn2SystemMsgs[0].content.map((p) => p.text).join('');
    assert.match(sysText, /You are an assistant\./);
    assert.match(
      sysText,
      /Some earlier messages in this conversation have been removed/
    );
  });

  it('reports non-zero inputTokensBefore when safety cap fires with maxInputTokens unset', async () => {
    const ai = genkit({});
    const pm = ai.defineModel({ name: 'safetyCapTokensModel' }, async () => ({
      message: { role: 'model', content: [{ text: 'ok' }] },
      usage: { inputTokens: 300 },
    }));

    const response = (await ai.generate({
      model: pm,
      messages: [
        { role: 'user', content: [{ text: 'fetch' }] },
        {
          role: 'tool',
          content: [
            {
              toolResponse: {
                name: 'dump',
                output: 'A'.repeat(1000),
              },
            },
          ],
        },
      ],
      use: [
        contextCompression({
          maxToolResponseChars: 200,
        }),
      ],
    })) as any;

    assert.strictEqual(response.custom?.contextCompression?.triggered, true);
    assert.ok(response.custom?.contextCompression?.inputTokensBefore > 0);
  });

  it('clears multipart toolResponse.content when deduplicating older tool responses', async () => {
    const ai = genkit({});
    let capturedRequest: GenerateRequest | undefined;

    const pm = ai.defineModel({ name: 'multipartDedupModel' }, async (req) => {
      capturedRequest = req;
      return {
        message: { role: 'model', content: [{ text: 'done' }] },
        usage: { inputTokens: 50 },
      };
    });

    const oldMediaPart = {
      media: {
        url: 'data:image/png;base64,OLD_SCREENSHOT',
        contentType: 'image/png',
      },
    };
    const newMediaPart = {
      media: {
        url: 'data:image/png;base64,NEW_SCREENSHOT',
        contentType: 'image/png',
      },
    };

    await ai.generate({
      model: pm,
      messages: [
        { role: 'user', content: [{ text: 'take screenshots' }] },
        {
          role: 'model',
          content: [
            {
              toolRequest: {
                name: 'screenshot',
                ref: 'shot_1',
                input: { page: 'home' },
              },
            },
          ],
        },
        {
          role: 'tool',
          content: [
            {
              toolResponse: {
                name: 'screenshot',
                ref: 'shot_1',
                output: 'Captured screenshot 1 ' + 'X'.repeat(200),
                content: [oldMediaPart],
              },
            },
          ],
        },
        {
          role: 'model',
          content: [
            {
              toolRequest: {
                name: 'screenshot',
                ref: 'shot_2',
                input: { page: 'home' },
              },
            },
          ],
        },
        {
          role: 'tool',
          content: [
            {
              toolResponse: {
                name: 'screenshot',
                ref: 'shot_2',
                output: 'Captured screenshot 2 ' + 'Y'.repeat(200),
                content: [newMediaPart],
              },
            },
          ],
        },
      ],
      use: [
        contextCompression({
          maxInputTokens: 50,
          deduplicateToolResponses: { matchBy: 'name-and-input' },
        }),
      ],
    });

    const toolMessages = capturedRequest!.messages.filter(
      (m) => m.role === 'tool'
    );
    assert.strictEqual(toolMessages.length, 2);
    // Older duplicate has output replaced and multipart content removed
    assert.match(
      String(toolMessages[0].content[0].toolResponse?.output),
      /Deduplicated/
    );
    assert.strictEqual(
      toolMessages[0].content[0].toolResponse?.content,
      undefined
    );
    assert.strictEqual(
      'content' in toolMessages[0].content[0].toolResponse!,
      false
    );
    // Most recent occurrence keeps both output and multipart content
    assert.ok(
      String(toolMessages[1].content[0].toolResponse?.output).startsWith(
        'Captured screenshot 2 '
      )
    );
    assert.deepStrictEqual(toolMessages[1].content[0].toolResponse?.content, [
      newMediaPart,
    ]);
  });

  it('validates keepRecent as positive integer in schema and clamps non-positive values at runtime', async () => {
    assert.strictEqual(
      DeduplicateToolResponsesOptionsSchema.safeParse({ keepRecent: 1 })
        .success,
      true
    );
    assert.strictEqual(
      DeduplicateToolResponsesOptionsSchema.safeParse({ keepRecent: 0 })
        .success,
      false
    );
    assert.strictEqual(
      DeduplicateToolResponsesOptionsSchema.safeParse({ keepRecent: -1 })
        .success,
      false
    );
    assert.strictEqual(
      DeduplicateToolResponsesOptionsSchema.safeParse({ keepRecent: 1.5 })
        .success,
      false
    );

    const ai = genkit({});
    let capturedRequest: GenerateRequest | undefined;

    const pm = ai.defineModel({ name: 'clampKeepRecentModel' }, async (req) => {
      capturedRequest = req;
      return {
        message: { role: 'model', content: [{ text: 'done' }] },
        usage: { inputTokens: 50 },
      };
    });

    await ai.generate({
      model: pm,
      messages: [
        { role: 'user', content: [{ text: 'check single tool call' }] },
        {
          role: 'model',
          content: [
            {
              toolRequest: {
                name: 'fetch',
                ref: 'call_1',
                input: { id: '1' },
              },
            },
          ],
        },
        {
          role: 'tool',
          content: [
            {
              toolResponse: {
                name: 'fetch',
                ref: 'call_1',
                output: 'Only response ' + 'X'.repeat(200),
              },
            },
          ],
        },
      ],
      use: [
        contextCompression({
          maxInputTokens: 10,
          deduplicateToolResponses: { keepRecent: 0 },
        }),
      ],
    });

    const toolMsg = capturedRequest!.messages.find((m) => m.role === 'tool');
    // Clamped to 1 so a tool's sole response is never replaced
    assert.ok(
      String(toolMsg!.content[0].toolResponse?.output).startsWith(
        'Only response '
      )
    );
  });

  it('accepts model name string, ModelReference, and ModelAction for summarize.model', async () => {
    const ai = genkit({});
    let stringModelCalls = 0;
    let refModelCalls = 0;

    ai.defineModel({ name: 'stringSummarizer' }, async () => {
      stringModelCalls++;
      return {
        message: { role: 'model', content: [{ text: 'Summary via string' }] },
        finishReason: 'stop',
      };
    });

    ai.defineModel({ name: 'refSummarizer' }, async () => {
      refModelCalls++;
      return {
        message: { role: 'model', content: [{ text: 'Summary via modelRef' }] },
        finishReason: 'stop',
      };
    });

    const pm = ai.defineModel({ name: 'mainModel' }, async () => ({
      message: { role: 'model', content: [{ text: 'done' }] },
      usage: { inputTokens: 50 },
    }));

    await ai.generate({
      model: pm,
      messages: [
        { role: 'user', content: [{ text: 'u1 ' + 'X'.repeat(200) }] },
        { role: 'model', content: [{ text: 'm1 ' + 'X'.repeat(200) }] },
        { role: 'user', content: [{ text: 'u2' }] },
      ],
      use: [
        contextCompression({
          maxInputTokens: 20,
          summarize: {
            model: 'stringSummarizer',
            preserveRecent: 1,
          },
        }),
      ],
    });

    const typedRef = modelRef({
      name: 'refSummarizer',
      config: { temperature: 0.1 },
    });
    await ai.generate({
      model: pm,
      messages: [
        { role: 'user', content: [{ text: 'u1 ' + 'X'.repeat(200) }] },
        { role: 'model', content: [{ text: 'm1 ' + 'X'.repeat(200) }] },
        { role: 'user', content: [{ text: 'u2' }] },
      ],
      use: [
        contextCompression({
          maxInputTokens: 20,
          summarize: {
            model: typedRef,
            preserveRecent: 1,
          },
        }),
      ],
    });

    assert.strictEqual(stringModelCalls, 1);
    assert.strictEqual(refModelCalls, 1);
  });

  it('validates preserveRecent and skipSummarizationThreshold in schema and clamps preserveRecent at runtime', async () => {
    assert.strictEqual(
      ContextCompressionOptionsSchema.safeParse({
        preserveRecent: 0,
      }).success,
      false
    );
    assert.strictEqual(
      ContextCompressionOptionsSchema.safeParse({
        summarize: { model: 'some-model', preserveRecent: 0 },
      }).success,
      false
    );
    assert.strictEqual(
      ContextCompressionOptionsSchema.safeParse({
        skipSummarizationThreshold: 1.5,
      }).success,
      false
    );
    assert.strictEqual(
      ContextCompressionOptionsSchema.safeParse({
        skipSummarizationThreshold: -0.1,
      }).success,
      false
    );
    assert.strictEqual(
      ContextCompressionOptionsSchema.safeParse({
        preserveRecent: 4,
        summarize: { model: 'some-model', preserveRecent: 2 },
        skipSummarizationThreshold: 0.25,
      }).success,
      true
    );

    // Verify runtime clamping prevents slice(-0) from duplicating all messages
    const ai = genkit({});
    let capturedRequest: GenerateRequest | undefined;
    ai.defineModel({ name: 'clampSummarizer' }, async () => ({
      message: { role: 'model', content: [{ text: 'Condensed' }] },
      finishReason: 'stop',
    }));
    const pm = ai.defineModel({ name: 'clampMain' }, async (req) => {
      capturedRequest = req;
      return {
        message: { role: 'model', content: [{ text: 'ok' }] },
        usage: { inputTokens: 50 },
      };
    });

    await ai.generate({
      model: pm,
      messages: [
        { role: 'user', content: [{ text: 'msg 1 ' + 'X'.repeat(200) }] },
        { role: 'model', content: [{ text: 'msg 2 ' + 'X'.repeat(200) }] },
        { role: 'user', content: [{ text: 'msg 3' }] },
      ],
      use: [
        contextCompression({
          maxInputTokens: 20,
          summarize: {
            model: 'clampSummarizer',
            preserveRecent: 0 as number,
          },
        }),
      ],
    });

    // Clamped to 1: [summary, msg 3] = 2 messages (never [summary, msg 1, msg 2, msg 3])
    assert.strictEqual(capturedRequest?.messages.length, 2);
  });

  it('does not skip summarization when cheap strategies save >= threshold but remain over maxInputTokens', async () => {
    const ai = genkit({});
    let summaryCalled = false;

    const summaryModel = ai.defineModel(
      { name: 'stillOverBudgetSummarizer' },
      async () => {
        summaryCalled = true;
        return {
          message: { role: 'model', content: [{ text: 'Summary' }] },
          finishReason: 'stop',
        };
      }
    );

    const pm = ai.defineModel(
      { name: 'stillOverBudgetMainModel' },
      async () => ({
        message: { role: 'model', content: [{ text: 'done' }] },
        usage: { inputTokens: 50 },
      })
    );

    // Total before cheap strategies: ~1400 chars (~400 tokens).
    // Cheap tool truncation reduces 700-char tool output to ~130 chars (~40% savings >= 0.25),
    // but remaining context is ~830 chars (~238 tokens), still exceeding maxInputTokens: 150.
    const response = await ai.generate({
      model: pm,
      messages: [
        {
          role: 'tool',
          content: [
            {
              toolResponse: {
                name: 'heavy',
                ref: '1',
                output: 'T'.repeat(700),
              },
            },
          ],
        },
        { role: 'user', content: [{ text: 'U'.repeat(350) }] },
        { role: 'model', content: [{ text: 'M'.repeat(350) }] },
        { role: 'user', content: [{ text: 'latest question' }] },
      ],
      use: [
        contextCompression({
          maxInputTokens: 150,
          toolResponses: { maxChars: 100, preserveRecent: 0 },
          summarize: {
            model: summaryModel,
            preserveRecent: 1,
          },
          skipSummarizationThreshold: 0.25,
        }),
      ],
    });

    assert.strictEqual(summaryCalled, true);
    const meta = (response.custom as Record<string, unknown> | undefined)
      ?.contextCompression as Record<string, unknown> | undefined;
    assert.strictEqual(meta?.summarized, true);
    assert.strictEqual(meta?.summarizationSkipped, false);
  });

  it('renders reasoning, media, multipart tool responses, resource, and data parts for summarization', async () => {
    const ai = genkit({});
    let capturedSummaryPrompt = '';

    const summaryModel = ai.defineModel(
      { name: 'richPartSummarizer' },
      async (req) => {
        capturedSummaryPrompt = req.messages[0]?.content[0]?.text ?? '';
        return {
          message: { role: 'model', content: [{ text: 'Rich summary' }] },
          finishReason: 'stop',
        };
      }
    );

    const pm = ai.defineModel({ name: 'richPartMain' }, async () => ({
      message: { role: 'model', content: [{ text: 'ok' }] },
      usage: { inputTokens: 50 },
    }));

    const base64Payload = 'QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVo=';
    await ai.generate({
      model: pm,
      messages: [
        {
          role: 'user',
          content: [
            { text: 'Check these attachments ' + 'X'.repeat(200) },
            {
              media: {
                url: `data:image/png;base64,${base64Payload}`,
              },
            },
            {
              media: {
                contentType: 'application/pdf',
                url: 'https://example.com/spec.pdf',
              },
            },
            {
              resource: { uri: 'file:///workspace/README.md' },
            },
          ],
        },
        {
          role: 'model',
          content: [
            { reasoning: 'Thinking through the spec carefully' },
            { data: { status: 'analyzed' } },
            { toolRequest: { name: 'inspect', input: { target: 'spec' } } },
          ],
        },
        {
          role: 'tool',
          content: [
            {
              toolResponse: {
                name: 'inspect',
                output: { ok: true },
                content: [{ text: 'Multipart tool content detail' }],
              },
            },
          ],
        },
        { role: 'user', content: [{ text: 'Final question' }] },
      ],
      use: [
        contextCompression({
          maxInputTokens: 20,
          summarize: {
            model: summaryModel,
            preserveRecent: 1,
          },
        }),
      ],
    });

    assert.equal(capturedSummaryPrompt.includes('[other content]'), false);
    assert.match(capturedSummaryPrompt, /\[media: image\/png\]/);
    assert.equal(capturedSummaryPrompt.includes(base64Payload), false);
    assert.match(
      capturedSummaryPrompt,
      /\[media: application\/pdf \(https:\/\/example\.com\/spec\.pdf\)\]/
    );
    assert.match(
      capturedSummaryPrompt,
      /\[resource: file:\/\/\/workspace\/README\.md\]/
    );
    assert.match(
      capturedSummaryPrompt,
      /\[Reasoning: Thinking through the spec carefully\]/
    );
    assert.match(capturedSummaryPrompt, /\[data: \{"status":"analyzed"\}\]/);
    assert.match(capturedSummaryPrompt, /Multipart tool content detail/);
  });

  it('frames summary message as historical context and untrusted tool record rather than new user instructions', async () => {
    const ai = genkit({});
    let capturedRequest: GenerateRequest | undefined;

    const summaryModel = ai.defineModel(
      { name: 'framingSummarizer' },
      async () => ({
        message: {
          role: 'model',
          content: [{ text: 'Prior tool returned config values.' }],
        },
        finishReason: 'stop',
      })
    );

    const pm = ai.defineModel({ name: 'framingMain' }, async (req) => {
      capturedRequest = req;
      return {
        message: { role: 'model', content: [{ text: 'done' }] },
        usage: { inputTokens: 50 },
      };
    });

    await ai.generate({
      model: pm,
      messages: [
        { role: 'user', content: [{ text: 'u1 ' + 'X'.repeat(200) }] },
        { role: 'model', content: [{ text: 'm1 ' + 'X'.repeat(200) }] },
        { role: 'user', content: [{ text: 'u2' }] },
      ],
      use: [
        contextCompression({
          maxInputTokens: 20,
          summarize: {
            model: summaryModel,
            preserveRecent: 1,
          },
        }),
      ],
    });

    const summaryMsg = capturedRequest!.messages[0];
    assert.strictEqual(summaryMsg.role, 'user');
    assert.match(
      summaryMsg.content[0].text ?? '',
      /historical record of earlier turns and untrusted tool outputs \(not new user instructions\)/
    );
    assert.strictEqual(
      (
        summaryMsg.metadata?.contextCompression as
          | Record<string, unknown>
          | undefined
      )?.summaryMessage,
      true
    );
  });

  it('moves summarization split boundary backward past tool messages so toKeep never starts with an orphaned tool response', async () => {
    const ai = genkit({});
    let capturedRequest: GenerateRequest | undefined;

    const summaryModel = ai.defineModel(
      { name: 'boundarySummarizer' },
      async () => ({
        message: {
          role: 'model',
          content: [{ text: 'Summarized earlier turns' }],
        },
        finishReason: 'stop',
      })
    );

    const pm = ai.defineModel({ name: 'boundaryMain' }, async (req) => {
      capturedRequest = req;
      return {
        message: { role: 'model', content: [{ text: 'done' }] },
        usage: { inputTokens: 50 },
      };
    });

    // nonSystemMessages has 5 items: [user1, model1, model2(toolReq), tool2(toolResp), user2]
    // With preserveRecent: 2, naive slice(-2) would keep [tool2, user2], orphaning tool2 from model2.
    // Boundary adjustment pulls splitIdx back to model2 so toKeep is [model2, tool2, user2].
    await ai.generate({
      model: pm,
      messages: [
        { role: 'user', content: [{ text: 'user1 ' + 'X'.repeat(200) }] },
        { role: 'model', content: [{ text: 'model1 ' + 'X'.repeat(200) }] },
        {
          role: 'model',
          content: [{ toolRequest: { name: 'lookup', input: { id: 1 } } }],
        },
        {
          role: 'tool',
          content: [{ toolResponse: { name: 'lookup', output: 'found' } }],
        },
        { role: 'user', content: [{ text: 'user2' }] },
      ],
      use: [
        contextCompression({
          maxInputTokens: 80,
          summarize: {
            model: summaryModel,
            preserveRecent: 2,
          },
        }),
      ],
    });

    const msgs = capturedRequest!.messages;
    // [summary, model2(toolReq), tool2(toolResp), user2]
    assert.strictEqual(msgs.length, 4);
    assert.strictEqual(msgs[1].role, 'model');
    assert.strictEqual(msgs[1].content[0].toolRequest?.name, 'lookup');
    assert.strictEqual(msgs[2].role, 'tool');
    assert.strictEqual(msgs[3].role, 'user');
  });

  it('caps oversized summarizer input and appends conversation when custom prompt omits {conversation}', async () => {
    const ai = genkit({});
    let capturedPrompt = '';

    const summaryModel = ai.defineModel(
      { name: 'cappedInputSummarizer' },
      async (req) => {
        capturedPrompt = req.messages[0]?.content[0]?.text ?? '';
        return {
          message: { role: 'model', content: [{ text: 'Capped summary' }] },
          finishReason: 'stop',
        };
      }
    );

    const pm = ai.defineModel({ name: 'cappedInputMain' }, async () => ({
      message: { role: 'model', content: [{ text: 'ok' }] },
      usage: { inputTokens: 50 },
    }));

    await ai.generate({
      model: pm,
      messages: [
        { role: 'user', content: [{ text: 'H'.repeat(250_000) }] },
        { role: 'model', content: [{ text: 'T'.repeat(250_000) }] },
        { role: 'user', content: [{ text: 'Recent question' }] },
      ],
      use: [
        contextCompression({
          maxInputTokens: 100,
          summarize: {
            model: summaryModel,
            preserveRecent: 1,
            prompt: 'Custom prompt without placeholder.',
          },
        }),
      ],
    });

    assert.match(capturedPrompt, /^Custom prompt without placeholder\./);
    assert.match(
      capturedPrompt,
      /\.\.\.\[\d+ chars of conversation omitted\]\.\.\./
    );
    assert.ok(capturedPrompt.length < 410_000);
  });

  it('forwards abortSignal, context, and default maxOutputTokens to summarizer and falls back on non-stop or empty summary', async () => {
    const ai = genkit({});
    let capturedSummaryReq: GenerateRequest | undefined;
    let capturedSummaryCtx: Record<string, unknown> | undefined;

    const summaryModel = ai.defineModel(
      { name: 'ctxCheckSummarizer', apiVersion: 'v2' },
      async (req, ctx) => {
        capturedSummaryReq = req;
        capturedSummaryCtx = ctx as unknown as Record<string, unknown>;
        return {
          message: { role: 'model', content: [{ text: 'Valid summary' }] },
          finishReason: 'stop',
        };
      }
    );

    const pm = ai.defineModel({ name: 'ctxCheckMain' }, async () => ({
      message: { role: 'model', content: [{ text: 'done' }] },
      usage: { inputTokens: 50 },
    }));

    const controller = new AbortController();
    await ai.generate({
      model: pm,
      abortSignal: controller.signal,
      context: { auth: { uid: 'user-123' } },
      messages: [
        { role: 'user', content: [{ text: 'u1 ' + 'X'.repeat(200) }] },
        { role: 'model', content: [{ text: 'm1 ' + 'X'.repeat(200) }] },
        { role: 'user', content: [{ text: 'u2' }] },
      ],
      use: [
        contextCompression({
          maxInputTokens: 20,
          summarize: {
            model: summaryModel,
            preserveRecent: 1,
          },
        }),
      ],
    });

    assert.strictEqual(
      (capturedSummaryReq?.config as Record<string, unknown> | undefined)
        ?.maxOutputTokens,
      4096
    );
    assert.deepStrictEqual(
      (capturedSummaryCtx?.context as Record<string, unknown> | undefined)
        ?.auth,
      { uid: 'user-123' }
    );
    assert.ok(capturedSummaryCtx?.abortSignal);

    // Now test fallback when summarizer returns finishReason: 'blocked' or empty text
    const blockedSummarizer = ai.defineModel(
      { name: 'blockedSummarizer' },
      async () => ({
        message: { role: 'model', content: [{ text: 'Partial' }] },
        finishReason: 'blocked',
      })
    );

    let fallbackCapturedReq: GenerateRequest | undefined;
    const pmFallback = ai.defineModel({ name: 'fallbackMain' }, async (req) => {
      fallbackCapturedReq = req;
      return {
        message: { role: 'model', content: [{ text: 'ok' }] },
        usage: { inputTokens: 50 },
      };
    });

    await ai.generate({
      model: pmFallback,
      messages: [
        { role: 'user', content: [{ text: 'u1 ' + 'X'.repeat(200) }] },
        { role: 'model', content: [{ text: 'm1 ' + 'X'.repeat(200) }] },
        { role: 'user', content: [{ text: 'u2' }] },
        { role: 'model', content: [{ text: 'm2' }] },
        { role: 'user', content: [{ text: 'u3' }] },
      ],
      use: [
        contextCompression({
          maxInputTokens: 100,
          preserveRecent: 3,
          insertTruncationNotice: false,
          summarize: {
            model: blockedSummarizer,
            preserveRecent: 3,
          },
        }),
      ],
    });

    // Summarization failed due to finishReason: 'blocked', so message truncation
    // fallback preserved the 3 recent user-anchored messages ([u2, m2, u3]).
    assert.strictEqual(fallbackCapturedReq?.messages.length, 3);
    assert.strictEqual(fallbackCapturedReq?.messages[0].content[0].text, 'u2');
  });

  it('does not drop a newly generated summary when summarize and maxMessages are both configured', async () => {
    const ai = genkit({});
    let capturedRequest: GenerateRequest | undefined;

    const summaryModel = ai.defineModel(
      { name: 'coexistSummarizer' },
      async () => ({
        message: {
          role: 'model',
          content: [{ text: 'Preserved summary text' }],
        },
        finishReason: 'stop',
      })
    );

    const pm = ai.defineModel({ name: 'coexistMain' }, async (req) => {
      capturedRequest = req;
      return {
        message: { role: 'model', content: [{ text: 'done' }] },
        usage: { inputTokens: 50 },
      };
    });

    await ai.generate({
      model: pm,
      messages: Array.from({ length: 10 }, (_, i) => ({
        role: i % 2 === 0 ? ('user' as const) : ('model' as const),
        content: [{ text: `Turn ${i}: ${'X'.repeat(50)}` }],
      })),
      use: [
        contextCompression({
          maxInputTokens: 150,
          maxMessages: 6,
          summarize: {
            model: summaryModel,
            preserveRecent: 6,
          },
        }),
      ],
    });

    const msgs = capturedRequest!.messages;
    // Total messages must respect maxMessages: 6 and keep the summary at index 0
    assert.strictEqual(msgs.length, 6);
    assert.match(msgs[0].content[0].text ?? '', /Preserved summary text/);
  });

  it('truncates older messages to preserveRecent when maxInputTokens is exceeded without summarize or maxMessages', async () => {
    const ai = genkit({});
    let capturedRequest: GenerateRequest | undefined;

    const pm = ai.defineModel({ name: 'preserveRecentOnly' }, async (req) => {
      capturedRequest = req;
      return {
        message: { role: 'model', content: [{ text: 'done' }] },
        usage: { inputTokens: 50 },
      };
    });

    await ai.generate({
      model: pm,
      messages: [
        { role: 'user', content: [{ text: 'u1 ' + 'X'.repeat(100) }] },
        { role: 'model', content: [{ text: 'm1 ' + 'X'.repeat(100) }] },
        { role: 'user', content: [{ text: 'u2 ' + 'X'.repeat(100) }] },
        { role: 'model', content: [{ text: 'm2 ' + 'X'.repeat(100) }] },
        { role: 'user', content: [{ text: 'u3' }] },
        { role: 'model', content: [{ text: 'm3' }] },
        { role: 'user', content: [{ text: 'u4' }] },
      ],
      use: [
        contextCompression({
          maxInputTokens: 100,
          preserveRecent: 3,
          insertTruncationNotice: false,
        }),
      ],
    });

    const msgs = capturedRequest!.messages;
    assert.strictEqual(msgs.length, 3);
    assert.strictEqual(msgs[0].content[0].text, 'u3');
    assert.strictEqual(msgs[1].content[0].text, 'm3');
    assert.strictEqual(msgs[2].content[0].text, 'u4');
  });
});
