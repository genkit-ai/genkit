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

import { genkit, z, type GenerateRequest } from 'genkit';
import assert from 'node:assert';
import { describe, it } from 'node:test';
import {
  contextCompression,
  resolveCompressedHistory,
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
    // effectiveMaxMessages = max(2+1, 6 - (4 - 2)) = 4
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

    // Truncation should have clamped to 4 messages rather than 6 due to severe overshoot
    assert.strictEqual(capturedRequest?.messages.length, 4);
  });

  it('attaches compressedHistory to message metadata and keeps original history in request.messages when preserveOriginalMessages: true (default)', async () => {
    const ai = genkit({});
    const longText = 'TOOL_OUTPUT_'.repeat(50);
    let modelReceivedMessages: any[] = [];

    const pm = ai.defineModel(
      { name: 'compressedHistoryModel' },
      async (req) => {
        modelReceivedMessages = req.messages;
        return {
          message: { role: 'model', content: [{ text: 'done' }] },
          usage: { inputTokens: 50 },
        };
      }
    );

    const response = (await ai.generate({
      model: pm,
      messages: [
        { role: 'user', content: [{ text: 'run heavy tool' }] },
        {
          role: 'model',
          content: [{ toolRequest: { name: 'heavyTool', input: {} } }],
        },
        {
          role: 'tool',
          content: [{ toolResponse: { name: 'heavyTool', output: longText } }],
        },
      ],
      use: [
        contextCompression({
          maxInputTokens: 100,
          toolResponses: { maxChars: 10, preserveRecent: 0 },
        }),
      ],
    })) as any;

    assert.strictEqual(response.text, 'done');
    assert.ok(response.custom?.contextCompression);
    assert.strictEqual(response.custom.contextCompression.triggered, true);

    // 1. Check model received messages: The tool message received by the model WAS compressed/truncated
    assert.strictEqual(modelReceivedMessages.length, 3);
    const modelToolPart = modelReceivedMessages[2].content[0].toolResponse;
    assert.ok(modelToolPart.output.includes('[Truncated '));

    // 2. Check response.request.messages: Contains original full uncompressed tool text with metadata.compressedHistory
    const reqMessages = response.request.messages;
    assert.strictEqual(reqMessages.length, 3);
    assert.strictEqual(reqMessages[2].content[0].toolResponse.output, longText);
    assert.ok(reqMessages[2].metadata?.compressedHistory);
    assert.strictEqual(
      (reqMessages[2].metadata.compressedHistory as any[])[2].content[0]
        .toolResponse.output.length < longText.length,
      true
    );

    // 3. Check response.messages: Contains all 3 original messages + the 4th model response
    assert.strictEqual(response.messages.length, 4);
    assert.strictEqual(
      response.messages[2].content[0].toolResponse.output,
      longText
    );
  });

  it('attaches compressedHistory on cut message during summarization and resolves cleanly in model hook', async () => {
    const ai = genkit({});
    const longPrompt = 'original user research prompt '.repeat(20);
    let modelReceivedMessages: any[] = [];

    const summaryModel = ai.defineModel(
      { name: 'mockSummaryModel' },
      async () => ({
        message: {
          role: 'model',
          content: [{ text: 'MOCK_SUMMARY_TEXT' }],
        },
      })
    );

    const pm = ai.defineModel(
      { name: 'summarizeResolutionModel' },
      async (req) => {
        modelReceivedMessages = req.messages;
        return {
          message: { role: 'model', content: [{ text: 'done all' }] },
          usage: { inputTokens: 50 },
        };
      }
    );

    const response = (await ai.generate({
      model: pm,
      messages: [
        { role: 'user', content: [{ text: longPrompt }] },
        {
          role: 'model',
          content: [{ toolRequest: { name: 'tool1', input: {} } }],
        },
        {
          role: 'tool',
          content: [
            { toolResponse: { name: 'tool1', output: 'TOOL_1_RESULT' } },
          ],
        },
        { role: 'user', content: [{ text: 'followup question' }] },
      ],
      use: [
        contextCompression({
          maxInputTokens: 100,
          summarize: {
            model: summaryModel,
            preserveRecent: 1,
          },
        }),
      ],
    })) as any;

    assert.strictEqual(response.text, 'done all');
    assert.ok(response.custom?.contextCompression);
    assert.strictEqual(response.custom.contextCompression.summarized, true);

    // Model should receive: Summary Message + last preserved message ('followup question')
    assert.strictEqual(modelReceivedMessages.length, 2);
    assert.ok(
      modelReceivedMessages[0].content[0].text.includes('MOCK_SUMMARY_TEXT')
    );
    assert.strictEqual(
      modelReceivedMessages[1].content[0].text,
      'followup question'
    );

    // request.messages retains all 4 original messages with metadata.compressedHistory on cut message (index 2)
    const reqMsgs = response.request.messages;
    assert.strictEqual(reqMsgs.length, 4);
    assert.strictEqual(reqMsgs[0].content[0].text, longPrompt);
    assert.ok(reqMsgs[2].metadata?.compressedHistory);
    assert.strictEqual(
      (reqMsgs[2].metadata.compressedHistory as any[]).length,
      1
    );
  });

  it('overwrites request.messages with compressed messages when preserveOriginalMessages: false', async () => {
    const ai = genkit({});
    let modelReceivedMessages: any[] = [];

    const pm = ai.defineModel(
      { name: 'disabledPreserveModel' },
      async (req) => {
        modelReceivedMessages = req.messages;
        return {
          message: { role: 'model', content: [{ text: 'done' }] },
          usage: { inputTokens: 50 },
        };
      }
    );

    const response = (await ai.generate({
      model: pm,
      messages: [
        { role: 'user', content: [{ text: 'A'.repeat(500) }] },
        { role: 'model', content: [{ text: 'response 1' }] },
        { role: 'user', content: [{ text: 'user 2' }] },
      ],
      use: [
        contextCompression({
          maxInputTokens: 50,
          preserveOriginalMessages: false,
          maxMessages: 1,
          preserveRecent: 0,
          insertTruncationNotice: false,
        }),
      ],
    })) as any;

    assert.strictEqual(response.text, 'done');
    assert.ok(response.custom?.contextCompression);
    assert.strictEqual(response.custom.contextCompression.triggered, true);

    // Model received the compressed history (1 message)
    assert.strictEqual(modelReceivedMessages.length, 1);

    // When preserveOriginalMessages is false and insertTruncationNotice is false, 1 message remains
    assert.strictEqual(response.request.messages.length, 1);
    assert.strictEqual(
      response.request.messages[0].content[0].text,
      'user 2'
    );
  });

  it('prevents stale compressedHistory from shadowing newer compressions in multi-turn history', async () => {
    const ai = genkit({});
    let modelReceivedMessages: any[] = [];

    let summaryCount = 0;
    const summaryModel = ai.defineModel(
      { name: 'staleShadowSummaryModel' },
      async () => {
        summaryCount++;
        return {
          message: {
            role: 'model',
            content: [{ text: `SUMMARY_${summaryCount}` }],
          },
        };
      }
    );

    const pm = ai.defineModel({ name: 'staleShadowModel' }, async (req) => {
      modelReceivedMessages = req.messages;
      return {
        message: { role: 'model', content: [{ text: 'done' }] },
        usage: { inputTokens: 500 },
      };
    });

    // Simulated Turn 1: 4 messages get summarized down to 2 active messages
    const r1 = await ai.generate({
      model: pm,
      messages: [
        { role: 'user', content: [{ text: 'M1 '.repeat(50) }] },
        { role: 'model', content: [{ text: 'R1 '.repeat(50) }] },
        { role: 'user', content: [{ text: 'M2 '.repeat(50) }] },
        { role: 'model', content: [{ text: 'R2 '.repeat(50) }] },
      ],
      use: [
        contextCompression({
          maxInputTokens: 50,
          summarize: { model: summaryModel, preserveRecent: 1 },
        }),
      ],
    });

    // Turn 2: append new user message onto full uncompressed history returned by r1
    const fullHistoryTurn2 = [
      ...r1.messages,
      { role: 'user' as const, content: [{ text: 'M3 '.repeat(50) }] },
    ];

    const r2 = await ai.generate({
      model: pm,
      messages: fullHistoryTurn2,
      use: [
        contextCompression({
          maxInputTokens: 50,
          summarize: { model: summaryModel, preserveRecent: 1 },
        }),
      ],
    });

    // Model on turn 2 should receive SUMMARY_2 and only the most recent message
    assert.strictEqual(modelReceivedMessages.length, 2);
    assert.ok(modelReceivedMessages[0].content[0].text.includes('SUMMARY_2'));

    // resolveCompressedHistory should resolve to the latest summary, not the stale SUMMARY_1
    const resolved = resolveCompressedHistory(r2.messages);
    assert.strictEqual(resolved.length, 3);
    assert.ok(resolved[0].content[0].text.includes('SUMMARY_2'));
  });
});
