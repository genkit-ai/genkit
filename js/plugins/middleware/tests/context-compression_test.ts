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
import { contextCompression } from '../src/context-compression.js';

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
      /\[TRUNCATED:/
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

    assert.match(String(firstToolOutput), /\[TRUNCATED:/);
    assert.strictEqual(String(secondToolOutput).includes('[TRUNCATED:'), false);
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
          maxMessages: 3,
          insertTruncationNotice: true,
        }),
      ],
    });

    assert.strictEqual(response.text, 'done');
    const msgs = capturedRequest!.messages;
    // Notice message + 2 kept messages = 3 total messages
    assert.strictEqual(msgs.length, 3);
    assert.match(msgs[0].content[0].text!, /\[NOTE\] Some earlier messages/);
    assert.strictEqual(msgs[1].content[0].text, 'msg 4');
    assert.strictEqual(msgs[2].content[0].text, 'msg 5');
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
          preserveRecent: 0,
          insertTruncationNotice: false,
        }),
      ],
    })) as any;

    assert.strictEqual(response.text, 'done');
    assert.ok(response.custom?.contextCompression);
    assert.strictEqual(response.custom.contextCompression.triggered, true);
    assert.strictEqual(response.custom.contextCompression.messagesOriginal, 3);
    assert.strictEqual(response.custom.contextCompression.messagesAfter, 2);
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
    // Turn 1 (outer generate) did not compress, so strict per-turn isolation ensures custom is clean
    assert.strictEqual(response.custom?.contextCompression, undefined);
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
      preserveRecent: 0,
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
    assert.strictEqual(resp1.custom.contextCompression.messagesAfter, 2);
    assert.strictEqual(resp2.custom?.contextCompression, undefined);
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
              { toolRequest: { name: 'search', input: { query: 'first-query' } } },
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
              { toolRequest: { name: 'search', input: { query: 'second-query' } } },
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
    assert.strictEqual(String(toolMessages[0].content[0].toolResponse?.output).includes('Deduplicated'), false);
    assert.strictEqual(String(toolMessages[1].content[0].toolResponse?.output).includes('Deduplicated'), false);
  });

  it('summarizes older messages using summary model', async () => {
    const ai = genkit({});
    let turn = 0;
    const capturedRequests: GenerateRequest[] = [];

    const summaryModel = ai.defineModel(
      { name: 'summaryModel' },
      async () => ({
        message: {
          role: 'model',
          content: [{ text: 'Summary of past events: steps were executed.' }],
        },
      })
    );

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
});
