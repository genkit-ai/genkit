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

import type { UIMessage } from 'ai';
import type { MessageData } from 'genkit';
import assert from 'node:assert';
import { describe, it } from 'node:test';
import {
  extractResolvedToolResults,
  findLastUserMessage,
  mapGenkitMessageToUI,
  mapUIMessageToGenkit,
  mapUIPartToGenkit,
  messagesFromSnapshot,
} from '../src/mapping.js';

// ---------------------------------------------------------------------------
// UIMessage → Genkit
// ---------------------------------------------------------------------------

describe('mapUIPartToGenkit', () => {
  it('maps text parts', () => {
    const result = mapUIPartToGenkit({ type: 'text', text: 'hello' });
    assert.deepStrictEqual(result, [{ text: 'hello' }]);
  });

  it('maps v6 tool part (input-available state)', () => {
    const result = mapUIPartToGenkit({
      type: 'tool-getWeather',
      toolCallId: 'call-1',
      state: 'input-available',
      input: { location: 'London' },
    } as UIMessage['parts'][number]);
    assert.deepStrictEqual(result, [
      {
        toolRequest: {
          ref: 'call-1',
          name: 'getWeather',
          input: { location: 'London' },
        },
      },
    ]);
  });

  it('maps v6 tool part (output-available state) to request + response', () => {
    const result = mapUIPartToGenkit({
      type: 'tool-getWeather',
      toolCallId: 'call-2',
      state: 'output-available',
      input: { location: 'Paris' },
      output: { weather: 'sunny' },
    } as UIMessage['parts'][number]);
    assert.strictEqual(result.length, 2);
    assert.deepStrictEqual(result[0], {
      toolRequest: {
        ref: 'call-2',
        name: 'getWeather',
        input: { location: 'Paris' },
      },
    });
    assert.deepStrictEqual(result[1], {
      toolResponse: {
        ref: 'call-2',
        name: 'getWeather',
        output: { weather: 'sunny' },
      },
    });
  });

  it('maps dynamic-tool part', () => {
    const result = mapUIPartToGenkit({
      type: 'dynamic-tool',
      toolName: 'myTool',
      toolCallId: 'dc-1',
      state: 'input-available',
      input: { x: 1 },
    } as UIMessage['parts'][number]);
    assert.deepStrictEqual(result, [
      {
        toolRequest: {
          ref: 'dc-1',
          name: 'myTool',
          input: { x: 1 },
        },
      },
    ]);
  });

  it('maps file parts', () => {
    const result = mapUIPartToGenkit({
      type: 'file',
      url: 'https://example.com/img.png',
      mediaType: 'image/png',
    });
    assert.deepStrictEqual(result, [
      {
        media: { url: 'https://example.com/img.png', contentType: 'image/png' },
      },
    ]);
  });

  it('maps reasoning parts to Genkit reasoning parts', () => {
    const result = mapUIPartToGenkit({
      type: 'reasoning',
      text: 'Let me think about this...',
    } as UIMessage['parts'][number]);
    assert.deepStrictEqual(result, [
      { reasoning: 'Let me think about this...' },
    ]);
  });

  it('drops reasoning parts with empty text', () => {
    assert.deepStrictEqual(
      mapUIPartToGenkit({
        type: 'reasoning',
        text: '',
      } as UIMessage['parts'][number]),
      []
    );
  });

  it('returns empty for unsupported types', () => {
    assert.deepStrictEqual(
      mapUIPartToGenkit({ type: 'step-start' } as UIMessage['parts'][number]),
      []
    );
  });

  it('preserves part metadata on text parts (UI → Genkit)', () => {
    const result = mapUIPartToGenkit({
      type: 'text',
      text: 'hello',
      metadata: { foo: 'bar' },
    } as unknown as UIMessage['parts'][number]);
    assert.deepStrictEqual(result, [
      { text: 'hello', metadata: { foo: 'bar' } },
    ]);
  });

  it('preserves part metadata on tool parts (request + response)', () => {
    const result = mapUIPartToGenkit({
      type: 'tool-getWeather',
      toolCallId: 'call-9',
      state: 'output-available',
      input: { location: 'Paris' },
      output: { weather: 'sunny' },
      metadata: { filesystemMiddlewareTool: true },
    } as unknown as UIMessage['parts'][number]);
    assert.strictEqual(result.length, 2);
    assert.deepStrictEqual(result[0].metadata, {
      filesystemMiddlewareTool: true,
    });
    assert.deepStrictEqual(result[1].metadata, {
      filesystemMiddlewareTool: true,
    });
  });

  it('ignores non-object metadata', () => {
    const result = mapUIPartToGenkit({
      type: 'text',
      text: 'hello',
      metadata: 'not-an-object',
    } as unknown as UIMessage['parts'][number]);
    assert.deepStrictEqual(result, [{ text: 'hello' }]);
  });
});

describe('mapUIMessageToGenkit', () => {
  it('maps user message with text parts', () => {
    const msg: UIMessage = {
      id: 'msg-1',
      role: 'user',
      parts: [{ type: 'text', text: 'Hello' }],
    };
    const result = mapUIMessageToGenkit(msg);
    assert.strictEqual(result.role, 'user');
    assert.deepStrictEqual(result.content, [{ text: 'Hello' }]);
  });

  it('maps assistant role to model', () => {
    const msg: UIMessage = {
      id: 'msg-2',
      role: 'assistant',
      parts: [{ type: 'text', text: 'Hi there' }],
    };
    const result = mapUIMessageToGenkit(msg);
    assert.strictEqual(result.role, 'model');
  });

  it('returns empty content when parts is empty', () => {
    const msg: UIMessage = {
      id: 'msg-3',
      role: 'user',
      parts: [],
    };
    const result = mapUIMessageToGenkit(msg);
    assert.deepStrictEqual(result.content, []);
  });
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

describe('extractResolvedToolResults', () => {
  it('extracts results from the last assistant message', () => {
    const messages: UIMessage[] = [
      { id: '1', role: 'user', parts: [{ type: 'text', text: 'do it' }] },
      {
        id: '2',
        role: 'assistant',
        parts: [
          {
            type: 'tool-approve',
            toolCallId: 'call-x',
            state: 'output-available',
            input: { action: 'transfer' },
            output: { approved: true },
          } as UIMessage['parts'][number],
        ],
      },
    ];
    const results = extractResolvedToolResults(messages);
    assert.strictEqual(results.length, 1);
    assert.strictEqual(results[0].toolCallId, 'call-x');
    assert.strictEqual(results[0].toolName, 'approve');
    assert.deepStrictEqual(results[0].result, { approved: true });
  });

  it('returns empty when no results', () => {
    const messages: UIMessage[] = [
      { id: '1', role: 'user', parts: [{ type: 'text', text: 'hi' }] },
    ];
    assert.strictEqual(extractResolvedToolResults(messages).length, 0);
  });

  it('ignores tool parts without output-available state', () => {
    const messages: UIMessage[] = [
      {
        id: '2',
        role: 'assistant',
        parts: [
          {
            type: 'tool-approve',
            toolCallId: 'call-y',
            state: 'input-available',
            input: {},
          } as UIMessage['parts'][number],
        ],
      },
    ];
    assert.strictEqual(extractResolvedToolResults(messages).length, 0);
  });

  it('extracts v6 per-tool format with explicit toolName', () => {
    const messages: UIMessage[] = [
      {
        id: '1',
        role: 'assistant',
        parts: [
          {
            type: 'tool-userApproval',
            toolCallId: 'tc-1',
            toolName: 'userApproval',
            state: 'output-available',
            output: { approved: true },
          } as unknown as UIMessage['parts'][number],
        ],
      },
    ];
    const results = extractResolvedToolResults(messages);
    assert.strictEqual(results.length, 1);
    assert.strictEqual(results[0].toolCallId, 'tc-1');
    assert.strictEqual(results[0].toolName, 'userApproval');
    assert.deepStrictEqual(results[0].result, { approved: true });
  });

  it('derives toolName from type field when toolName is absent (SDK v6)', () => {
    const messages: UIMessage[] = [
      {
        id: '1',
        role: 'assistant',
        parts: [
          {
            type: 'tool-userApproval',
            toolCallId: 'tc-2',
            // No toolName property — SDK v6 ToolUIPart doesn't have one
            state: 'output-available',
            output: { approved: false, feedback: 'User rejected' },
          } as UIMessage['parts'][number],
        ],
      },
    ];
    const results = extractResolvedToolResults(messages);
    assert.strictEqual(results.length, 1);
    assert.strictEqual(results[0].toolCallId, 'tc-2');
    assert.strictEqual(results[0].toolName, 'userApproval');
    assert.deepStrictEqual(results[0].result, {
      approved: false,
      feedback: 'User rejected',
    });
  });

  it('ignores v6 tool parts without output-available state', () => {
    const messages: UIMessage[] = [
      {
        id: '1',
        role: 'assistant',
        parts: [
          {
            type: 'tool-getWeather',
            toolCallId: 'tc-3',
            state: 'input-available',
            input: { location: 'NYC' },
          } as UIMessage['parts'][number],
        ],
      },
    ];
    assert.strictEqual(extractResolvedToolResults(messages).length, 0);
  });

  it('extracts dynamic-tool parts (with explicit toolName)', () => {
    const messages: UIMessage[] = [
      {
        id: '1',
        role: 'assistant',
        parts: [
          {
            type: 'dynamic-tool',
            toolName: 'userApproval',
            toolCallId: 'dyn-1',
            state: 'output-available',
            input: { action: 'transfer' },
            output: { approved: true },
          } as unknown as UIMessage['parts'][number],
        ],
      },
    ];
    const results = extractResolvedToolResults(messages);
    assert.strictEqual(results.length, 1);
    assert.strictEqual(results[0].toolCallId, 'dyn-1');
    assert.strictEqual(results[0].toolName, 'userApproval');
    assert.deepStrictEqual(results[0].input, { action: 'transfer' });
    assert.deepStrictEqual(results[0].result, { approved: true });
  });
});

describe('findLastUserMessage', () => {
  it('finds the last user message', () => {
    const messages: UIMessage[] = [
      { id: '1', role: 'user', parts: [{ type: 'text', text: 'first' }] },
      {
        id: '2',
        role: 'assistant',
        parts: [{ type: 'text', text: 'reply' }],
      },
      { id: '3', role: 'user', parts: [{ type: 'text', text: 'second' }] },
    ];
    const result = findLastUserMessage(messages);
    assert.strictEqual(result?.id, '3');
  });

  it('returns undefined when no user messages', () => {
    const messages: UIMessage[] = [
      { id: '1', role: 'assistant', parts: [{ type: 'text', text: 'hi' }] },
    ];
    assert.strictEqual(findLastUserMessage(messages), undefined);
  });
});

// ---------------------------------------------------------------------------
// Genkit → UIMessage (session restore)
// ---------------------------------------------------------------------------

describe('mapGenkitMessageToUI', () => {
  it('maps a model text message to an assistant UIMessage', () => {
    const msg: MessageData = {
      role: 'model',
      content: [{ text: 'Hello there' }],
    };
    const ui = mapGenkitMessageToUI(msg, undefined, 'm-1');
    assert.strictEqual(ui.id, 'm-1');
    assert.strictEqual(ui.role, 'assistant');
    assert.deepStrictEqual(ui.parts, [{ type: 'text', text: 'Hello there' }]);
  });

  it('maps a user message', () => {
    const ui = mapGenkitMessageToUI(
      { role: 'user', content: [{ text: 'hi' }] },
      undefined,
      'm-2'
    );
    assert.strictEqual(ui.role, 'user');
    assert.deepStrictEqual(ui.parts, [{ type: 'text', text: 'hi' }]);
  });

  it('maps reasoning and media parts', () => {
    const ui = mapGenkitMessageToUI(
      {
        role: 'model',
        content: [
          { reasoning: 'thinking' },
          {
            media: {
              url: 'data:image/png;base64,xxx',
              contentType: 'image/png',
            },
          },
        ],
      },
      undefined,
      'm-3'
    );
    assert.deepStrictEqual(ui.parts, [
      { type: 'reasoning', text: 'thinking' },
      {
        type: 'file',
        url: 'data:image/png;base64,xxx',
        mediaType: 'image/png',
      },
    ]);
  });

  it('drops empty reasoning parts', () => {
    const ui = mapGenkitMessageToUI(
      { role: 'model', content: [{ reasoning: '' }, { text: 'answer' }] },
      undefined,
      'm-4'
    );
    assert.deepStrictEqual(ui.parts, [{ type: 'text', text: 'answer' }]);
  });

  it('emits an unresolved tool request as input-available', () => {
    const ui = mapGenkitMessageToUI(
      {
        role: 'model',
        content: [
          {
            toolRequest: {
              ref: 'r-1',
              name: 'getWeather',
              input: { city: 'NYC' },
            },
          },
        ],
      },
      undefined,
      'm-5'
    );
    assert.deepStrictEqual(ui.parts, [
      {
        type: 'tool-getWeather',
        toolCallId: 'r-1',
        input: { city: 'NYC' },
        state: 'input-available',
      },
    ]);
  });

  it('pairs a tool request with its response into output-available', () => {
    const responses = new Map<string, unknown>([['r-2', { temp: 72 }]]);
    const ui = mapGenkitMessageToUI(
      {
        role: 'model',
        content: [
          {
            toolRequest: {
              ref: 'r-2',
              name: 'getWeather',
              input: { city: 'NYC' },
            },
          },
        ],
      },
      responses,
      'm-6'
    );
    assert.deepStrictEqual(ui.parts, [
      {
        type: 'tool-getWeather',
        toolCallId: 'r-2',
        input: { city: 'NYC' },
        state: 'output-available',
        output: { temp: 72 },
      },
    ]);
  });

  it('preserves part metadata (Genkit → UI)', () => {
    const ui = mapGenkitMessageToUI(
      {
        role: 'model',
        content: [{ text: 'hi', metadata: { foo: 'bar' } }],
      },
      undefined,
      'm-7'
    );
    assert.deepStrictEqual(ui.parts, [
      { type: 'text', text: 'hi', metadata: { foo: 'bar' } },
    ]);
  });
});

describe('messagesFromSnapshot', () => {
  it('maps a simple user/model conversation', () => {
    const messages: MessageData[] = [
      { role: 'user', content: [{ text: 'hello' }] },
      { role: 'model', content: [{ text: 'hi there' }] },
    ];
    const ui = messagesFromSnapshot(messages);
    assert.strictEqual(ui.length, 2);
    assert.strictEqual(ui[0].role, 'user');
    assert.strictEqual(ui[1].role, 'assistant');
    assert.deepStrictEqual(ui[1].parts, [{ type: 'text', text: 'hi there' }]);
  });

  it('merges tool responses (in separate tool messages) into request parts', () => {
    const messages: MessageData[] = [
      { role: 'user', content: [{ text: 'weather?' }] },
      {
        role: 'model',
        content: [
          {
            toolRequest: {
              ref: 'r-1',
              name: 'getWeather',
              input: { city: 'NYC' },
            },
          },
        ],
      },
      {
        role: 'tool',
        content: [
          {
            toolResponse: {
              ref: 'r-1',
              name: 'getWeather',
              output: { temp: 72 },
            },
          },
        ],
      },
      { role: 'model', content: [{ text: 'It is 72F.' }] },
    ];
    const ui = messagesFromSnapshot(messages);
    // tool-role message is omitted; its response is merged into the request.
    assert.strictEqual(ui.length, 3);
    assert.strictEqual(ui[0].role, 'user');
    assert.deepStrictEqual(ui[1].parts, [
      {
        type: 'tool-getWeather',
        toolCallId: 'r-1',
        input: { city: 'NYC' },
        state: 'output-available',
        output: { temp: 72 },
      },
    ]);
    assert.deepStrictEqual(ui[2].parts, [{ type: 'text', text: 'It is 72F.' }]);
  });

  it('emits an unresolved interrupt as input-available', () => {
    const messages: MessageData[] = [
      { role: 'user', content: [{ text: 'transfer $100' }] },
      {
        role: 'model',
        content: [
          {
            toolRequest: {
              ref: 'r-9',
              name: 'userApproval',
              input: { amount: 100 },
            },
          },
        ],
      },
    ];
    const ui = messagesFromSnapshot(messages);
    assert.strictEqual(ui.length, 2);
    assert.deepStrictEqual(ui[1].parts, [
      {
        type: 'tool-userApproval',
        toolCallId: 'r-9',
        input: { amount: 100 },
        state: 'input-available',
      },
    ]);
  });

  it('skips messages that produce no renderable parts', () => {
    const messages: MessageData[] = [
      { role: 'user', content: [{ text: 'hi' }] },
      // A tool-only message becomes empty after merging — and is skipped.
      {
        role: 'tool',
        content: [{ toolResponse: { ref: 'x', name: 'foo', output: {} } }],
      },
    ];
    const ui = messagesFromSnapshot(messages);
    assert.strictEqual(ui.length, 1);
    assert.strictEqual(ui[0].role, 'user');
  });

  it('assigns stable, unique ids', () => {
    const ui = messagesFromSnapshot([
      { role: 'user', content: [{ text: 'a' }] },
      { role: 'model', content: [{ text: 'b' }] },
    ]);
    assert.strictEqual(ui[0].id, 'restored-0');
    assert.strictEqual(ui[1].id, 'restored-1');
  });
});
