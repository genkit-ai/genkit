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
import assert from 'node:assert';
import { describe, it } from 'node:test';
import {
  extractResolvedToolResults,
  findLastUserMessage,
  mapUIMessageToGenkit,
  mapUIPartToGenkit,
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

  it('returns empty for unsupported types', () => {
    assert.deepStrictEqual(
      mapUIPartToGenkit({ type: 'reasoning', text: '' } as UIMessage['parts'][number]),
      []
    );
    assert.deepStrictEqual(
      mapUIPartToGenkit({ type: 'step-start' } as UIMessage['parts'][number]),
      []
    );
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
