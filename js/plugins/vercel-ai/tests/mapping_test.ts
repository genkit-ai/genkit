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

import assert from 'node:assert';
import { describe, it } from 'node:test';
import {
  extractResolvedToolResults,
  findLastUserMessage,
  mapGenkitMessageToUI,
  mapGenkitPartToUI,
  mapUIMessageToGenkit,
  mapUIPartToGenkit,
  type UIMessage,
} from '../src/mapping.js';

// ---------------------------------------------------------------------------
// UIMessage → Genkit
// ---------------------------------------------------------------------------

describe('mapUIPartToGenkit', () => {
  it('maps text parts', () => {
    const result = mapUIPartToGenkit({ type: 'text', text: 'hello' });
    assert.deepStrictEqual(result, [{ text: 'hello' }]);
  });

  it('maps tool-invocation (call state)', () => {
    const result = mapUIPartToGenkit({
      type: 'tool-invocation',
      toolInvocation: {
        toolCallId: 'call-1',
        toolName: 'getWeather',
        args: { location: 'London' },
        state: 'call',
      },
    });
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

  it('maps tool-invocation (result state) to request + response', () => {
    const result = mapUIPartToGenkit({
      type: 'tool-invocation',
      toolInvocation: {
        toolCallId: 'call-2',
        toolName: 'getWeather',
        args: { location: 'Paris' },
        state: 'result',
        result: { weather: 'sunny' },
      },
    });
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
    assert.deepStrictEqual(mapUIPartToGenkit({ type: 'reasoning' }), []);
    assert.deepStrictEqual(mapUIPartToGenkit({ type: 'step-start' }), []);
    assert.deepStrictEqual(mapUIPartToGenkit({ type: 'source-url' }), []);
  });
});

describe('mapUIMessageToGenkit', () => {
  it('maps user message with text parts', () => {
    const msg: UIMessage = {
      id: 'msg-1',
      role: 'user',
      content: '',
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
      content: 'Hi there',
      parts: [{ type: 'text', text: 'Hi there' }],
    };
    const result = mapUIMessageToGenkit(msg);
    assert.strictEqual(result.role, 'model');
  });

  it('falls back to content string when parts is empty', () => {
    const msg: UIMessage = {
      id: 'msg-3',
      role: 'user',
      content: 'Legacy content',
      parts: [],
    };
    const result = mapUIMessageToGenkit(msg);
    assert.deepStrictEqual(result.content, [{ text: 'Legacy content' }]);
  });
});

// ---------------------------------------------------------------------------
// Genkit → UIMessage
// ---------------------------------------------------------------------------

describe('mapGenkitPartToUI', () => {
  it('maps text', () => {
    assert.deepStrictEqual(mapGenkitPartToUI({ text: 'hello' }), {
      type: 'text',
      text: 'hello',
    });
  });

  it('maps toolRequest', () => {
    const result = mapGenkitPartToUI({
      toolRequest: { ref: 'c1', name: 'myTool', input: { x: 1 } },
    });
    assert.strictEqual(result?.type, 'tool-invocation');
    assert.strictEqual(result?.toolInvocation?.toolCallId, 'c1');
    assert.strictEqual(result?.toolInvocation?.toolName, 'myTool');
    assert.strictEqual(result?.toolInvocation?.state, 'call');
  });

  it('maps toolResponse', () => {
    const result = mapGenkitPartToUI({
      toolResponse: { ref: 'c1', name: 'myTool', output: { y: 2 } },
    });
    assert.strictEqual(result?.type, 'tool-invocation');
    assert.strictEqual(result?.toolInvocation?.state, 'result');
    assert.deepStrictEqual(result?.toolInvocation?.result, { y: 2 });
  });

  it('maps media', () => {
    const result = mapGenkitPartToUI({
      media: { url: 'https://a.com/b.png', contentType: 'image/png' },
    });
    assert.deepStrictEqual(result, {
      type: 'file',
      url: 'https://a.com/b.png',
      mediaType: 'image/png',
    });
  });

  it('returns null for data/custom parts', () => {
    assert.strictEqual(mapGenkitPartToUI({ data: { foo: 1 } }), null);
  });
});

describe('mapGenkitMessageToUI', () => {
  it('maps model message to assistant', () => {
    const result = mapGenkitMessageToUI(
      { role: 'model', content: [{ text: 'Hi' }] },
      'test-id'
    );
    assert.strictEqual(result.role, 'assistant');
    assert.strictEqual(result.id, 'test-id');
    assert.strictEqual(result.content, 'Hi');
    assert.strictEqual(result.parts.length, 1);
  });
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

describe('extractResolvedToolResults', () => {
  it('extracts results from the last assistant message', () => {
    const messages: UIMessage[] = [
      { id: '1', role: 'user', content: 'do it', parts: [] },
      {
        id: '2',
        role: 'assistant',
        content: '',
        parts: [
          {
            type: 'tool-invocation',
            toolInvocation: {
              toolCallId: 'call-x',
              toolName: 'approve',
              args: { action: 'transfer' },
              state: 'result',
              result: { approved: true },
            },
          },
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
      { id: '1', role: 'user', content: 'hi', parts: [] },
    ];
    assert.strictEqual(extractResolvedToolResults(messages).length, 0);
  });

  it('ignores unresolved tool invocations', () => {
    const messages: UIMessage[] = [
      {
        id: '2',
        role: 'assistant',
        content: '',
        parts: [
          {
            type: 'tool-invocation',
            toolInvocation: {
              toolCallId: 'call-y',
              toolName: 'approve',
              args: {},
              state: 'call',
            },
          },
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
        content: '',
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
        content: '',
        parts: [
          {
            type: 'tool-userApproval',
            toolCallId: 'tc-2',
            // No toolName property — SDK v6 ToolUIPart doesn't have one
            state: 'output-available',
            output: { approved: false, feedback: 'User rejected' },
          } as unknown as UIMessage['parts'][number],
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
        content: '',
        parts: [
          {
            type: 'tool-getWeather',
            toolCallId: 'tc-3',
            state: 'input-available',
            input: { location: 'NYC' },
          } as unknown as UIMessage['parts'][number],
        ],
      },
    ];
    assert.strictEqual(extractResolvedToolResults(messages).length, 0);
  });
});

describe('findLastUserMessage', () => {
  it('finds the last user message', () => {
    const messages: UIMessage[] = [
      { id: '1', role: 'user', content: 'first', parts: [] },
      { id: '2', role: 'assistant', content: 'reply', parts: [] },
      { id: '3', role: 'user', content: 'second', parts: [] },
    ];
    const result = findLastUserMessage(messages);
    assert.strictEqual(result?.id, '3');
  });

  it('returns undefined when no user messages', () => {
    const messages: UIMessage[] = [
      { id: '1', role: 'assistant', content: 'hi', parts: [] },
    ];
    assert.strictEqual(findLastUserMessage(messages), undefined);
  });
});
