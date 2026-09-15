/**
 * Copyright 2025 Google LLC
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

import * as assert from 'assert';
import type { MessageData } from 'genkit/model';
import { describe, it } from 'node:test';
import {
  mapMessage,
  mapOutputMessage,
  mapPart,
  mapRole,
  normalizeMessages,
} from '../src/genai/gen-ai-message-mapping.js';

describe('mapRole', () => {
  it('maps model to assistant', () => {
    assert.strictEqual(mapRole('model'), 'assistant');
  });
  it('passes other roles through', () => {
    assert.strictEqual(mapRole('user'), 'user');
    assert.strictEqual(mapRole('tool'), 'tool');
  });
});

describe('mapPart', () => {
  it('maps text parts', () => {
    assert.deepStrictEqual(mapPart({ text: 'hi' }), {
      type: 'text',
      content: 'hi',
    });
  });

  it('maps reasoning parts', () => {
    assert.deepStrictEqual(mapPart({ reasoning: 'because' }), {
      type: 'reasoning',
      content: 'because',
    });
  });

  it('maps tool_call parts', () => {
    assert.deepStrictEqual(
      mapPart({
        toolRequest: { ref: 'r1', name: 'getWeather', input: { city: 'SF' } },
      }),
      {
        type: 'tool_call',
        id: 'r1',
        name: 'getWeather',
        arguments: { city: 'SF' },
      }
    );
  });

  it('maps tool_call_response parts', () => {
    assert.deepStrictEqual(
      mapPart({
        toolResponse: { ref: 'r1', name: 'getWeather', output: { temp: 20 } },
      }),
      {
        type: 'tool_call_response',
        id: 'r1',
        response: { temp: 20 },
      }
    );
  });

  it('maps media parts', () => {
    assert.deepStrictEqual(
      mapPart({ media: { url: 'data:...', contentType: 'image/png' } }),
      {
        type: 'media',
        content: 'data:...',
        content_type: 'image/png',
      }
    );
  });

  it('falls back to text for opaque parts', () => {
    const result = mapPart({ custom: { foo: 'bar' } });
    assert.strictEqual(result.type, 'text');
    assert.match(result.content as string, /foo/);
  });
});

describe('normalizeMessages', () => {
  it('splits system instructions from conversation', () => {
    const messages: MessageData[] = [
      { role: 'system', content: [{ text: 'be nice' }] },
      { role: 'user', content: [{ text: 'hello' }] },
      { role: 'model', content: [{ text: 'hi there' }] },
    ];
    const result = normalizeMessages(messages);
    assert.deepStrictEqual(result.systemInstructions, [
      { type: 'text', content: 'be nice' },
    ]);
    assert.strictEqual(result.messages.length, 2);
    assert.strictEqual(result.messages[0].role, 'user');
    assert.strictEqual(result.messages[1].role, 'assistant');
  });
});

describe('mapMessage / mapOutputMessage', () => {
  it('maps a message with parts', () => {
    assert.deepStrictEqual(
      mapMessage({ role: 'user', content: [{ text: 'hi' }] }),
      { role: 'user', parts: [{ type: 'text', content: 'hi' }] }
    );
  });

  it('attaches finish_reason to output messages', () => {
    const result = mapOutputMessage(
      { role: 'model', content: [{ text: 'done' }] },
      'stop'
    );
    assert.strictEqual(result.role, 'assistant');
    assert.strictEqual(result.finish_reason, 'stop');
  });
});
