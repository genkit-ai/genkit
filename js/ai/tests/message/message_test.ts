/**
 * Copyright 2024 Google LLC
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
import { describe, it } from 'node:test';
import { Message } from '../../src/message.js';

describe('Message', () => {
  describe('.parseData()', () => {
    const testCases = [
      {
        desc: 'convert string to user message',
        input: 'i am a user message',
        want: { role: 'user', content: [{ text: 'i am a user message' }] },
      },
      {
        desc: 'convert string content to Part[] content',
        input: {
          role: 'system',
          content: 'i am a system message',
          metadata: { extra: true },
        },
        want: {
          role: 'system',
          content: [{ text: 'i am a system message' }],
          metadata: { extra: true },
        },
      },
      {
        desc: 'leave valid MessageData alone',
        input: { role: 'model', content: [{ text: 'i am a model message' }] },
        want: { role: 'model', content: [{ text: 'i am a model message' }] },
      },
    ];

    for (const t of testCases) {
      it(t.desc, () => {
        assert.deepStrictEqual(Message.parseData(t.input as any), t.want);
      });
    }
  });
});
