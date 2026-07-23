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
import { actionToMessage } from '../src/client.js';
import { isA2uiPart } from '../src/part.js';
import { A2UI_MIME_TYPE, type A2uiClientAction } from '../src/types.js';

function makeAction(
  overrides: Partial<A2uiClientAction> = {}
): A2uiClientAction {
  return {
    name: 'submit',
    surfaceId: 'surface-1',
    sourceComponentId: 'btn',
    timestamp: '2026-01-01T00:00:00.000Z',
    context: {},
    ...overrides,
  };
}

describe('actionToMessage', () => {
  it('builds a user message summarizing the action', () => {
    const message = actionToMessage(makeAction());
    assert.strictEqual(message.role, 'user');
    const summary = message.content[0] as { text: string };
    assert.match(summary.text, /submit/);
    assert.match(summary.text, /surface-1/);
  });

  it('attaches the full action as an a2ui data part', () => {
    const action = makeAction();
    const message = actionToMessage(action);
    const dataPart = message.content[1];
    assert.ok(isA2uiPart(dataPart));
    assert.strictEqual(
      (dataPart as { metadata: { mimeType: string } }).metadata.mimeType,
      A2UI_MIME_TYPE
    );
    assert.deepStrictEqual((dataPart as { data: unknown[] }).data, [
      { action },
    ]);
  });

  it('includes the context in the summary when present', () => {
    const message = actionToMessage(makeAction({ context: { city: 'Tokyo' } }));
    const summary = message.content[0] as { text: string };
    assert.match(summary.text, /context/);
    assert.match(summary.text, /Tokyo/);
  });

  it('omits the context clause when context is empty', () => {
    const message = actionToMessage(makeAction());
    const summary = message.content[0] as { text: string };
    assert.doesNotMatch(summary.text, /context/);
  });
});
