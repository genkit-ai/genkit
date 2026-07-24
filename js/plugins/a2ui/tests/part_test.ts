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
import { a2uiEnvelopesFromParts, a2uiPart, isA2uiPart } from '../src/part.js';
import { A2UI_MIME_TYPE, type A2uiEnvelope } from '../src/types.js';

const sampleEnvelope: A2uiEnvelope = {
  createSurface: { surfaceId: 's1', catalogId: 'c1' },
  version: 'v0.9',
};

describe('a2uiPart', () => {
  it('wraps envelopes in a data part tagged with the a2ui mime type', () => {
    const part = a2uiPart([sampleEnvelope]);
    assert.deepStrictEqual(part.data, { envelopes: [sampleEnvelope] });
    assert.strictEqual(part.metadata.mimeType, A2UI_MIME_TYPE);
  });
});

describe('isA2uiPart', () => {
  it('accepts a well-formed a2ui data part', () => {
    assert.strictEqual(isA2uiPart(a2uiPart([sampleEnvelope])), true);
  });

  it('rejects a plain text part', () => {
    assert.strictEqual(isA2uiPart({ text: 'hi' }), false);
  });

  it('rejects a data part with a different mime type', () => {
    assert.strictEqual(
      isA2uiPart({
        data: { envelopes: [] },
        metadata: { mimeType: 'application/json' },
      }),
      false
    );
  });

  it('rejects a data part whose data.envelopes is not an array', () => {
    assert.strictEqual(
      isA2uiPart({
        data: { envelopes: {} },
        metadata: { mimeType: A2UI_MIME_TYPE },
      }),
      false
    );
  });

  it('rejects the legacy bare-array data shape', () => {
    assert.strictEqual(
      isA2uiPart({ data: [], metadata: { mimeType: A2UI_MIME_TYPE } }),
      false
    );
  });

  it('rejects null / non-objects', () => {
    assert.strictEqual(isA2uiPart(null), false);
    assert.strictEqual(isA2uiPart(undefined), false);
    assert.strictEqual(isA2uiPart('nope'), false);
  });
});

describe('a2uiEnvelopesFromParts', () => {
  it('extracts envelopes from a single a2ui part', () => {
    assert.deepStrictEqual(
      a2uiEnvelopesFromParts([a2uiPart([sampleEnvelope])]),
      [sampleEnvelope]
    );
  });

  it("extracts envelopes from a message's content", () => {
    const content = [{ text: 'hi' }, a2uiPart([sampleEnvelope])];
    assert.deepStrictEqual(a2uiEnvelopesFromParts(content), [sampleEnvelope]);
  });

  it('collects across multiple parts', () => {
    const content = [
      { text: 'hi' },
      a2uiPart([sampleEnvelope]),
      a2uiPart([sampleEnvelope]),
    ];
    assert.deepStrictEqual(a2uiEnvelopesFromParts(content), [
      sampleEnvelope,
      sampleEnvelope,
    ]);
  });

  it('returns [] for prose / non-a2ui content', () => {
    assert.deepStrictEqual(a2uiEnvelopesFromParts([{ text: 'hi' }]), []);
  });

  it('returns [] for a nullish parts list', () => {
    assert.deepStrictEqual(a2uiEnvelopesFromParts(null), []);
    assert.deepStrictEqual(a2uiEnvelopesFromParts(undefined), []);
  });
});
