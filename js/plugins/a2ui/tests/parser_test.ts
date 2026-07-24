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
import { basicCatalog } from '../src/catalog.js';
import { A2uiStreamParser } from '../src/parser.js';
import type { A2uiEnvelope } from '../src/types.js';

function fixedId() {
  return 'surface-1';
}

/** Runs `fn` while capturing `console.warn` output, restoring it afterwards. */
function captureWarnings(fn: () => void): string[] {
  const warnings: string[] = [];
  const original = console.warn;
  console.warn = (...args: unknown[]) => {
    warnings.push(args.map(String).join(' '));
  };
  try {
    fn();
  } finally {
    console.warn = original;
  }
  return warnings;
}

function collect(parser: A2uiStreamParser, chunks: string[]) {
  let prose = '';
  const batches: A2uiEnvelope[][] = [];
  for (const c of chunks) {
    const r = parser.push(c);
    prose += r.prose;
    batches.push(...r.envelopeBatches);
  }
  const f = parser.flush();
  prose += f.prose;
  batches.push(...f.envelopeBatches);
  return { prose, batches };
}

const SAMPLE_BLOCK = `\`\`\`a2ui
[
  { "createSurface": { "surfaceId": "SURFACE_ID", "catalogId": "${basicCatalog.id}" } },
  { "updateComponents": { "surfaceId": "SURFACE_ID", "components": [
    { "id": "root", "component": "Text", "text": "hi" }
  ] } }
]
\`\`\`
`;

describe('A2uiStreamParser', () => {
  it('separates prose from a complete a2ui block', () => {
    const parser = new A2uiStreamParser({
      catalog: basicCatalog,
      surfaceId: fixedId,
    });
    const { prose, batches } = collect(parser, [
      'Here is the weather:\n',
      SAMPLE_BLOCK,
    ]);
    assert.match(prose, /Here is the weather/);
    assert.doesNotMatch(prose, /createSurface/);
    assert.strictEqual(batches.length, 1);
    assert.strictEqual(batches[0].length, 2);
  });

  it('substitutes SURFACE_ID placeholder with the generated id', () => {
    const parser = new A2uiStreamParser({
      catalog: basicCatalog,
      surfaceId: fixedId,
    });
    const { batches } = collect(parser, [SAMPLE_BLOCK]);
    const create = batches[0][0] as any;
    assert.strictEqual(create.createSurface.surfaceId, 'surface-1');
    const update = batches[0][1] as any;
    assert.strictEqual(update.updateComponents.surfaceId, 'surface-1');
  });

  it('stamps the protocol version', () => {
    const parser = new A2uiStreamParser({
      catalog: basicCatalog,
      surfaceId: fixedId,
      version: 'v0.9',
    });
    const { batches } = collect(parser, [SAMPLE_BLOCK]);
    assert.strictEqual((batches[0][0] as any).version, 'v0.9');
  });

  it('handles a block split across many tiny chunks', () => {
    const parser = new A2uiStreamParser({
      catalog: basicCatalog,
      surfaceId: fixedId,
    });
    const chunks = SAMPLE_BLOCK.match(/.{1,3}/gs) ?? [];
    const { prose, batches } = collect(parser, ['prefix ', ...chunks]);
    assert.match(prose, /prefix/);
    assert.strictEqual(batches.length, 1);
    assert.strictEqual(batches[0].length, 2);
  });

  it('does not leak a partial fence into prose', () => {
    const parser = new A2uiStreamParser({
      catalog: basicCatalog,
      surfaceId: fixedId,
    });
    // Push text ending in a partial fence, then complete it.
    const r1 = parser.push('hello ```a2');
    assert.doesNotMatch(r1.prose, /```a2/);
    const { batches } = collect(parser, [
      'ui\n[{"createSurface":{"surfaceId":"SURFACE_ID","catalogId":"' +
        basicCatalog.id +
        '"}}]\n```\n',
    ]);
    assert.strictEqual(batches.length, 1);
  });

  it('emits prose with no blocks unchanged', () => {
    const parser = new A2uiStreamParser({
      catalog: basicCatalog,
      surfaceId: fixedId,
    });
    const { prose, batches } = collect(parser, ['just ', 'text ', 'here']);
    assert.strictEqual(prose, 'just text here');
    assert.strictEqual(batches.length, 0);
  });

  it('throws in strict mode on unknown component', () => {
    const parser = new A2uiStreamParser({
      catalog: basicCatalog,
      surfaceId: fixedId,
      validate: 'strict',
    });
    const bad = `\`\`\`a2ui
[{ "updateComponents": { "surfaceId": "SURFACE_ID", "components": [
  { "id": "root", "component": "NotAThing" }
] } }]
\`\`\`
`;
    assert.throws(() => collect(parser, [bad]), /not in catalog/);
  });

  it('throws in strict mode when root is missing', () => {
    const parser = new A2uiStreamParser({
      catalog: basicCatalog,
      surfaceId: fixedId,
      validate: 'strict',
    });
    const bad = `\`\`\`a2ui
[{ "updateComponents": { "surfaceId": "SURFACE_ID", "components": [
  { "id": "x", "component": "Text", "text": "hi" }
] } }]
\`\`\`
`;
    assert.throws(() => collect(parser, [bad]), /root/);
  });

  it('validate:off does not throw on bad JSON', () => {
    const parser = new A2uiStreamParser({
      catalog: basicCatalog,
      surfaceId: fixedId,
      validate: 'off',
    });
    const bad = '```a2ui\n{not json}\n```\n';
    const { batches } = collect(parser, [bad]);
    assert.strictEqual(batches.length, 0);
  });

  it('validate:warn drops an unknown component without throwing', () => {
    const parser = new A2uiStreamParser({
      catalog: basicCatalog,
      surfaceId: fixedId,
      validate: 'warn',
    });
    const bad = `\`\`\`a2ui
[{ "updateComponents": { "surfaceId": "SURFACE_ID", "components": [
  { "id": "root", "component": "NotAThing" }
] } }]
\`\`\`
`;
    const warnings = captureWarnings(() => {
      const { batches } = collect(parser, [bad]);
      assert.strictEqual(batches.length, 0);
    });
    assert.ok(
      warnings.some((w) => /not in catalog/.test(w)),
      'expected a warning about the unknown component'
    );
  });

  it('validate:warn drops bad JSON without throwing', () => {
    const parser = new A2uiStreamParser({
      catalog: basicCatalog,
      surfaceId: fixedId,
      validate: 'warn',
    });
    const bad = '```a2ui\n{not json}\n```\n';
    const warnings = captureWarnings(() => {
      const { batches } = collect(parser, [bad]);
      assert.strictEqual(batches.length, 0);
    });
    assert.ok(
      warnings.some((w) => /JSON/.test(w)),
      'expected a warning about the malformed JSON'
    );
  });

  it('prepends a createSurface when a block only has updates', () => {
    const parser = new A2uiStreamParser({
      catalog: basicCatalog,
      surfaceId: fixedId,
    });
    const updateOnly = `\`\`\`a2ui
[{ "updateComponents": { "surfaceId": "SURFACE_ID", "components": [
  { "id": "root", "component": "Text", "text": "refreshed" }
] } }]
\`\`\`
`;
    const { batches } = collect(parser, [updateOnly]);
    assert.strictEqual(batches.length, 1);
    const first = batches[0][0] as any;
    assert.ok(
      first.createSurface,
      'expected a synthesized createSurface first'
    );
    assert.strictEqual(first.createSurface.surfaceId, 'surface-1');
    assert.strictEqual(first.createSurface.catalogId, basicCatalog.id);
    // The update targets the same surface id.
    const update = batches[0][1] as any;
    assert.strictEqual(update.updateComponents.surfaceId, 'surface-1');
  });

  it('does not add a second createSurface when one is present', () => {
    const parser = new A2uiStreamParser({
      catalog: basicCatalog,
      surfaceId: fixedId,
    });
    const { batches } = collect(parser, [SAMPLE_BLOCK]);
    const createCount = batches[0].filter((e: any) => e.createSurface).length;
    assert.strictEqual(createCount, 1);
  });

  it('handles two separate blocks in one turn', () => {
    const parser = new A2uiStreamParser({
      catalog: basicCatalog,
      surfaceId: fixedId,
    });
    const { batches } = collect(parser, [
      SAMPLE_BLOCK,
      'some text between\n',
      SAMPLE_BLOCK,
    ]);
    assert.strictEqual(batches.length, 2);
  });

  it('preserves prose/block order in segments (prose after a block)', () => {
    const parser = new A2uiStreamParser({
      catalog: basicCatalog,
      surfaceId: fixedId,
    });
    // A single push containing: intro prose, a block, then trailing prose.
    const r = parser.push('before\n' + SAMPLE_BLOCK + 'after');
    const f = parser.flush();
    const segments = [...r.segments, ...f.segments];
    // Expect order: prose("before"), envelopes, prose("after").
    assert.strictEqual(segments.length, 3);
    assert.ok('prose' in segments[0] && /before/.test(segments[0].prose));
    assert.ok('envelopes' in segments[1]);
    assert.ok('prose' in segments[2] && /after/.test(segments[2].prose));
  });
});
