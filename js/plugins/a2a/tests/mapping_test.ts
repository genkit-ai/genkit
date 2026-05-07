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

import * as assert from 'assert';
import { describe, it } from 'node:test';

import {
  mapA2AArtifactToGenkit,
  mapA2APartToGenkit,
  mapGenkitArtifactToA2A,
  mapGenkitPartToA2A,
  type GenkitArtifact,
} from '../src/mapping.js';

// ---------------------------------------------------------------------------
// mapGenkitPartToA2A
// ---------------------------------------------------------------------------

describe('mapGenkitPartToA2A', () => {
  it('maps a text part', () => {
    const result = mapGenkitPartToA2A({ text: 'hello world' });
    assert.deepStrictEqual(result, { kind: 'text', text: 'hello world' });
  });

  it('maps an empty text part', () => {
    const result = mapGenkitPartToA2A({ text: '' });
    assert.deepStrictEqual(result, { kind: 'text', text: '' });
  });

  it('maps a media part with a remote URL', () => {
    const result = mapGenkitPartToA2A({
      media: { url: 'https://example.com/image.png', contentType: 'image/png' },
    });
    assert.strictEqual(result.kind, 'file');
    const file = (result as any).file;
    assert.strictEqual(file.uri, 'https://example.com/image.png');
    assert.strictEqual(file.mimeType, 'image/png');
    assert.strictEqual(file.name, 'remote_file');
  });

  it('maps a media part with a data: URI (base64)', () => {
    const dataUri = 'data:image/jpeg;base64,/9j/4AAQSkZJRg==';
    const result = mapGenkitPartToA2A({
      media: { url: dataUri, contentType: 'image/jpeg' },
    });
    assert.strictEqual(result.kind, 'file');
    const file = (result as any).file;
    assert.strictEqual(file.bytes, '/9j/4AAQSkZJRg==');
    assert.strictEqual(file.mimeType, 'image/jpeg');
    assert.strictEqual(file.name, 'inline_file');
  });

  it('maps a toolRequest part to a data part with genkit_type metadata', () => {
    const result = mapGenkitPartToA2A({
      toolRequest: {
        ref: 'call-1',
        name: 'get_weather',
        input: { location: 'London' },
      },
    });
    assert.strictEqual(result.kind, 'data');
    const dataPart = result as any;
    assert.strictEqual(dataPart.data.id, 'call-1');
    assert.strictEqual(dataPart.data.name, 'get_weather');
    assert.deepStrictEqual(dataPart.data.args, { location: 'London' });
    assert.strictEqual(dataPart.metadata.genkit_type, 'function_call');
  });

  it('maps a toolResponse part to a data part with genkit_type metadata', () => {
    const result = mapGenkitPartToA2A({
      toolResponse: {
        ref: 'call-1',
        name: 'get_weather',
        output: { temp: 72, condition: 'sunny' },
      },
    });
    assert.strictEqual(result.kind, 'data');
    const dataPart = result as any;
    assert.strictEqual(dataPart.data.id, 'call-1');
    assert.strictEqual(dataPart.data.name, 'get_weather');
    assert.deepStrictEqual(dataPart.data.response, {
      temp: 72,
      condition: 'sunny',
    });
    assert.strictEqual(dataPart.metadata.genkit_type, 'function_response');
  });

  it('maps an unknown part as a data fallback', () => {
    const weirdPart = { data: { foo: 'bar' } } as any;
    const result = mapGenkitPartToA2A(weirdPart);
    assert.strictEqual(result.kind, 'data');
    // The entire Genkit part object is wrapped as the data payload
    assert.deepStrictEqual((result as any).data, { data: { foo: 'bar' } });
  });
});

// ---------------------------------------------------------------------------
// mapA2APartToGenkit
// ---------------------------------------------------------------------------

describe('mapA2APartToGenkit', () => {
  it('maps a text part', () => {
    const result = mapA2APartToGenkit({ kind: 'text', text: 'hello' } as any);
    assert.deepStrictEqual(result, { text: 'hello' });
  });

  it('maps a file part with URI', () => {
    const result = mapA2APartToGenkit({
      kind: 'file',
      file: {
        uri: 'https://example.com/doc.pdf',
        mimeType: 'application/pdf',
      },
    } as any);
    assert.deepStrictEqual(result, {
      media: {
        url: 'https://example.com/doc.pdf',
        contentType: 'application/pdf',
      },
    });
  });

  it('maps a file part with bytes to data: URI', () => {
    const result = mapA2APartToGenkit({
      kind: 'file',
      file: {
        bytes: 'AAAA',
        mimeType: 'image/png',
      },
    } as any);
    assert.deepStrictEqual(result, {
      media: {
        url: 'data:image/png;base64,AAAA',
        contentType: 'image/png',
      },
    });
  });

  it('maps a file part with bytes and no mimeType', () => {
    const result = mapA2APartToGenkit({
      kind: 'file',
      file: {
        bytes: 'BBBB',
      },
    } as any);
    assert.deepStrictEqual(result, {
      media: {
        url: 'data:application/octet-stream;base64,BBBB',
        contentType: undefined,
      },
    });
  });

  it('maps a data part with genkit_type=function_call to toolRequest', () => {
    const result = mapA2APartToGenkit({
      kind: 'data',
      data: { id: 'c1', name: 'search', args: { q: 'test' } },
      metadata: { genkit_type: 'function_call' },
    } as any);
    assert.deepStrictEqual(result, {
      toolRequest: {
        ref: 'c1',
        name: 'search',
        input: { q: 'test' },
      },
    });
  });

  it('maps a data part with genkit_type=function_response to toolResponse', () => {
    const result = mapA2APartToGenkit({
      kind: 'data',
      data: { id: 'c1', name: 'search', response: { results: [] } },
      metadata: { genkit_type: 'function_response' },
    } as any);
    assert.deepStrictEqual(result, {
      toolResponse: {
        ref: 'c1',
        name: 'search',
        output: { results: [] },
      },
    });
  });

  it('maps a data part whose data is a Genkit Part back to that part', () => {
    const result = mapA2APartToGenkit({
      kind: 'data',
      data: { text: 'restored' },
    } as any);
    assert.deepStrictEqual(result, { text: 'restored' });
  });

  it('maps a data part with a "custom" key (which is a Genkit Part key) back as-is', () => {
    // 'custom' is in the GENKIT_PART_KEYS set, so isGenkitPart returns true
    const result = mapA2APartToGenkit({
      kind: 'data',
      data: { custom: 'payload' },
    } as any);
    assert.deepStrictEqual(result, { custom: 'payload' });
  });

  it('maps a truly unknown data part to { data: ... }', () => {
    // 'unknownKey' is NOT in GENKIT_PART_KEYS, so it falls through to { data: ... }
    const result = mapA2APartToGenkit({
      kind: 'data',
      data: { unknownKey: 'payload' },
    } as any);
    assert.deepStrictEqual(result, { data: { unknownKey: 'payload' } });
  });

  it('falls back to JSON text for unknown kinds', () => {
    const weird = { kind: 'something-else', value: 42 } as any;
    const result = mapA2APartToGenkit(weird);
    assert.strictEqual(result.text, JSON.stringify(weird));
  });
});

// ---------------------------------------------------------------------------
// Round-trip: Genkit → A2A → Genkit
// ---------------------------------------------------------------------------

describe('round-trip mapping', () => {
  it('text round-trips', () => {
    const original = { text: 'hello' };
    const result = mapA2APartToGenkit(mapGenkitPartToA2A(original));
    assert.deepStrictEqual(result, original);
  });

  it('remote media round-trips', () => {
    const original = {
      media: { url: 'https://example.com/img.jpg', contentType: 'image/jpeg' },
    };
    const result = mapA2APartToGenkit(mapGenkitPartToA2A(original));
    assert.deepStrictEqual(result, original);
  });

  it('inline media round-trips', () => {
    const original = {
      media: {
        url: 'data:audio/mp3;base64,SGVsbG8=',
        contentType: 'audio/mp3',
      },
    };
    const result = mapA2APartToGenkit(mapGenkitPartToA2A(original));
    assert.deepStrictEqual(result, original);
  });

  it('toolRequest round-trips', () => {
    const original = {
      toolRequest: {
        ref: 'abc',
        name: 'myTool',
        input: { x: 1 },
      },
    };
    const result = mapA2APartToGenkit(mapGenkitPartToA2A(original));
    assert.deepStrictEqual(result, original);
  });

  it('toolResponse round-trips', () => {
    const original = {
      toolResponse: {
        ref: 'abc',
        name: 'myTool',
        output: { y: 2 },
      },
    };
    const result = mapA2APartToGenkit(mapGenkitPartToA2A(original));
    assert.deepStrictEqual(result, original);
  });
});

// ---------------------------------------------------------------------------
// Artifact mapping
// ---------------------------------------------------------------------------

describe('mapGenkitArtifactToA2A', () => {
  it('maps a basic artifact', () => {
    const artifact: GenkitArtifact = {
      name: 'report',
      parts: [{ text: 'Some report content' }],
    };
    const result = mapGenkitArtifactToA2A(artifact);
    assert.strictEqual(result.artifactId, 'report');
    assert.strictEqual(result.name, 'report');
    assert.strictEqual(result.parts.length, 1);
    assert.deepStrictEqual(result.parts[0], {
      kind: 'text',
      text: 'Some report content',
    });
  });

  it('forwards a2a metadata overrides', () => {
    const artifact: GenkitArtifact = {
      name: 'report',
      parts: [{ text: 'content' }],
      metadata: {
        a2a: { name: 'Pretty Report', description: 'A report' },
        custom: 'value',
      },
    };
    const result = mapGenkitArtifactToA2A(artifact);
    assert.strictEqual(result.artifactId, 'report');
    assert.strictEqual(result.name, 'Pretty Report');
    assert.strictEqual((result as any).description, 'A report');
    // Non-a2a metadata is forwarded
    assert.strictEqual((result as any).metadata?.custom, 'value');
  });

  it('throws when name is missing', () => {
    const artifact: GenkitArtifact = {
      parts: [{ text: 'content' }],
    };
    assert.throws(
      () => mapGenkitArtifactToA2A(artifact),
      /Artifact\.name is required/
    );
  });
});

describe('mapA2AArtifactToGenkit', () => {
  it('maps a basic A2A artifact', () => {
    const a2aArtifact = {
      artifactId: 'doc-1',
      name: 'Document',
      parts: [{ kind: 'text', text: 'Hello' }],
    } as any;
    const result = mapA2AArtifactToGenkit(a2aArtifact);
    assert.strictEqual(result.name, 'doc-1');
    assert.strictEqual(result.parts.length, 1);
    assert.deepStrictEqual(result.parts[0], { text: 'Hello' });
    assert.strictEqual(result.metadata?.a2a?.name, 'Document');
  });

  it('preserves A2A-specific fields in metadata.a2a', () => {
    const a2aArtifact = {
      artifactId: 'img-1',
      name: 'Image',
      description: 'A cool image',
      parts: [
        {
          kind: 'file',
          file: { uri: 'https://example.com/img.png', mimeType: 'image/png' },
        },
      ],
      metadata: { source: 'external' },
    } as any;
    const result = mapA2AArtifactToGenkit(a2aArtifact);
    assert.strictEqual(result.name, 'img-1');
    assert.strictEqual(result.metadata?.a2a?.name, 'Image');
    assert.strictEqual(result.metadata?.a2a?.description, 'A cool image');
    assert.deepStrictEqual(result.metadata?.a2a?.metadata, {
      source: 'external',
    });
  });
});

// ---------------------------------------------------------------------------
// Artifact round-trip
// ---------------------------------------------------------------------------

describe('artifact round-trip', () => {
  it('Genkit → A2A → Genkit preserves content', () => {
    const original: GenkitArtifact = {
      name: 'summary',
      parts: [
        { text: 'Summary line 1' },
        {
          media: {
            url: 'https://example.com/chart.png',
            contentType: 'image/png',
          },
        },
      ],
    };
    const a2a = mapGenkitArtifactToA2A(original);
    const restored = mapA2AArtifactToGenkit(a2a);
    assert.strictEqual(restored.name, original.name);
    assert.deepStrictEqual(restored.parts, original.parts);
  });
});
