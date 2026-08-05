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

import type { Message as A2AMessage, Part as A2APart } from '@a2a-js/sdk';
import type { Part } from 'genkit/beta';
import assert from 'node:assert';
import { describe, it } from 'node:test';
import {
  A2A_METADATA,
  GenkitPartType,
  a2aMessageToGenkit,
  a2aMessageToResumeInput,
  a2aPartToGenkit,
  a2aStateToFinishReason,
  genkitPartToA2A,
  genkitResumeToA2AParts,
} from '../src/mapping.js';

// ---------------------------------------------------------------------------
// Genkit Part -> A2A Part
// ---------------------------------------------------------------------------

describe('genkitPartToA2A', () => {
  it('maps a plain text part to a clean A2A text part', () => {
    assert.deepStrictEqual(genkitPartToA2A({ text: 'hello' }), {
      kind: 'text',
      text: 'hello',
    });
  });

  it('maps a reasoning part to a text part with the reasoning flag', () => {
    assert.deepStrictEqual(genkitPartToA2A({ reasoning: 'thinking' }), {
      kind: 'text',
      text: 'thinking',
      metadata: {
        [A2A_METADATA.TYPE]: GenkitPartType.REASONING,
        [A2A_METADATA.REASONING]: true,
      },
    });
  });

  it('maps a remote media part to a file (uri) part', () => {
    assert.deepStrictEqual(
      genkitPartToA2A({
        media: { url: 'https://example.com/a.png', contentType: 'image/png' },
      }),
      {
        kind: 'file',
        file: { uri: 'https://example.com/a.png', mimeType: 'image/png' },
        metadata: { [A2A_METADATA.TYPE]: 'file' },
      }
    );
  });

  it('maps a base64 data: media part to a file (bytes) part', () => {
    const result = genkitPartToA2A({
      media: { url: 'data:image/png;base64,QUJD', contentType: 'image/png' },
    }) as A2APart & { file: { bytes: string; mimeType?: string } };
    assert.strictEqual(result.kind, 'file');
    assert.strictEqual(result.file.bytes, 'QUJD');
    assert.strictEqual(result.file.mimeType, 'image/png');
  });

  it('maps a toolRequest to a data part with interrupt metadata', () => {
    const result = genkitPartToA2A({
      toolRequest: { ref: 'r1', name: 'getWeather', input: { city: 'NYC' } },
      metadata: { interrupt: { reason: 'confirm' } },
    });
    assert.deepStrictEqual(result, {
      kind: 'data',
      data: { ref: 'r1', name: 'getWeather', input: { city: 'NYC' } },
      metadata: {
        [A2A_METADATA.TYPE]: GenkitPartType.TOOL_REQUEST,
        [A2A_METADATA.INTERRUPT]: { reason: 'confirm' },
        [A2A_METADATA.PART_METADATA]: { interrupt: { reason: 'confirm' } },
      },
    });
  });

  it('maps a toolResponse to a data part', () => {
    assert.deepStrictEqual(
      genkitPartToA2A({
        toolResponse: { ref: 'r1', name: 'getWeather', output: { temp: 20 } },
      }),
      {
        kind: 'data',
        data: { ref: 'r1', name: 'getWeather', output: { temp: 20 } },
        metadata: { [A2A_METADATA.TYPE]: GenkitPartType.TOOL_RESPONSE },
      }
    );
  });
});

// ---------------------------------------------------------------------------
// A2A Part -> Genkit Part (round trips)
// ---------------------------------------------------------------------------

describe('a2aPartToGenkit', () => {
  it('round-trips a text part', () => {
    const genkit: Part = { text: 'hello' };
    const a2a = genkitPartToA2A(genkit)!;
    assert.deepStrictEqual(a2aPartToGenkit(a2a), genkit);
  });

  it('round-trips a reasoning part', () => {
    const genkit: Part = { reasoning: 'thinking' };
    const a2a = genkitPartToA2A(genkit)!;
    assert.deepStrictEqual(a2aPartToGenkit(a2a), genkit);
  });

  it('round-trips a toolRequest with interrupt metadata', () => {
    const genkit: Part = {
      toolRequest: { ref: 'r1', name: 'getWeather', input: { city: 'NYC' } },
      metadata: { interrupt: { reason: 'confirm' } },
    };
    const a2a = genkitPartToA2A(genkit)!;
    assert.deepStrictEqual(a2aPartToGenkit(a2a), genkit);
  });

  it('maps a generic (non-genkit) text part to a text part', () => {
    assert.deepStrictEqual(a2aPartToGenkit({ kind: 'text', text: 'hi' }), {
      text: 'hi',
    });
  });

  it('maps a generic A2A data part to a Genkit data part', () => {
    assert.deepStrictEqual(
      a2aPartToGenkit({ kind: 'data', data: { foo: 'bar' } }),
      { data: { foo: 'bar' } }
    );
  });

  it('maps a file (bytes) part to a data: media part', () => {
    const result = a2aPartToGenkit({
      kind: 'file',
      file: { bytes: 'QUJD', mimeType: 'image/png' },
    }) as Part;
    assert.deepStrictEqual(result, {
      media: { url: 'data:image/png;base64,QUJD', contentType: 'image/png' },
    });
  });
});

// ---------------------------------------------------------------------------
// Message-level mapping
// ---------------------------------------------------------------------------

describe('a2aMessageToGenkit', () => {
  it('maps an agent message to a model message', () => {
    const msg: A2AMessage = {
      kind: 'message',
      messageId: 'm1',
      role: 'agent',
      parts: [{ kind: 'text', text: 'hi' }],
    };
    assert.deepStrictEqual(a2aMessageToGenkit(msg), {
      role: 'model',
      content: [{ text: 'hi' }],
    });
  });

  it('maps a user message to a user message', () => {
    const msg: A2AMessage = {
      kind: 'message',
      messageId: 'm1',
      role: 'user',
      parts: [{ kind: 'text', text: 'hi' }],
    };
    assert.deepStrictEqual(a2aMessageToGenkit(msg), {
      role: 'user',
      content: [{ text: 'hi' }],
    });
  });
});

// ---------------------------------------------------------------------------
// Resume detection
// ---------------------------------------------------------------------------

describe('a2aMessageToResumeInput', () => {
  it('treats a plain text message as a fresh user turn', () => {
    const msg: A2AMessage = {
      kind: 'message',
      messageId: 'm1',
      role: 'user',
      parts: [{ kind: 'text', text: 'hello' }],
    };
    assert.deepStrictEqual(a2aMessageToResumeInput(msg), {
      message: { role: 'user', content: [{ text: 'hello' }] },
    });
  });

  it('builds a respond resume from a toolResponse data part', () => {
    const msg: A2AMessage = {
      kind: 'message',
      messageId: 'm1',
      role: 'user',
      parts: [
        {
          kind: 'data',
          data: { ref: 'r1', name: 'approve', output: { ok: true } },
          metadata: { [A2A_METADATA.TYPE]: GenkitPartType.TOOL_RESPONSE },
        },
      ],
    };
    assert.deepStrictEqual(a2aMessageToResumeInput(msg), {
      resume: {
        respond: [
          {
            toolResponse: { ref: 'r1', name: 'approve', output: { ok: true } },
          },
        ],
      },
    });
  });

  it('builds a restart resume from a flagged toolRequest data part', () => {
    const msg: A2AMessage = {
      kind: 'message',
      messageId: 'm1',
      role: 'user',
      parts: [
        {
          kind: 'data',
          data: { ref: 'r1', name: 'getRate', input: { from: 'USD' } },
          metadata: {
            [A2A_METADATA.TYPE]: GenkitPartType.TOOL_REQUEST,
            [A2A_METADATA.RESTART]: { confirmedAt: 123 },
          },
        },
      ],
    };
    assert.deepStrictEqual(a2aMessageToResumeInput(msg), {
      resume: {
        restart: [
          {
            toolRequest: { ref: 'r1', name: 'getRate', input: { from: 'USD' } },
            metadata: { resumed: { confirmedAt: 123 } },
          },
        ],
      },
    });
  });
});

// ---------------------------------------------------------------------------
// Genkit resume payload -> A2A parts (outbound, client side)
// ---------------------------------------------------------------------------

describe('genkitResumeToA2AParts', () => {
  it('maps a respond entry to a tagged toolResponse data part', () => {
    assert.deepStrictEqual(
      genkitResumeToA2AParts({
        respond: [
          {
            toolResponse: { ref: 'r1', name: 'approve', output: { ok: true } },
          },
        ],
      }),
      [
        {
          kind: 'data',
          data: { ref: 'r1', name: 'approve', output: { ok: true } },
          metadata: { [A2A_METADATA.TYPE]: GenkitPartType.TOOL_RESPONSE },
        },
      ]
    );
  });

  it('maps a restart entry to a tagged toolRequest data part', () => {
    assert.deepStrictEqual(
      genkitResumeToA2AParts({
        restart: [
          {
            toolRequest: { ref: 'r1', name: 'getRate', input: { from: 'USD' } },
            metadata: { resumed: { confirmedAt: 123 } },
          },
        ],
      }),
      [
        {
          kind: 'data',
          data: { ref: 'r1', name: 'getRate', input: { from: 'USD' } },
          metadata: {
            [A2A_METADATA.TYPE]: GenkitPartType.TOOL_REQUEST,
            [A2A_METADATA.RESTART]: { confirmedAt: 123 },
          },
        },
      ]
    );
  });

  it('round-trips through a2aMessageToResumeInput', () => {
    const resume = {
      respond: [
        { toolResponse: { ref: 'r1', name: 'approve', output: { ok: true } } },
      ],
    };
    const parts = genkitResumeToA2AParts(resume);
    const roundTripped = a2aMessageToResumeInput({
      kind: 'message',
      messageId: 'm1',
      role: 'user',
      parts,
    });
    assert.deepStrictEqual(roundTripped, { resume });
  });
});

// ---------------------------------------------------------------------------
// A2A task state -> Genkit finish reason
// ---------------------------------------------------------------------------

describe('a2aStateToFinishReason', () => {
  it('maps terminal states to finish reasons', () => {
    assert.strictEqual(a2aStateToFinishReason('completed'), 'stop');
    assert.strictEqual(a2aStateToFinishReason('input-required'), 'interrupted');
    assert.strictEqual(a2aStateToFinishReason('failed'), 'failed');
    assert.strictEqual(a2aStateToFinishReason('canceled'), 'aborted');
    assert.strictEqual(a2aStateToFinishReason('rejected'), 'blocked');
  });

  it('falls back to unknown for non-terminal states', () => {
    assert.strictEqual(a2aStateToFinishReason('working'), 'unknown');
    assert.strictEqual(a2aStateToFinishReason('submitted'), 'unknown');
  });
});
