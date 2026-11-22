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
import type { GenerateRequest } from 'genkit';
import { describe, it } from 'node:test';
import {
  buildState,
  mediaUrl,
  stringifyBody,
  systemPreamble,
  type Reads,
} from '../src/state.js';

const none: Reads = { images: false, audio: false, video: false };
const all: Reads = { images: true, audio: true, video: true };

function request(partial: Partial<GenerateRequest>): GenerateRequest {
  return { messages: [], ...partial } as GenerateRequest;
}

describe('state', () => {
  it('is one message as its value', () => {
    assert.deepStrictEqual(
      buildState(
        request({ messages: [{ role: 'user', content: [{ text: 'hi' }] }] }),
        false,
        none
      ).state,
      'hi'
    );
    assert.deepStrictEqual(
      buildState(
        request({
          messages: [{ role: 'user', content: [{ data: { ticket: 'x' } }] }],
        }),
        false,
        none
      ).state,
      { ticket: 'x' }
    );
  });

  it('is several messages as turns, without system messages or plumbing', () => {
    const { state } = buildState(
      request({
        messages: [
          { role: 'system', content: [{ text: 'You judge.' }] },
          { role: 'user', content: [{ text: 'Refund me.' }] },
          {
            role: 'model',
            content: [
              { text: 'Sure.' },
              { text: 'Output JSON', metadata: { purpose: 'output' } },
            ],
          },
        ],
      }),
      false,
      none
    );
    assert.deepStrictEqual(state, [
      { role: 'user', content: 'Refund me.' },
      { role: 'model', content: 'Sure.' },
    ]);
  });

  it('is messages and context with documents', () => {
    const { state } = buildState(
      request({
        messages: [{ role: 'user', content: [{ text: 'Is it covered?' }] }],
        docs: [
          { content: [{ text: 'Policy A' }], metadata: { title: 'A' } },
          { content: [{ text: 'Policy B' }] },
        ],
      }),
      false,
      none
    );
    assert.deepStrictEqual(state, {
      messages: [{ role: 'user', content: 'Is it covered?' }],
      context: [{ content: 'Policy A', metadata: { title: 'A' } }, 'Policy B'],
    });
  });

  it('sends a media-only document with metadata as null content', () => {
    const { state } = buildState(
      request({
        messages: [{ role: 'user', content: [{ text: 'Is it damaged?' }] }],
        docs: [
          {
            content: [{ media: { url: 'data:image/png;base64,AA' } }],
            metadata: { source: 'photo' },
          },
        ],
      }),
      false,
      all
    );
    assert.deepStrictEqual(state, {
      messages: [{ role: 'user', content: 'Is it damaged?' }],
      context: [{ content: null, metadata: { source: 'photo' } }],
    });
  });

  it('collects media by kind, a media-only message an empty turn', () => {
    const png = 'data:image/png;base64,AAAA';
    const { state, media } = buildState(
      request({
        messages: [
          { role: 'user', content: [{ media: { url: png } }] },
          {
            role: 'user',
            content: [
              { text: 'Listen.' },
              { media: { url: 'data:audio/wav;base64,BBBB' } },
              { media: { url: 'data:video/mp4;base64,CCCC' } },
            ],
          },
        ],
        docs: [
          { content: [{ media: { url: png, contentType: 'image/png' } }] },
        ],
      }),
      false,
      all
    );
    assert.deepStrictEqual(media, {
      images: [png, png],
      audio: ['data:audio/wav;base64,BBBB'],
      videos: ['data:video/mp4;base64,CCCC'],
    });
    assert.deepStrictEqual(state, [
      { role: 'user', content: '' },
      { role: 'user', content: 'Listen.' },
    ]);
  });

  it('refuses media the model does not read, URLs, and other kinds', () => {
    const msg = (url: string) =>
      request({ messages: [{ role: 'user', content: [{ media: { url } }] }] });
    assert.throws(
      () => buildState(msg('data:image/png;base64,AA'), false, none),
      /reads no image\/png media/
    );
    assert.throws(
      () => buildState(msg('https://example.com/a.png'), false, all),
      /is a URL/
    );
    assert.throws(
      () => buildState(msg('data:application/pdf;base64,AA'), false, all),
      /not an image, audio, or video/
    );
  });

  it('reads the scheme, ;BASE64, and the media type in any case', () => {
    assert.deepStrictEqual(mediaUrl('DATA:IMAGE/PNG;BASE64,AAAA'), {
      dataUrl: 'data:image/png;base64,AAAA',
      contentType: 'image/png',
    });
  });

  it('never quotes a refused URL, which can carry a token', () => {
    assert.throws(
      () => mediaUrl('HTTPS://cdn.example.com/a.png?token=secret'),
      (e: Error) =>
        /the media at https:\/\/cdn\.example\.com is a URL/.test(e.message) &&
        !e.message.includes('secret')
    );
  });

  it('refuses video to a model that reads images only', () => {
    assert.throws(
      () =>
        buildState(
          request({
            messages: [
              {
                role: 'user',
                content: [
                  { text: 'Look.' },
                  { media: { url: 'data:video/mp4;base64,AA' } },
                ],
              },
            ],
          }),
          false,
          { images: true, audio: false, video: false }
        ),
      /reads no video\/mp4 media/
    );
  });

  it('percent-decodes a non-base64 data URL', () => {
    assert.deepStrictEqual(mediaUrl('DATA:Image/SVG+XML,%3Csvg%2F%3Eé'), {
      dataUrl: `data:image/svg+xml;base64,${Buffer.from('<svg/>é').toString('base64')}`,
      contentType: 'image/svg+xml',
    });
    assert.throws(
      () => mediaUrl('data:image/svg+xml,%zz'),
      /invalid URL escape/
    );
  });

  it('keeps stateJSON numbers verbatim', () => {
    const { state } = buildState(
      request({
        messages: [
          { role: 'user', content: [{ text: '{"id": 12345678901234567890}' }] },
        ],
      }),
      true,
      none
    );
    assert.strictEqual(
      stringifyBody({ state }),
      '{"state":{"id": 12345678901234567890}}'
    );
    assert.throws(
      () =>
        buildState(
          request({
            messages: [{ role: 'user', content: [{ text: 'prose' }] }],
          }),
          true,
          none
        ),
      /stateJSON is set but the text is not JSON/
    );
  });

  it('refuses a request with no state, and parts that cannot be state', () => {
    assert.throws(
      () =>
        buildState(
          request({
            messages: [{ role: 'system', content: [{ text: 'Q?' }] }],
          }),
          false,
          none
        ),
      /carries no state/
    );
    assert.throws(
      () =>
        buildState(
          request({
            messages: [
              {
                role: 'model',
                content: [{ toolRequest: { name: 't', input: {} } }],
              },
            ],
          }),
          false,
          none
        ),
      /a tool request part cannot be state/
    );
  });
});

describe('system preamble', () => {
  it('joins system texts and skips plumbing', () => {
    assert.strictEqual(
      systemPreamble([
        { role: 'system', content: [{ text: ' First. ' }] },
        { role: 'user', content: [{ text: 'state' }] },
        {
          role: 'system',
          content: [
            { text: 'Second.' },
            { text: 'Output ONLY', metadata: { purpose: 'output' } },
          ],
        },
      ]),
      'First.\n\nSecond.'
    );
  });

  it('is text only', () => {
    assert.throws(
      () =>
        systemPreamble([
          {
            role: 'system',
            content: [{ media: { url: 'data:image/png;base64,A' } }],
          },
        ]),
      /a media part cannot be instructions/
    );
  });
});
