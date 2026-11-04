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

import Anthropic from '@anthropic-ai/sdk';
import * as assert from 'assert';
import type { Part } from 'genkit';
import { describe, it } from 'node:test';
import { BetaRunner } from '../src/runner/beta.js';
import { Runner } from '../src/runner/stable.js';

for (const RunnerClass of [Runner, BetaRunner]) {
  describe(`${RunnerClass.name} multipart tool responses`, () => {
    const runner = new RunnerClass({
      name: 'anthropic/claude-haiku-4-5',
      client: new Anthropic({ apiKey: 'unused' }),
    });
    const convert = (response: NonNullable<Part['toolResponse']>) =>
      (runner as any).toAnthropicMessageContent({ toolResponse: response });
    const image = {
      media: { contentType: 'image/png', url: 'data:image/png;base64,AAAA' },
    };
    const imageMetadata =
      RunnerClass === Runner ? { cache_control: undefined } : {};
    const textBlock = (text: string) =>
      RunnerClass === Runner
        ? { type: 'text', text, cache_control: undefined, citations: null }
        : { type: 'text', text };
    const imageBlock = {
      ...imageMetadata,
      type: 'image',
      source: { type: 'base64', media_type: 'image/png', data: 'AAAA' },
    };
    const response = { name: 'screenshot', ref: 'call_1' };

    it('preserves text and image order without adding empty output', () => {
      const result = convert({
        ...response,
        content: [{ text: 'before' }, image, { text: 'after' }],
      });
      assert.strictEqual(result.type, 'tool_result');
      assert.strictEqual(result.tool_use_id, 'call_1');
      assert.deepStrictEqual(result.content, [
        textBlock('before'),
        imageBlock,
        textBlock('after'),
      ]);
    });

    it('preserves structured output alongside multipart content', () => {
      assert.deepStrictEqual(
        convert({ ...response, output: { ok: true }, content: [image] })
          .content,
        [{ type: 'text', text: '{"ok":true}' }, imageBlock]
      );
    });

    it('skips empty text without dropping adjacent media', () => {
      assert.deepStrictEqual(
        convert({ ...response, content: [{ text: '' }, image] }).content,
        [imageBlock]
      );
    });

    it('converts PDF content into document blocks', () => {
      const result = convert({
        ...response,
        content: [
          {
            media: {
              contentType: 'application/pdf',
              url: 'data:application/pdf;base64,AAAA',
            },
          },
        ],
      });
      assert.strictEqual(result.content[0].type, 'document');
      assert.deepStrictEqual(result.content[0].source, {
        type: 'base64',
        media_type: 'application/pdf',
        data: 'AAAA',
      });
    });

    it('supports URL image content', () => {
      assert.deepStrictEqual(
        convert({
          ...response,
          content: [
            {
              media: {
                contentType: 'image/png',
                url: 'https://example.com/image.png',
              },
            },
          ],
        }).content,
        [
          {
            ...imageMetadata,
            type: 'image',
            source: { type: 'url', url: 'https://example.com/image.png' },
          },
        ]
      );
    });

    for (const output of ['plain text', { value: 42 }, false, 0]) {
      it(`preserves legacy output ${JSON.stringify(output)}`, () => {
        for (const content of [undefined, []]) {
          assert.deepStrictEqual(
            convert({ ...response, output, content }).content,
            [
              {
                type: 'text',
                text:
                  typeof output === 'string' ? output : JSON.stringify(output),
              },
            ]
          );
        }
      });
    }

    it('rejects unsupported nested resource parts', () => {
      assert.throws(
        () =>
          convert({
            ...response,
            content: [{ resource: { uri: 'file:///image.png' } }],
          }),
        /Unsupported tool response content/
      );
    });

    it('retains multipart content in the outgoing request body', () => {
      const body = (runner as any).toAnthropicRequestBody('claude-haiku-4-5', {
        messages: [
          {
            role: 'model',
            content: [
              {
                toolRequest: {
                  name: response.name,
                  ref: response.ref,
                  input: {},
                },
              },
            ],
          },
          {
            role: 'tool',
            content: [{ toolResponse: { ...response, content: [image] } }],
          },
        ],
      });
      assert.deepStrictEqual(body.messages[1].content[0].content, [imageBlock]);
    });
  });
}
