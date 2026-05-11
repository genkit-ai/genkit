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

import { describe, expect, it } from '@jest/globals';
import type { MessageData } from '../../src/types/model';
import type { PromptFrontmatter } from '../../src/types/prompt';
import { fromMessages } from '../../src/utils/prompt';

describe('fromMessages', () => {
  it('builds a template from messages', () => {
    const frontmatter: PromptFrontmatter = {
      name: 'my-prompt',
      model: 'googleai/gemini-pro',
      config: {
        temperature: 0.5,
      },
    };
    const messages: MessageData[] = [
      { role: 'user', content: [{ text: 'Who are you?' }] },
      {
        role: 'model',
        content: [
          { text: 'I am Oz -- the Great and Powerful.' },
          { media: { url: 'https://example.com/image.jpg' } },
        ],
      },
    ];
    const expected =
      '---\n' +
      'name: my-prompt\n' +
      'model: googleai/gemini-pro\n' +
      'config:\n' +
      '  temperature: 0.5\n' +
      '---\n' +
      '\n' +
      '{{role "user"}}\n' +
      'Who are you?\n' +
      '\n' +
      '{{role "model"}}\n' +
      'I am Oz -- the Great and Powerful.{{media url:https://example.com/image.jpg}}\n';
    expect(fromMessages(frontmatter, messages)).toStrictEqual(expected);
  });

  it('handles toolRequest by omitting the entire message', () => {
    const frontmatter: PromptFrontmatter = {
      model: 'googleai/gemini-pro',
      use: [{ name: 'test-middleware', config: { foo: 'bar' } }],
    };
    const messages: MessageData[] = [
      {
        role: 'user',
        content: [
          { text: 'Hello' },
          { reasoning: 'Thinking...' } as any,
          { toolRequest: { name: 'myTool' } } as any,
        ],
      },
    ];

    const expected =
      '---\n' +
      'model: googleai/gemini-pro\n' +
      'use:\n' +
      '  - name: test-middleware\n' +
      '    config:\n' +
      '      foo: bar\n' +
      '---\n' +
      '\n' +
      '{{! Some advanced message types, such as tool requests/responses, have been omitted from the history. See comments inline for more details. }}\n' +
      '\n' +
      '{{! message with role "user" omitted (toolRequest). }}\n';

    expect(fromMessages(frontmatter, messages)).toStrictEqual(expected);
  });

  it('omits messages entirely composed of unsupported parts', () => {
    const frontmatter: PromptFrontmatter = { model: 'model' };
    const messages: MessageData[] = [
      {
        role: 'model',
        content: [
          { toolResponse: { name: 'myTool', output: 'result' } } as any,
        ],
      },
    ];

    const expected =
      '---\n' +
      'model: model\n' +
      '---\n' +
      '\n' +
      '{{! Some advanced message types, such as tool requests/responses, have been omitted from the history. See comments inline for more details. }}\n' +
      '\n' +
      '{{! message with role "model" omitted (toolResponse). }}\n';

    expect(fromMessages(frontmatter, messages)).toStrictEqual(expected);
  });

  it('omits messages composed of other unsupported parts with "unsupported content" reason', () => {
    const frontmatter: PromptFrontmatter = { model: 'model' };
    const messages: MessageData[] = [
      {
        role: 'model',
        content: [{ reasoning: 'Thinking...' } as any],
      },
    ];

    const expected =
      '---\n' +
      'model: model\n' +
      '---\n' +
      '\n' +
      '{{! Some advanced message types, such as tool requests/responses, have been omitted from the history. See comments inline for more details. }}\n' +
      '\n' +
      '{{! message with role "model" omitted (unsupported content). }}\n';

    expect(fromMessages(frontmatter, messages)).toStrictEqual(expected);
  });

  it('handles mixed support messages without toolRequest by commenting parts', () => {
    const frontmatter: PromptFrontmatter = { model: 'model' };
    const messages: MessageData[] = [
      {
        role: 'user',
        content: [
          { text: 'Here is data: ' },
          { data: { foo: 'bar' } } as any,
          { text: ' and more text.' },
        ],
      },
    ];

    const expected =
      '---\n' +
      'model: model\n' +
      '---\n' +
      '\n' +
      '{{! Some advanced message types, such as tool requests/responses, have been omitted from the history. See comments inline for more details. }}\n' +
      '\n' +
      '{{role "user"}}\n' +
      'Here is data: {{! data part omitted }} and more text.\n';

    expect(fromMessages(frontmatter, messages)).toStrictEqual(expected);
  });

  it('recursively cleans empty objects and arrays from frontmatter', () => {
    const frontmatter: any = {
      model: 'googleai/gemini-pro',
      use: [
        {
          name: 'fallback',
          config: {},
        },
      ],
      tools: [],
      config: {
        safetySettings: [],
      },
    };
    const messages: any[] = [];

    const expected =
      '---\n' +
      'model: googleai/gemini-pro\n' +
      'use:\n' +
      '  - name: fallback\n' +
      '---\n';

    expect(fromMessages(frontmatter, messages)).toStrictEqual(expected);
  });
});
