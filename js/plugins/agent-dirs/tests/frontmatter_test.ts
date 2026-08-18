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

import * as assert from 'node:assert';
import { test } from 'node:test';
import { parseInstructionsSource } from '../src/frontmatter.js';

test('file without frontmatter is all body', () => {
  const parsed = parseInstructionsSource('You are a helpful assistant.\n');
  assert.deepStrictEqual(parsed.frontmatter, {});
  assert.strictEqual(parsed.body, 'You are a helpful assistant.');
});

test('frontmatter and body split, body trimmed', () => {
  const parsed = parseInstructionsSource(
    '---\ndescription: A helper.\nmodel: vertexai/gemini-2.5-flash\nconfig:\n  temperature: 0.2\ndelegates:\n  - shipping\n---\n\nYou are a helper.\n'
  );
  assert.deepStrictEqual(parsed.frontmatter, {
    description: 'A helper.',
    model: 'vertexai/gemini-2.5-flash',
    config: { temperature: 0.2 },
    delegates: ['shipping'],
  });
  assert.strictEqual(parsed.body, 'You are a helper.');
});

test('empty frontmatter block yields empty mapping', () => {
  const parsed = parseInstructionsSource('---\n---\nBody.\n');
  assert.deepStrictEqual(parsed.frontmatter, {});
  assert.strictEqual(parsed.body, 'Body.');
});

test('frontmatter-only file yields empty body', () => {
  const parsed = parseInstructionsSource('---\ndescription: X\n---\n');
  assert.deepStrictEqual(parsed.frontmatter, { description: 'X' });
  assert.strictEqual(parsed.body, '');
});

test('CRLF line endings', () => {
  const parsed = parseInstructionsSource(
    '---\r\ndescription: X\r\n---\r\nBody.\r\n'
  );
  assert.deepStrictEqual(parsed.frontmatter, { description: 'X' });
  assert.strictEqual(parsed.body, 'Body.');
});

test('closing fence at end of file without trailing newline', () => {
  const parsed = parseInstructionsSource('---\ndescription: X\n---');
  assert.deepStrictEqual(parsed.frontmatter, { description: 'X' });
  assert.strictEqual(parsed.body, '');
});

test('a thematic break (----) does not open frontmatter', () => {
  const parsed = parseInstructionsSource('----\nBody.\n');
  assert.deepStrictEqual(parsed.frontmatter, {});
  assert.strictEqual(parsed.body, '----\nBody.');
});

test('unterminated fence throws', () => {
  assert.throws(
    () => parseInstructionsSource('---\ndescription: X\nBody without close.'),
    /frontmatter is not closed/
  );
});

test('invalid YAML throws', () => {
  assert.throws(
    () => parseInstructionsSource('---\ndescription: [unclosed\n---\nBody.'),
    /invalid frontmatter YAML/
  );
});

test('non-mapping frontmatter throws', () => {
  assert.throws(
    () => parseInstructionsSource('---\n- a\n- b\n---\nBody.'),
    /must be a YAML mapping.*list/
  );
  assert.throws(
    () => parseInstructionsSource('---\nhello\n---\nBody.'),
    /must be a YAML mapping/
  );
});

test('handlebars in body survive verbatim', () => {
  const parsed = parseInstructionsSource('---\n---\nUse {{tone}} always.');
  assert.strictEqual(parsed.body, 'Use {{tone}} always.');
});
