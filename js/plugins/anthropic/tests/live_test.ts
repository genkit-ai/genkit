/**
 * Copyright 2025 Google LLC
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

/**
 * Live integration tests that call the real Anthropic API.
 * Only runs when ANTHROPIC_API_KEY environment variable is set.
 *
 * Run with: ANTHROPIC_API_KEY=your-key pnpm test:live
 */

import * as assert from 'assert';
import { genkit, z } from 'genkit';
import { describe, it } from 'node:test';
import {
  anthropic,
  anthropicDocument,
  type AnthropicCitation,
} from '../src/index.js';

const API_KEY = process.env.ANTHROPIC_API_KEY;

describe('Live Anthropic API Tests', { skip: !API_KEY }, () => {
  it('should work with short model name claude-haiku-4-5', async () => {
    const ai = genkit({
      plugins: [anthropic({ apiKey: API_KEY })],
    });

    const result = await ai.generate({
      model: 'anthropic/claude-haiku-4-5',
      prompt: 'Say "hello" and nothing else.',
    });

    assert.ok(result.text.toLowerCase().includes('hello'));
  });

  it('should work with anthropic.model() helper', async () => {
    const ai = genkit({
      plugins: [anthropic({ apiKey: API_KEY })],
    });

    const result = await ai.generate({
      model: anthropic.model('claude-haiku-4-5'),
      prompt: 'Say "hello" and nothing else.',
    });

    assert.ok(result.text.toLowerCase().includes('hello'));
  });

  it('should return structured output matching the schema', async () => {
    const ai = genkit({
      plugins: [anthropic({ apiKey: API_KEY, apiVersion: 'beta' })],
    });

    const schema = z.object({
      name: z.string(),
      age: z.number(),
      city: z.string(),
      isStudent: z.boolean(),
      isEmployee: z.boolean(),
      isRetired: z.boolean(),
      isUnemployed: z.boolean(),
      isDisabled: z.boolean(),
    });

    const result = await ai.generate({
      model: 'anthropic/claude-sonnet-4-5',
      prompt:
        'Generate a fictional person with name "Alice", age 30, and city "New York". Return only the JSON.',
      output: { schema, format: 'json', constrained: true },
    });

    const parsed = result.output;
    assert.ok(parsed, 'Should have parsed output');
    assert.deepStrictEqual(
      { name: parsed.name, age: parsed.age, city: parsed.city },
      { name: 'Alice', age: 30, city: 'New York' }
    );

    // Check that boolean fields are present and are actually booleans
    for (const key of [
      'isStudent',
      'isEmployee',
      'isRetired',
      'isUnemployed',
      'isDisabled',
    ]) {
      assert.strictEqual(
        typeof parsed[key],
        'boolean',
        `Field ${key} should be a boolean but got: ${typeof parsed[key]}`
      );
    }
  });

  it('should return citations from a plain text document', async () => {
    const ai = genkit({
      plugins: [anthropic({ apiKey: API_KEY, apiVersion: 'beta' })],
    });

    const result = await ai.generate({
      model: anthropic.model('claude-sonnet-4-5'),
      messages: [
        {
          role: 'user',
          content: [
            anthropicDocument({
              source: {
                type: 'text',
                data: 'The grass is green. The sky is blue. Water is wet.',
              },
              title: 'Basic Facts',
              citations: { enabled: true },
            }),
            { text: 'What color is the grass? Cite your source.' },
          ],
        },
      ],
    });

    assert.ok(result.text, 'Should have response text');
    assert.ok(
      result.text.toLowerCase().includes('green'),
      'Response should mention green'
    );

    // Extract citations from response parts
    const citations =
      result.message?.content?.flatMap(
        (part) => (part.metadata?.citations as AnthropicCitation[]) || []
      ) ?? [];

    assert.ok(
      citations && citations.length > 0,
      'Should have at least one citation'
    );

    // Verify citation structure
    const citation = citations[0];
    assert.strictEqual(
      citation.type,
      'char_location',
      'Should be a char_location citation'
    );
    assert.ok(citation.citedText, 'Citation should have cited text');
    assert.strictEqual(
      citation.documentIndex,
      0,
      'Should reference first document'
    );
  });

  it('should return citations with streaming enabled', async () => {
    const ai = genkit({
      plugins: [anthropic({ apiKey: API_KEY, apiVersion: 'beta' })],
    });

    const streamedChunks: string[] = [];

    const result = await ai.generate({
      model: anthropic.model('claude-sonnet-4-5'),
      messages: [
        {
          role: 'user',
          content: [
            anthropicDocument({
              source: {
                type: 'text',
                data: 'Cats are mammals. Dogs are also mammals. Birds have feathers.',
              },
              title: 'Animal Facts',
              citations: { enabled: true },
            }),
            { text: 'Are cats mammals? Cite your source.' },
          ],
        },
      ],
      streamingCallback: (chunk) => {
        if (chunk.text) {
          streamedChunks.push(chunk.text);
        }
      },
    });

    assert.ok(result.text, 'Should have response text');
    assert.ok(
      streamedChunks.length > 0,
      'Should have received streaming chunks'
    );

    // Extract citations from final response
    const citations =
      result.message?.content?.flatMap(
        (part) => (part.metadata?.citations as AnthropicCitation[]) || []
      ) ?? [];

    assert.ok(
      citations && citations.length > 0,
      'Should have at least one citation'
    );
  });

  it('should return citations using stable API', async () => {
    const ai = genkit({
      plugins: [anthropic({ apiKey: API_KEY })], // No apiVersion = stable
    });

    const result = await ai.generate({
      model: anthropic.model('claude-sonnet-4-5'),
      messages: [
        {
          role: 'user',
          content: [
            anthropicDocument({
              source: {
                type: 'text',
                data: 'The ocean is blue. The sun is yellow. Snow is white.',
              },
              title: 'Color Facts',
              citations: { enabled: true },
            }),
            { text: 'What color is the ocean? Cite your source.' },
          ],
        },
      ],
    });

    assert.ok(result.text, 'Should have response text');
    assert.ok(
      result.text.toLowerCase().includes('blue'),
      'Response should mention blue'
    );

    // Extract citations from response parts
    const citations =
      result.message?.content?.flatMap(
        (part) => (part.metadata?.citations as AnthropicCitation[]) || []
      ) ?? [];

    assert.ok(
      citations && citations.length > 0,
      'Should have at least one citation with stable API'
    );

    // Verify citation structure
    const citation = citations[0];
    assert.strictEqual(
      citation.type,
      'char_location',
      'Should be a char_location citation'
    );
    assert.ok(citation.citedText, 'Citation should have cited text');
  });

  it('should return citations from a custom content document with content_block_location', async () => {
    const ai = genkit({
      plugins: [anthropic({ apiKey: API_KEY, apiVersion: 'beta' })],
    });

    const result = await ai.generate({
      model: anthropic.model('claude-sonnet-4-5'),
      messages: [
        {
          role: 'user',
          content: [
            anthropicDocument({
              source: {
                type: 'content',
                content: [
                  { type: 'text', text: 'Fact 1: Dogs are mammals.' },
                  { type: 'text', text: 'Fact 2: Cats are also mammals.' },
                  { type: 'text', text: 'Fact 3: Birds have feathers.' },
                  { type: 'text', text: 'Fact 4: Fish live in water.' },
                ],
              },
              title: 'Animal Facts',
              citations: { enabled: true },
            }),
            {
              text: 'What do dogs and cats have in common? Cite your source with block references.',
            },
          ],
        },
      ],
    });

    assert.ok(result.text, 'Should have response text');
    assert.ok(
      result.text.toLowerCase().includes('mammal'),
      'Response should mention mammals'
    );

    // Extract citations from response parts
    const citations =
      result.message?.content?.flatMap(
        (part) => (part.metadata?.citations as AnthropicCitation[]) || []
      ) ?? [];

    assert.ok(
      citations && citations.length > 0,
      'Should have at least one citation'
    );

    // Verify at least one citation is content_block_location type
    const contentBlockCitations = citations.filter(
      (c) => c.type === 'content_block_location'
    );
    assert.ok(
      contentBlockCitations.length > 0,
      'Should have at least one content_block_location citation'
    );

    // Verify content_block_location citation structure
    const contentBlockCitation = contentBlockCitations[0];
    assert.strictEqual(
      contentBlockCitation.type,
      'content_block_location',
      'Should be a content_block_location citation'
    );
    assert.ok(
      contentBlockCitation.citedText,
      'Citation should have cited text'
    );
    assert.strictEqual(
      contentBlockCitation.documentIndex,
      0,
      'Should reference first document'
    );
    assert.ok(
      typeof contentBlockCitation.startBlockIndex === 'number',
      'Citation should have startBlockIndex'
    );
    assert.ok(
      typeof contentBlockCitation.endBlockIndex === 'number',
      'Citation should have endBlockIndex'
    );
    assert.ok(
      contentBlockCitation.startBlockIndex >= 0,
      'startBlockIndex should be non-negative'
    );
    assert.ok(
      contentBlockCitation.endBlockIndex >=
        contentBlockCitation.startBlockIndex,
      'endBlockIndex should be >= startBlockIndex'
    );
  });

  it('should return citations from multiple documents with correct document indexing', async () => {
    const ai = genkit({
      plugins: [anthropic({ apiKey: API_KEY, apiVersion: 'beta' })],
    });

    const result = await ai.generate({
      model: anthropic.model('claude-sonnet-4-5'),
      messages: [
        {
          role: 'user',
          content: [
            anthropicDocument({
              source: {
                type: 'text',
                data: 'The capital of France is Paris. The capital of Germany is Berlin.',
              },
              title: 'European Capitals',
              citations: { enabled: true },
            }),
            anthropicDocument({
              source: {
                type: 'text',
                data: 'The capital of Japan is Tokyo. The capital of China is Beijing.',
              },
              title: 'Asian Capitals',
              citations: { enabled: true },
            }),
            {
              text: 'What are the capitals of France and Japan? Cite your sources for each.',
            },
          ],
        },
      ],
    });

    assert.ok(result.text, 'Should have response text');
    assert.ok(
      result.text.toLowerCase().includes('paris'),
      'Response should mention Paris'
    );
    assert.ok(
      result.text.toLowerCase().includes('tokyo'),
      'Response should mention Tokyo'
    );

    // Extract citations from response parts
    const citations =
      result.message?.content?.flatMap(
        (part) => (part.metadata?.citations as AnthropicCitation[]) || []
      ) ?? [];

    assert.ok(
      citations && citations.length > 0,
      'Should have at least one citation'
    );

    // Verify citations reference different documents
    const documentIndices = new Set(citations.map((c) => c.documentIndex));
    assert.ok(
      documentIndices.size >= 1,
      'Should have citations from at least one document'
    );
    assert.ok(
      documentIndices.has(0) || documentIndices.has(1),
      'Should have citations from document 0 or 1'
    );

    // Verify citation structure for each document index
    for (const citation of citations) {
      assert.ok(
        citation.documentIndex === 0 || citation.documentIndex === 1,
        `Citation documentIndex should be 0 or 1, got ${citation.documentIndex}`
      );
      assert.ok(citation.citedText, 'Citation should have cited text');
      assert.strictEqual(
        citation.type,
        'char_location',
        'Text document citations should be char_location type'
      );

      // Verify char_location specific fields
      if (citation.type === 'char_location') {
        assert.ok(
          typeof citation.startCharIndex === 'number',
          'Citation should have startCharIndex'
        );
        assert.ok(
          typeof citation.endCharIndex === 'number',
          'Citation should have endCharIndex'
        );
        assert.ok(
          citation.endCharIndex >= citation.startCharIndex,
          'endCharIndex should be >= startCharIndex'
        );
      }
    }

    // If we have citations from both documents, verify they reference different content
    if (documentIndices.size === 2) {
      const doc0Citations = citations.filter((c) => c.documentIndex === 0);
      const doc1Citations = citations.filter((c) => c.documentIndex === 1);
      assert.ok(
        doc0Citations.length > 0,
        'Should have citations from document 0'
      );
      assert.ok(
        doc1Citations.length > 0,
        'Should have citations from document 1'
      );
    }
  });

  it('should throw descriptive error for invalid image media types in document content', async () => {
    const ai = genkit({
      plugins: [anthropic({ apiKey: API_KEY, apiVersion: 'beta' })],
    });

    // Test that invalid media type throws descriptive error
    await assert.rejects(
      async () => {
        await ai.generate({
          model: anthropic.model('claude-sonnet-4-5'),
          messages: [
            {
              role: 'user',
              content: [
                anthropicDocument({
                  source: {
                    type: 'content',
                    content: [
                      {
                        type: 'image',
                        source: {
                          type: 'base64',
                          mediaType: 'image/bmp', // Invalid - not supported
                          data: 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
                        },
                      },
                    ],
                  },
                  citations: { enabled: true },
                }),
                { text: 'What is in this image?' },
              ],
            },
          ],
        });
      },
      (error: Error) => {
        assert.ok(
          error.message.includes('Unsupported image media type'),
          'Error should mention unsupported media type'
        );
        assert.ok(
          error.message.includes('image/bmp'),
          'Error should include the invalid media type'
        );
        assert.ok(
          error.message.includes('image/jpeg') ||
            error.message.includes('image/png') ||
            error.message.includes('image/gif') ||
            error.message.includes('image/webp'),
          'Error should include supported types'
        );
        return true;
      }
    );
  });

  it('should handle valid image media types in document content gracefully', async () => {
    const ai = genkit({
      plugins: [anthropic({ apiKey: API_KEY, apiVersion: 'beta' })],
    });

    // Test that valid media types work correctly
    // Using a minimal valid PNG (1x1 transparent pixel)
    const validPngBase64 =
      'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

    const result = await ai.generate({
      model: anthropic.model('claude-sonnet-4-5'),
      messages: [
        {
          role: 'user',
          content: [
            anthropicDocument({
              source: {
                type: 'content',
                content: [
                  { type: 'text', text: 'This document contains an image.' },
                  {
                    type: 'image',
                    source: {
                      type: 'base64',
                      mediaType: 'image/png', // Valid media type
                      data: validPngBase64,
                    },
                  },
                ],
              },
              title: 'Document with Image',
              citations: { enabled: true },
            }),
            { text: 'What does the document say?' },
          ],
        },
      ],
    });

    // Should not throw and should return a response
    assert.ok(result.text, 'Should have response text');
    assert.ok(result.text.length > 0, 'Response should not be empty');
  });
});
