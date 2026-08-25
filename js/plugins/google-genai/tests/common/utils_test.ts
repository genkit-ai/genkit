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

import * as assert from 'assert';
import { GenkitError, embedderRef, modelRef } from 'genkit';
import { GenerateRequest } from 'genkit/model';
import { describe, it } from 'node:test';
import {
  FinishReason,
  GenerateContentResponse,
} from '../../src/common/types.js';
import {
  TEST_ONLY,
  checkModelName,
  checkSupportedMimeType,
  cleanSchema,
  displayUrl,
  extractErrMsg,
  extractMedia,
  extractMimeType,
  extractText,
  extractVersion,
  httpStatusToGenkitStatus,
  modelName,
  parseRetryAfterMs,
  parseStreamErrorText,
  processStream,
} from '../../src/common/utils.js';

const { aggregateResponses } = TEST_ONLY;

describe('Common Utils', () => {
  describe('extractErrMsg', () => {
    it('extracts message from an Error object', () => {
      const error = new Error('This is a test error.');
      assert.strictEqual(extractErrMsg(error), 'This is a test error.');
    });

    it('returns the string if error is a string', () => {
      const error = 'A simple string error.';
      assert.strictEqual(extractErrMsg(error), 'A simple string error.');
    });

    it('stringifies other error types', () => {
      const error = { code: 500, message: 'Object error' };
      assert.strictEqual(
        extractErrMsg(error),
        '{"code":500,"message":"Object error"}'
      );
    });
  });

  describe('extractVersion', () => {
    it('should return version from modelRef if present', () => {
      const ref = modelRef({
        name: 'vertexai/gemini-1.5-pro',
        version: 'gemini-1.5-pro-001',
      });
      assert.strictEqual(extractVersion(ref), 'gemini-1.5-pro-001');
    });

    it('should extract version from name if version field is missing', () => {
      const ref = modelRef({ name: 'vertexai/gemini-2.5-flash' });
      assert.strictEqual(extractVersion(ref), 'gemini-2.5-flash');
    });

    it('should work with embedderRef', () => {
      const ref = embedderRef({ name: 'vertexai/gemini-embedding-001' });
      assert.strictEqual(extractVersion(ref), 'gemini-embedding-001');
    });
  });

  describe('modelName', () => {
    it('extracts model name from a full path', () => {
      assert.strictEqual(
        modelName('models/googleai/gemini-1.5-pro'),
        'gemini-1.5-pro'
      );
      assert.strictEqual(
        modelName('vertexai/gemini-2.5-flash'),
        'gemini-2.5-flash'
      );
      assert.strictEqual(modelName('model/foo'), 'foo');
      assert.strictEqual(modelName('embedders/bar'), 'bar');
      assert.strictEqual(modelName('background-model/baz'), 'baz');
    });

    it('returns the name if no known prefix is present', () => {
      assert.strictEqual(modelName('gemini-1.0-ultra'), 'gemini-1.0-ultra');
    });

    it('handles undefined input', () => {
      assert.strictEqual(modelName(undefined), undefined);
    });

    it('handles empty string input', () => {
      assert.strictEqual(modelName(''), '');
    });
  });

  describe('checkModelName', () => {
    it('extracts model name from a full path', () => {
      const name = 'models/vertexai/gemini-1.5-pro';
      assert.strictEqual(checkModelName(name), 'gemini-1.5-pro');
    });

    it('returns name if no prefix', () => {
      assert.strictEqual(checkModelName('foo-bar'), 'foo-bar');
    });

    it('throws an error for undefined input', () => {
      assert.throws(
        () => checkModelName(undefined),
        (err: any) => {
          assert.ok(err instanceof GenkitError, 'Expected GenkitError');
          assert.strictEqual(err.status, 'INVALID_ARGUMENT');
          assert.strictEqual(
            err.message,
            'INVALID_ARGUMENT: Model name is required.'
          );
          return true;
        }
      );
    });

    it('throws an error for an empty string', () => {
      assert.throws(
        () => checkModelName(''),
        (err: any) => {
          assert.ok(err instanceof GenkitError, 'Expected GenkitError');
          assert.strictEqual(err.status, 'INVALID_ARGUMENT');
          assert.strictEqual(
            err.message,
            'INVALID_ARGUMENT: Model name is required.'
          );
          return true;
        }
      );
    });
  });

  describe('extractText', () => {
    it('extracts text from the last message', () => {
      const request: GenerateRequest = {
        messages: [
          { role: 'user', content: [{ text: 'Hello there.' }] },
          { role: 'model', content: [{ text: 'How can I help?' }] },
          { role: 'user', content: [{ text: 'Tell me a joke.' }] },
        ],
      };
      assert.strictEqual(extractText(request), 'Tell me a joke.');
    });

    it('concatenates multiple text parts in the last message', () => {
      const request: GenerateRequest = {
        messages: [
          {
            role: 'user',
            content: [{ text: 'Part 1. ' }, { text: 'Part 2.' }],
          },
        ],
      };
      assert.strictEqual(extractText(request), 'Part 1. Part 2.');
    });

    it('ignores non-text parts in the last message', () => {
      const request: GenerateRequest = {
        messages: [
          {
            role: 'user',
            content: [
              { text: 'A ' },
              { media: { url: 'data:image/jpeg;base64,IMAGEDATA' } },
              { text: 'B' },
            ],
          },
        ],
      };
      assert.strictEqual(extractText(request), 'A B');
    });

    it('returns an empty string if there are no text parts in the last message', () => {
      const request: GenerateRequest = {
        messages: [
          {
            role: 'user',
            content: [{ media: { url: 'data:image/jpeg;base64,IMAGEDATA' } }],
          },
        ],
      };
      assert.strictEqual(extractText(request), '');
    });

    it('returns an empty string if there are no messages', () => {
      const request: GenerateRequest = {
        messages: [],
      };
      assert.strictEqual(extractText(request), '');
    });
  });

  describe('extractMimeType', () => {
    it('extracts from data URL with base64', () => {
      assert.strictEqual(
        extractMimeType('data:image/png;base64,iVBORw0KGgoAAAANSUhEUgA...'),
        'image/png'
      );
      assert.strictEqual(
        extractMimeType('data:application/pdf;base64,JVBERi0xLjQKJ...'),
        'application/pdf'
      );
    });

    it('returns empty string for invalid data URL format', () => {
      assert.strictEqual(extractMimeType('data:image/png'), '');
      assert.strictEqual(extractMimeType('data:,text'), '');
    });

    it('extracts from known file extensions', () => {
      assert.strictEqual(extractMimeType('image.jpg'), 'image/jpeg');
      assert.strictEqual(extractMimeType('path/to/document.png'), 'image/png');
      assert.strictEqual(extractMimeType('video.mp4'), 'video/mp4');
    });

    it('returns empty string for unknown file extensions', () => {
      assert.strictEqual(extractMimeType('file.unknown'), '');
      assert.strictEqual(extractMimeType('archive.zip'), '');
    });

    it('returns empty string for URL without extension', () => {
      assert.strictEqual(extractMimeType('http://example.com/image'), '');
    });

    it('returns empty string for undefined or empty input', () => {
      assert.strictEqual(extractMimeType(undefined), '');
      assert.strictEqual(extractMimeType(''), '');
    });
  });

  describe('checkSupportedMimeType', () => {
    const supported = ['image/jpeg', 'image/png'];
    it('should not throw for supported mime types', () => {
      assert.doesNotThrow(() =>
        checkSupportedMimeType(
          { url: 'test.jpg', contentType: 'image/jpeg' },
          supported
        )
      );
      assert.doesNotThrow(() =>
        checkSupportedMimeType(
          { url: 'test.png', contentType: 'image/png' },
          supported
        )
      );
    });

    it('should throw GenkitError for unsupported mime types', () => {
      try {
        checkSupportedMimeType(
          { url: 'test.gif', contentType: 'image/gif' },
          supported
        );
        assert.fail('Should have thrown');
      } catch (e: any) {
        assert.ok(e instanceof GenkitError, 'Expected GenkitError');
        assert.strictEqual(e.status, 'INVALID_ARGUMENT');
        assert.ok(
          e.message.includes('Invalid mimeType for test.gif: "image/gif"')
        );
        assert.ok(
          e.message.includes('Supported mimeTypes: image/jpeg, image/png')
        );
      }
    });

    it('should throw GenkitError if contentType is missing', () => {
      try {
        checkSupportedMimeType({ url: 'test.jpg' }, supported);
        assert.fail('Should have thrown');
      } catch (e: any) {
        assert.ok(e instanceof GenkitError, 'Expected GenkitError');
        assert.strictEqual(e.status, 'INVALID_ARGUMENT');
        assert.ok(
          e.message.includes('Invalid mimeType for test.jpg: "undefined"')
        );
      }
    });
  });

  describe('displayUrl', () => {
    it('should return the full URL if short', () => {
      const url = 'http://example.com/short';
      assert.strictEqual(displayUrl(url), url);
    });

    it('should truncate long URLs', () => {
      const longUrl =
        'http://example.com/this/is/a/very/long/url/that/needs/truncation/to/fit';
      const expected = 'http://example.com/this/i...t/needs/truncation/to/fit';
      assert.strictEqual(displayUrl(longUrl), expected);
    });

    it('should handle URLs exactly at the limit', () => {
      const url = 'a'.repeat(50);
      assert.strictEqual(displayUrl(url), url);
    });
  });

  describe('extractMedia', () => {
    const imageMedia = {
      url: 'data:image/png;base64,IMAGEDATA',
      contentType: 'image/png',
    };
    const videoMedia = {
      url: 'data:video/mp4;base64,VIDEODATA',
      contentType: 'video/mp4',
    };

    it('extracts any media from the last message if no params', () => {
      const request: GenerateRequest = {
        messages: [
          { role: 'user', content: [{ text: 'A ' }, { media: imageMedia }] },
        ],
      };
      assert.deepStrictEqual(extractMedia(request, {}), imageMedia);
    });

    it('extracts media matching metadataType', () => {
      const request: GenerateRequest = {
        messages: [
          {
            role: 'user',
            content: [
              { media: imageMedia, metadata: { type: 'image' } },
              { media: videoMedia, metadata: { type: 'video' } },
            ],
          },
        ],
      };
      assert.deepStrictEqual(
        extractMedia(request, { metadataType: 'video' }),
        videoMedia
      );
      assert.deepStrictEqual(
        extractMedia(request, { metadataType: 'image' }),
        imageMedia
      );
    });

    it('extracts media with no metadata type if isDefault is true', () => {
      const request: GenerateRequest = {
        messages: [
          {
            role: 'user',
            content: [
              { media: imageMedia },
              { media: videoMedia, metadata: { type: 'video' } },
            ],
          },
        ],
      };
      assert.deepStrictEqual(
        extractMedia(request, { metadataType: 'image', isDefault: true }),
        imageMedia
      );
    });

    it('does not extract media with different metadataType even if isDefault is true', () => {
      const request: GenerateRequest = {
        messages: [
          {
            role: 'user',
            content: [{ media: videoMedia, metadata: { type: 'video' } }],
          },
        ],
      };
      assert.strictEqual(
        extractMedia(request, { metadataType: 'image', isDefault: true }),
        undefined
      );
    });

    it('returns undefined if no media matches metadataType', () => {
      const request: GenerateRequest = {
        messages: [
          {
            role: 'user',
            content: [{ media: imageMedia, metadata: { type: 'image' } }],
          },
        ],
      };
      assert.strictEqual(
        extractMedia(request, { metadataType: 'video' }),
        undefined
      );
    });

    it('infers contentType if missing', () => {
      const request: GenerateRequest = {
        messages: [
          {
            role: 'user',
            content: [{ media: { url: 'data:image/jpeg;base64,DATA' } }],
          },
        ],
      };
      const result = extractMedia(request, {});
      assert.deepStrictEqual(result, {
        url: 'data:image/jpeg;base64,DATA',
        contentType: 'image/jpeg',
      });
    });

    it('returns undefined if no media parts in the last message', () => {
      const request: GenerateRequest = {
        messages: [{ role: 'user', content: [{ text: 'No media' }] }],
      };
      assert.strictEqual(extractMedia(request, {}), undefined);
    });

    it('returns undefined for empty messages array', () => {
      const request: GenerateRequest = { messages: [] };
      assert.strictEqual(extractMedia(request, {}), undefined);
    });
  });

  describe('cleanSchema', () => {
    it('strips $schema and additionalProperties', () => {
      const schema = {
        type: 'object',
        properties: { name: { type: 'string' } },
        $schema: 'http://json-schema.org/draft-07/schema#',
        additionalProperties: false,
      };
      const cleaned = cleanSchema(schema);
      assert.deepStrictEqual(cleaned, {
        type: 'object',
        properties: { name: { type: 'string' } },
      });
    });

    it('handles nested objects', () => {
      const schema = {
        type: 'object',
        properties: {
          user: {
            type: 'object',
            properties: { id: { type: 'number' } },
            additionalProperties: true,
          },
        },
      };
      const cleaned = cleanSchema(schema);
      assert.deepStrictEqual(cleaned, {
        type: 'object',
        properties: {
          user: {
            type: 'object',
            properties: { id: { type: 'number' } },
          },
        },
      });
    });

    it('converts type ["string", "null"] to "string"', () => {
      const schema = {
        type: 'object',
        properties: {
          name: { type: ['string', 'null'] },
          age: { type: ['number', 'null'] },
        },
      };
      const cleaned = cleanSchema(schema);
      assert.deepStrictEqual(cleaned, {
        type: 'object',
        properties: {
          name: { type: 'string' },
          age: { type: 'number' },
        },
      });
    });

    it('converts type ["null", "boolean"] to "boolean"', () => {
      const schema = {
        type: 'object',
        properties: {
          isActive: { type: ['null', 'boolean'] },
        },
      };
      const cleaned = cleanSchema(schema);
      assert.deepStrictEqual(cleaned, {
        type: 'object',
        properties: {
          isActive: { type: 'boolean' },
        },
      });
    });

    it('leaves other properties untouched', () => {
      const schema = {
        type: 'string',
        description: 'A name',
        maxLength: 100,
      };
      const cleaned = cleanSchema(schema);
      assert.deepStrictEqual(cleaned, schema);
    });
  });

  describe('aggregateResponses', () => {
    it('should aggregate streaming function call parts', () => {
      const responses: GenerateContentResponse[] = [
        {
          candidates: [
            {
              index: 0,
              content: {
                role: 'model',
                parts: [
                  {
                    functionCall: {
                      name: 'findFlights',
                      id: '1234',
                      willContinue: true,
                    },
                    thoughtSignature: 'thoughtSignature1234',
                  },
                ],
              },
            },
          ],
        },
        {
          candidates: [
            {
              index: 0,
              content: {
                role: 'model',
                parts: [
                  {
                    functionCall: {
                      willContinue: true,
                      partialArgs: [
                        {
                          jsonPath: '$.flights[0].departure_airport',
                          stringValue: 'SFO',
                        },
                      ],
                    },
                  },
                ],
              },
            },
          ],
        },
        {
          candidates: [
            {
              index: 0,
              content: {
                role: 'model',
                parts: [
                  {
                    functionCall: {
                      willContinue: true,
                      partialArgs: [
                        {
                          jsonPath: '$.flights[0].arrival_airport',
                          stringValue: 'JFK',
                        },
                      ],
                    },
                  },
                ],
              },
            },
          ],
        },
        {
          candidates: [
            {
              index: 0,
              content: {
                role: 'model',
                parts: [
                  {
                    functionCall: {
                      name: 'findFlights',
                    },
                  },
                ],
              },
            },
          ],
        },
      ];

      const aggregated = aggregateResponses(responses);

      const expected = {
        candidates: [
          {
            index: 0,
            content: {
              role: 'model',
              parts: [
                {
                  functionCall: {
                    name: 'findFlights',
                    id: '1234',
                    args: {
                      flights: [
                        {
                          departure_airport: 'SFO',
                          arrival_airport: 'JFK',
                        },
                      ],
                    },
                  },
                  thoughtSignature: 'thoughtSignature1234',
                },
              ],
            },
          },
        ],
      };

      assert.deepStrictEqual(aggregated, expected);
    });

    it('should correctly aggregate toolCall and toolResponse parts across chunks', () => {
      const responses: GenerateContentResponse[] = [
        {
          candidates: [
            {
              index: 0,
              content: {
                role: 'model',
                parts: [
                  {
                    thoughtSignature: 'sig1',
                    toolCall: {
                      toolType: 'GOOGLE_SEARCH_WEB',
                      args: { queries: ['Canada'] },
                      id: 'goccvdqb',
                    },
                  },
                  { text: '' },
                ],
              },
            },
          ],
        },
        {
          candidates: [
            {
              index: 0,
              content: {
                role: 'model',
                parts: [
                  {
                    thoughtSignature: 'sig2',
                    toolResponse: {
                      toolType: 'GOOGLE_SEARCH_WEB',
                      response: { search_suggestions: '...' },
                      id: 'goccvdqb',
                    },
                  },
                ],
              },
            },
          ],
        },
        {
          candidates: [
            {
              index: 0,
              content: {
                role: 'model',
                parts: [
                  {
                    thoughtSignature: 'sig3',
                    functionCall: {
                      name: 'getWeather',
                      args: { location: 'Iqaluit, NU' },
                      id: 'c46t8dh5',
                    },
                  },
                  { text: '' },
                ],
              },
              finishReason: FinishReason.STOP,
            },
          ],
        },
      ];

      const aggregated = aggregateResponses(responses);

      const expected = {
        candidates: [
          {
            index: 0,
            finishReason: FinishReason.STOP,
            content: {
              role: 'model',
              parts: [
                {
                  thoughtSignature: 'sig1',
                  toolCall: {
                    toolType: 'GOOGLE_SEARCH_WEB',
                    args: { queries: ['Canada'] },
                    id: 'goccvdqb',
                  },
                },
                { text: '' },
                {
                  thoughtSignature: 'sig2',
                  toolResponse: {
                    toolType: 'GOOGLE_SEARCH_WEB',
                    response: { search_suggestions: '...' },
                    id: 'goccvdqb',
                  },
                },
                {
                  thoughtSignature: 'sig3',
                  functionCall: {
                    name: 'getWeather',
                    args: { location: 'Iqaluit, NU' },
                    id: 'c46t8dh5',
                  },
                },
                { text: '' },
              ],
            },
          },
        ],
      };

      assert.deepStrictEqual(aggregated, expected);
    });

    it('should properly aggregate citationMetadata and groundingMetadata arrays', () => {
      const responses: GenerateContentResponse[] = [
        {
          candidates: [
            {
              index: 0,
              content: { role: 'model', parts: [{ text: 'Hello' }] },
              citationMetadata: {
                citationSources: [{ uri: 'https://example.com/1' }],
              },
              groundingMetadata: {
                groundingChunks: [{ web: { uri: 'https://example.com/a' } }],
                webSearchQueries: ['query1'],
              },
            },
          ],
        },
        {
          candidates: [
            {
              index: 0,
              content: { role: 'model', parts: [{ text: ' World' }] },
              citationMetadata: {
                citationSources: [{ uri: 'https://example.com/2' }],
              },
              groundingMetadata: {
                groundingChunks: [{ web: { uri: 'https://example.com/b' } }],
                webSearchQueries: ['query2'],
              },
            },
          ],
        },
      ];

      const aggregated = aggregateResponses(responses);

      assert.deepStrictEqual(
        aggregated.candidates?.[0].citationMetadata?.citationSources,
        [{ uri: 'https://example.com/1' }, { uri: 'https://example.com/2' }]
      );
      assert.deepStrictEqual(
        aggregated.candidates?.[0].groundingMetadata?.groundingChunks,
        [
          { web: { uri: 'https://example.com/a' } },
          { web: { uri: 'https://example.com/b' } },
        ]
      );
      assert.deepStrictEqual(
        aggregated.candidates?.[0].groundingMetadata?.webSearchQueries,
        ['query1', 'query2']
      );
      assert.strictEqual(
        aggregated.candidates?.[0].content.parts[1].text,
        ' World'
      );
    });
  });

  describe('parseRetryAfterMs', () => {
    it('parses delay-seconds to milliseconds', () => {
      assert.strictEqual(parseRetryAfterMs('60'), 60_000);
      assert.strictEqual(parseRetryAfterMs('0'), 0);
      assert.strictEqual(parseRetryAfterMs('120'), 120_000);
    });

    it('parses fractional delay-seconds', () => {
      assert.strictEqual(parseRetryAfterMs('1.5'), 1_500);
    });

    it('parses HTTP-date format', () => {
      const futureDate = new Date(Date.now() + 30_000);
      const result = parseRetryAfterMs(futureDate.toUTCString());
      assert.ok(result !== undefined);
      // Should be approximately 30 seconds (allow some tolerance for test execution time)
      assert.ok(
        result > 28_000 && result <= 31_000,
        `Expected ~30000ms, got ${result}ms`
      );
    });

    it('returns 0 for HTTP-date in the past', () => {
      const pastDate = new Date(Date.now() - 60_000);
      assert.strictEqual(parseRetryAfterMs(pastDate.toUTCString()), 0);
    });

    it('returns 0 for negative delay-seconds (parsed as ancient date)', () => {
      // '-5' fails the seconds >= 0 check, but JS Date parses it as year -5,
      // which is in the past, so Math.max(0, past - now) = 0.
      assert.strictEqual(parseRetryAfterMs('-5'), 0);
    });

    it('returns undefined for unparseable values', () => {
      assert.strictEqual(parseRetryAfterMs('not-a-number-or-date'), undefined);
    });

    it('returns undefined for empty string', () => {
      assert.strictEqual(parseRetryAfterMs(''), undefined);
    });

    it('returns undefined for whitespace-only string', () => {
      assert.strictEqual(parseRetryAfterMs('   '), undefined);
    });
  });

  describe('processStream', () => {
    it('throws if response body is not found', () => {
      const mockResponse = new Response(null);
      assert.throws(
        () => processStream(mockResponse),
        /Error processing stream because response.body not found/
      );
    });

    it('processes a valid stream into async generator and final aggregated response', async () => {
      const encoder = new TextEncoder();
      const chunks = [
        'data: {"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"}}]}\n\n',
        'data: {"candidates":[{"content":{"parts":[{"text":" World"}],"role":"model"}}]}\n\n',
      ];

      const stream = new ReadableStream({
        start(controller) {
          for (const chunk of chunks) {
            controller.enqueue(encoder.encode(chunk));
          }
          controller.close();
        },
      });

      const mockResponse = new Response(stream);
      const { stream: asyncStream, response } = processStream(mockResponse);

      const yieldedValues: GenerateContentResponse[] = [];
      for await (const val of asyncStream) {
        yieldedValues.push(val);
      }

      assert.strictEqual(yieldedValues.length, 2);
      assert.strictEqual(
        yieldedValues[0].candidates?.[0].content.parts[0].text,
        'Hello'
      );
      assert.strictEqual(
        yieldedValues[1].candidates?.[0].content.parts[0].text,
        ' World'
      );

      const finalResponse = await response;
      assert.strictEqual(
        finalResponse.candidates?.[0].content.parts[0].text,
        'Hello'
      );
      assert.strictEqual(
        finalResponse.candidates?.[0].content.parts[1].text,
        ' World'
      );
    });

    it('throws an error if JSON is malformed in the stream', async () => {
      const encoder = new TextEncoder();
      const chunks = [
        'data: {"candidates":[\n\n', // broken json
      ];

      const stream = new ReadableStream({
        start(controller) {
          for (const chunk of chunks) {
            controller.enqueue(encoder.encode(chunk));
          }
          controller.close();
        },
      });

      const mockResponse = new Response(stream);
      const { stream: asyncStream, response } = processStream(mockResponse);

      // Silence the parallel promise rejection so it doesn't fail the test runner asynchronously
      response.catch(() => {});

      try {
        for await (const val of asyncStream) {
          // should throw
        }
        assert.fail('Should have thrown on malformed JSON');
      } catch (err: any) {
        assert.ok(err.message.includes('Error parsing JSON response:'));
      }
    });

    it('throws an error if stream yields trailing text without proper data formatting', async () => {
      const encoder = new TextEncoder();
      const chunks = [
        'data: {"candidates":[]}\n\n',
        'trailing unformatted string',
      ];

      const stream = new ReadableStream({
        start(controller) {
          for (const chunk of chunks) {
            controller.enqueue(encoder.encode(chunk));
          }
          controller.close();
        },
      });

      const mockResponse = new Response(stream);
      const { stream: asyncStream, response } = processStream(mockResponse);

      // Silence the parallel promise rejection so it doesn't fail the test runner asynchronously
      response.catch(() => {});

      try {
        for await (const val of asyncStream) {
          // First one is yielded, but then trailing fails parsing
        }
        assert.fail('Should have thrown on trailing data');
      } catch (err: any) {
        assert.ok(err.message.includes('Failed to parse stream'));
      }
    });

    it('surfaces a JSON error body (HTTP 200 with error) as a GenkitError', async () => {
      // Overloaded models sometimes return HTTP 200 and put the real error in
      // the body as plain JSON (not an SSE `data:` frame).
      const encoder = new TextEncoder();
      const errorBody = JSON.stringify({
        error: {
          code: 503,
          message:
            'This model is currently experiencing high demand. Spikes in demand are usually temporary. Please try again later.',
          status: 'UNAVAILABLE',
        },
      });

      const stream = new ReadableStream({
        start(controller) {
          controller.enqueue(encoder.encode(errorBody));
          controller.close();
        },
      });

      const mockResponse = new Response(stream);
      const { stream: asyncStream, response } = processStream(mockResponse);

      // Silence the parallel promise rejection so it doesn't fail the test runner asynchronously
      response.catch(() => {});

      try {
        for await (const val of asyncStream) {
          // should throw
        }
        assert.fail('Should have thrown on error body');
      } catch (err: any) {
        assert.ok(err instanceof GenkitError, 'Expected GenkitError');
        assert.strictEqual(err.status, 'UNAVAILABLE');
        assert.ok(err.message.includes('high demand'));
      }
    });

    it('does not cause an unhandled rejection when only the stream is consumed on error', async () => {
      // Regression test: previously the teed `response` promise would reject
      // with no handler attached (the caller only awaits `stream`), producing
      // an unhandled promise rejection that crashed the process.
      const encoder = new TextEncoder();
      const errorBody = JSON.stringify({
        error: { code: 503, message: 'overloaded', status: 'UNAVAILABLE' },
      });

      const rejections: unknown[] = [];
      const onRejection = (reason: unknown) => rejections.push(reason);
      process.on('unhandledRejection', onRejection);

      try {
        const stream = new ReadableStream({
          start(controller) {
            controller.enqueue(encoder.encode(errorBody));
            controller.close();
          },
        });

        const mockResponse = new Response(stream);
        // Intentionally ignore `response` to simulate the streaming consumer.
        const { stream: asyncStream } = processStream(mockResponse);

        await assert.rejects(async () => {
          for await (const val of asyncStream) {
            // should throw
          }
        });

        // Give any pending microtasks/rejections a chance to surface.
        await new Promise((resolve) => setTimeout(resolve, 10));
        assert.strictEqual(
          rejections.length,
          0,
          `Expected no unhandled rejections, got: ${rejections}`
        );
      } finally {
        process.off('unhandledRejection', onRejection);
      }
    });
  });

  describe('httpStatusToGenkitStatus', () => {
    it('maps known HTTP status codes to Genkit statuses', () => {
      assert.strictEqual(httpStatusToGenkitStatus(400), 'INVALID_ARGUMENT');
      assert.strictEqual(httpStatusToGenkitStatus(401), 'UNAUTHENTICATED');
      assert.strictEqual(httpStatusToGenkitStatus(403), 'PERMISSION_DENIED');
      assert.strictEqual(httpStatusToGenkitStatus(404), 'NOT_FOUND');
      assert.strictEqual(httpStatusToGenkitStatus(429), 'RESOURCE_EXHAUSTED');
      assert.strictEqual(httpStatusToGenkitStatus(499), 'CANCELLED');
      assert.strictEqual(httpStatusToGenkitStatus(500), 'INTERNAL');
      assert.strictEqual(httpStatusToGenkitStatus(503), 'UNAVAILABLE');
      assert.strictEqual(httpStatusToGenkitStatus(504), 'DEADLINE_EXCEEDED');
    });

    it('returns UNKNOWN for unmapped or missing codes', () => {
      assert.strictEqual(httpStatusToGenkitStatus(418), 'UNKNOWN');
      assert.strictEqual(httpStatusToGenkitStatus(undefined), 'UNKNOWN');
    });
  });

  describe('parseStreamErrorText', () => {
    it('returns a GenkitError with status from the error status field', () => {
      const text = JSON.stringify({
        error: { code: 503, message: 'overloaded', status: 'UNAVAILABLE' },
      });
      const err = parseStreamErrorText(text);
      assert.ok(err instanceof GenkitError, 'Expected GenkitError');
      assert.strictEqual(err.status, 'UNAVAILABLE');
      assert.ok(err.message.includes('overloaded'));
    });

    it('falls back to the HTTP code when status is not a valid StatusName', () => {
      const text = JSON.stringify({
        error: { code: 429, message: 'too many' },
      });
      const err = parseStreamErrorText(text);
      assert.ok(err instanceof GenkitError, 'Expected GenkitError');
      assert.strictEqual(err.status, 'RESOURCE_EXHAUSTED');
    });

    it('coerces a stringified HTTP code to a number when mapping status', () => {
      const text = JSON.stringify({
        error: { code: '503', message: 'overloaded' },
      });
      const err = parseStreamErrorText(text);
      assert.ok(err instanceof GenkitError, 'Expected GenkitError');
      assert.strictEqual(err.status, 'UNAVAILABLE');
    });

    it('uses a default message when the error message is not a string', () => {
      const text = JSON.stringify({
        error: { code: 500, message: { nested: 'oops' } },
      });
      const err = parseStreamErrorText(text);
      assert.ok(err instanceof GenkitError, 'Expected GenkitError');
      assert.strictEqual(err.status, 'INTERNAL');
      assert.ok(err.message.includes('Error streaming from the model'));
    });

    it('truncates long non-JSON payloads in the fallback message', () => {
      const longText = 'x'.repeat(1000);
      const err = parseStreamErrorText(longText);
      assert.ok(!(err instanceof GenkitError));
      assert.ok(err.message.includes('...'));
      // 'Failed to parse stream: ' + 500 chars + '...'
      assert.ok(
        err.message.length < 600,
        `Expected truncated message, got length ${err.message.length}`
      );
    });

    it('returns a generic Error for non-JSON text', () => {
      const err = parseStreamErrorText('not json at all');
      assert.ok(!(err instanceof GenkitError));
      assert.ok(err.message.includes('Failed to parse stream'));
      assert.ok(err.message.includes('not json at all'));
    });

    it('returns a generic Error for JSON that is not an error body', () => {
      const err = parseStreamErrorText(JSON.stringify({ hello: 'world' }));
      assert.ok(!(err instanceof GenkitError));
      assert.ok(err.message.includes('Failed to parse stream'));
    });
  });
});
