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
  interactionErrorCodeToGenkitStatus,
  interactionProcessStream,
  isInteractionContentBlockCode,
  modelName,
  parseInteractionStreamErrorText,
  parseRetryAfterMs,
  parseStreamErrorText,
  processStream,
} from '../../src/common/utils.js';
import { fromInteractionSync } from '../../src/googleai/interaction-converters.js';
import { InteractionSseEvent } from '../../src/googleai/interaction-types.js';

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

  describe('interactionProcessStream', () => {
    it('throws if response body is not found', () => {
      const mockResponse = new Response(null);
      assert.throws(
        () => interactionProcessStream(mockResponse),
        /Error processing stream because response.body not found/
      );
    });

    it('processes a valid stream into async generator and final aggregated response (happy path)', async () => {
      const encoder = new TextEncoder();
      const chunks = [
        'event: interaction.created\ndata: {"event_type":"interaction.created","interaction":{"id":"int-1","status":"in_progress"}}\n\n',
        'event: step.start\ndata: {"event_type":"step.start","index":0,"step":{"type":"model_output","content":[]}}\n\n',
        'event: step.delta\ndata: {"event_type":"step.delta","index":0,"delta":{"type":"text","text":"Hello"}}\n\n',
        'event: step.delta\ndata: {"event_type":"step.delta","index":0,"delta":{"type":"text","text":" World"}}\n\n',
        'event: step.stop\ndata: {"event_type":"step.stop","index":0}\n\n',
        'event: interaction.completed\ndata: {"event_type":"interaction.completed","interaction":{"id":"int-1","status":"completed"}}\n\n',
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
      const { stream: asyncStream, response } =
        interactionProcessStream(mockResponse);

      const events: InteractionSseEvent[] = [];
      for await (const event of asyncStream) {
        events.push(event);
      }

      assert.strictEqual(events.length, 6);
      assert.strictEqual(events[0].event_type, 'interaction.created');
      assert.strictEqual(events[2].event_type, 'step.delta');

      const finalInteraction = await response;
      assert.strictEqual(finalInteraction.id, 'int-1');
      assert.strictEqual(finalInteraction.status, 'completed');
      assert.strictEqual(finalInteraction.steps?.length, 1);
      assert.deepStrictEqual(finalInteraction.steps?.[0], {
        type: 'model_output',
        content: [{ type: 'text', text: 'Hello World' }],
      });
    });

    describe('assembling the final interaction from step deltas', () => {
      async function finalInteractionFrom(events: object[]) {
        const encoder = new TextEncoder();
        const stream = new ReadableStream({
          start(controller) {
            for (const event of events) {
              controller.enqueue(
                encoder.encode(`data: ${JSON.stringify(event)}\n\n`)
              );
            }
            controller.close();
          },
        });
        const { stream: asyncStream, response } = interactionProcessStream(
          new Response(stream)
        );
        for await (const _ of asyncStream) {
        }
        return response;
      }

      const start = (step: object) => ({
        event_type: 'step.start',
        index: 0,
        step,
      });
      const delta = (d: object) => ({
        event_type: 'step.delta',
        index: 0,
        delta: d,
      });
      const stop = { event_type: 'step.stop', index: 0 };

      it('keeps streamed image deltas in the final interaction', async () => {
        const interaction = await finalInteractionFrom([
          start({ type: 'model_output', content: [] }),
          delta({ type: 'text', text: 'Here is your image:' }),
          delta({ type: 'image', mime_type: 'image/png', data: 'AAAA' }),
          stop,
        ]);
        assert.deepStrictEqual(interaction.steps?.[0], {
          type: 'model_output',
          content: [
            { type: 'text', text: 'Here is your image:' },
            { type: 'image', mime_type: 'image/png', data: 'AAAA' },
          ],
        });
      });

      it('keeps audio, video and document deltas', async () => {
        const interaction = await finalInteractionFrom([
          start({ type: 'model_output', content: [] }),
          delta({ type: 'audio', mime_type: 'audio/wav', data: 'AU' }),
          delta({ type: 'video', uri: 'gs://bucket/v.mp4' }),
          delta({ type: 'document', mime_type: 'application/pdf', data: 'PD' }),
          stop,
        ]);
        assert.deepStrictEqual(
          (interaction.steps?.[0] as any).content.map((c: any) => c.type),
          ['audio', 'video', 'document']
        );
      });

      it('starts a new text block after a non-text block', async () => {
        const interaction = await finalInteractionFrom([
          start({ type: 'model_output', content: [] }),
          delta({ type: 'text', text: 'Before ' }),
          delta({ type: 'text', text: 'image.' }),
          delta({ type: 'image', mime_type: 'image/png', data: 'AAAA' }),
          delta({ type: 'text', text: 'After ' }),
          delta({ type: 'text', text: 'image.' }),
          stop,
        ]);
        assert.deepStrictEqual((interaction.steps?.[0] as any).content, [
          { type: 'text', text: 'Before image.' },
          { type: 'image', mime_type: 'image/png', data: 'AAAA' },
          { type: 'text', text: 'After image.' },
        ]);
      });

      it('attaches text_annotation_delta citations to the text block', async () => {
        const citation = {
          type: 'url_citation',
          url: 'https://example.com',
          title: 'Example',
          start_index: 0,
          end_index: 5,
        };
        const interaction = await finalInteractionFrom([
          start({ type: 'model_output', content: [] }),
          delta({ type: 'text', text: 'Paris ' }),
          delta({ type: 'text', text: 'is the capital.' }),
          delta({ type: 'text_annotation_delta', annotations: [citation] }),
          stop,
        ]);
        assert.deepStrictEqual((interaction.steps?.[0] as any).content, [
          {
            type: 'text',
            text: 'Paris is the capital.',
            annotations: [citation],
          },
        ]);
      });

      it('merges built-in tool deltas (google search) into their steps', async () => {
        // Event shapes observed from a live gemini-3.6-flash streaming call:
        // step.start carries only ids and an empty signature; arguments,
        // results and the real signature arrive in a delta of the same type.
        const at = (index: number, e: object) => ({ ...e, index });
        const interaction = await finalInteractionFrom([
          at(0, {
            event_type: 'step.start',
            step: {
              id: 'call_1',
              signature: '',
              type: 'google_search_call',
              search_type: 'web_search',
            },
          }),
          at(0, {
            event_type: 'step.delta',
            delta: {
              type: 'google_search_call',
              signature: 'sig-call',
              arguments: { queries: ['Nobel Prize in Physics winner'] },
            },
          }),
          at(0, { event_type: 'step.stop' }),
          at(1, {
            event_type: 'step.start',
            step: {
              call_id: 'call_1',
              signature: '',
              type: 'google_search_result',
            },
          }),
          at(1, {
            event_type: 'step.delta',
            delta: {
              type: 'google_search_result',
              signature: 'sig-result',
              result: [{ search_suggestions: '<div>...</div>' }],
              is_error: false,
            },
          }),
          at(1, { event_type: 'step.stop' }),
        ]);
        assert.deepStrictEqual(interaction.steps, [
          {
            id: 'call_1',
            signature: 'sig-call',
            type: 'google_search_call',
            search_type: 'web_search',
            arguments: { queries: ['Nobel Prize in Physics winner'] },
          },
          {
            call_id: 'call_1',
            signature: 'sig-result',
            type: 'google_search_result',
            result: [{ search_suggestions: '<div>...</div>' }],
            is_error: false,
          },
        ]);
      });

      it('merges code execution deltas into their steps', async () => {
        const at = (index: number, e: object) => ({ ...e, index });
        const interaction = await finalInteractionFrom([
          at(0, {
            event_type: 'step.start',
            step: { id: 'c1', signature: '', type: 'code_execution_call' },
          }),
          at(0, {
            event_type: 'step.delta',
            delta: {
              type: 'code_execution_call',
              arguments: { code: 'print(1)', language: 'python' },
            },
          }),
          at(1, {
            event_type: 'step.start',
            step: {
              call_id: 'c1',
              signature: '',
              type: 'code_execution_result',
            },
          }),
          at(1, {
            event_type: 'step.delta',
            delta: { type: 'code_execution_result', result: '1\n' },
          }),
        ]);
        assert.deepStrictEqual((interaction.steps?.[0] as any).arguments, {
          code: 'print(1)',
          language: 'python',
        });
        assert.strictEqual((interaction.steps?.[1] as any).result, '1\n');
      });

      it('does not overwrite streamed steps with interaction.completed', async () => {
        const interaction = await finalInteractionFrom([
          start({ type: 'model_output', content: [] }),
          delta({ type: 'text', text: 'streamed' }),
          stop,
          {
            event_type: 'interaction.completed',
            interaction: {
              id: 'v1_abc123',
              status: 'completed',
              steps: [],
              usage: { total_tokens: 3 },
            },
          },
        ]);
        assert.strictEqual(interaction.id, 'v1_abc123');
        assert.strictEqual(interaction.status, 'completed');
        assert.strictEqual(interaction.usage?.total_tokens, 3);
        assert.deepStrictEqual((interaction.steps?.[0] as any).content, [
          { type: 'text', text: 'streamed' },
        ]);
      });

      it('uses interaction.completed steps when none were streamed', async () => {
        const interaction = await finalInteractionFrom([
          {
            event_type: 'interaction.completed',
            interaction: {
              id: 'v1_abc123',
              status: 'completed',
              steps: [
                {
                  type: 'model_output',
                  content: [{ type: 'text', text: 'only here' }],
                },
              ],
            },
          },
        ]);
        assert.deepStrictEqual((interaction.steps?.[0] as any).content, [
          { type: 'text', text: 'only here' },
        ]);
      });
    });

    // End-to-end through the streaming path: a stream that ends with a failed
    // `interaction.completed` must reach `fromInteractionSync` with `status`
    // and `errors[]` intact, giving the same result as the non-streaming path.
    // The server includes `errors[]` on `interaction.completed` when populated.
    describe('failed interaction.completed through fromInteractionSync', () => {
      async function streamToFinalInteraction(chunks: string[]) {
        const encoder = new TextEncoder();
        const stream = new ReadableStream({
          start(controller) {
            for (const chunk of chunks) {
              controller.enqueue(encoder.encode(chunk));
            }
            controller.close();
          },
        });
        const { stream: asyncStream, response } = interactionProcessStream(
          new Response(stream)
        );
        for await (const _ of asyncStream) {
        }
        return response;
      }

      it('returns finishReason blocked for a post-execution safety block', async () => {
        const id = 'v1_abc123';
        const safetyMessage =
          'Request blocked due to safety violations (harmful content). Please modify your input and retry.';
        const completed = {
          event_type: 'interaction.completed',
          interaction: {
            id,
            status: 'failed',
            errors: [{ code: 'safety', message: safetyMessage }],
            usage: { total_input_tokens: 12, total_tokens: 12 },
          },
        };
        const interaction = await streamToFinalInteraction([
          `event: interaction.created\ndata: {"event_type":"interaction.created","interaction":{"id":"${id}","status":"in_progress"}}\n\n`,
          `event: interaction.completed\ndata: ${JSON.stringify(completed)}\n\n`,
        ]);
        assert.strictEqual(interaction.status, 'failed');
        assert.deepStrictEqual(interaction.errors, [
          { code: 'safety', message: safetyMessage },
        ]);

        const result = fromInteractionSync(interaction);
        assert.strictEqual(result.finishReason, 'blocked');
        assert.strictEqual(result.finishMessage, safetyMessage);
        assert.strictEqual(result.message?.metadata?.interactionId, id);
        assert.strictEqual(result.usage?.inputTokens, 12);
        assert.strictEqual(result.usage?.totalTokens, 12);
      });

      it('throws a retryable ABORTED GenkitError for malformed_function_call', async () => {
        const interaction = await streamToFinalInteraction([
          'event: interaction.completed\ndata: {"event_type":"interaction.completed","interaction":{"id":"int-1","status":"failed","errors":[{"code":"malformed_function_call","message":"Model generated invalid JSON syntax. Please retry the request."}]}}\n\n',
        ]);
        assert.throws(
          () => fromInteractionSync(interaction),
          (err: any) =>
            err instanceof GenkitError &&
            err.status === 'ABORTED' &&
            err.message.includes('Please retry the request.')
        );
      });

      it('throws a non-retryable UNKNOWN GenkitError when errors[] is absent', async () => {
        // `errors[]` is only included when populated, so a failed interaction
        // may arrive without it.
        const interaction = await streamToFinalInteraction([
          'event: interaction.completed\ndata: {"event_type":"interaction.completed","interaction":{"id":"int-1","status":"failed"}}\n\n',
        ]);
        assert.throws(
          () => fromInteractionSync(interaction),
          (err: any) =>
            err instanceof GenkitError &&
            err.status === 'UNKNOWN' &&
            err.message.includes('Interaction failed')
        );
      });
    });

    it('surfaces an error event in final response', async () => {
      const encoder = new TextEncoder();
      const chunks = [
        'event: error\ndata: {"event_type":"error","error":{"code":"quota_exceeded","message":"Quota exceeded"}}\n\n',
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
      const { stream: asyncStream, response } =
        interactionProcessStream(mockResponse);

      for await (const _ of asyncStream) {
      }

      await assert.rejects(
        async () => {
          await response;
        },
        (err: any) => {
          assert.ok(err instanceof GenkitError);
          assert.strictEqual(err.status, 'RESOURCE_EXHAUSTED');
          assert.ok(err.message.includes('[quota_exceeded] Quota exceeded'));
          return true;
        }
      );
    });

    it('maps an Interactions-style error event code (service_unavailable) to UNAVAILABLE', async () => {
      const encoder = new TextEncoder();
      const chunks = [
        'event: error\ndata: {"event_type":"error","error":{"code":"service_unavailable","message":"high demand"}}\n\n',
      ];
      const stream = new ReadableStream({
        start(controller) {
          for (const chunk of chunks) {
            controller.enqueue(encoder.encode(chunk));
          }
          controller.close();
        },
      });

      const { stream: asyncStream, response } = interactionProcessStream(
        new Response(stream)
      );
      for await (const _ of asyncStream) {
      }

      await assert.rejects(response, (err: any) => {
        assert.ok(err instanceof GenkitError);
        assert.strictEqual(err.status, 'UNAVAILABLE');
        assert.ok(err.message.includes('high demand'));
        return true;
      });
    });

    it('falls back to INTERNAL for an unrecognized error event code', async () => {
      const encoder = new TextEncoder();
      const chunks = [
        'event: error\ndata: {"event_type":"error","error":{"code":"something_weird","message":"boom"}}\n\n',
      ];
      const stream = new ReadableStream({
        start(controller) {
          for (const chunk of chunks) {
            controller.enqueue(encoder.encode(chunk));
          }
          controller.close();
        },
      });

      const { stream: asyncStream, response } = interactionProcessStream(
        new Response(stream)
      );
      for await (const _ of asyncStream) {
      }

      await assert.rejects(response, (err: any) => {
        assert.ok(err instanceof GenkitError);
        assert.strictEqual(err.status, 'INTERNAL');
        return true;
      });
    });

    it('surfaces an Interactions JSON error body (service_unavailable) as a retryable UNAVAILABLE GenkitError', async () => {
      // Real-world shape observed from /v1beta/interactions when overloaded.
      const encoder = new TextEncoder();
      const errorBody = JSON.stringify({
        error: {
          message:
            'gemini-flash-latest is currently experiencing high demand, spikes in demand are usually temporary. Please try again later.',
          code: 'service_unavailable',
        },
      });
      const stream = new ReadableStream({
        start(controller) {
          controller.enqueue(encoder.encode(errorBody));
          controller.close();
        },
      });

      const { stream: asyncStream, response } = interactionProcessStream(
        new Response(stream)
      );
      response.catch(() => {});

      await assert.rejects(
        async () => {
          for await (const _ of asyncStream) {
          }
        },
        (err: any) => {
          assert.ok(err instanceof GenkitError, 'Expected GenkitError');
          assert.strictEqual(err.status, 'UNAVAILABLE');
          assert.ok(err.message.includes('high demand'));
          return true;
        }
      );
    });

    it('surfaces a JSON error body (HTTP 200 with error) using parseInteractionStreamErrorText', async () => {
      const encoder = new TextEncoder();
      const errorBody = JSON.stringify({
        error: {
          code: 503,
          message: 'The model is overloaded. Please try again later.',
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
      const { stream: asyncStream, response } =
        interactionProcessStream(mockResponse);
      response.catch(() => {});

      try {
        for await (const _ of asyncStream) {
        }
        assert.fail('Should have thrown on error body');
      } catch (err: any) {
        assert.ok(err instanceof GenkitError);
        assert.strictEqual(err.status, 'UNAVAILABLE');
        assert.ok(err.message.includes('overloaded'));
      }
    });

    it('reassembles arguments_delta across step.delta events and parses JSON at step.stop', async () => {
      const encoder = new TextEncoder();
      const chunks = [
        'event: step.start\ndata: {"event_type":"step.start","index":0,"step":{"type":"function_call","name":"getWeather","id":"call-1"}}\n\n',
        'event: step.delta\ndata: {"event_type":"step.delta","index":0,"delta":{"type":"arguments_delta","arguments":"{\\"city\\": "}}\n\n',
        'event: step.delta\ndata: {"event_type":"step.delta","index":0,"delta":{"type":"arguments_delta","arguments":"\\"Seattle\\"}"}}\n\n',
        'event: step.stop\ndata: {"event_type":"step.stop","index":0}\n\n',
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
      const { stream: asyncStream, response } =
        interactionProcessStream(mockResponse);

      for await (const _ of asyncStream) {
      }

      const finalInteraction = await response;
      assert.strictEqual(finalInteraction.steps?.length, 1);
      const step: any = finalInteraction.steps?.[0];
      assert.strictEqual(step.type, 'function_call');
      assert.deepStrictEqual(step.arguments, { city: 'Seattle' });
    });

    it('does not cause an unhandled rejection when only the stream is consumed on error', async () => {
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
        const { stream: asyncStream } = interactionProcessStream(mockResponse);

        await assert.rejects(async () => {
          for await (const _ of asyncStream) {
          }
        });

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

  describe('interactionErrorCodeToGenkitStatus', () => {
    it('maps every known Interactions error code', () => {
      const expected: Record<string, string> = {
        service_unavailable: 'UNAVAILABLE',
        rate_limit_exceeded: 'RESOURCE_EXHAUSTED',
        quota_exceeded: 'RESOURCE_EXHAUSTED',
        invalid_request: 'INVALID_ARGUMENT',
        parameter_unknown: 'INVALID_ARGUMENT',
        failed_precondition: 'FAILED_PRECONDITION',
        out_of_range: 'OUT_OF_RANGE',
        agent_max_token_limit: 'INVALID_ARGUMENT',
        model_not_found: 'NOT_FOUND',
        not_found: 'NOT_FOUND',
        authentication: 'UNAUTHENTICATED',
        permission_denied: 'PERMISSION_DENIED',
        already_exists: 'ALREADY_EXISTS',
        aborted: 'ABORTED',
        cancelled: 'CANCELLED',
        deadline_exceeded: 'DEADLINE_EXCEEDED',
        api_error: 'INTERNAL',
        unimplemented: 'UNIMPLEMENTED',
        malformed_function_call: 'ABORTED',
        unexpected_tool_call: 'ABORTED',
        no_image: 'ABORTED',
        safety: 'FAILED_PRECONDITION',
        recitation: 'FAILED_PRECONDITION',
        language: 'FAILED_PRECONDITION',
        prohibited_content: 'FAILED_PRECONDITION',
        spii: 'FAILED_PRECONDITION',
        blocklist: 'FAILED_PRECONDITION',
        image_safety: 'FAILED_PRECONDITION',
        image_prohibited_content: 'FAILED_PRECONDITION',
        image_recitation: 'FAILED_PRECONDITION',
        image_other: 'FAILED_PRECONDITION',
        content_blocked: 'FAILED_PRECONDITION',
        jailbreak: 'FAILED_PRECONDITION',
        model_armor: 'FAILED_PRECONDITION',
      };
      for (const [code, status] of Object.entries(expected)) {
        assert.strictEqual(
          interactionErrorCodeToGenkitStatus(code),
          status,
          `code "${code}"`
        );
      }
    });

    it('matches codes case-insensitively and ignores surrounding whitespace', () => {
      assert.strictEqual(
        interactionErrorCodeToGenkitStatus('SERVICE_UNAVAILABLE'),
        'UNAVAILABLE'
      );
      assert.strictEqual(
        interactionErrorCodeToGenkitStatus(' rate_limit_exceeded '),
        'RESOURCE_EXHAUSTED'
      );
    });

    it('does not match inherited object properties', () => {
      assert.strictEqual(
        interactionErrorCodeToGenkitStatus('constructor'),
        undefined
      );
      assert.strictEqual(
        interactionErrorCodeToGenkitStatus('__proto__'),
        undefined
      );
    });

    it('maps numeric and stringified numeric HTTP codes', () => {
      assert.strictEqual(
        interactionErrorCodeToGenkitStatus(503),
        'UNAVAILABLE'
      );
      assert.strictEqual(
        interactionErrorCodeToGenkitStatus('429'),
        'RESOURCE_EXHAUSTED'
      );
    });

    it('returns undefined for unmappable codes', () => {
      assert.strictEqual(
        interactionErrorCodeToGenkitStatus('weird'),
        undefined
      );
      assert.strictEqual(interactionErrorCodeToGenkitStatus(418), undefined);
      assert.strictEqual(interactionErrorCodeToGenkitStatus(''), undefined);
      assert.strictEqual(
        interactionErrorCodeToGenkitStatus(undefined),
        undefined
      );
      assert.strictEqual(interactionErrorCodeToGenkitStatus({}), undefined);
    });
  });

  describe('isInteractionContentBlockCode', () => {
    it('recognizes content-policy codes case-insensitively', () => {
      assert.strictEqual(isInteractionContentBlockCode('safety'), true);
      assert.strictEqual(isInteractionContentBlockCode('MODEL_ARMOR'), true);
      assert.strictEqual(isInteractionContentBlockCode('image_other'), true);
    });

    it('rejects non-content-policy codes and non-strings', () => {
      assert.strictEqual(
        isInteractionContentBlockCode('malformed_function_call'),
        false
      );
      assert.strictEqual(
        isInteractionContentBlockCode('service_unavailable'),
        false
      );
      assert.strictEqual(isInteractionContentBlockCode(undefined), false);
      assert.strictEqual(isInteractionContentBlockCode(400), false);
    });
  });

  describe('parseInteractionStreamErrorText', () => {
    it('maps an Interactions string code to a GenkitError status', () => {
      const err = parseInteractionStreamErrorText(
        JSON.stringify({
          error: { code: 'service_unavailable', message: 'high demand' },
        })
      );
      assert.ok(err instanceof GenkitError, 'Expected GenkitError');
      assert.strictEqual(err.status, 'UNAVAILABLE');
      assert.ok(err.message.includes('high demand'));
    });

    it('prefers an explicit Google-style status field', () => {
      const err = parseInteractionStreamErrorText(
        JSON.stringify({
          error: { code: 503, message: 'overloaded', status: 'UNAVAILABLE' },
        })
      );
      assert.ok(err instanceof GenkitError, 'Expected GenkitError');
      assert.strictEqual(err.status, 'UNAVAILABLE');
    });

    it('returns UNKNOWN for an unrecognized string code', () => {
      const err = parseInteractionStreamErrorText(
        JSON.stringify({ error: { code: 'weird', message: 'hmm' } })
      );
      assert.ok(err instanceof GenkitError, 'Expected GenkitError');
      assert.strictEqual(err.status, 'UNKNOWN');
    });

    it('returns a generic Error for non-JSON text', () => {
      const err = parseInteractionStreamErrorText('not json at all');
      assert.ok(!(err instanceof GenkitError));
      assert.ok(err.message.includes('Failed to parse stream'));
    });
  });
});
