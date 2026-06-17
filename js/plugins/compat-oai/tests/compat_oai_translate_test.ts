/**
 * Copyright 2024 The Fire Company
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

import { describe, expect, it, jest } from '@jest/globals';
import { GenerateRequest } from 'genkit';
import OpenAI from 'openai';
import { Translation } from 'openai/resources/audio/translations.mjs';
import {
  defineCompatOpenAITranslationModel,
  toTranslationRequest,
  translationToGenerateResponse,
} from '../src/translate';

jest.mock('genkit/model', () => {
  const originalModule =
    jest.requireActual<typeof import('genkit/model')>('genkit/model');
  return {
    ...originalModule,
    defineModel: jest.fn((_, runner) => {
      return runner;
    }),
  };
});

describe('toTranslationRequest', () => {
  it('should create translation request from base64 audio', () => {
    const request = {
      messages: [
        {
          role: 'user',
          content: [
            {
              media: {
                contentType: 'audio/wav',
                url: 'data:audio/wav;base64,aGVsbG8=',
              },
            },
            { text: 'Translate this file' },
          ],
        },
      ],
      output: { format: 'text' },
    } as GenerateRequest;

    const actualOutput = toTranslationRequest('whisper-1', request);
    expect(actualOutput).toStrictEqual({
      model: 'whisper-1',
      file: expect.any(File),
      prompt: 'Translate this file',
      response_format: 'text',
    });
  });

  it('should allow verbose_json when output.format is json', () => {
    const request = {
      messages: [
        {
          role: 'user',
          content: [
            {
              media: {
                contentType: 'audio/wav',
                url: 'data:audio/wav;base64,aGVsbG8=',
              },
            },
          ],
        },
      ],
      output: { format: 'json' },
      config: { response_format: 'verbose_json' },
    } as GenerateRequest;

    const actualOutput = toTranslationRequest('whisper-1', request);
    expect(actualOutput).toStrictEqual({
      model: 'whisper-1',
      file: expect.any(File),
      prompt: '',
      response_format: 'verbose_json',
    });
  });

  it('should throw error when media.url is missing', () => {
    const request = {
      messages: [
        {
          role: 'user',
          content: [
            {
              media: {
                contentType: 'audio/wav',
              },
            },
          ],
        },
      ],
      output: { format: 'text' },
    } as GenerateRequest;

    expect(() => toTranslationRequest('whisper-1', request)).toThrowError(
      'No media found in the request'
    );
  });

  it('should throw error when output.format is json but custom format is incompatible', () => {
    const request = {
      messages: [
        {
          role: 'user',
          content: [
            {
              media: {
                contentType: 'audio/wav',
                url: 'data:audio/wav;base64,aGVsbG8=',
              },
            },
          ],
        },
      ],
      output: { format: 'json' },
      config: { response_format: 'srt' },
    } as GenerateRequest;

    expect(() => toTranslationRequest('whisper-1', request)).toThrowError(
      'Custom response format srt is not compatible with output format json'
    );
  });

  it('should throw error when output.format is media', () => {
    const request = {
      messages: [
        {
          role: 'user',
          content: [
            {
              media: {
                contentType: 'audio/wav',
                url: 'data:audio/wav;base64,aGVsbG8=',
              },
            },
          ],
        },
      ],
      output: { format: 'media' },
    } as GenerateRequest;

    expect(() => toTranslationRequest('whisper-1', request)).toThrow(
      'Output format media is not supported.'
    );
  });

  it('should run with requestBuilder', () => {
    const requestBuilder = jest.fn((_, params) => {
      (params as any).foo = 'bar';
    });

    const request = {
      messages: [
        {
          role: 'user',
          content: [
            {
              media: {
                contentType: 'audio/wav',
                url: 'data:audio/wav;base64,aGVsbG8=',
              },
            },
          ],
        },
      ],
      output: { format: 'text' },
    } as GenerateRequest;

    const actualOutput = toTranslationRequest(
      'whisper-1',
      request,
      requestBuilder
    );

    expect(requestBuilder).toHaveBeenCalledTimes(1);
    expect(actualOutput).toHaveProperty('foo', 'bar');
  });
});

describe('translationToGenerateResponse', () => {
  it('should transform translation result correctly when result is Translation object', () => {
    const result: Translation = {
      text: 'Hello',
    };

    const actualOutput = translationToGenerateResponse(result);
    expect(actualOutput).toStrictEqual({
      message: {
        role: 'model',
        content: [{ text: 'Hello' }],
      },
      finishReason: 'stop',
      raw: result,
    });
  });

  it('should transform translation result correctly when result is string', () => {
    const result = 'Hello';

    const actualOutput = translationToGenerateResponse(result);
    expect(actualOutput).toStrictEqual({
      message: {
        role: 'model',
        content: [{ text: 'Hello' }],
      },
      finishReason: 'stop',
      raw: result,
    });
  });
});

describe('defineCompatOpenAITranslationModel runner', () => {
  it('should correctly run Translation requests', async () => {
    const result: Translation = {
      text: 'Hello',
    };

    const openaiClient = {
      audio: {
        translations: {
          create: jest.fn(async () => result),
        },
      },
    };
    const abortSignal = jest.fn();
    const runner = defineCompatOpenAITranslationModel({
      name: 'whisper-1',
      client: openaiClient as unknown as OpenAI,
    });
    await runner(
      {
        messages: [
          {
            role: 'user',
            content: [
              {
                media: {
                  url: 'data:audio/wav;base64,aGVsbG8=',
                  contentType: 'audio/wav',
                },
              },
            ],
          },
        ],
      },
      {
        abortSignal: abortSignal as unknown as AbortSignal,
      }
    );
    expect(openaiClient.audio.translations.create).toHaveBeenCalledWith(
      {
        model: 'whisper-1',
        file: expect.any(File),
        prompt: '',
        response_format: 'text',
      },
      { signal: abortSignal }
    );
  });
});
