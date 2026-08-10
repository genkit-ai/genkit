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
import { Response } from 'openai/core.mjs';
import { Transcription } from 'openai/resources/audio/transcriptions.mjs';
import {
  defineCompatOpenAISpeechModel,
  defineCompatOpenAITranscriptionModel,
  speechToGenerateResponse,
  toSttRequest,
  toTTSRequest,
  transcriptionToGenerateResponse,
} from '../src/audio';

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

describe('toTTSRequest', () => {
  it('should create speech request from text', () => {
    const request = {
      messages: [{ role: 'user', content: [{ text: 'Hello' }] }],
    } as GenerateRequest;

    const actualOutput = toTTSRequest('gpt-4o-mini-tts', request);
    expect(actualOutput).toStrictEqual({
      model: 'gpt-4o-mini-tts',
      input: 'Hello',
      voice: 'alloy',
    });
  });

  it('should remove undefined keys from options in toTTSRequest', () => {
    const request = {
      messages: [{ role: 'user', content: [{ text: 'Hello' }] }],
      config: {
        voice: 'echo',
        speed: undefined,
      },
    } as GenerateRequest;

    const actualOutput = toTTSRequest('gpt-4o-mini-tts', request);
    expect(actualOutput).toStrictEqual({
      model: 'gpt-4o-mini-tts',
      input: 'Hello',
      voice: 'echo',
    });
  });

  it('should run with requestBuilder', () => {
    const requestBuilder = jest.fn((_, params) => {
      (params as any).foo = 'bar';
    });

    const request = {
      messages: [{ role: 'user', content: [{ text: 'Hello' }] }],
      config: {
        voice: 'alloy',
      },
    } as GenerateRequest;

    const actualOutput = toTTSRequest(
      'gpt-4o-mini-tts',
      request,
      requestBuilder
    );

    expect(requestBuilder).toHaveBeenCalledTimes(1);
    expect(actualOutput).toHaveProperty('foo', 'bar');
  });
});

describe('speechToGenerateResponse', () => {
  it('should transform media correctly', async () => {
    const response = {
      arrayBuffer: async () => new Uint8Array([1, 2, 3, 4]).buffer,
    } as Response;

    const actualOutput = await speechToGenerateResponse(response);
    expect(actualOutput).toStrictEqual({
      message: {
        role: 'model',
        content: [
          {
            media: {
              contentType: 'audio/mpeg',
              url: 'data:audio/mpeg;base64,AQIDBA==',
            },
          },
        ],
      },
      finishReason: 'stop',
      raw: response,
    });
  });
});

describe('defineCompatOpenAISpeechModel runner', () => {
  it('should correctly run speech generation requests', async () => {
    const response = {
      arrayBuffer: async () => new Uint8Array([1, 2, 3, 4]).buffer,
    } as Response;

    const openaiClient = {
      audio: {
        speech: {
          create: jest.fn(async () => response),
        },
      },
    };
    const abortSignal = jest.fn();
    const runner = defineCompatOpenAISpeechModel({
      name: 'tts-1',
      client: openaiClient as unknown as OpenAI,
      pluginOptions: { name: 'openai' },
    });
    await runner(
      {
        messages: [{ role: 'user', content: [{ text: 'Hello' }] }],
      },
      {
        abortSignal: abortSignal as unknown as AbortSignal,
      }
    );
    expect(openaiClient.audio.speech.create).toHaveBeenCalledWith(
      {
        model: 'tts-1',
        input: 'Hello',
        voice: 'alloy',
      },
      { signal: abortSignal }
    );
  });
});

describe('toSttRequest', () => {
  it('should create transcription request from base64 audio', () => {
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

    const actualOutput = toSttRequest('whisper-1', request);
    expect(actualOutput).toStrictEqual({
      model: 'whisper-1',
      file: expect.any(File),
      prompt: '',
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

    const actualOutput = toSttRequest('whisper-1', request);
    expect(actualOutput).toStrictEqual({
      model: 'whisper-1',
      file: expect.any(File),
      prompt: '',
      response_format: 'verbose_json',
    });
  });

  it('should parse content type from data url when media.contentType is missing', () => {
    const request = {
      messages: [
        {
          role: 'user',
          content: [
            {
              media: {
                url: 'data:audio/wav;base64,aGVsbG8=',
              },
            },
          ],
        },
      ],
      output: { format: 'text' },
    } as GenerateRequest;

    const actualOutput = toSttRequest('whisper-1', request);
    expect(actualOutput).toStrictEqual({
      model: 'whisper-1',
      file: expect.any(File),
      prompt: '',
      response_format: 'text',
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

    expect(() => toSttRequest('whisper-1', request)).toThrowError(
      'No media found in the request'
    );
  });

  it('should throw error when output.format is json but custom format is incompatible (e.g., srt)', () => {
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

    expect(() => toSttRequest('whisper-1', request)).toThrowError(
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

    expect(() => toSttRequest('whisper-1', request)).toThrow(
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

    const actualOutput = toSttRequest('whisper-1', request, requestBuilder);

    expect(requestBuilder).toHaveBeenCalledTimes(1);
    expect(actualOutput).toHaveProperty('foo', 'bar');
  });
});

describe('transcriptionToGenerateResponse', () => {
  it('should transform transcription result correctly when result is Transcription object', () => {
    const result: Transcription = {
      text: 'Hello',
    };

    const actualOutput = transcriptionToGenerateResponse(result);
    expect(actualOutput).toStrictEqual({
      message: {
        role: 'model',
        content: [{ text: 'Hello' }],
      },
      finishReason: 'stop',
      raw: result,
    });
  });

  it('should transform transcription result correctly when result is string', () => {
    const result = 'Hello';

    const actualOutput = transcriptionToGenerateResponse(result);
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

describe('defineCompatOpenAITranscriptionModel runner', () => {
  it('should correctly run Transcription requests', async () => {
    const result: Transcription = {
      text: 'Hello',
    };

    const openaiClient = {
      audio: {
        transcriptions: {
          create: jest.fn(async () => result),
        },
      },
    };
    const abortSignal = jest.fn();
    const runner = defineCompatOpenAITranscriptionModel({
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
    expect(openaiClient.audio.transcriptions.create).toHaveBeenCalledWith(
      {
        model: 'whisper-1',
        file: expect.any(File),
        prompt: '',
        response_format: 'text',
        stream: false,
      },
      { signal: abortSignal }
    );
  });
});
