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
import OpenAI from 'openai';
import { Transcription } from 'openai/resources/audio/transcriptions.mjs';
import { Translation } from 'openai/resources/audio/translations.mjs';
import { defineOpenAIWhisperModel } from '../src/openai/whisper';

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

describe('defineOpenAIWhisperModel runner — transcription (default)', () => {
  it('should call transcriptions.create when translate is not set', async () => {
    const result: Transcription = {
      text: 'Hello world',
    };

    const openaiClient = {
      audio: {
        transcriptions: {
          create: jest.fn(async () => result),
        },
        translations: {
          create: jest.fn(async () => ({ text: 'should not be called' })),
        },
      },
    };
    const abortSignal = jest.fn();

    const runner = defineOpenAIWhisperModel({
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
    expect(openaiClient.audio.translations.create).not.toHaveBeenCalled();
  });

  it('should call transcriptions.create when translate is explicitly false', async () => {
    const result: Transcription = {
      text: 'transcribed text',
    };

    const openaiClient = {
      audio: {
        transcriptions: {
          create: jest.fn(async () => result),
        },
        translations: {
          create: jest.fn(async () => ({ text: 'should not be called' })),
        },
      },
    };
    const abortSignal = jest.fn();

    const runner = defineOpenAIWhisperModel({
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
        config: { translate: false },
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
    expect(openaiClient.audio.translations.create).not.toHaveBeenCalled();
  });
});

describe('defineOpenAIWhisperModel runner — translation (translate: true)', () => {
  it('should call translations.create when translate is true', async () => {
    const result: Translation = {
      text: 'Hello in English',
    };

    const openaiClient = {
      audio: {
        transcriptions: {
          create: jest.fn(async () => ({ text: 'should not be called' })),
        },
        translations: {
          create: jest.fn(async () => result),
        },
      },
    };
    const abortSignal = jest.fn();

    const runner = defineOpenAIWhisperModel({
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
        config: { translate: true },
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
    expect(openaiClient.audio.transcriptions.create).not.toHaveBeenCalled();
  });
});
