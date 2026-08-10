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

import { afterEach, describe, expect, it, jest } from '@jest/globals';
import { modelRef } from 'genkit/model';
import type OpenAI from 'openai';
import {
  SpeechConfigSchema,
  TranscriptionConfigSchema,
  defineCompatOpenAISpeechModel,
  defineCompatOpenAITranscriptionModel,
} from '../src/audio';

describe('audioModel', () => {
  afterEach(() => {
    jest.clearAllMocks();
  });

  it('should correctly define supported audio tts models', () => {
    const model = defineCompatOpenAISpeechModel({
      name: 'openai/tts-1',
      client: {} as OpenAI,
      pluginOptions: { name: 'openai' },
      modelRef: modelRef({
        name: 'openai/tts-1',
        info: {
          supports: {
            media: false,
            output: ['media'],
            multiturn: false,
            systemRole: false,
            tools: false,
          },
        },
        configSchema: SpeechConfigSchema,
      }),
    });
    expect({
      name: model.__action.name,
      supports: model.__action.metadata?.model.supports,
    }).toStrictEqual({
      name: 'openai/tts-1',
      supports: {
        media: false,
        output: ['media'],
        multiturn: false,
        systemRole: false,
        tools: false,
      },
    });
  });

  it('should correctly define supported audio stt models', () => {
    const model = defineCompatOpenAITranscriptionModel({
      name: 'openai/whisper-1',
      client: {} as OpenAI,
      pluginOptions: { name: 'openai' },
      modelRef: modelRef({
        name: 'openai/whisper-1',
        info: {
          supports: {
            media: true,
            output: ['text', 'json'],
            multiturn: false,
            systemRole: false,
            tools: false,
          },
        },
        configSchema: TranscriptionConfigSchema,
      }),
    });
    expect({
      name: model.__action.name,
      supports: model.__action.metadata?.model.supports,
    }).toStrictEqual({
      name: 'openai/whisper-1',
      supports: {
        media: true,
        output: ['text', 'json'],
        multiturn: false,
        systemRole: false,
        tools: false,
      },
    });
  });
});
