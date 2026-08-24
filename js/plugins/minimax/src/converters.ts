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

import { GenkitError, type MessageData } from 'genkit';
import {
  getBasicUsageStats,
  type GenerateRequest,
  type GenerateResponseData,
} from 'genkit/model';
import type { MiniMaxMusicConfigSchemaType } from './music.js';
import {
  DEFAULT_AUDIO_FORMAT,
  DEFAULT_OUTPUT_FORMAT,
  MUSIC_STREAM_OUTPUT_FORMATS,
  MUSIC_TASK_STATUS,
  type MiniMaxRegion,
  type MusicAudioFormat,
  type MusicGenerationRequest,
  type MusicGenerationResponse,
  type MusicOutputFormat,
} from './types.js';

const AUDIO_CONTENT_TYPES: Record<MusicAudioFormat, string> = {
  mp3: 'audio/mpeg',
  wav: 'audio/wav',
  pcm: 'audio/pcm',
};

/**
 * Maps an audio container format to the media content type reported on the
 * returned media part.
 */
export function audioContentType(format: MusicAudioFormat): string {
  return AUDIO_CONTENT_TYPES[format];
}

/**
 * Joins the text parts of the request messages into a single prompt.
 */
function promptFromMessages(messages: MessageData[]): string {
  return messages
    .flatMap((message) => message.content)
    .map((part) => part.text ?? '')
    .join('')
    .trim();
}

/**
 * Builds the music generation request body.
 *
 * The prompt is taken from the request messages, falling back to
 * `config.prompt` when the messages carry no text.
 */
export function toMusicGenerationRequest(
  model: string,
  request: GenerateRequest<MiniMaxMusicConfigSchemaType>,
  region: MiniMaxRegion
): MusicGenerationRequest {
  const {
    apiKey: _apiKey,
    region: _region,
    baseUrl: _baseUrl,
    prompt,
    stream,
    aigc_watermark,
    ...rest
  } = request.config ?? {};

  if (stream) {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message:
        'Streaming music generation is not supported by this model. Remove `stream` from the config, ' +
        `or request one of the streaming output formats (${MUSIC_STREAM_OUTPUT_FORMATS.join(
          ', '
        )}) through a direct API call.`,
    });
  }

  if (aigc_watermark !== undefined && region !== 'cn') {
    throw new GenkitError({
      status: 'INVALID_ARGUMENT',
      message: `The \`aigc_watermark\` config is only accepted by the \`cn\` region, but the request targets \`${region}\`.`,
    });
  }

  const resolvedPrompt = promptFromMessages(request.messages) || prompt;

  return {
    model,
    ...(resolvedPrompt ? { prompt: resolvedPrompt } : {}),
    ...(aigc_watermark !== undefined ? { aigc_watermark } : {}),
    ...rest,
  };
}

/**
 * Decodes a hexadecimal audio payload into a base64 string.
 */
function hexToBase64(hex: string): string {
  const normalized = hex.trim();
  if (normalized.length % 2 !== 0 || !/^[0-9a-fA-F]+$/.test(normalized)) {
    throw new GenkitError({
      status: 'UNKNOWN',
      message:
        'The response audio was not a valid hexadecimal string. Set `output_format` to `url` to receive a download URL instead.',
    });
  }
  return Buffer.from(normalized, 'hex').toString('base64');
}

/**
 * Converts a music generation response into a Genkit response holding a single
 * audio media part.
 *
 * With `output_format: 'url'` the returned URL is passed through unchanged;
 * with `hex` the audio bytes are inlined as a data URL.
 */
export function fromMusicGenerationResponse(
  request: GenerateRequest<MiniMaxMusicConfigSchemaType>,
  response: MusicGenerationResponse,
  outputFormat: MusicOutputFormat = DEFAULT_OUTPUT_FORMAT,
  audioFormat: MusicAudioFormat = DEFAULT_AUDIO_FORMAT
): GenerateResponseData {
  const status = response.data?.status;
  if (status === MUSIC_TASK_STATUS.IN_PROGRESS) {
    throw new GenkitError({
      status: 'UNAVAILABLE',
      message:
        'The music generation is still in progress. The endpoint does not expose a way to poll for the result, so please retry the request.',
    });
  }

  const audio = response.data?.audio;
  if (!audio) {
    throw new GenkitError({
      status: 'UNKNOWN',
      message: 'The response did not contain any generated audio.',
    });
  }

  const contentType = audioContentType(audioFormat);
  const url =
    outputFormat === 'url'
      ? audio
      : `data:${contentType};base64,${hexToBase64(audio)}`;

  const message: MessageData = {
    role: 'model',
    content: [{ media: { url, contentType } }],
  };

  return {
    finishReason: 'stop',
    message,
    usage: getBasicUsageStats(request.messages, message),
    custom: response,
  };
}
