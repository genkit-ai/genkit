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

import type { Genkit } from 'genkit';
import { logger } from 'genkit/logging';
import {
  getBasicUsageStats,
  type GenerateRequest,
  type GenerateResponseData,
  type MessageData,
} from 'genkit/model';
import {
  TwelveLabsConfigSchema,
  type AnalyzeResponse,
  type ModelDefinition,
  type TwelveLabsConfig,
} from './types.js';

interface DefineModelParams {
  apiKey: string;
  baseUrl: string;
  model: ModelDefinition;
}

/**
 * Registers a TwelveLabs Pegasus video-understanding model against the
 * `/analyze` endpoint.
 *
 * The video is taken from a media part in the request (a direct `http(s)` URL
 * to a raw media file) and the text parts form the prompt. Pegasus fetches the
 * video server-side, so no prior indexing is required.
 */
export function defineTwelveLabsModel(
  ai: Genkit,
  { apiKey, baseUrl, model }: DefineModelParams
) {
  return ai.defineModel(
    {
      name: `twelvelabs/${model.name}`,
      label: `TwelveLabs - ${model.name}`,
      configSchema: TwelveLabsConfigSchema,
      supports: {
        multiturn: false,
        media: true,
        tools: false,
        systemRole: false,
      },
    },
    async (input, streamingCallback) => {
      const config = (input.config ?? {}) as TwelveLabsConfig;
      const videoUrl = getVideoUrl(input);
      if (!videoUrl) {
        throw new Error(
          'TwelveLabs Pegasus requires a video. Pass it as a media part with a ' +
            'direct http(s) URL, e.g. { media: { url: "https://.../video.mp4", contentType: "video/mp4" } }.'
        );
      }

      const request = {
        video: { type: 'url', url: videoUrl },
        prompt: getPrompt(input),
        model_name: config.modelName ?? model.modelName ?? model.name,
        temperature: config.temperature,
        max_tokens: config.maxTokens,
        stream: !!streamingCallback,
      };
      logger.debug(request, 'twelvelabs /analyze request');

      const response = await fetch(`${baseUrl}/analyze`, {
        method: 'POST',
        headers: {
          'x-api-key': apiKey,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(request),
      });

      if (!response.ok || !response.body) {
        const errMsg = await safeErrorMessage(response);
        throw new Error(
          `Error from TwelveLabs /analyze: ${response.statusText}. ${errMsg}`
        );
      }

      let text: string;
      let outputTokens: number | undefined;

      if (streamingCallback) {
        text = '';
        for await (const event of readSseEvents(response.body)) {
          if (event.event_type === 'text_generation' && event.text) {
            text += event.text;
            streamingCallback({
              index: 0,
              content: [{ text: event.text }],
            });
          } else if (event.event_type === 'stream_end') {
            outputTokens = event.metadata?.usage?.output_tokens;
          }
        }
      } else {
        const json = (await response.json()) as AnalyzeResponse;
        text = json.data;
        outputTokens = json.usage?.output_tokens;
      }

      const message: MessageData = {
        role: 'model',
        content: [{ text }],
      };

      return {
        message,
        finishReason: 'stop',
        usage: { ...getBasicUsageStats(input.messages, message), outputTokens },
      } as GenerateResponseData;
    }
  );
}

/** Extracts the first media (video) URL from the request, if any. */
function getVideoUrl(input: GenerateRequest): string | undefined {
  for (const message of input.messages) {
    for (const part of message.content) {
      if (part.media?.url) {
        return part.media.url;
      }
    }
  }
  return undefined;
}

/** Concatenates the text parts of the request into a single prompt. */
function getPrompt(input: GenerateRequest): string {
  return input.messages
    .flatMap((m) => m.content)
    .map((c) => c.text ?? '')
    .join('')
    .trim();
}

interface SseEvent {
  event_type?: string;
  text?: string;
  metadata?: { usage?: { output_tokens?: number } };
}

/**
 * Parses the newline-delimited JSON stream returned by `/analyze` when
 * `stream: true`. Each line is a JSON object with an `event_type`.
 */
async function* readSseEvents(
  body: ReadableStream<Uint8Array>
): AsyncGenerator<SseEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newlineIndex: number;
      while ((newlineIndex = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, newlineIndex).trim();
        buffer = buffer.slice(newlineIndex + 1);
        if (line) yield JSON.parse(line) as SseEvent;
      }
    }
    const tail = buffer.trim();
    if (tail) yield JSON.parse(tail) as SseEvent;
  } finally {
    reader.cancel().catch(() => {});
  }
}

async function safeErrorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { message?: string };
    return body.message ?? '';
  } catch {
    return '';
  }
}
