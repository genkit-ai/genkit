/**
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

import type { StreamingCallback } from '@genkit-ai/core';
import type { Registry } from '@genkit-ai/core/registry';
import {
  defineModel,
  type GenerateRequest,
  type GenerateResponseChunkData,
  type GenerateResponseData,
  type ModelAction,
  type ModelInfo,
} from '../src/model';

export async function runAsync<O>(fn: () => O): Promise<O> {
  return new Promise((resolve) => {
    setTimeout(() => resolve(fn()), 0);
  });
}

export type ProgrammableModel = ModelAction & {
  handleResponse: (
    req: GenerateRequest,
    streamingCallback?: StreamingCallback<GenerateResponseChunkData>
  ) => Promise<GenerateResponseData>;

  lastRequest?: GenerateRequest;
  requestCount: number;
};

export function defineProgrammableModel(
  registry: Registry,
  info?: ModelInfo,
  name?: string
): ProgrammableModel {
  const pm = defineModel(
    registry,
    {
      apiVersion: 'v2',
      ...(info as any),
      name: name ?? 'programmableModel',
    },
    async (request, { sendChunk }) => {
      pm.requestCount++;
      pm.lastRequest = JSON.parse(JSON.stringify(request));
      return pm.handleResponse(request, sendChunk);
    }
  ) as ProgrammableModel;

  pm.requestCount = 0;
  return pm;
}

export function defineEchoModel(registry: Registry): ModelAction {
  const model = defineModel(
    registry,
    {
      name: 'echoModel',
    },
    async (request, streamingCallback) => {
      (model as any).__test__lastRequest = request;
      (model as any).__test__lastStreamingCallback = streamingCallback;
      if (streamingCallback) {
        streamingCallback({
          content: [
            {
              text: '3',
            },
          ],
        });
        streamingCallback({
          content: [
            {
              text: '2',
            },
          ],
        });
        streamingCallback({
          content: [
            {
              text: '1',
            },
          ],
        });
      }
      return {
        message: {
          role: 'model',
          content: [
            {
              text:
                'Echo: ' +
                request.messages
                  .map(
                    (m) =>
                      (m.role === 'user' || m.role === 'model'
                        ? ''
                        : `${m.role}: `) + m.content.map((c) => c.text).join()
                  )
                  .join(),
            },
            {
              text: '; config: ' + JSON.stringify(request.config),
            },
          ],
        },
        finishReason: 'stop',
      };
    }
  );
  return model;
}
