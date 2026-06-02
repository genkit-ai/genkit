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

import type { HasRegistry, Registry } from '@genkit-ai/core/registry';
import { Message } from '../message.js';
import {
  defineModel,
  type GenerateRequest,
  type GenerateResponseChunkData,
  type GenerateResponseData,
  type ModelAction,
  type ModelInfo,
  type Part,
} from '../model.js';

/**
 * A streamed chunk a mock model may emit. A bare string is shorthand for a
 * single text part (`{ content: [{ text }] }`).
 */
export type MockChunk = string | GenerateResponseChunkData;

/** Context passed to a {@link MockModelOptions.respond} callback. */
export interface MockContext {
  /**
   * Emit a streamed chunk. A no-op unless the caller used `generateStream` /
   * `{ onChunk }`. A string is shorthand for a text part.
   */
  sendChunk: (chunk: MockChunk) => void;
}

/**
 * What a {@link MockModelOptions.respond} callback may return. From lightest to
 * fullest control:
 * - a `string` — shorthand for a single text response;
 * - an object with any of `text` / `toolRequests` / `content` — the common
 *   cases, assembled into a well-formed response for you;
 * - a full {@link GenerateResponseData} (anything with a `message`) — used as-is.
 */
export interface MockResponseObject {
  /** Text content of the model message. */
  text?: string;
  /** Tool/function calls to emit, by tool name. */
  toolRequests?: Array<{ name: string; input?: unknown; ref?: string }>;
  /** Escape hatch: raw message parts, prepended before `text`/`toolRequests`. */
  content?: Part[];
  finishReason?: GenerateResponseData['finishReason'];
  usage?: GenerateResponseData['usage'];
}

export type MockResponse = string | MockResponseObject | GenerateResponseData;

/** Options for {@link mockModel}. */
export interface MockModelOptions {
  /** Registered model name. Defaults to `'mockModel'`. */
  name?: string;
  /** Model metadata (e.g. `supports`, `versions`). */
  info?: ModelInfo;
  /**
   * Called once per `generate` call. Return the response to give back. Receives
   * the resolved request and a {@link MockContext} for streaming. Defaults to
   * returning empty text.
   */
  respond?: (
    request: GenerateRequest,
    context: MockContext
  ) => MockResponse | Promise<MockResponse>;
}

/**
 * A mock model with typed inspection of the calls it received. The extra
 * members are read-only views over the recorded calls.
 */
export type MockModel = ModelAction & {
  /** The request from the most recent call, or `undefined` if never called. */
  readonly lastRequest: GenerateRequest | undefined;
  /**
   * The final message of the most recent request, wrapped as a {@link Message}
   * so you can read it the same way you read a response — `.text`, `.media`,
   * `.toolRequests`, etc. `undefined` if the model was never called.
   *
   * ```ts
   * assert.match(model.lastMessage!.text, /Summarize: long text/);
   * ```
   */
  readonly lastMessage: Message | undefined;
  /** Every request this model received, oldest first. */
  readonly requests: GenerateRequest[];
  /** How many times this model was called. */
  readonly requestCount: number;
};

function resolveRegistry(registry: Registry | HasRegistry): Registry {
  return (registry as HasRegistry).registry ?? (registry as Registry);
}

function toChunkData(chunk: MockChunk): GenerateResponseChunkData {
  return typeof chunk === 'string' ? { content: [{ text: chunk }] } : chunk;
}

function toResponseData(response: MockResponse): GenerateResponseData {
  if (typeof response === 'string') {
    return {
      message: { role: 'model', content: [{ text: response }] },
      finishReason: 'stop',
    };
  }
  if ('message' in response && response.message) {
    return { finishReason: 'stop', ...(response as GenerateResponseData) };
  }
  const obj = response as MockResponseObject;
  const content: Part[] = [...(obj.content ?? [])];
  if (obj.text !== undefined) {
    content.push({ text: obj.text });
  }
  for (const tool of obj.toolRequests ?? []) {
    content.push({
      toolRequest: { name: tool.name, input: tool.input ?? {}, ref: tool.ref },
    });
  }
  return {
    message: { role: 'model', content },
    finishReason: obj.finishReason ?? 'stop',
    usage: obj.usage,
  };
}

/**
 * Defines a programmable mock model for tests. Drive each call's response with
 * `respond`, and inspect what the model was called with via `lastRequest` /
 * `requests` / `requestCount`.
 *
 * ```ts
 * const model = mockModel(ai, { respond: () => ({ text: 'a summary' }) });
 * const out = (await ai.generate({ model, prompt: 'Summarize: ...' })).text;
 * assert.match(model.lastRequest!.messages.at(-1)!.content[0].text!, /Summarize/);
 * ```
 *
 * Streaming and tool calls are first-class:
 *
 * ```ts
 * mockModel(ai, {
 *   respond: (req, { sendChunk }) => {
 *     sendChunk('hel');
 *     sendChunk('lo');
 *     return { text: 'hello' };
 *   },
 * });
 *
 * mockModel(ai, {
 *   respond: () => ({ toolRequests: [{ name: 'lookup', input: { id: 1 } }] }),
 * });
 * ```
 *
 * @param registry a `Genkit` instance (or anything holding a `Registry`).
 * @param options model name, metadata, and the `respond` callback.
 */
export function mockModel(
  registry: Registry | HasRegistry,
  options: MockModelOptions = {}
): MockModel {
  const requests: GenerateRequest[] = [];
  const respond = options.respond ?? (() => ({ text: '' }));

  const model = defineModel(
    resolveRegistry(registry),
    {
      apiVersion: 'v2',
      // info widened to satisfy defineModel's overload resolution, mirroring
      // the internal helpers in js/ai/tests/helpers.ts.
      ...((options.info as any) ?? {}),
      name: options.name ?? 'mockModel',
    },
    async (request, { sendChunk }) => {
      // Snapshot so later mutation of the request can't alter recorded history.
      requests.push(JSON.parse(JSON.stringify(request)));
      const context: MockContext = {
        sendChunk: (chunk) => sendChunk?.(toChunkData(chunk)),
      };
      return toResponseData(await respond(request, context));
    }
  ) as MockModel;

  Object.defineProperties(model, {
    requests: { get: () => requests },
    lastRequest: { get: () => requests[requests.length - 1] },
    lastMessage: {
      get: () => {
        const last = requests[requests.length - 1]?.messages.at(-1);
        return last ? new Message(last) : undefined;
      },
    },
    requestCount: { get: () => requests.length },
  });
  return model;
}

/** Options for {@link echoModel}. */
export interface EchoModelOptions {
  /** Registered model name. Defaults to `'echoModel'`. */
  name?: string;
  /** Model metadata. */
  info?: ModelInfo;
}

/**
 * Defines a zero-config mock model that echoes the rendered request back as
 * text — useful for asserting prompt and message assembly (what the model
 * *would have seen*). Supports the same inspection members as {@link mockModel}.
 *
 * ```ts
 * const model = echoModel(ai);
 * const res = await ai.generate({ model, system: 'Be terse', prompt: 'hi' });
 * assert.match(res.text, /system: Be terse/);
 * ```
 *
 * @param registry a `Genkit` instance (or anything holding a `Registry`).
 * @param options model name and metadata.
 */
export function echoModel(
  registry: Registry | HasRegistry,
  options: EchoModelOptions = {}
): MockModel {
  return mockModel(registry, {
    name: options.name ?? 'echoModel',
    info: options.info,
    respond: (request) => ({
      content: [
        {
          text:
            'Echo: ' +
            request.messages
              .map(
                (m) =>
                  (m.role === 'user' || m.role === 'model'
                    ? ''
                    : `${m.role}: `) +
                  m.content.map((c) => c.text ?? '').join('')
              )
              .join(''),
        },
        { text: '; config: ' + JSON.stringify(request.config) },
      ],
    }),
  });
}
