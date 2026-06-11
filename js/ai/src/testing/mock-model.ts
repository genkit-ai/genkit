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
   * assert.match(model.lastRequestMessage!.text, /Summarize: long text/);
   * ```
   */
  readonly lastRequestMessage: Message | undefined;
  /**
   * The full assembled conversation of the most recent request, flattened to a
   * single string (system + every message, in order). Use it for prompt-
   * assembly assertions on any mock — including ones returning structured
   * output, where {@link echoModel} can't be used. `undefined` if never called.
   *
   * ```ts
   * assert.match(model.lastRequestText!, /system: Be terse/);
   * ```
   */
  readonly lastRequestText: string | undefined;
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

/**
 * Renders a single request part to text for {@link renderRequestText}. Non-text
 * parts (media, tool requests/responses, reasoning, resource, data, custom) are
 * rendered as a labelled placeholder rather than silently dropped.
 */
function renderPart(part: Part): string {
  if (part.text !== undefined) return part.text;
  if (part.media) {
    const type = part.media.contentType ? ` ${part.media.contentType}` : '';
    return `[media${type}: ${part.media.url}]`;
  }
  if (part.toolRequest) {
    const { name, input } = part.toolRequest;
    return `[toolRequest ${name}(${JSON.stringify(input)})]`;
  }
  if (part.toolResponse) {
    const { name, output } = part.toolResponse;
    return `[toolResponse ${name}: ${JSON.stringify(output)}]`;
  }
  if (part.reasoning !== undefined) return `[reasoning: ${part.reasoning}]`;
  if (part.resource) return `[resource: ${part.resource.uri}]`;
  if (part.data !== undefined) return `[data: ${JSON.stringify(part.data)}]`;
  if (part.custom !== undefined) return `[custom: ${JSON.stringify(part.custom)}]`;
  return '';
}

/**
 * Flattens a request's full message list to text — system and tool messages are
 * prefixed with their role; `user`/`model` are not. This is the assembled
 * conversation the model would have seen, used by both {@link echoModel} and
 * {@link MockModel.lastRequestText}.
 */
function renderRequestText(request: GenerateRequest): string {
  return request.messages
    .map(
      (m) =>
        (m.role === 'user' || m.role === 'model' ? '' : `${m.role}: `) +
        m.content.map(renderPart).join('')
    )
    .join('');
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
      toolRequest: { name: tool.name, input: tool.input, ref: tool.ref },
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
      name: options.name ?? 'mockModel',
      // Forward only the metadata fields defineModel accepts; ModelInfo's
      // `configSchema`/`stage` aren't part of DefineModelOptions.
      versions: options.info?.versions,
      label: options.info?.label,
      supports: options.info?.supports,
    },
    async (request, { sendChunk }) => {
      // Snapshot so later mutation of the request can't alter recorded history.
      requests.push(structuredClone(request));
      const context: MockContext = {
        sendChunk: (chunk) => sendChunk?.(toChunkData(chunk)),
      };
      return toResponseData(await respond(request, context));
    }
  ) as MockModel;

  Object.defineProperties(model, {
    requests: { get: () => [...requests] },
    lastRequest: { get: () => requests[requests.length - 1] },
    lastRequestMessage: {
      get: () => {
        const last = requests[requests.length - 1]?.messages.at(-1);
        return last ? new Message(last) : undefined;
      },
    },
    lastRequestText: {
      get: () => {
        const last = requests[requests.length - 1];
        return last ? renderRequestText(last) : undefined;
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
 * A {@link mockModel} preset for *text* paths: a zero-config model that echoes
 * the rendered request back as text, for asserting prompt and message assembly
 * (what the model *would have seen*). Supports the same inspection members as
 * {@link mockModel}.
 *
 * ```ts
 * const model = echoModel(ai);
 * const res = await ai.generate({ model, system: 'Be terse', prompt: 'hi' });
 * assert.match(res.text, /system: Be terse/);
 * ```
 *
 * Because it returns text, it can't satisfy a structured **output schema** —
 * Genkit derives `output` by parsing the response text and validating it, which
 * prose can't pass. If the request carries an output schema, `echoModel` throws
 * an explanatory error. For structured-output paths, use {@link mockModel} with
 * a conforming response and assert assembly via {@link MockModel.lastRequestText}
 * / {@link MockModel.lastRequest} instead.
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
    // Declare native constrained support so the framework hands the output
    // schema to the model directly (in `request.output.schema`) rather than
    // injecting it as prompt text — that lets the guard below detect it
    // reliably, and keeps the echo free of framework-injected schema blobs.
    info: {
      ...options.info,
      supports: { ...options.info?.supports, constrained: 'all' },
    },
    respond: (request) => {
      if (request.output?.schema) {
        throw new Error(
          "echoModel returns text and can't satisfy an output schema: this " +
            'request asks for structured output. Either move `output: { schema }` ' +
            'to the generate()/flow call site so the prompt stays text-only, or ' +
            'use mockModel(...) with a conforming response and assert prompt ' +
            'assembly via model.lastRequestText / model.lastRequest.'
        );
      }
      return {
        content: [
          { text: 'Echo: ' + renderRequestText(request) },
          { text: '; config: ' + JSON.stringify(request.config) },
        ],
      };
    },
  });
}
