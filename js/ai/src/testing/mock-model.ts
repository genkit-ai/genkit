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

/** A `respond` callback: given the request, returns the response to emit. */
export type MockRespondFn = (
  request: GenerateRequest,
  context: MockContext
) => MockResponse | Promise<MockResponse>;

/** Options for {@link mockModel}. */
export interface MockModelOptions {
  /** Registered model name. Defaults to `'mockModel'`. */
  name?: string;
  /**
   * Model metadata (e.g. `supports`, `versions`).
   *
   * `supports.constrained` defaults to `'all'` (native constrained generation),
   * so a `generate` with an `output.schema` reaches `respond` with that schema
   * on `request.output.schema`. Pass `supports: { constrained: 'none' }` to
   * instead exercise the framework's simulated path, where the schema is
   * injected into the prompt (and thus visible in `lastRequestText`) and
   * stripped from what `respond` sees.
   */
  info?: ModelInfo;
  /**
   * What to respond with on each `generate` call. Defaults to empty text.
   *
   * - A **single response** (string / object) is returned on every call.
   * - A **callback** `(request, { sendChunk }) => response` is invoked once per
   *   call, so you can branch on request history (for tool loops) and stream via
   *   `sendChunk`.
   * - An **array** is a queue consumed one item per call, with the last item
   *   repeating once exhausted — handy for scripted multi-turn tests. A queued
   *   `Error` is thrown when reached, to inject a failure on a given turn.
   *   (Queued items are static, so streaming needs the callback form.)
   *
   * ```ts
   * mockModel(ai, { respond: 'always this' });
   * mockModel(ai, { respond: ['first', 'second'] });
   * mockModel(ai, { respond: ['ok', new Error('rate limited')] });
   * ```
   */
  respond?: MockRespond;
}

/**
 * Anything accepted as respond behavior: a single {@link MockResponse}, a
 * per-call callback, or a queue of responses/errors. See
 * {@link MockModelOptions.respond}.
 */
export type MockRespond =
  | MockRespondFn
  | MockResponse
  | Array<MockResponse | Error>;

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
  /**
   * The tool results fed back to the model in the most recent request, in order.
   * Use it to assert which tools ran and what they returned, without digging
   * through message content yourself. Empty if the model was never called or saw
   * no tool results.
   *
   * ```ts
   * assert.deepStrictEqual(model.toolResponses.map((t) => t.name), ['lookup']);
   * ```
   */
  readonly toolResponses: Array<{
    name: string;
    ref?: string;
    output: unknown;
  }>;
  /** Every request this model received, oldest first. */
  readonly requests: GenerateRequest[];
  /** How many times this model was called. */
  readonly requestCount: number;
  /**
   * Replaces the respond behavior for subsequent calls. Accepts everything
   * {@link MockModelOptions.respond} does; an array re-arms as a fresh queue.
   * Recorded history is untouched — use {@link MockModel.reset} for that.
   *
   * Together with `reset()` this supports the register-once idiom: define the
   * mock once per test file, then give each test its own behavior.
   *
   * ```ts
   * const model = mockModel(ai, { name: 'menuModel' });
   * beforeEach(() => model.reset());
   *
   * test('...', async () => {
   *   model.respondWith({ text: 'scripted' });
   *   // ...
   * });
   * ```
   */
  respondWith(respond: MockRespond): void;
  /**
   * Clears recorded history (`requests`, `requestCount`, …) and restores the
   * respond behavior given at construction, re-arming a queued `respond` from
   * its first item. Call it in `beforeEach` so tests sharing a mock stay
   * order-independent.
   */
  reset(): void;
};

function resolveRegistry(registry: Registry | HasRegistry): Registry {
  return (registry as HasRegistry).registry ?? (registry as Registry);
}

/**
 * Snapshots a request so later mutation can't alter recorded history. Prefers a
 * deep `structuredClone`, but falls back to a message/part-level copy when the
 * request carries non-serializable values (e.g. a function or class instance in
 * `config`), which would otherwise throw a `DataCloneError`.
 */
function cloneRequest(request: GenerateRequest): GenerateRequest {
  try {
    return structuredClone(request);
  } catch {
    return {
      ...request,
      messages: request.messages.map((m) => ({
        ...m,
        content: m.content.map((c) => ({ ...c })),
      })),
    };
  }
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
  if (part.custom !== undefined)
    return `[custom: ${JSON.stringify(part.custom)}]`;
  return '';
}

/**
 * Flattens a request's full message list to text — system and tool messages are
 * prefixed with their role; `user`/`model` are not. Messages are newline-
 * separated so adjacent messages' text can't fuse into a single token (which
 * would silently break boundary-spanning assertions). This is the assembled
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
    .join('\n');
}

/**
 * Normalizes the `respond` option into a single callback. A single response is
 * returned on every call; an array becomes a queue consumed one item per call,
 * with the last item repeating once exhausted; a queued `Error` is thrown when
 * reached.
 */
function toRespondFn(respond: MockRespond | undefined): MockRespondFn {
  if (respond === undefined) {
    return () => ({ text: '' });
  }
  if (typeof respond === 'function') {
    return respond;
  }
  const queue = Array.isArray(respond) ? respond : [respond];
  if (queue.length === 0) {
    return () => ({ text: '' });
  }
  let i = 0;
  return () => {
    const item = queue[Math.min(i, queue.length - 1)];
    i++;
    if (item instanceof Error) throw item;
    return item;
  };
}

function toResponseData(response: MockResponse): GenerateResponseData {
  // A `respond` that streams but returns nothing (void) yields an empty message
  // rather than throwing on the `'message' in response` check below. Checked
  // against null/undefined only, so `respond: ''` still means empty *text*.
  if (response === undefined || response === null) {
    return { message: { role: 'model', content: [] }, finishReason: 'stop' };
  }
  if (typeof response === 'string') {
    return {
      message: { role: 'model', content: [{ text: response }] },
      finishReason: 'stop',
    };
  }
  if ('message' in response && response.message) {
    const data = response as GenerateResponseData;
    // Default finishReason without letting an explicit `undefined` clobber it.
    return { ...data, finishReason: data.finishReason ?? 'stop' };
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
 * For structured output, the mock defaults to native constrained generation
 * (`supports.constrained: 'all'`), so `respond` sees `request.output.schema`
 * and no schema blob is injected into the prompt. Override with
 * `supports: { constrained: 'none' }` to test the simulated path.
 *
 * @param registry a `Genkit` instance (or anything holding a `Registry`).
 * @param options model name, metadata, and the `respond` callback.
 */
export function mockModel(
  registry: Registry | HasRegistry,
  options: MockModelOptions = {}
): MockModel {
  const requests: GenerateRequest[] = [];
  let respond = toRespondFn(options.respond);

  const model = defineModel(
    resolveRegistry(registry),
    {
      apiVersion: 'v2',
      name: options.name ?? 'mockModel',
      // Forward only the metadata fields defineModel accepts; ModelInfo's
      // `configSchema`/`stage` aren't part of DefineModelOptions.
      versions: options.info?.versions,
      label: options.info?.label,
      // Default to native constrained generation (like modern provider models),
      // so a structured-output request reaches `respond` with `output.schema`
      // intact instead of the framework injecting a schema blob into the prompt
      // and stripping it. Spread last so callers can opt out with
      // `supports: { constrained: 'none' }` to exercise the simulated path.
      supports: { constrained: 'all', ...options.info?.supports },
    },
    async (request, { sendChunk }) => {
      // Snapshot so later mutation of the request can't alter recorded history.
      requests.push(cloneRequest(request));
      const context: MockContext = {
        sendChunk: (chunk) => sendChunk?.(toChunkData(chunk)),
      };
      return toResponseData(await respond(request, context));
    }
  ) as MockModel;

  Object.defineProperties(model, {
    // Return clones so callers can't mutate recorded history through a view.
    requests: { get: () => requests.map((r) => cloneRequest(r)) },
    lastRequest: {
      get: () => {
        const last = requests[requests.length - 1];
        return last ? cloneRequest(last) : undefined;
      },
    },
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
    toolResponses: {
      get: () =>
        (requests[requests.length - 1]?.messages ?? [])
          .flatMap((m) => m.content)
          .filter((p) => p.toolResponse)
          .map((p) => ({
            name: p.toolResponse!.name,
            ref: p.toolResponse!.ref,
            output: p.toolResponse!.output,
          })),
    },
    requestCount: { get: () => requests.length },
    respondWith: {
      value: (next: MockRespond) => {
        respond = toRespondFn(next);
      },
    },
    reset: {
      value: () => {
        requests.length = 0;
        respond = toRespondFn(options.respond);
      },
    },
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
