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

/**
 * The `a2ui()` generate middleware — the whole server-side integration.
 *
 * Add it to `defineAgent({ use: [...] })` or `ai.generate({ use: [...] })` and
 * the agent gains the ability to render A2UI surfaces. The middleware:
 *
 * 1. Injects the catalog's capabilities into the system prompt so the model
 *    knows what UI it may render.
 * 2. Intercepts the model's output (both the streamed chunks and the final
 *    message), extracts any `a2ui` fenced blocks, validates them against the
 *    catalog, and rewrites them into the canonical a2ui data part.
 *
 * Downstream (client transport, `@a2ui/web_core`) only ever sees a2ui parts —
 * "pure vs mixed" turns are a prompting choice, not a separate code path.
 *
 * Implemented with `generateMiddleware` and the `model` hook, so it wraps each
 * raw model call in the agent's tool loop.
 *
 * @module
 */

import { generateMiddleware, z, type GenerateMiddleware } from 'genkit';
import type {
  GenerateRequest,
  GenerateResponseChunkData,
  GenerateResponseData,
  MessageData,
  Part,
  TextPart,
} from 'genkit/model';
import {
  DEFAULT_CATALOG_ID,
  renderCatalogInstructions,
  type A2uiCatalog,
} from './catalog.js';
import { resolveCatalog } from './loader.js';
import { A2uiStreamParser, type ParseResult } from './parser.js';
import { a2uiPart, isA2uiPart } from './part.js';
import {
  A2UI_VERSION,
  SUPPORTED_VERSIONS,
  type A2uiEnvelope,
} from './types.js';

/** Zod schema for the {@link a2ui} middleware configuration. */
export const A2uiOptionsSchema = z.object({
  /**
   * The id of the catalog describing what the agent may render. Defaults to
   * `'basic'` (the bundled basic catalog). Register additional catalogs with
   * `loadCatalog(ai, { id, catalog | file })` and reference them by id here.
   */
  catalog: z.string().optional(),

  /**
   * Where to inject the catalog's capabilities. `'system'` (default) appends
   * A2UI instructions to the system prompt; `'none'` injects nothing (useful if
   * you supply your own instructions).
   */
  instructions: z.enum(['system', 'none']).optional(),

  /**
   * Validate emitted envelopes against the catalog. `'warn'` (default) logs a
   * warning and drops the offending block/envelope, keeping the rest of the
   * turn alive; `'strict'` throws on malformed JSON or unknown components (best
   * for development); `'off'` passes them through unchecked.
   *
   * IMPORTANT: this validates envelope structure and component *type names*
   * against the catalog only. It does NOT validate component props or
   * data-model values. Model-controlled values such as `Image.url` and `Text`
   * (inline Markdown, which the renderer may turn into HTML) pass through
   * untouched even under `'strict'`. `'strict'` is a well-formedness check, not
   * a security boundary — see the "Security / trust boundary" section of the
   * README. Prop sanitization is the renderer/catalog's responsibility, and
   * hosts should CSP-restrict image and other remote sources.
   */
  validate: z.enum(['strict', 'warn', 'off']).optional(),

  /**
   * Surface id policy. Provide a fixed id to reuse for every surface. Defaults
   * to a fresh UUID per surface.
   */
  surfaceId: z.string().optional(),

  /**
   * Protocol version stamped on emitted envelopes. Defaults to `'v0.9'`.
   * Constrained to the versions the renderer understands so a typo can't emit
   * envelopes that fail at runtime.
   */
  version: z
    .enum(SUPPORTED_VERSIONS as unknown as [string, ...string[]])
    .optional(),
});

/** Configuration for the {@link a2ui} middleware. */
export type A2uiOptions = z.infer<typeof A2uiOptionsSchema>;

/** Type guard: is this part a text part? */
function isTextPart(part: Part): part is TextPart {
  return (
    !!part &&
    typeof part === 'object' &&
    typeof (part as { text?: unknown }).text === 'string'
  );
}

/** Resolves the configured surface-id policy into a factory. */
function surfaceIdFactory(policy: A2uiOptions['surfaceId']): () => string {
  if (typeof policy === 'string') return () => policy;
  // Use the Web Crypto API (available on Node >=16 and all supported "exotic"
  // runtimes, e.g. Cloudflare Workers) rather than a hard `node:crypto` import,
  // keeping the middleware portable.
  return () => globalThis.crypto.randomUUID();
}

/**
 * Wraps a surface-id factory so a single model turn's streamed parse and its
 * final-message parse mint the *same* surface ids.
 *
 * A turn is parsed twice: once incrementally over the streamed chunks, and once
 * over the aggregated final message (so consumers that read `response.message`
 * see a2ui parts too). Each parse pulls surface ids from the factory. With the
 * default `randomUUID` policy those two parses would otherwise produce different
 * ids for the same surface. So while streaming we generate and record ids in
 * order (`next`); before re-parsing the final message we `reset`, then
 * `replayNext` hands back the recorded ids in the same order (only generating a
 * fresh id if the final parse yields more blocks than the stream did).
 */
function replayableSurfaceIds(base: () => string): {
  next: () => string;
  replayNext: () => string;
  reset: () => void;
} {
  const generated: string[] = [];
  let cursor = 0;
  const next = () => {
    const id = base();
    generated.push(id);
    return id;
  };
  const replayNext = () =>
    cursor < generated.length ? generated[cursor++] : next();
  const reset = () => {
    cursor = 0;
  };
  return { next, replayNext, reset };
}

/**
 * Turns a parse result into ordered prose + a2ui parts, preserving the exact
 * source order (so prose after a block stays after it). Returns the new parts
 * to substitute for the original text part.
 */
function partsFromParse(result: ParseResult): Part[] {
  const out: Part[] = [];
  for (const seg of result.segments) {
    if ('prose' in seg) {
      if (seg.prose) out.push({ text: seg.prose });
    } else {
      out.push(a2uiPart(seg.envelopes));
    }
  }
  return out;
}

/**
 * The A2UI generate middleware.
 *
 * @example
 * ```ts
 * import { a2ui } from '@genkit-ai/a2ui';
 *
 * export const uiAgent = ai.defineAgent({
 *   name: 'uiAgent',
 *   model: 'googleai/gemini-flash-latest',
 *   system: 'You help users. Render UI when it is clearer than prose.',
 *   use: [a2ui()], // defaults to the bundled 'basic' catalog
 * });
 * ```
 */
export const a2ui: GenerateMiddleware<typeof A2uiOptionsSchema> =
  generateMiddleware(
    {
      name: 'a2ui',
      description:
        'Adds A2UI (Agent-to-UI) streaming UI support: injects catalog ' +
        'capabilities into the prompt and rewrites emitted UI blocks into a2ui ' +
        'data parts.',
      configSchema: A2uiOptionsSchema,
    },
    (options) => {
      const { ai, config } = options;
      const {
        catalog: catalogId = DEFAULT_CATALOG_ID,
        instructions = 'system',
        validate = 'warn',
        version = A2UI_VERSION,
      } = config ?? {};
      const nextSurfaceId = surfaceIdFactory(config?.surfaceId);

      return {
        model: async (req, ctx, next) => {
          // Resolve the catalog by id from the registry (falls back to the
          // bundled basic catalog for the default id).
          const catalog = await resolveCatalog(ai, catalogId);

          // Share surface ids between the streamed parse and the final-message
          // parse of this single turn, so the same surface gets the same id in
          // both (see replayableSurfaceIds).
          const surfaceIds = replayableSurfaceIds(nextSurfaceId);

          // 0) Sanitize any inbound a2ui data parts (e.g. a surface action sent
          //    back as the next turn, or replayed history) into model-readable
          //    text, so the underlying model's converter never sees the a2ui
          //    mime type.
          const sanitized = sanitizeInboundA2ui(req);

          // 1) Inject catalog instructions into the system prompt.
          const request =
            instructions === 'none'
              ? sanitized
              : injectInstructions(sanitized, catalog);

          // 2) Wrap the streaming callback so streamed text is split into prose
          //    deltas + whole a2ui parts as blocks complete.
          const streamParser = new A2uiStreamParser({
            catalog,
            validate,
            version,
            surfaceId: surfaceIds.next,
          });

          const originalOnChunk = ctx?.onChunk;
          const wrappedCtx = originalOnChunk
            ? {
                ...ctx,
                onChunk: (chunk: GenerateResponseChunkData) => {
                  const transformed = transformChunk(chunk, streamParser);
                  if (transformed) originalOnChunk(transformed);
                },
              }
            : ctx;

          // 3) Run downstream model, then flush the stream parser so the last
          //    withheld prose tail (the parser holds back up to a partial
          //    opening fence) and any unterminated trailing block still reach
          //    the streaming consumer. Without this, clients that render purely
          //    from stream deltas would show truncated prose / miss a final
          //    block (the aggregated message recovers it, but the stream would
          //    not).
          const response = await next(request, wrappedCtx);
          if (originalOnChunk) {
            const tail = partsFromParse(streamParser.flush());
            if (tail.length > 0) originalOnChunk({ content: tail });
          }

          // 4) Transform the final message. The final parse replays the same
          //    surface ids the stream minted.
          surfaceIds.reset();
          return transformResponse(response, {
            catalog,
            validate,
            version,
            surfaceId: surfaceIds.replayNext,
          });
        },
      };
    }
  );

/** Appends A2UI instructions to (or creates) the system message. */
function injectInstructions(
  req: GenerateRequest,
  catalog: A2uiCatalog
): GenerateRequest {
  const text = renderCatalogInstructions(catalog);
  const messages: MessageData[] = [...req.messages];
  const sysIdx = messages.findIndex((m) => m.role === 'system');
  if (sysIdx >= 0) {
    const sys = messages[sysIdx];
    messages[sysIdx] = {
      ...sys,
      content: [...sys.content, { text: '\n\n' + text }],
    };
  } else {
    messages.unshift({ role: 'system', content: [{ text }] });
  }
  return { ...req, messages };
}

/** Transforms a single streamed chunk; returns null if nothing to emit. */
function transformChunk(
  chunk: GenerateResponseChunkData,
  parser: A2uiStreamParser
): GenerateResponseChunkData | null {
  if (!chunk?.content || chunk.content.length === 0) return chunk;
  const newContent: Part[] = [];
  for (const part of chunk.content) {
    if (isTextPart(part) && part.text !== '') {
      newContent.push(...partsFromParse(parser.push(part.text)));
    } else {
      newContent.push(part);
    }
  }
  if (newContent.length === 0) return null;
  return { ...chunk, content: newContent };
}

/** Transforms the final response message: prose text + a2ui parts. */
function transformResponse(
  response: GenerateResponseData,
  opts: {
    catalog: A2uiCatalog;
    validate: 'strict' | 'warn' | 'off';
    version: string;
    surfaceId: () => string;
  }
): GenerateResponseData {
  // Real models (e.g. google-genai) return the candidates shape, not a
  // top-level `message`; only the GenerateResponse constructor later collapses
  // `message ?? candidates[0].message`. Read from whichever the model used so
  // the transform runs on the real path, not just when a caller hands us a
  // pre-collapsed `message`.
  const message = response.message ?? response.candidates?.[0]?.message;
  if (!message?.content) return response;

  const parser = new A2uiStreamParser(opts);

  const newContent: Part[] = [];

  // Drains whatever the parser is still holding (a withheld prose tail, or an
  // unterminated trailing block) and appends it. Called at every non-text
  // boundary and once at the end.
  const flushHeld = () => {
    const tail = parser.flush();
    newContent.push(...partsFromParse(tail));
  };

  for (const part of message.content) {
    if (isTextPart(part) && part.text !== '') {
      // Push WITHOUT flushing between consecutive text parts so an a2ui block
      // that spans several adjacent text parts is stitched back together. The
      // model's final message is not guaranteed to coalesce adjacent text: the
      // Gemini plugin, for instance, aggregates a turn into many separate text
      // parts (fence, JSON body split many ways, close fence, then a trailing
      // empty-text part carrying the thought signature), so a per-part flush
      // would reset the parser mid-block and leak the whole surface back out as
      // raw prose. This mirrors the streaming path, which shares one parser
      // across all chunks and flushes only once at the end.
      newContent.push(...partsFromParse(parser.push(part.text)));
    } else {
      // A non-text part (e.g. a toolRequest) or an empty-text part (e.g. the
      // trailing thought-signature carrier) is a boundary: flush any held tail
      // so it lands before this part, preserving order. This is what a
      // tool-calling turn [text("Checking the weather."), toolRequest] relies
      // on so the prose is not reordered behind the toolRequest. Then carry the
      // part through untouched so its metadata survives.
      flushHeld();
      newContent.push(part);
    }
  }
  flushHeld();

  const newMessage = { ...message, content: newContent };

  // Write the transformed message back to WHICHEVER shape(s) the response
  // carries, so no consumer sees the raw prose. Real providers (e.g.
  // google-genai) use the candidates shape, but a response could carry a
  // top-level `message` and a `candidates` array at once (e.g. a prior
  // middleware pre-populated `message` while keeping `candidates`); updating
  // only one would leave the other holding the untransformed fence text. Only
  // candidate 0 is transformed: the framework collapses to
  // `candidates[0].message` and candidateCount defaults to 1, and each
  // candidate would otherwise consume the shared surface-id replay state.
  const result = { ...response };
  if (result.message) {
    result.message = newMessage;
  }
  if (result.candidates?.[0]) {
    result.candidates = result.candidates.map((c, i) =>
      i === 0 ? { ...c, message: newMessage } : c
    );
  }
  return result;
}

/**
 * Converts inbound a2ui data parts in the request into model-readable text.
 *
 * The a2ui data part (mime `application/a2ui+json`) is meaningful to the client
 * renderer, but the underlying model's message converter (e.g. Gemini) does not
 * understand it. When a rendered surface's action is sent back as the next
 * turn's input — or when prior assistant turns containing surfaces are replayed
 * as history — we replace those parts with a compact text summary so the model
 * can reason about them.
 */
function sanitizeInboundA2ui(req: GenerateRequest): GenerateRequest {
  let changed = false;
  const messages: MessageData[] = [];
  for (const message of req.messages) {
    if (!Array.isArray(message.content)) {
      messages.push(message);
      continue;
    }
    let msgChanged = false;
    const content: Part[] = [];
    for (const part of message.content) {
      if (isA2uiPart(part)) {
        msgChanged = true;
        const text = summarizeA2uiPart(part.data.envelopes);
        if (text) content.push({ text });
      } else {
        content.push(part as Part);
      }
    }
    if (!msgChanged) {
      messages.push(message);
      continue;
    }
    changed = true;
    // Drop a message that sanitizing emptied out. This happens when its only
    // content was an a2ui part whose envelopes all summarized to nothing (e.g.
    // an empty or all-unrecognized envelope array). Sending `content: []`
    // downstream would make providers like Gemini and Vertex reject the
    // request, so skip the message entirely instead.
    if (content.length === 0) continue;
    messages.push({ ...message, content });
  }
  return changed ? { ...req, messages } : req;
}

/**
 * Converts an array of a2ui envelopes from an inbound message part back into
 * model-readable text — the inverse of the outbound block-to-part transform.
 *
 * The two envelope kinds are handled differently on purpose:
 *
 * - Assistant-authored surface envelopes (`createSurface`, `updateComponents`,
 *   `updateDataModel`, `deleteSurface`) are reconstructed as the canonical
 *   `a2ui` fenced block the model originally emitted. Replaying a prior turn's
 *   surface as history therefore shows the model its own UI output in the exact
 *   format it is asked to produce, reinforcing correct behavior. (Summarizing it
 *   to a sentinel like `[rendered UI surface]` instead taught the model to emit
 *   that literal string in place of a real block.)
 * - Client-synthesized `action` envelopes never had a block form, so they
 *   become a short text summary the model can reason about.
 *
 * Consecutive surface envelopes are grouped into a single block (one surface is
 * usually several envelopes: create + update(s)). Unknown envelope shapes are
 * dropped, so an all-unrecognized (or empty) envelope array summarizes to an
 * empty string; {@link sanitizeInboundA2ui} then drops the emptied message
 * rather than sending empty content downstream.
 */
function summarizeA2uiPart(envelopes: A2uiEnvelope[]): string {
  const out: string[] = [];
  let pendingSurface: A2uiEnvelope[] = [];

  const flushSurface = () => {
    if (pendingSurface.length === 0) return;
    // Keep the real surface ids verbatim. The model may not reuse them for a
    // NEW surface: the parser forces a fresh id onto every `createSurface`
    // block (see `finalizeBlock`), so a copied id can't overwrite a prior
    // surface. Keeping the real ids lets the model correlate a replayed action
    // (`[UI action ... on surface <id>]`) with the surface it targeted — which
    // matters when several surfaces are on screen at once.
    //
    // Encode compactly (not pretty-printed): fewer tokens, and it collapses the
    // payload to a single line so the block is exactly three lines (open fence,
    // JSON, close fence). Because JSON escapes any newline inside a string as
    // `\n`, an A2UI `Text` value containing a fenced code sample can't put a
    // literal ``` at the start of a line, so it can never prematurely close this
    // block (the parser's close fence is line-anchored). The block text can
    // still contain ``` characters mid-line; a fully robust emitter would use a
    // variable-length fence, but that also requires the parser's fixed
    // three-backtick open fence to become count-aware, so it is deferred.
    out.push('```a2ui\n' + JSON.stringify(pendingSurface) + '\n```');
    pendingSurface = [];
  };

  for (const env of envelopes) {
    if (!env || typeof env !== 'object') continue;
    if ('action' in env && env.action) {
      // Emit any buffered surface block before the action, preserving order.
      flushSurface();
      const a = env.action;
      const ctx =
        a.context && Object.keys(a.context).length
          ? ` context=${JSON.stringify(a.context)}`
          : '';
      out.push(`[UI action "${a.name}" on surface ${a.surfaceId}${ctx}]`);
    } else if (
      ('createSurface' in env && env.createSurface) ||
      ('updateComponents' in env && env.updateComponents) ||
      ('updateDataModel' in env && env.updateDataModel) ||
      ('deleteSurface' in env && env.deleteSurface)
    ) {
      pendingSurface.push(env);
    }
  }
  flushSurface();
  return out.join('\n');
}
