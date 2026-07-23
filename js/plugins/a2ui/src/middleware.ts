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
import { randomUUID } from 'node:crypto';
import {
  DEFAULT_CATALOG_ID,
  renderCatalogInstructions,
  type A2uiCatalog,
} from './catalog.js';
import { resolveCatalog } from './loader.js';
import { A2uiStreamParser } from './parser.js';
import { a2uiPart, isA2uiPart } from './part.js';
import {
  A2UI_VERSION,
  type A2uiClientAction,
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
   * Validate emitted envelopes against the catalog. `'strict'` (default) throws
   * on malformed JSON or unknown components; `'off'` passes them through.
   */
  validate: z.enum(['strict', 'off']).optional(),

  /**
   * Surface id policy. Provide a fixed id, or a factory called once per surface.
   * Defaults to a fresh UUID per surface. (A function is not serializable, so it
   * only applies for in-process `use`.)
   */
  surfaceId: z
    .union([z.string(), z.custom<() => string>((v) => typeof v === 'function')])
    .optional(),

  /** Protocol version stamped on emitted envelopes. Defaults to `'v0.9'`. */
  version: z.string().optional(),
});

/** Configuration for the {@link a2ui} middleware. */
export type A2uiOptions = z.infer<typeof A2uiOptionsSchema>;

/** Type guard: is this part a text part? */
function isTextPart(part: Part): part is TextPart {
  return typeof (part as { text?: unknown }).text === 'string';
}

/** Resolves the configured surface-id policy into a factory. */
function surfaceIdFactory(policy: A2uiOptions['surfaceId']): () => string {
  if (typeof policy === 'string') return () => policy;
  if (typeof policy === 'function') return policy;
  return randomUUID;
}

/**
 * Turns a text run into prose + a2ui parts using a parser, preserving order.
 * Returns the new parts to substitute for the original text part.
 */
function partsFromParse(prose: string, batches: A2uiEnvelope[][]): Part[] {
  const out: Part[] = [];
  if (prose) out.push({ text: prose });
  for (const batch of batches) {
    out.push(a2uiPart(batch));
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
        validate = 'strict',
        version = A2UI_VERSION,
      } = config ?? {};
      const nextSurfaceId = surfaceIdFactory(config?.surfaceId);

      return {
        model: async (req, ctx, next) => {
          // Resolve the catalog by id from the registry (falls back to the
          // bundled basic catalog for the default id).
          const catalog = await resolveCatalog(ai, catalogId);

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
            surfaceId: nextSurfaceId,
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

          // 3) Run downstream model, then transform the final message.
          const response = await next(request, wrappedCtx);
          return transformResponse(response, {
            catalog,
            validate,
            version,
            surfaceId: nextSurfaceId,
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
      const { prose, envelopeBatches } = parser.push(part.text);
      newContent.push(...partsFromParse(prose, envelopeBatches));
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
    validate: 'strict' | 'off';
    version: string;
    surfaceId: () => string;
  }
): GenerateResponseData {
  const message = response.message;
  if (!message?.content) return response;

  const parser = new A2uiStreamParser(opts);
  const newContent: Part[] = [];
  for (const part of message.content) {
    if (isTextPart(part)) {
      const pushed = parser.push(part.text);
      const flushed = parser.flush();
      newContent.push(
        ...partsFromParse(pushed.prose + flushed.prose, [
          ...pushed.envelopeBatches,
          ...flushed.envelopeBatches,
        ])
      );
    } else {
      newContent.push(part);
    }
  }
  return {
    ...response,
    message: { ...message, content: newContent },
  };
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
  const messages = req.messages.map((message) => {
    if (!Array.isArray(message.content)) return message;
    let msgChanged = false;
    const content: Part[] = [];
    for (const part of message.content) {
      if (isA2uiPart(part)) {
        msgChanged = true;
        const text = summarizeA2uiPart(part.data);
        if (text) content.push({ text });
      } else {
        content.push(part as Part);
      }
    }
    if (!msgChanged) return message;
    changed = true;
    return { ...message, content };
  });
  return changed ? { ...req, messages } : req;
}

/** The shapes {@link summarizeA2uiPart} narrows inbound envelope values into. */
interface SummarizableEnvelope {
  action?: A2uiClientAction;
  createSurface?: { surfaceId: string };
  updateComponents?: unknown;
  updateDataModel?: unknown;
  deleteSurface?: unknown;
}

/** Summarizes an array of a2ui envelopes / actions into a short text string. */
function summarizeA2uiPart(envelopes: unknown[]): string {
  const lines: string[] = [];
  for (const env of envelopes) {
    if (!env || typeof env !== 'object') continue;
    const e = env as SummarizableEnvelope;
    if (e.action) {
      const a = e.action;
      const ctx =
        a.context && Object.keys(a.context).length
          ? ` context=${JSON.stringify(a.context)}`
          : '';
      lines.push(`[UI action "${a.name}" on surface ${a.surfaceId}${ctx}]`);
    } else if (e.createSurface) {
      lines.push(`[UI surface ${e.createSurface.surfaceId} created]`);
    } else if (e.updateComponents || e.updateDataModel || e.deleteSurface) {
      // Prior assistant surface content — summarize as a rendered surface.
      lines.push('[rendered UI surface]');
    }
  }
  // Collapse repeated "[rendered UI surface]" lines from one assistant turn.
  return [...new Set(lines)].join(' ');
}
