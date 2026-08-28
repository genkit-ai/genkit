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
 * Helpers for working with the canonical "a2ui part" — a Genkit `data` part
 * whose `data` is an object `{ envelopes }` wrapping an array of A2UI envelopes,
 * tagged with {@link A2UI_MIME_TYPE}.
 *
 * These helpers operate on plain part-shaped objects (no Genkit runtime import),
 * so they are safe to use on both the server and the browser. To pull envelopes
 * off an agent stream, pass the chunk's parts:
 * `a2uiEnvelopesFromParts(chunk.raw.modelChunk?.content)`.
 *
 * @module
 */

import { type Part } from 'genkit';
import {
  A2UI_MIME_TYPE,
  type A2uiEnvelope,
  type A2uiPart,
  type A2uiServerEnvelope,
} from './types.js';

/** A minimal structural view of a Genkit part for these helpers. */
interface PartLike {
  data?: unknown;
  metadata?: Record<string, unknown> | null;
  [key: string]: unknown;
}

/** Builds an a2ui data part wrapping the given envelopes. */
export function a2uiPart(envelopes: A2uiEnvelope[]): A2uiPart {
  return {
    data: { envelopes },
    metadata: { mimeType: A2UI_MIME_TYPE },
  };
}

/** Type guard: is this part an a2ui data part? */
export function isA2uiPart(part: unknown): part is A2uiPart {
  const p = part as PartLike | null | undefined;
  const data = p?.data as { envelopes?: unknown } | undefined;
  return (
    !!p &&
    typeof p === 'object' &&
    !!data &&
    typeof data === 'object' &&
    Array.isArray(data.envelopes) &&
    (p as PartLike).metadata?.mimeType === A2UI_MIME_TYPE
  );
}

/**
 * Extracts all A2UI envelopes carried by the given `parts`.
 *
 * This is the single entry point for reading envelopes off any part-carrying
 * value: pass a message's, chunk's, or response's `content`. For example, to
 * consume an agent stream:
 *
 * ```ts
 * for await (const chunk of turn.stream) {
 *   const envelopes = a2uiEnvelopesFromParts(chunk.raw.modelChunk?.content);
 * }
 * ```
 *
 * `parts` is nullable purely for call-site convenience (a chunk's content can
 * be `undefined`); a nullish list is treated as empty.
 *
 * Returns `[]` for content that carries no a2ui parts (e.g. plain prose).
 *
 * The result is typed as {@link A2uiServerEnvelope}[] — the surfaces the agent
 * renders — because that is all a render stream carries. Any inbound
 * {@link ActionEnvelope} (a user action echoed in history) is filtered out, so
 * the returned envelopes can be handed straight to a renderer's message
 * processor without a cast.
 */
export function a2uiEnvelopesFromParts(
  parts: readonly Part[] | null | undefined
): A2uiServerEnvelope[] {
  if (!Array.isArray(parts)) return [];
  const out: A2uiServerEnvelope[] = [];
  for (const part of parts) {
    if (!isA2uiPart(part)) continue;
    for (const env of part.data.envelopes) {
      // Keep only server → client surface envelopes. Inbound action envelopes
      // (a user action echoed in history) are dropped, and any null /
      // non-object garbage from malformed model output is skipped so the
      // `A2uiServerEnvelope[]` return type stays honest. Arrays are excluded
      // explicitly since `typeof [] === 'object'`.
      if (!env || typeof env !== 'object' || Array.isArray(env)) continue;
      if (isActionEnvelope(env)) continue;
      out.push(env);
    }
  }
  return out;
}

/** Narrows an envelope to a client → server action envelope. */
function isActionEnvelope(
  env: A2uiEnvelope
): env is Extract<A2uiEnvelope, { action: unknown }> {
  return 'action' in env;
}
