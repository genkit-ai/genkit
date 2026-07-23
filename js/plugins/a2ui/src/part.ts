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
 * carrying an array of A2UI envelopes tagged with {@link A2UI_MIME_TYPE}.
 *
 * These helpers operate on plain part-shaped objects (no Genkit runtime import),
 * so they are safe to use on both the server and the browser.
 *
 * @module
 */

import { A2UI_MIME_TYPE, type A2uiEnvelope, type A2uiPart } from './types.js';

/** A minimal structural view of a Genkit part for these helpers. */
interface PartLike {
  data?: unknown;
  metadata?: Record<string, unknown> | null;
  [key: string]: unknown;
}

/** Builds an a2ui data part wrapping the given envelopes. */
export function a2uiPart(envelopes: A2uiEnvelope[]): A2uiPart {
  return {
    data: envelopes,
    metadata: { mimeType: A2UI_MIME_TYPE },
  };
}

/** Type guard: is this part an a2ui data part? */
export function isA2uiPart(part: unknown): part is A2uiPart {
  const p = part as PartLike | null | undefined;
  return (
    !!p &&
    typeof p === 'object' &&
    Array.isArray((p as PartLike).data) &&
    (p as PartLike).metadata?.mimeType === A2UI_MIME_TYPE
  );
}

/**
 * Extracts all A2UI envelopes from a message-, chunk-, or part-shaped value.
 * Returns `[]` for anything that carries no a2ui parts (e.g. plain prose).
 *
 * Accepts:
 * - a single part
 * - an object with a `content: Part[]` array (a message)
 * - an `AgentChunk`-shaped object with `modelChunk.content`
 */
export function a2uiEnvelopes(value: unknown): A2uiEnvelope[] {
  if (!value || typeof value !== 'object') return [];

  const v = value as {
    modelChunk?: { content?: unknown };
    content?: unknown;
  };

  // AgentChunk: { modelChunk: { content: Part[] } }
  if (v.modelChunk?.content) {
    return collectFromParts(v.modelChunk.content);
  }
  // Message / GenerateResponseChunk: { content: Part[] }
  if (Array.isArray(v.content)) {
    return collectFromParts(v.content);
  }
  // A single part.
  if (isA2uiPart(value)) {
    return [...value.data];
  }
  return [];
}

function collectFromParts(parts: unknown): A2uiEnvelope[] {
  if (!Array.isArray(parts)) return [];
  const out: A2uiEnvelope[] = [];
  for (const part of parts) {
    if (isA2uiPart(part)) out.push(...part.data);
  }
  return out;
}
