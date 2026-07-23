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
 * Streaming parser that extracts A2UI envelopes from model output.
 *
 * The model emits A2UI as a fenced code block tagged `a2ui` containing a JSON
 * array of envelopes (see {@link renderCatalogInstructions}). This parser scans
 * a text stream incrementally, separating ordinary prose (which streams through
 * as deltas) from complete A2UI blocks (which are buffered until the closing
 * fence, then parsed into whole, validated envelopes — the protocol requires
 * ordered, complete messages, so we never emit half-parsed JSON).
 *
 * This module is Node-free so it can be unit-tested and reused anywhere.
 *
 * @module
 */

import type { A2uiCatalog } from './catalog.js';
import { SURFACE_ID_PLACEHOLDER } from './catalog.js';
import {
  A2UI_VERSION,
  type A2uiComponent,
  type A2uiEnvelope,
  type CreateSurfaceEnvelope,
  type DeleteSurfaceEnvelope,
  type UpdateComponentsEnvelope,
  type UpdateDataModelEnvelope,
} from './types.js';

/** A parsed-but-not-yet-normalized envelope, as read from the model's JSON. */
interface RawEnvelope {
  version?: string;
  createSurface?: CreateSurfaceEnvelope['createSurface'];
  updateComponents?: UpdateComponentsEnvelope['updateComponents'];
  updateDataModel?: UpdateDataModelEnvelope['updateDataModel'];
  deleteSurface?: DeleteSurfaceEnvelope['deleteSurface'];
}

/** Opening fence, matched case-insensitively (```a2ui). */
const OPEN_FENCE_RE = /```[ \t]*a2ui[ \t]*\r?\n/i;
/** The longest prefix of an opening fence, used to hold back a partial fence. */
const MAX_PARTIAL_FENCE = '```a2ui\n'.length;
/** Closing fence: ``` on its own (optionally indented) line, or end of text. */
const CLOSE_FENCE_RE = /```/;

/** Result of feeding text to {@link A2uiStreamParser.push}. */
export interface ParseResult {
  /** Prose text ready to stream through (never contains A2UI blocks). */
  prose: string;
  /** Zero or more fully-parsed A2UI envelope batches (one per completed block). */
  envelopeBatches: A2uiEnvelope[][];
}

/** Options controlling how the parser finalizes envelopes. */
export interface A2uiParserOptions {
  /** Catalog used to validate component references. */
  catalog?: A2uiCatalog;
  /** `'strict'` throws on unknown components / bad JSON; `'off'` skips. */
  validate?: 'strict' | 'off';
  /** Protocol version stamped onto envelopes lacking one. */
  version?: string;
  /** Produces the surface id substituted for the model's placeholder. */
  surfaceId: () => string;
}

/**
 * Incremental A2UI extractor. Create one per model turn, `push()` text deltas as
 * they arrive, and `flush()` at the end to drain any trailing block.
 */
export class A2uiStreamParser {
  private buffer = '';
  private inBlock = false;
  /** Stable surface id for the current block (placeholders map to this). */
  private currentSurfaceId: string | null = null;

  constructor(private readonly options: A2uiParserOptions) {}

  /** Feeds a chunk of model text, returning prose + any completed blocks. */
  push(text: string): ParseResult {
    this.buffer += text;
    return this.drain(false);
  }

  /** Drains any remaining buffered content at end of stream. */
  flush(): ParseResult {
    return this.drain(true);
  }

  private drain(final: boolean): ParseResult {
    let prose = '';
    const envelopeBatches: A2uiEnvelope[][] = [];

    // Loop because a single push may contain multiple prose/block transitions.
    // Each iteration makes progress or returns.
    for (;;) {
      if (!this.inBlock) {
        const open = this.buffer.match(OPEN_FENCE_RE);
        if (!open) {
          // No opening fence (yet). Emit prose, but hold back a tail that could
          // be the start of an incomplete opening fence, unless finalizing.
          if (final) {
            prose += this.buffer;
            this.buffer = '';
          } else {
            const keep = Math.min(MAX_PARTIAL_FENCE, this.buffer.length);
            const safeLen = this.buffer.length - keep;
            if (safeLen > 0) {
              prose += this.buffer.slice(0, safeLen);
              this.buffer = this.buffer.slice(safeLen);
            }
          }
          break;
        }
        // Emit prose before the fence, then enter the block.
        prose += this.buffer.slice(0, open.index);
        this.buffer = this.buffer.slice(open.index! + open[0].length);
        this.inBlock = true;
        this.currentSurfaceId = this.options.surfaceId();
        continue;
      }

      // In a block: look for the closing fence.
      const close = this.buffer.match(CLOSE_FENCE_RE);
      if (!close) {
        if (final) {
          // Unterminated block at end of stream — try to parse what we have.
          const batch = this.finalizeBlock(this.buffer);
          if (batch) envelopeBatches.push(batch);
          this.buffer = '';
          this.inBlock = false;
        }
        break;
      }
      const blockText = this.buffer.slice(0, close.index);
      this.buffer = this.buffer.slice(close.index! + close[0].length);
      // Consume an optional trailing newline after the closing fence.
      this.buffer = this.buffer.replace(/^[ \t]*\r?\n/, '');
      this.inBlock = false;
      const batch = this.finalizeBlock(blockText);
      if (batch) envelopeBatches.push(batch);
      continue;
    }

    return { prose, envelopeBatches };
  }

  /** Parses, validates, and normalizes one block's JSON into envelopes. */
  private finalizeBlock(raw: string): A2uiEnvelope[] | null {
    const surfaceId = this.currentSurfaceId ?? this.options.surfaceId();
    this.currentSurfaceId = null;
    const strict = this.options.validate !== 'off';

    const text = raw.trim();
    if (!text) return null;

    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch (e) {
      if (strict) {
        throw new Error(
          `A2UI: failed to parse envelope block as JSON: ${(e as Error).message}`
        );
      }
      return null;
    }

    const envelopes: unknown[] = Array.isArray(parsed) ? parsed : [parsed];
    const out: A2uiEnvelope[] = [];
    for (const env of envelopes) {
      const normalized = this.normalizeEnvelope(env, surfaceId, strict);
      if (normalized) out.push(normalized);
    }
    if (out.length === 0) return null;

    // Guarantee the block opens with a `createSurface`, so the client always
    // has a surface before any update targets it. Models often emit only
    // `updateComponents`/`updateDataModel` on a follow-up (e.g. a "refresh")
    // turn; without this the renderer would drop those updates as "surface not
    // found". Idempotent re-creation is fine — it resets the surface.
    const hasCreate = out.some(
      (e) => (e as CreateSurfaceEnvelope).createSurface !== undefined
    );
    if (!hasCreate) {
      out.unshift({
        version: this.options.version ?? A2UI_VERSION,
        createSurface: {
          surfaceId,
          catalogId: this.options.catalog?.id ?? '',
        },
      });
    }
    return out;
  }

  /**
   * Validates a single envelope, substitutes the real surface id for the
   * placeholder, and stamps the protocol version.
   */
  private normalizeEnvelope(
    env: unknown,
    surfaceId: string,
    strict: boolean
  ): A2uiEnvelope | null {
    if (typeof env !== 'object' || env === null) {
      if (strict) throw new Error('A2UI: envelope must be an object.');
      return null;
    }
    const e = env as RawEnvelope;
    const version = e.version ?? this.options.version ?? A2UI_VERSION;

    const swapSurfaceId = (payload: { surfaceId?: string } | undefined) => {
      if (!payload) return;
      if (
        payload.surfaceId === undefined ||
        payload.surfaceId === SURFACE_ID_PLACEHOLDER ||
        payload.surfaceId === ''
      ) {
        payload.surfaceId = surfaceId;
      }
    };

    if (e.createSurface) {
      swapSurfaceId(e.createSurface);
      return { version, createSurface: e.createSurface };
    }
    if (e.updateComponents) {
      swapSurfaceId(e.updateComponents);
      if (strict) this.validateComponents(e.updateComponents.components);
      return { version, updateComponents: e.updateComponents };
    }
    if (e.updateDataModel) {
      swapSurfaceId(e.updateDataModel);
      return { version, updateDataModel: e.updateDataModel };
    }
    if (e.deleteSurface) {
      swapSurfaceId(e.deleteSurface);
      return { version, deleteSurface: e.deleteSurface };
    }
    if (strict) {
      throw new Error(
        `A2UI: unknown envelope type (keys: ${Object.keys(e).join(', ')}).`
      );
    }
    return null;
  }

  /** Ensures every component references a known catalog component. */
  private validateComponents(components: unknown): void {
    const catalog = this.options.catalog;
    if (!catalog) return;
    if (!Array.isArray(components)) {
      throw new Error('A2UI: updateComponents.components must be an array.');
    }
    const known = new Set(catalog.components.map((c) => c.name));
    const hasRoot = (components as A2uiComponent[]).some(
      (c) => c.id === 'root'
    );
    if (!hasRoot) {
      throw new Error(
        'A2UI: component list must contain a component id "root".'
      );
    }
    for (const c of components as A2uiComponent[]) {
      if (!c || typeof c.component !== 'string') {
        throw new Error('A2UI: every component needs a "component" type name.');
      }
      if (!known.has(c.component)) {
        throw new Error(
          `A2UI: component "${c.component}" is not in catalog "${catalog.id}".`
        );
      }
    }
  }
}
