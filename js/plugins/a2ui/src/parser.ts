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
  type SupportedVersion,
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

/**
 * A single ordered piece of parsed output: either a run of prose or one
 * completed A2UI envelope batch. Segments preserve the exact source order, so
 * prose that appears *after* a block is not reordered ahead of it.
 */
export type ParseSegment = { prose: string } | { envelopes: A2uiEnvelope[] };

/** Result of feeding text to {@link A2uiStreamParser.push}. */
export interface ParseResult {
  /**
   * Ordered prose/envelope segments exactly as they appear in the source text.
   * Prefer this over {@link ParseResult.prose}/{@link ParseResult.envelopeBatches}
   * when order between prose and blocks matters.
   */
  segments: ParseSegment[];
  /**
   * Convenience: all prose runs concatenated (never contains A2UI blocks). Loses
   * the relative order of prose vs. blocks — use {@link ParseResult.segments}
   * when that matters.
   */
  prose: string;
  /** Convenience: the fully-parsed A2UI envelope batches, in order. */
  envelopeBatches: A2uiEnvelope[][];
}

/** Options controlling how the parser finalizes envelopes. */
export interface A2uiParserOptions {
  /** Catalog used to validate component references. */
  catalog?: A2uiCatalog;
  /**
   * `'strict'` throws on unknown components / bad JSON; `'warn'` logs a warning
   * and drops the offending block/envelope; `'off'` skips validation entirely.
   */
  validate?: 'strict' | 'warn' | 'off';
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
    const segments: ParseSegment[] = [];
    // Accumulates prose across loop iterations so consecutive prose runs (e.g.
    // when a partial fence is held back) coalesce into a single segment.
    let proseBuf = '';
    const flushProse = () => {
      if (proseBuf) {
        segments.push({ prose: proseBuf });
        proseBuf = '';
      }
    };

    // Loop because a single push may contain multiple prose/block transitions.
    // Each iteration makes progress or returns.
    for (;;) {
      if (!this.inBlock) {
        const open = this.buffer.match(OPEN_FENCE_RE);
        if (!open) {
          // No opening fence (yet). Emit prose, but hold back a tail that could
          // be the start of an incomplete opening fence, unless finalizing.
          if (final) {
            proseBuf += this.buffer;
            this.buffer = '';
          } else {
            const keep = Math.min(MAX_PARTIAL_FENCE, this.buffer.length);
            const safeLen = this.buffer.length - keep;
            if (safeLen > 0) {
              proseBuf += this.buffer.slice(0, safeLen);
              this.buffer = this.buffer.slice(safeLen);
            }
          }
          break;
        }
        // Emit prose before the fence, then enter the block.
        proseBuf += this.buffer.slice(0, open.index);
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
          if (batch) {
            flushProse();
            segments.push({ envelopes: batch });
          }
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
      if (batch) {
        // Emit any prose seen before this block first, preserving source order.
        flushProse();
        segments.push({ envelopes: batch });
      }
      continue;
    }
    flushProse();

    // Derive the convenience fields from the ordered segments.
    let prose = '';
    const envelopeBatches: A2uiEnvelope[][] = [];
    for (const seg of segments) {
      if ('prose' in seg) prose += seg.prose;
      else envelopeBatches.push(seg.envelopes);
    }
    return { segments, prose, envelopeBatches };
  }

  /**
   * Handles a validation failure according to the configured `validate` mode:
   * throws in `'strict'`, logs a warning in `'warn'` (the middleware default),
   * and is silent in `'off'`. Always returns `null` so callers can
   * `return this.reject(...)`.
   */
  private reject(message: string): null {
    const full = `A2UI: ${message}`;
    if (this.options.validate === 'off') return null;
    if (this.options.validate === 'warn') {
      // Keep this module Node-free / browser-safe: use console, not the logger.
      console.warn(`${full} (dropping block/envelope)`);
      return null;
    }
    throw new Error(full);
  }

  /** Parses, validates, and normalizes one block's JSON into envelopes. */
  private finalizeBlock(raw: string): A2uiEnvelope[] | null {
    const surfaceId = this.currentSurfaceId ?? this.options.surfaceId();
    this.currentSurfaceId = null;

    const text = raw.trim();
    if (!text) return null;

    let parsed: unknown;
    try {
      parsed = JSON.parse(text);
    } catch (e) {
      return this.reject(
        `failed to parse envelope block as JSON: ${(e as Error).message}`
      );
    }

    const envelopes: unknown[] = Array.isArray(parsed) ? parsed : [parsed];
    const out: A2uiEnvelope[] = [];
    for (const env of envelopes) {
      const normalized = this.normalizeEnvelope(env, surfaceId);
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
        version: (this.options.version ?? A2UI_VERSION) as SupportedVersion,
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
    surfaceId: string
  ): A2uiEnvelope | null {
    if (typeof env !== 'object' || env === null) {
      return this.reject('envelope must be an object.');
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
      return {
        version: version as SupportedVersion,
        createSurface: e.createSurface,
      };
    }
    if (e.updateComponents) {
      swapSurfaceId(e.updateComponents);
      if (this.options.validate !== 'off') {
        const err = this.validateComponents(e.updateComponents.components);
        if (err) return this.reject(err);
      }
      return {
        version: version as SupportedVersion,
        updateComponents: e.updateComponents,
      };
    }
    if (e.updateDataModel) {
      swapSurfaceId(e.updateDataModel);
      return {
        version: version as SupportedVersion,
        updateDataModel: e.updateDataModel,
      };
    }
    if (e.deleteSurface) {
      swapSurfaceId(e.deleteSurface);
      return {
        version: version as SupportedVersion,
        deleteSurface: e.deleteSurface,
      };
    }
    return this.reject(
      `unknown envelope type (keys: ${Object.keys(e).join(', ')}).`
    );
  }

  /**
   * Ensures every component references a known catalog component. Returns an
   * error message describing the first problem found, or `null` if valid.
   */
  private validateComponents(components: unknown): string | null {
    const catalog = this.options.catalog;
    if (!catalog) return null;
    if (!Array.isArray(components)) {
      return 'updateComponents.components must be an array.';
    }
    const known = new Set(catalog.components.map((c) => c.name));
    const hasRoot = (components as A2uiComponent[]).some(
      (c) => c.id === 'root'
    );
    if (!hasRoot) {
      return 'component list must contain a component id "root".';
    }
    for (const c of components as A2uiComponent[]) {
      if (!c || typeof c.component !== 'string') {
        return 'every component needs a "component" type name.';
      }
      if (!known.has(c.component)) {
        return `component "${c.component}" is not in catalog "${catalog.id}".`;
      }
    }
    return null;
  }
}
