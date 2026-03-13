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

import {
  SpanData,
  TraceDataSchema,
  type TraceData,
  type TraceQueryFilter,
} from '@genkit-ai/tools-common';
import { logger } from '@genkit-ai/tools-common/utils';
import { Mutex } from 'async-mutex';
import fs from 'fs';
import lockfile from 'lockfile';
import path from 'path';
import type { TraceQuery, TraceQueryResponse, TraceStore } from './types';
import { version as currentVersion } from './utils/version';

const MAX_TRACES = 1000;
const MAX_INDEX_FILES = 10;
const MAX_LIST_ATTR_LENGTH = 1000;

/**
 * Implementation of trace store that persists traces on local disk.
 */
export class LocalFileTraceStore implements TraceStore {
  private readonly storeRoot;
  private readonly indexRoot;
  private mutexes: Record<string, Mutex> = {};
  private filters: Record<string, string>;
  private readonly index: Index;

  static defaultFilters: Record<string, string> = {
    // Prevent prompt rendering from spamming local trace store
    'genkit:metadata:subtype': 'prompt',
  };

  constructor(options: {
    filters?: Record<string, string>;
    storeRoot: string;
    indexRoot: string;
  }) {
    this.storeRoot = path.resolve(options.storeRoot, `.genkit/traces`);
    fs.mkdirSync(this.storeRoot, { recursive: true });
    this.indexRoot = path.resolve(options.indexRoot, `.genkit/traces_idx`);
    fs.mkdirSync(this.indexRoot, { recursive: true });
    logger.debug(
      `[Telemetry Server] initialized local file trace store at root: ${this.storeRoot}`
    );
    this.filters = options?.filters ?? LocalFileTraceStore.defaultFilters;
    this.index = new Index(this.indexRoot);
  }

  async init() {
    const metadata = this.index.getMetadata();
    // if the metadata file doesn't exist or it was for the older version or if
    // there are too many index files we recreate the index.
    if (
      !metadata ||
      metadata.version !== currentVersion ||
      this.index.listIndexFiles().length > MAX_INDEX_FILES
    ) {
      await this.reIndex();
    }
  }

  private async reIndex() {
    this.index.clear();
    const time = Date.now();
    // Index only the last MAX_TRACES traces.
    const list = await this.listFromFiles({ limit: MAX_TRACES });
    for (const trace of list.traces.reverse()) {
      const hasRootSpan = !!Object.values(trace.spans).find(
        (s) => !s.parentSpanId
      );
      if (!hasRootSpan) continue;
      this.index.add(trace);
    }
    logger.info(
      `Indexed ${list.traces.length} traces in ${Date.now() - time}ms in ${this.indexRoot}`
    );
  }

  async load(id: string): Promise<TraceData | undefined> {
    const filePath = path.resolve(this.storeRoot, `${id}`);
    if (!fs.existsSync(filePath)) {
      return undefined;
    }
    const data = fs.readFileSync(filePath, 'utf8');
    const parsed = JSON.parse(data);
    // For backwards compatibility, new field.
    if (!parsed.traceId) {
      parsed.traceId = id;
    }
    return TraceDataSchema.parse(parsed);
  }

  getMutex(id: string): Mutex {
    if (!this.mutexes[id]) {
      this.mutexes[id] = new Mutex();
    }
    return this.mutexes[id];
  }

  async save(id: string, rawTrace: TraceData): Promise<void> {
    let trace = this.filter(rawTrace);
    // if everything is filtered, it's probably the root.
    const possibleRoot = Object.keys(trace.spans).length === 0;
    const mutex = this.getMutex(id);
    await mutex.waitForUnlock();
    const release = await mutex.acquire();
    try {
      const existing = (await this.load(id)) || trace;
      if (existing) {
        Object.keys(trace.spans).forEach((spanId) => {
          // If the existing span is already completed (has endTime), and the incoming
          // span is incomplete (no endTime), we assume the incoming span is a stale
          // "start" event that arrived out of order. We ignore it to prevent overwriting
          // the completed span with incomplete data.
          if (
            existing.spans[spanId] &&
            existing.spans[spanId].endTime &&
            !trace.spans[spanId].endTime
          ) {
            return;
          }
          existing.spans[spanId] = trace.spans[spanId];
        });
        // If it's one of those weird roots (internal span that we filter) we try to fix
        // whoever was referencing it by making them root.
        if (possibleRoot) {
          Object.keys(existing.spans).forEach((spanId) => {
            const span = existing.spans[spanId];
            if (
              possibleRoot &&
              span.parentSpanId &&
              !existing.spans[span.parentSpanId]
            ) {
              delete span.parentSpanId;
            }
          });
        }
        // Only update metadata if new values are defined
        // Prevents overwriting with undefined when batches arrive without root span
        if (trace.displayName !== undefined) {
          existing.displayName = trace.displayName;
        }
        if (trace.startTime !== undefined) {
          existing.startTime = trace.startTime;
        }
        if (trace.endTime !== undefined) {
          existing.endTime = trace.endTime;
        }
        trace = existing;
      }
      fs.writeFileSync(
        path.resolve(this.storeRoot, `${id}`),
        JSON.stringify(trace)
      );
      const hasRootSpan = !!Object.values(rawTrace.spans).find(
        (s) => !s.parentSpanId
      );
      if (this.index && hasRootSpan) {
        // re-load the full trace, there are likely spans written there previously.
        const fullTrace = await this.load(rawTrace.traceId);
        if (!fullTrace) {
          throw new Error(
            'unable to read the trace that was just written... "this should never happen"'
          );
        }
        this.index.add(fullTrace);
      }
    } finally {
      release();
    }
  }

  async list(query?: TraceQuery): Promise<TraceQueryResponse> {
    const searchResult = this.index.search({
      limit: query?.limit ?? 10,
      startFromIndex: query?.continuationToken
        ? Number.parseInt(query?.continuationToken)
        : undefined,
      filter: query?.filter,
    });

    const loadedTraces = await Promise.all(
      searchResult.data.map((d) => this.load(d['id']).then(trucateTraceDetails))
    );

    return {
      traces: loadedTraces.filter((t) => !!t) as TraceData[],
      continuationToken: searchResult.pageLastIndex
        ? `${searchResult.pageLastIndex}`
        : undefined,
    };
  }

  private async listFromFiles(query?: TraceQuery): Promise<TraceQueryResponse> {
    const files = fs.readdirSync(this.storeRoot);
    files.sort((a, b) => {
      return (
        fs.statSync(path.resolve(this.storeRoot, `${b}`)).mtime.getTime() -
        fs.statSync(path.resolve(this.storeRoot, `${a}`)).mtime.getTime()
      );
    });
    const startFrom = query?.continuationToken
      ? Number.parseInt(query?.continuationToken)
      : 0;
    const stopAt = startFrom + (query?.limit || 10);
    const traces = files.slice(startFrom, stopAt).map((id) => {
      const filePath = path.resolve(this.storeRoot, `${id}`);
      const data = fs.readFileSync(filePath, 'utf8');
      const parsed = JSON.parse(data);
      // For backwards compatibility, new field.
      if (!parsed.traceId) {
        parsed.traceId = id;
      }
      return TraceDataSchema.parse(parsed);
    });
    return {
      traces,
      continuationToken: files.length > stopAt ? stopAt.toString() : undefined,
    };
  }

  private filter(trace: TraceData): TraceData {
    // Delete any spans that match the filter criteria
    Object.keys(trace.spans).forEach((spanId) => {
      const span = trace.spans[spanId];
      Object.keys(this.filters).forEach((f) => {
        if (span.attributes[f] === this.filters[f]) {
          delete trace.spans[spanId];
        }
      });
    });
    // Delete the root wrapper if it's the only span left
    if (Object.keys(trace.spans).length === 1) {
      Object.keys(trace.spans).forEach((spanId) => {
        const span = trace.spans[spanId];
        if (span.attributes['genkit:name'] === 'dev-run-action-wrapper') {
          delete trace.spans[spanId];
        }
      });
    }
    return trace;
  }
}

function trucateTraceDetails(t?: TraceData): TraceData | undefined {
  if (!t) return t;

  const { spans: originalSpans, ...restOfTrace } = t;

  const spans = {} as Record<string, SpanData>;
  for (const spanId of Object.keys(originalSpans)) {
    if (!originalSpans[spanId].parentSpanId) {
      const { attributes: originalAttributes, ...restOfSpan } =
        originalSpans[spanId];
      spans[spanId] = {
        attributes: trucateLargeAttrs(originalAttributes),
        ...restOfSpan,
      } as SpanData;
      break;
    }
  }

  return { spans, ...restOfTrace };
}

export function trucateLargeAttrs<T>(input: T): T {
  if (
    input === undefined ||
    input === null ||
    Array.isArray(input) ||
    typeof input !== 'object'
  ) {
    return input;
  }
  for (const key in input) {
    if (
      typeof input[key] === 'string' &&
      (input[key] as string).length > MAX_LIST_ATTR_LENGTH
    ) {
      input[key] = ((input[key] as string).substring(0, MAX_LIST_ATTR_LENGTH) +
        '...') as any;
    }
  }
  return input;
}

export interface IndexSearchResult {
  pageLastIndex?: number;
  data: Record<string, string>[];
}

/**
 * A super basic searchable index implementation. It's not particularly efficient,
 * but should not be worse than reading individual trace files from disk.
 */
export class Index {
  private currentIndexFile: string;

  constructor(private indexRoot: string) {
    // TODO: do something about index getting too big. Delete/forget old stuff, etc.
    this.currentIndexFile = path.resolve(
      this.indexRoot,
      this.newIndexFileName()
    );
    fs.mkdirSync(this.indexRoot, { recursive: true });
  }

  clear() {
    fs.rmSync(this.indexRoot, { recursive: true, force: true });
    fs.mkdirSync(this.indexRoot, { recursive: true });
    fs.appendFileSync(
      this.metadataFileName(),
      JSON.stringify({ version: currentVersion })
    );
  }

  metadataFileName() {
    return path.resolve(this.indexRoot, 'genkit.metadata');
  }

  getMetadata(): { version: string } | undefined {
    try {
      return JSON.parse(
        fs.readFileSync(this.metadataFileName(), { encoding: 'utf8' })
      );
    } catch (e) {
      return undefined;
    }
  }

  private newIndexFileName() {
    return `idx_${(Date.now() + '').padStart(17, '0')}.json`;
  }

  listIndexFiles() {
    return fs.readdirSync(this.indexRoot).filter((f) => f.startsWith('idx_'));
  }

  add(traceData: TraceData) {
    const rootSpans = Object.values(traceData.spans).filter(
      (s) => !s.parentSpanId
    );
    const rootSpan = rootSpans.length > 0 ? rootSpans[0] : undefined;
    const indexData = {
      id: traceData.traceId,
    } as Record<string, string | number>;
    indexData['type'] =
      `${rootSpan?.attributes?.['genkit:metadata:subtype'] || rootSpan?.attributes?.['genkit:type'] || 'UNKNOWN'}`;
    if (rootSpan?.startTime) {
      indexData['start'] = rootSpan.startTime;
    }
    if (rootSpan?.displayName) {
      indexData['name'] = rootSpan.displayName;
    }
    if (rootSpan?.endTime) {
      indexData['end'] = rootSpan.endTime;
    }
    if (rootSpan?.displayName) {
      indexData['status'] = rootSpan.status?.code ?? 'UNKNOWN';
    }

    Object.keys(rootSpan?.attributes ?? {})
      .filter((k) => k.startsWith('genkitx:'))
      .forEach((k) => {
        indexData[k] = `${rootSpan!.attributes[k]}`;
      });

    try {
      lockfile.lockSync(lockFile(this.currentIndexFile));

      fs.appendFileSync(
        this.currentIndexFile,
        JSON.stringify(indexData) + '\n'
      );
    } finally {
      lockfile.unlockSync(lockFile(this.currentIndexFile));
    }
  }

  search(query: {
    limit: number;
    startFromIndex?: number;
    filter?: TraceQueryFilter;
  }): IndexSearchResult {
    const startFromIndex = query.startFromIndex ?? 0;

    const fullData = [] as Record<string, string | number>[];

    for (const idxFile of this.listIndexFiles()) {
      const idxTxt = fs.readFileSync(
        path.resolve(this.indexRoot, idxFile),
        'utf8'
      );
      const fileData = idxTxt
        .split('\n')
        .map((l) => {
          try {
            return JSON.parse(l) as Record<string, string | number>;
          } catch {
            return undefined;
          }
        })
        .filter((d) => (!d ? false : isFilterMatch(d, query?.filter)))
        .reverse() as Record<string, string | number>[];

      fullData.push(...fileData);
    }
    fullData
      // We must sort the data as chronological ordering is not guaranteed between
      // different index files.
      .sort((a, b) => (b!['start'] as number) - (a!['start'] as number));

    // Dedupe by trace ID, keeping the most recent entry (first seen after sorting)
    // TODO: Real-time span events pollute the index with duplicates. This deduplication
    // works but may degrade index performance over time. Revisit if we see issues.
    const deduped = [] as Record<string, string | number>[];
    const seenIds = new Set<string>();
    for (const entry of fullData) {
      const id = entry['id'] as string;
      if (!seenIds.has(id)) {
        seenIds.add(id);
        deduped.push(entry);
      }
    }

    const result = {
      data: deduped.slice(startFromIndex, startFromIndex + query.limit),
    } as IndexSearchResult;

    // if there are more results, populate stop index.
    if (startFromIndex + query.limit < deduped.length) {
      result.pageLastIndex = startFromIndex + query.limit;
    }

    return result;
  }
}

function isFilterMatch(
  d: Record<string, string | number>,
  filter?: TraceQueryFilter
): boolean {
  if (!filter) return true;
  const { eq, neq, gt, gte, lt, lte } = filter;
  if (eq) {
    for (const k of Object.keys(eq)) {
      const filterVal = eq[k];
      const val = d[k];
      if (
        Array.isArray(filterVal)
          ? !filterVal.includes(val as any)
          : val !== filterVal
      )
        return false;
    }
  }
  if (neq) {
    for (const k of Object.keys(neq)) {
      const filterVal = neq[k];
      const val = d[k];
      if (
        Array.isArray(filterVal)
          ? filterVal.includes(val as any)
          : val === filterVal
      )
        return false;
    }
  }
  if (gt) {
    for (const k of Object.keys(gt)) {
      const val = d[k];
      if (typeof val !== 'number' || val <= gt[k]) return false;
    }
  }
  if (gte) {
    for (const k of Object.keys(gte)) {
      const val = d[k];
      if (typeof val !== 'number' || val < gte[k]) return false;
    }
  }
  if (lt) {
    for (const k of Object.keys(lt)) {
      const val = d[k];
      if (typeof val !== 'number' || val >= lt[k]) return false;
    }
  }
  if (lte) {
    for (const k of Object.keys(lte)) {
      const val = d[k];
      if (typeof val !== 'number' || val > lte[k]) return false;
    }
  }
  return true;
}

function lockFile(file: string) {
  return `${file}.lock`;
}
