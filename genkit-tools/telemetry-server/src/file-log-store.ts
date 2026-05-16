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
  LogRecordData,
  type LogQuery,
  type LogQueryResponse,
  type LogStore,
} from '@genkit-ai/tools-common';
import { logger } from '@genkit-ai/tools-common/utils';
import { Mutex } from 'async-mutex';
import crypto from 'crypto';
import fs from 'fs';
import lockfile from 'lockfile';
import path from 'path';

function lockFile(file: string) {
  return `${file}.lock`;
}

export class LocalFileLogStore implements LogStore {
  private readonly storeRoot: string;
  private readonly indexRoot: string;
  private readonly index: LogIndex;
  private mutex = new Mutex();

  constructor(options: { storeRoot: string; indexRoot: string }) {
    this.storeRoot = path.resolve(options.storeRoot, '.genkit/logs');
    fs.mkdirSync(this.storeRoot, { recursive: true });
    this.indexRoot = path.resolve(options.indexRoot, '.genkit/logs_idx');
    fs.mkdirSync(this.indexRoot, { recursive: true });
    logger.debug(
      `[Telemetry Server] initialized local file log store at root: ${this.storeRoot}`
    );
    this.index = new LogIndex(this.indexRoot);
  }

  async init(): Promise<void> {
    // Indexes and store are append-only.
  }

  private getCurrentLogFile(): string {
    const now = new Date();
    // format YYYY-MM-DD-HH
    const pad = (n: number) => n.toString().padStart(2, '0');
    const dateStr = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}-${pad(now.getHours())}`;
    return path.resolve(this.storeRoot, `logs-${dateStr}.jsonl`);
  }

  async save(logs: LogRecordData[]): Promise<void> {
    if (logs.length === 0) return;

    await this.mutex.runExclusive(async () => {
      const logFile = this.getCurrentLogFile();
      const relativeFileName = path.basename(logFile);

      let currentOffset = 0;
      if (fs.existsSync(logFile)) {
        currentOffset = fs.statSync(logFile).size;
      }

      const indexEntries: LogIndexEntry[] = [];
      let offsetTracker = currentOffset;

      const linesToAppend = logs.map((log) => {
        // Ensure log id exists if not provided
        if (!log.logId) {
          log.logId = crypto.randomUUID();
        }
        const line = JSON.stringify(log) + '\n';
        const length = Buffer.byteLength(line, 'utf8');

        indexEntries.push({
          traceId: log.traceId,
          spanId: log.spanId,
          timestamp: log.timestamp,
          severityText: log.severityText,
          severityNumber: log.severityNumber,
          file: relativeFileName,
          offset: offsetTracker,
          length: length,
        });

        offsetTracker += length;
        return line;
      });

      fs.appendFileSync(logFile, linesToAppend.join(''));
      await this.index.add(indexEntries);
    });
  }

  async list(query?: LogQuery): Promise<LogQueryResponse> {
    const startFromIndex = query?.continuationToken
      ? Number.parseInt(query.continuationToken)
      : 0;
    const limit = query?.limit ?? 100;

    const searchResult = this.index.search({
      limit,
      startFromIndex,
      traceId: query?.traceId,
      spanId: query?.spanId,
    });

    const logs: LogRecordData[] = [];
    const fdCache = new Map<string, number>();

    try {
      for (const entry of searchResult.entries) {
        const logFile = path.resolve(this.storeRoot, entry.file);
        let fd = fdCache.get(logFile);

        if (fd === undefined) {
          if (fs.existsSync(logFile)) {
            try {
              fd = fs.openSync(logFile, 'r');
              fdCache.set(logFile, fd);
            } catch (e) {
              logger.error(`Error opening log file ${logFile}: ${e}`);
              continue;
            }
          }
        }

        if (fd !== undefined) {
          const buffer = Buffer.alloc(entry.length);
          try {
            fs.readSync(fd, buffer, 0, entry.length, entry.offset);
            logs.push(
              JSON.parse(buffer.toString('utf8').trim()) as LogRecordData
            );
          } catch (e) {
            logger.error(
              `Error reading or parsing log at ${entry.file}:${entry.offset}: ${e}`
            );
          }
        }
      }
    } finally {
      for (const fd of fdCache.values()) {
        try {
          fs.closeSync(fd);
        } catch (e) {
          logger.error(`Error closing file descriptor for log file: ${e}`);
        }
      }
    }

    return {
      logs,
      continuationToken: searchResult.pageLastIndex
        ? `${searchResult.pageLastIndex}`
        : undefined,
    };
  }
}

export interface LogIndexEntry {
  traceId?: string;
  spanId?: string;
  timestamp: number;
  severityText?: string;
  severityNumber?: number;
  file: string;
  offset: number;
  length: number;
}

export interface LogIndexSearchResult {
  pageLastIndex?: number;
  entries: LogIndexEntry[];
}

export class LogIndex {
  private currentIndexFile: string;

  constructor(private indexRoot: string) {
    this.currentIndexFile = path.resolve(
      this.indexRoot,
      this.newIndexFileName()
    );
    fs.mkdirSync(this.indexRoot, { recursive: true });
  }

  private newIndexFileName() {
    return `idx_${(Date.now() + '').padStart(17, '0')}.jsonl`;
  }

  listIndexFiles() {
    return fs.readdirSync(this.indexRoot).filter((f) => f.startsWith('idx_'));
  }

  async add(entries: LogIndexEntry[]) {
    if (entries.length === 0) return;
    try {
      await new Promise<void>((resolve, reject) => {
        lockfile.lock(
          lockFile(this.currentIndexFile),
          { wait: 1000, retries: 5, retryWait: 100 },
          (err) => {
            if (err) reject(err);
            else resolve();
          }
        );
      });
      const lines = entries.map((e) => JSON.stringify(e) + '\n').join('');
      fs.appendFileSync(this.currentIndexFile, lines);
    } catch (err) {
      logger.error(
        `Failed to lock log index file ${this.currentIndexFile}: ${err}`
      );
    } finally {
      if (fs.existsSync(lockFile(this.currentIndexFile))) {
        lockfile.unlockSync(lockFile(this.currentIndexFile));
      }
    }
  }

  search(query: {
    limit: number;
    startFromIndex: number;
    traceId?: string;
    spanId?: string;
  }): LogIndexSearchResult {
    const indexFiles = this.listIndexFiles().sort().reverse();

    let skipped = 0;
    const entries: LogIndexEntry[] = [];
    let hasMore = false;

    for (const idxFile of indexFiles) {
      if (hasMore) break;

      const idxTxt = fs.readFileSync(
        path.resolve(this.indexRoot, idxFile),
        'utf8'
      );
      const fileData = idxTxt
        .split('\n')
        .filter((l) => l.trim().length > 0)
        .map((l) => {
          try {
            return JSON.parse(l) as LogIndexEntry;
          } catch {
            return undefined;
          }
        })
        .filter((d): d is LogIndexEntry => {
          if (!d) return false;
          if (query.traceId && d.traceId !== query.traceId) return false;
          if (query.spanId && d.spanId !== query.spanId) return false;
          return true;
        });

      fileData.sort((a, b) => b.timestamp - a.timestamp); // Newest first

      for (const entry of fileData) {
        if (skipped < query.startFromIndex) {
          skipped++;
          continue;
        }

        if (entries.length < query.limit) {
          entries.push(entry);
        } else {
          hasMore = true;
          break;
        }
      }
    }

    const result: LogIndexSearchResult = {
      entries,
    };

    if (hasMore) {
      result.pageLastIndex = query.startFromIndex + query.limit;
    }

    return result;
  }
}
