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

import fs from 'fs';
import { readFile, unlink, writeFile } from 'fs/promises';
import path from 'path';
import { createInterface } from 'readline';
import type {
  ListEvalKeysRequest,
  ListEvalKeysResponse,
} from '../types/apis.js';
import {
  EvalRunKeySchema,
  EvalRunSchema,
  type EvalRun,
  type EvalRunKey,
  type EvalStore,
} from '../types/eval.js';
import { logger } from '../utils/logger.js';

/**
 * A local, file-based EvalStore implementation.
 */
export class LocalFileEvalStore implements EvalStore {
  private storeRoot: string = '';
  private indexFile: string = '';
  private static cachedEvalStore: LocalFileEvalStore | null = null;

  private async init() {
    this.storeRoot = this.generateRootPath();
    this.indexFile = await this.resolveIndexFile();
    fs.mkdirSync(this.storeRoot, { recursive: true });
    if (!fs.existsSync(this.indexFile)) {
      fs.writeFileSync(path.resolve(this.indexFile), JSON.stringify({}));
    }
    logger.debug(
      `Initialized local file eval store at root: ${this.storeRoot}`
    );
  }

  static async getEvalStore() {
    if (!this.cachedEvalStore) {
      this.cachedEvalStore = new LocalFileEvalStore();
      await this.cachedEvalStore.init();
    }
    return this.cachedEvalStore;
  }

  static reset() {
    this.cachedEvalStore = null;
  }

  async save(evalRun: EvalRun): Promise<void> {
    const fileName = this.resolveEvalFilename(evalRun.key.evalRunId);

    logger.debug(
      `Saving EvalRun ${evalRun.key.evalRunId} to ` +
        path.resolve(this.storeRoot, fileName)
    );
    await writeFile(
      path.resolve(this.storeRoot, fileName),
      JSON.stringify(evalRun)
    );

    const index = await this.getEvalsIndex();
    index[evalRun.key.evalRunId] = evalRun.key;
    await writeFile(
      path.resolve(this.indexFile),
      JSON.stringify(index, null, 2)
    );
  }

  async load(evalRunId: string): Promise<EvalRun | undefined> {
    const filePath = path.resolve(
      this.storeRoot,
      this.resolveEvalFilename(evalRunId)
    );
    if (!fs.existsSync(filePath)) {
      return undefined;
    }
    return await readFile(filePath, 'utf8').then((data) =>
      EvalRunSchema.parse(JSON.parse(data))
    );
  }

  async list(
    query?: ListEvalKeysRequest | undefined
  ): Promise<ListEvalKeysResponse> {
    logger.debug(`Listing keys for filter: ${JSON.stringify(query)}`);
    let keys = await this.getEvalsIndex().then((index) => Object.values(index));

    if (query?.filter?.actionRef) {
      keys = keys.filter((key) => key.actionRef === query?.filter?.actionRef);
    }

    return {
      evalRunKeys: keys,
    };
  }

  async delete(evalRunId: string): Promise<void> {
    const filePath = path.resolve(
      this.storeRoot,
      this.resolveEvalFilename(evalRunId)
    );
    if (fs.existsSync(filePath)) {
      await unlink(filePath);

      const index = await this.getEvalsIndex();
      delete index[evalRunId];
      await writeFile(
        path.resolve(this.indexFile),
        JSON.stringify(index, null, 2)
      );
    }
  }

  private resolveEvalFilename(evalRunId: string): string {
    return `${evalRunId}.json`;
  }

  private async resolveIndexFile(): Promise<string> {
    const txtPath = path.resolve(this.storeRoot, 'index.txt');
    const jsonPath = path.resolve(this.storeRoot, 'index.json');
    if (fs.existsSync(txtPath)) {
      // Copy over index, delete txt file
      const keys = await this.processLineByLine(txtPath);
      await writeFile(path.resolve(jsonPath), JSON.stringify(keys, null, 2));
      await unlink(txtPath);
    }
    return jsonPath;
  }

  private async processLineByLine(filePath: string) {
    const fileStream = fs.createReadStream(filePath);
    const keys: Record<string, EvalRunKey> = {};

    const rl = createInterface({
      input: fileStream,
      crlfDelay: Infinity,
    });
    for await (const line of rl) {
      try {
        const entry = JSON.parse(line);
        const runKey = EvalRunKeySchema.parse(entry);
        keys[runKey.evalRunId] = runKey;
      } catch (e) {
        logger.debug(`Error parsing ${line}:\n`, JSON.stringify(e));
      }
    }
    return keys;
  }

  private generateRootPath(): string {
    return path.resolve(process.cwd(), '.genkit', 'evals');
  }

  private async getEvalsIndex(): Promise<Record<string, EvalRunKey>> {
    if (!fs.existsSync(this.indexFile)) {
      return Promise.resolve({} as any);
    }
    return await readFile(path.resolve(this.indexFile), 'utf8').then((data) =>
      JSON.parse(data)
    );
  }
}
