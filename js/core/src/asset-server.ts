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

import express from 'express';
import fs from 'fs/promises';
import getPort, { makeRange } from 'get-port';
import type { Server } from 'http';
import path from 'path';
import { logger } from './logging.js';

/**
 * Translates a module URL into an accessible URL if it's a local file path.
 *
 * @hidden
 */
export function getAssetUrl(
  moduleUrl: string,
  projectRoot: string | null
): string {
  if (moduleUrl.startsWith('http://') || moduleUrl.startsWith('https://')) {
    return moduleUrl;
  }
  let absolutePath = moduleUrl;
  if (!path.isAbsolute(moduleUrl) && projectRoot) {
    absolutePath = path.resolve(projectRoot, moduleUrl);
  }
  const id = Buffer.from(absolutePath).toString('base64url');
  return `/api/ui/assets/${id}`;
}

/**
 * Dedicated server for serving UI plugin assets.
 *
 * @hidden
 */
export class AssetServer {
  private app = express();
  private server: Server | null = null;
  private port: number | null = null;
  private projectRoot: string | null = null;

  constructor(projectRoot: string | null) {
    this.projectRoot = projectRoot;
    this.app.get('/api/ui/assets/:id', async (req, res) => {
      try {
        const id = req.params.id;
        const filePath = Buffer.from(id, 'base64url').toString('utf8');
        // Simple security check: must be an absolute path
        if (!path.isAbsolute(filePath)) {
          res.status(403).send('Forbidden: only absolute paths are allowed');
          return;
        }
        await fs.access(filePath);
        res.sendFile(filePath);
      } catch (err) {
        res.status(404).send('Asset not found');
      }
    });
  }

  get url(): string | null {
    return this.port ? `http://localhost:${this.port}` : null;
  }

  getAssetUrl(moduleUrl: string): string {
    return getAssetUrl(moduleUrl, this.projectRoot);
  }

  async start() {
    this.port = await getPort({ port: makeRange(3500, 3600) });
    return new Promise<void>((resolve) => {
      this.server = this.app.listen(this.port, () => {
        logger.debug(`Asset server running on ${this.url}`);
        resolve();
      });
    });
  }

  async stop() {
    if (this.server) {
      return new Promise<void>((resolve) => {
        this.server!.close(() => {
          this.port = null;
          this.server = null;
          resolve();
        });
      });
    }
  }
}
