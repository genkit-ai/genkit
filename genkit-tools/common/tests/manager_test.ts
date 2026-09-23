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
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  jest,
} from '@jest/globals';
import fs from 'fs/promises';
import http from 'http';
import os from 'os';
import path from 'path';
import { RuntimeManager } from '../src/manager/manager';
import { REFLECTION_SECRET_HEADER } from '../src/manager/reflection-auth';
import { RuntimeEvent, type RuntimeInfo } from '../src/manager/types';

jest.mock('chokidar', () => ({
  watch: jest.fn().mockReturnValue({
    on: jest.fn(),
    close: jest.fn(),
  }),
}));

describe('RuntimeManager', () => {
  it('should allow unsubscribing from runtime events', async () => {
    const manager = await RuntimeManager.create({ projectRoot: '.' });
    const listener = jest.fn();

    // Subscribe
    const unsubscribe = manager.onRuntimeEvent(listener);

    // Simulate event
    (manager as any).eventEmitter.emit(RuntimeEvent.ADD, { id: '1' });
    expect(listener).toHaveBeenCalledTimes(1);

    // Unsubscribe
    unsubscribe();

    // Simulate event again
    (manager as any).eventEmitter.emit(RuntimeEvent.ADD, { id: '2' });
    expect(listener).toHaveBeenCalledTimes(1); // Should not have increased

    await manager.stop();
  });
});

describe('RuntimeManager reflection auth', () => {
  let server: http.Server;
  let serverUrl: string;
  let seenSecrets: (string | undefined)[];
  let projectRoot: string;
  let manager: RuntimeManager | undefined;

  beforeEach(async () => {
    seenSecrets = [];
    server = http.createServer((req, res) => {
      seenSecrets.push(
        req.headers[REFLECTION_SECRET_HEADER] as string | undefined
      );
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end('{}');
    });
    await new Promise<void>((resolve) =>
      server.listen(0, '127.0.0.1', resolve)
    );
    const address = server.address();
    if (typeof address === 'string' || address === null) {
      throw new Error('expected a TCP address');
    }
    serverUrl = `http://127.0.0.1:${address.port}`;
    projectRoot = await fs.mkdtemp(path.join(os.tmpdir(), 'genkit-mgr-'));
  });

  afterEach(async () => {
    await manager?.stop();
    manager = undefined;
    await new Promise<void>((resolve) => server.close(() => resolve()));
    await fs.rm(projectRoot, { recursive: true, force: true });
  });

  /** Writes a discovery file and returns once the manager has picked it up. */
  async function withRuntimeFile(
    contents: Record<string, unknown>
  ): Promise<RuntimeManager> {
    const created = (await RuntimeManager.create({
      projectRoot,
      manageHealth: false,
      reflectionSecret: 'cli-secret',
    })) as RuntimeManager;
    manager = created;
    await fs.mkdir(path.join(projectRoot, '.genkit', 'runtimes'), {
      recursive: true,
    });
    const file = path.join(
      projectRoot,
      '.genkit',
      'runtimes',
      'test-runtime.json'
    );
    await fs.writeFile(file, JSON.stringify(contents));
    await (created as any).handleNewRuntime(file);
    return created;
  }

  const runtimeFile = (extra: Record<string, unknown> = {}) => ({
    id: 'rt-1',
    pid: 1234,
    reflectionServerUrl: serverUrl,
    timestamp: new Date().toISOString(),
    genkitVersion: 'nodejs/1.0.0',
    reflectionApiSpecVersion: 1,
    ...extra,
  });

  it('sends the runtime own secret from its discovery file', async () => {
    const mgr = await withRuntimeFile(
      runtimeFile({ reflectionSecret: 'runtime-secret' })
    );
    await mgr.listActions();
    expect(seenSecrets).toContain('runtime-secret');
  });

  it('keeps the secret out of RuntimeInfo', async () => {
    const mgr = await withRuntimeFile(
      runtimeFile({ reflectionSecret: 'runtime-secret' })
    );
    const runtimes: RuntimeInfo[] = mgr.listRuntimes();
    expect(runtimes).toHaveLength(1);
    expect(JSON.stringify(runtimes)).not.toContain('runtime-secret');
    expect('reflectionSecret' in runtimes[0]).toBe(false);
  });

  it('falls back to the configured secret when the file has none', async () => {
    const mgr = await withRuntimeFile(runtimeFile());
    await mgr.listActions();
    expect(seenSecrets).toContain('cli-secret');
  });

  it('explains a 401 from the runtime', async () => {
    const mgr = await withRuntimeFile(runtimeFile());
    server.removeAllListeners('request');
    server.on('request', (_req, res) => {
      res.writeHead(401);
      res.end();
    });
    await expect(mgr.listActions()).rejects.toThrow(
      /GENKIT_REFLECTION_SECRET_TOKEN/
    );
  });
});
