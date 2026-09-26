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

import { logger } from 'genkit/logging';
import { spawn } from 'node:child_process';
import { randomUUID } from 'node:crypto';
import { BOX_SELF_ID_ENV } from '../env.js';
import { SubprocessProvider } from '../providers/subprocess.js';
import type { SandboxProvider, SpawnHandle } from '../providers/types.js';
import { REFLECTION_SECRET_ENV } from '../reflection-auth.js';
import { ReflectionHost } from '../reflection-host.js';
import type {
  BoxConnection,
  BoxRunner,
  RunActionRequest,
  RunActionResult,
  RunOptions,
} from '../types.js';

/** Options for {@link execRunner}. */
export interface ExecRunnerOptions {
  /**
   * Command that starts the box entry point, e.g. `tsx src/boxed.ts`. Split on
   * whitespace into program + args. Mutually exclusive with `self`.
   */
  cmd?: string;
  /**
   * Re-spawn the current entry point (`process.argv`) as the box. Convenient
   * (one file) at the cost of weaker code isolation.
   */
  self?: boolean;
  /** Isolation strategy. Omit for a plain (trusted) subprocess. */
  isolate?: SandboxProvider;
  /** Working directory for spawned boxes. Defaults to the current cwd. */
  cwd?: string;
}

interface BoxInstance {
  runtimeId: string;
  handle: SpawnHandle;
  lastUsed: number;
}

/**
 * All live box child pids in this process, and a one-time set of process
 * handlers that terminate them when we exit. This is what makes nested cleanup
 * cascade: when a runtime (e.g. C1) is SIGTERM'd, its handler kills the boxes it
 * spawned (C2) instead of orphaning them into a reconnect loop.
 *
 * We only auto-install these in child box runtimes (identified by
 * GENKIT_BOX_SELF_ID). A user's top-level app is left alone: it owns its own
 * process and is expected to call `box.close()` explicitly, so the box library
 * never hijacks signals there.
 */
const liveChildren = new Set<number>();
let handlersInstalled = false;

function installCleanupHandlers() {
  if (handlersInstalled) return;
  handlersInstalled = true;
  if (!process.env[BOX_SELF_ID_ENV]) return; // top-level app: opt out
  const killAll = () => {
    for (const pid of liveChildren) {
      try {
        process.kill(pid, 'SIGTERM');
      } catch {
        // already gone
      }
    }
    liveChildren.clear();
  };
  // 'exit' must be synchronous; killAll is.
  process.on('exit', killAll);
  // Translate termination signals into a clean exit so 'exit' runs, then die.
  for (const sig of ['SIGTERM', 'SIGINT', 'SIGHUP'] as const) {
    process.on(sig, () => {
      killAll();
      process.exit(0);
    });
  }
}

function toHandle(child: ReturnType<typeof spawn>): SpawnHandle {
  return {
    pid: child.pid,
    kill: () =>
      new Promise<void>((resolve) => {
        if (child.exitCode !== null || child.signalCode !== null) {
          return resolve();
        }
        child.once('exit', () => resolve());
        try {
          child.kill('SIGTERM');
        } catch {
          resolve();
        }
      }),
    onExit: (cb) => child.once('exit', (code) => cb(code)),
  };
}

/**
 * A {@link BoxRunner} that runs each box as a local subprocess and talks to it
 * over the shared reflection host. Owns the routing-key -> instance map. This
 * is the runner behind self-mode and separate-entry boxes; `isolate:` layers
 * OS-level isolation on top without changing the lifecycle.
 */
export class ExecRunner implements BoxRunner {
  readonly name = 'exec';
  private host = new ReflectionHost();
  private hostStarted?: Promise<void>;
  private provider: SandboxProvider;
  private instances = new Map<string, BoxInstance>();
  private closed = false;
  /** Id of the owning box, passed to runtimes as {@link BOX_SELF_ID_ENV}. */
  private boxId?: string;

  constructor(private readonly options: ExecRunnerOptions) {
    if (options.self && options.cmd) {
      throw new Error('execRunner: `self` and `cmd` are mutually exclusive.');
    }
    if (!options.self && !options.cmd) {
      throw new Error('execRunner: one of `self` or `cmd` is required.');
    }
    this.provider = options.isolate ?? new SubprocessProvider();
  }

  attach(box: { readonly id: string }): void {
    this.boxId = box.id;
  }

  private ensureHost(): Promise<void> {
    if (!this.hostStarted) {
      this.hostStarted = this.host.start().then(() => undefined);
    }
    return this.hostStarted;
  }

  private baseSpawn(): { cmd: string; args: string[] } {
    if (this.options.self) {
      // Re-run the current entry point. Use execPath + execArgv so loader flags
      // (e.g. tsx's `--import`, source maps) are preserved; process.argv[0] is
      // the node binary and argv[1..] is the script + its args.
      return {
        cmd: process.execPath,
        args: [...process.execArgv, ...process.argv.slice(1)],
      };
    }
    const parts = this.options.cmd!.split(/\s+/).filter(Boolean);
    const [program, ...args] = parts;
    return { cmd: program, args };
  }

  async acquire(key: string, signal?: AbortSignal): Promise<BoxConnection> {
    if (this.closed) throw new Error('execRunner is closed.');
    await this.ensureHost();

    const existing = this.instances.get(key);
    if (existing && this.host.hasRuntime(existing.runtimeId)) {
      existing.lastUsed = Date.now();
      return this.connectionFor(existing.runtimeId, key);
    }

    const runtimeId = `box-${randomUUID()}`;
    const base = this.baseSpawn();
    const prepared = this.provider.prepare(
      {
        cmd: base.cmd,
        args: base.args,
        env: {
          GENKIT_REFLECTION_V2_SERVER: this.host.url,
          GENKIT_RUNTIME_ID: runtimeId,
          ...(this.boxId ? { [BOX_SELF_ID_ENV]: this.boxId } : {}),
        },
        cwd: this.options.cwd,
      },
      { url: this.host.url, port: this.host.port ?? 0 }
    );

    logger.debug(
      `Box spawning runtime ${runtimeId} via ${this.provider.name}: ` +
        `${prepared.cmd} ${prepared.args.join(' ')}`
    );
    // The child inherits this process's env (API keys, PATH, GENKIT_ENV), so
    // it runs in the same mode as its caller. Reflection settings are the
    // exception and must be overridden after the spread: under `genkit start`
    // we inherit the CLI's secret and v2 URL, which belong to the CLI, not to
    // our host. The v2 URL alone starts the box's reflection client, so no
    // GENKIT_ENV=dev is needed.
    const child = spawn(prepared.cmd, prepared.args, {
      env: {
        ...process.env,
        ...prepared.env,
        GENKIT_REFLECTION_V2_SERVER: prepared.reflectUrl,
        [REFLECTION_SECRET_ENV]: this.host.secret ?? '',
      },
      cwd: prepared.cwd,
      stdio: ['ignore', 'inherit', 'inherit'],
    });
    installCleanupHandlers();
    if (child.pid !== undefined) {
      const pid = child.pid;
      liveChildren.add(pid);
      child.once('exit', () => liveChildren.delete(pid));
    }
    const handle = toHandle(child);
    this.instances.set(key, { runtimeId, handle, lastUsed: Date.now() });

    await this.host.waitForRuntime(runtimeId).catch((e) => {
      handle.kill().catch(() => {});
      this.instances.delete(key);
      throw e;
    });
    if (signal?.aborted) {
      await this.release(key);
      throw new Error('Aborted before box became ready.');
    }
    return this.connectionFor(runtimeId, key);
  }

  private connectionFor(runtimeId: string, key: string): BoxConnection {
    return {
      runAction: <O = unknown>(req: RunActionRequest, opts?: RunOptions) => {
        const inst = this.instances.get(key);
        if (inst) inst.lastUsed = Date.now();
        return this.host.runAction<O>(runtimeId, req, opts) as Promise<
          RunActionResult<O>
        >;
      },
      listActions: () => this.host.listActions(runtimeId),
    };
  }

  async release(key: string): Promise<void> {
    const inst = this.instances.get(key);
    if (!inst) return;
    this.instances.delete(key);
    await inst.handle.kill().catch(() => {});
  }

  async close(): Promise<void> {
    this.closed = true;
    await Promise.all(
      Array.from(this.instances.values()).map((i) =>
        i.handle.kill().catch(() => {})
      )
    );
    this.instances.clear();
    await this.host.stop();
  }
}

/** Creates an {@link ExecRunner}. */
export function execRunner(options: ExecRunnerOptions): ExecRunner {
  return new ExecRunner(options);
}
