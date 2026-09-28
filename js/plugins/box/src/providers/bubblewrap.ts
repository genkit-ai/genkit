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

import { dirname } from 'node:path';
import type {
  PreparedSpawn,
  ReflectHost,
  SandboxProvider,
  SpawnSpec,
} from './types.js';

const SYSTEM_DIRS = ['/usr', '/bin', '/lib', '/lib64', '/etc'];

/**
 * The system dirs plus the directory of the running node binary, which
 * version managers (nvm, fnm, volta) keep under `$HOME`, outside the system
 * binds. The working directory is deliberately not included: keeping your
 * repo out of the box is the point. Add it (or just what the box needs) via
 * `roBind`.
 */
function defaultRoBinds(): string[] {
  const binds = [...SYSTEM_DIRS];
  const nodeDir = dirname(process.execPath);
  if (!binds.some((dir) => nodeDir === dir || nodeDir.startsWith(`${dir}/`))) {
    // The install root (`.../bin/node` -> `...`), so node finds its own lib.
    binds.push(dirname(nodeDir));
  }
  return binds;
}

/** Options for {@link bubblewrap}. */
export interface BubblewrapOptions {
  /**
   * Read-only bind mounts. Defaults to the system dirs plus the node install,
   * without the working directory. Setting it replaces the default.
   */
  roBind?: string[];
  /** Extra read-only binds added to the default (or to `roBind`). */
  extraRoBind?: string[];
  /** tmpfs mounts (writable, ephemeral). Defaults to `/tmp`. */
  tmpfs?: string[];
  /**
   * Isolate the network namespace. Default false, because the box must reach
   * the reflection host on loopback. Setting this true breaks the dial-back
   * unless you arrange another path, so leave it false for local boxes.
   */
  unshareNet?: boolean;
  /** Extra raw `bwrap` args appended before the command. */
  extraArgs?: string[];
}

/**
 * Linux `bubblewrap` (bwrap) provider. Wraps the command in a bwrap sandbox
 * with read-only system binds and a tmpfs, so the box's filesystem view is
 * confined (home/repo/credentials are not mounted).
 *
 * Caveats: network egress is fully open by default (the box shares the host
 * network so it can dial the reflection host on loopback; `unshareNet` would
 * isolate it but currently breaks the dial-back), and there are no CPU/memory or
 * seccomp limits. Stronger than the macOS provider for filesystem confinement,
 * but still not containment for hostile code. See the plugin README
 * ("Isolation levels"). Only available on linux.
 */
export class BubblewrapProvider implements SandboxProvider {
  readonly name = 'bubblewrap';
  readonly platforms: NodeJS.Platform[] = ['linux'];

  constructor(private readonly options: BubblewrapOptions = {}) {}

  prepare(spec: SpawnSpec, reflect: ReflectHost): PreparedSpawn {
    if (process.platform !== 'linux') {
      throw new Error(
        "The 'bubblewrap' provider is only available on Linux. Use " +
          'a container runner, or run without isolation.'
      );
    }
    return {
      ...spec,
      cmd: 'bwrap',
      args: [...this.bwrapArgs(), spec.cmd, ...spec.args],
      reflectUrl: reflect.url,
    };
  }

  /**
   * The `bwrap` flags, before the wrapped command. Pure (no platform check),
   * so it can be unit tested anywhere.
   */
  bwrapArgs(): string[] {
    const roBind = [
      ...(this.options.roBind ?? defaultRoBinds()),
      ...(this.options.extraRoBind ?? []),
    ];
    const tmpfs = this.options.tmpfs ?? ['/tmp'];

    const bwrapArgs: string[] = [
      '--die-with-parent',
      '--proc',
      '/proc',
      '--dev',
      '/dev',
    ];
    for (const dir of roBind) {
      // Skip missing paths so a minimal image doesn't fail the whole sandbox.
      bwrapArgs.push('--ro-bind-try', dir, dir);
    }
    for (const dir of tmpfs) {
      bwrapArgs.push('--tmpfs', dir);
    }
    if (this.options.unshareNet) {
      bwrapArgs.push('--unshare-net');
    }
    if (this.options.extraArgs) {
      bwrapArgs.push(...this.options.extraArgs);
    }
    return bwrapArgs;
  }
}

/** Creates a Linux bubblewrap isolation provider. */
export function bubblewrap(options?: BubblewrapOptions): BubblewrapProvider {
  return new BubblewrapProvider(options);
}
