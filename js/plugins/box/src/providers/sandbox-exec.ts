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

import type {
  PreparedSpawn,
  ReflectHost,
  SandboxProvider,
  SpawnSpec,
} from './types.js';

/** Options for {@link sandboxExec}. */
export interface SandboxExecOptions {
  /**
   * A complete seatbelt profile. When omitted, a coarse default is used that
   * denies filesystem writes (except /tmp) but allows network (needed for the
   * box to dial the reflection host).
   */
  profile?: string;
  /** Deny all filesystem writes except the temp dirs. Default true. */
  denyWrite?: boolean;
}

/**
 * macOS `sandbox-exec` (seatbelt) provider. Wraps the command in a seatbelt
 * profile.
 *
 * The default profile restricts filesystem WRITES only (allowed under /tmp).
 * Reads are unrestricted (a boxed tool can still read credentials, source, etc.)
 * and network egress is fully open (the profile allows all network, not just the
 * loopback the reflection dial-back needs). It is a coarse guardrail against
 * accidental writes, NOT containment for hostile code. See the plugin README
 * ("Isolation levels").
 *
 * `sandbox-exec` is also deprecated by Apple (still shipping on current macOS,
 * prints a warning). Only available on darwin.
 */
export class SandboxExecProvider implements SandboxProvider {
  readonly name = 'sandbox-exec';
  readonly platforms: NodeJS.Platform[] = ['darwin'];

  constructor(private readonly options: SandboxExecOptions = {}) {}

  private defaultProfile(): string {
    const lines = ['(version 1)', '(allow default)'];
    if (this.options.denyWrite !== false) {
      lines.push(
        '(deny file-write*)',
        '(allow file-write* (subpath "/private/tmp"))',
        '(allow file-write* (subpath "/tmp"))'
      );
    }
    // The reflection dial-back needs loopback network access.
    lines.push('(allow network*)');
    return lines.join('\n');
  }

  prepare(spec: SpawnSpec, reflect: ReflectHost): PreparedSpawn {
    if (process.platform !== 'darwin') {
      throw new Error(
        "The 'sandbox-exec' provider is only available on macOS. Use " +
          'a container runner, or run without isolation.'
      );
    }
    const profile = this.options.profile ?? this.defaultProfile();
    return {
      ...spec,
      cmd: 'sandbox-exec',
      args: ['-p', profile, spec.cmd, ...spec.args],
      reflectUrl: reflect.url,
    };
  }
}

/** Creates a macOS seatbelt isolation provider. */
export function sandboxExec(options?: SandboxExecOptions): SandboxExecProvider {
  return new SandboxExecProvider(options);
}
