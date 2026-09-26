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

/** A handle to a spawned box subprocess. */
export interface SpawnHandle {
  readonly pid?: number;
  /** Terminates the subprocess. Resolves once it has exited. */
  kill(): Promise<void>;
  /** Registers a callback invoked when the subprocess exits. */
  onExit(cb: (code: number | null) => void): void;
}

/** The base spawn a provider transforms. */
export interface SpawnSpec {
  cmd: string;
  args: string[];
  env: Record<string, string>;
  cwd?: string;
}

/** Where the reflection host is listening (what the box should dial). */
export interface ReflectHost {
  url: string;
  port: number;
}

/** The isolated spawn a provider produces. `reflectUrl` is what the box dials. */
export interface PreparedSpawn extends SpawnSpec {
  reflectUrl: string;
}

/**
 * Pluggable isolation strategy passed to `execRunner` as `isolate:`. Transforms
 * the base spawn into the isolated spawn. A provider MUST punch the loopback
 * hole so the box can reach the reflection host, and return the URL the box
 * should actually dial.
 */
export interface SandboxProvider {
  readonly name: string;
  /** Platforms this provider supports; used by `localSandbox()`. */
  readonly platforms?: NodeJS.Platform[];
  prepare(spec: SpawnSpec, reflect: ReflectHost): PreparedSpawn;
}
