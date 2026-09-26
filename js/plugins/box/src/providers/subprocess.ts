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

/**
 * The default provider: runs the command as-is with no OS-level isolation. Used
 * for trusted / cross-language / self boxes. The box can reach the reflection
 * host because no network restriction is applied.
 */
export class SubprocessProvider implements SandboxProvider {
  readonly name = 'subprocess';

  prepare(spec: SpawnSpec, reflect: ReflectHost): PreparedSpawn {
    return { ...spec, reflectUrl: reflect.url };
  }
}
