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

import type { GenkitError } from '../types/error';

export type Runtime = 'nodejs' | 'go' | undefined;

export class GenkitToolsError extends Error {
  public data?: GenkitError;

  constructor(msg: string, options?: ErrorOptions) {
    super(msg, options);
  }

  formatError(): string {
    let msg = this.message;
    if (this.data) {
      if (this.data?.message) {
        msg += `\Genkit Error: ${this.data?.message}`;
      }
      if (this.data?.stack) {
        msg += `\nStack Trace:\n${this.data?.stack}`;
      }
      if (this.data?.details) {
        msg += `\nDetails:\n${JSON.stringify(this.data?.details)}`;
      }
    } else if (this.stack) {
      msg += `\nStack Trace:\n${this.stack}`;
    }
    return msg;
  }
}

// Streaming callback function.
export type StreamingCallback<T> = (chunk: T) => void;

export interface RuntimeInfo {
  /** Runtime ID (either user-set or `pid`). */
  id: string;
  /** Process ID of the runtime. */
  pid: number;
  /** URL of the reflection server. */
  reflectionServerUrl: string;
  /** Timestamp when the runtime was started. */
  timestamp: string;
  /** Project name, typically basename of the root folder */
  projectName?: string;
  /** Display name for the Genkit instance, that will be shown in developer tooling. */
  name?: string;
  /** Genkit runtime library version. Ex: nodejs/0.9.5 or go/0.2.0 */
  genkitVersion?: string;
  /** Reflection API specification version. Ex: 1 */
  reflectionApiSpecVersion?: number;
}

export enum RuntimeEvent {
  ADD = 'add',
  REMOVE = 'remove',
}
