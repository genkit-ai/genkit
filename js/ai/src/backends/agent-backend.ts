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

import type { EditResult, LsResult, ReadResult, WriteResult } from './types.js';

export interface AgentBackend {
  readonly id: string;

  ls(
    dirPath: string,
    options?: {
      recursive?: boolean;
    }
  ): Promise<LsResult>;

  read(filePath: string, offset?: number, limit?: number): Promise<ReadResult>;

  write(filePath: string, content: string): Promise<WriteResult>;

  edit(
    filePath: string,
    oldString: string,
    newString: string,
    replaceAll?: boolean
  ): Promise<EditResult>;

  destroy?(): Promise<void>;
}
