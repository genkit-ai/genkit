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

export interface BackendFile {
  path: string;
  isDirectory: boolean;
}

export interface LsResult {
  error?: string;
  files?: BackendFile[];
}

export interface ReadResult {
  error?: string;
  content?: string | Uint8Array;
  mimeType?: string;
}

export interface WriteResult {
  error?: string;
  path?: string;
}

export interface EditResult {
  error?: string;
  path?: string;
  occurrences?: number;
}
