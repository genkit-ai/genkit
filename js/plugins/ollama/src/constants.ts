/**
 * Copyright 2025 Google LLC
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

import { ModelInfo } from 'genkit/model';

export const ANY_JSON_SCHEMA: Record<string, any> = {
  $schema: 'http://json-schema.org/draft-07/schema#',
};

export const GENERIC_MODEL_INFO = {
  supports: {
    multiturn: true,
    media: true,
    tools: true,
    toolChoice: true,
    systemRole: true,
    constrained: 'all',
  },
} as ModelInfo;

export const DEFAULT_OLLAMA_SERVER_ADDRESS = 'http://localhost:11434';
