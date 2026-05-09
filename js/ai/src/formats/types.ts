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

import type { JSONSchema } from '@genkit-ai/core';
import type { GenerateResponseChunk } from '../generate/chunk.js';
import type { Message } from '../message.js';
import type { ModelRequest } from '../model.js';

export type OutputContentTypes = 'application/json' | 'text/plain';

export interface Formatter<O = unknown, CO = unknown> {
  name: string;
  config: ModelRequest['output'] & {
    defaultInstructions?: false;
  };
  handler: (schema?: JSONSchema) => {
    parseMessage(message: Message): O;
    parseChunk?: (chunk: GenerateResponseChunk) => CO;
    instructions?: string;
  };
}
