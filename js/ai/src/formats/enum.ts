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

import { GenkitError } from '@genkit-ai/core';
import type { Formatter } from './types.js';

export const enumFormatter: Formatter<string, string> = {
  name: 'enum',
  config: {
    contentType: 'text/enum',
    constrained: true,
  },
  handler: (schema) => {
    if (schema && schema.type !== 'string' && schema.type !== 'enum') {
      throw new GenkitError({
        status: 'INVALID_ARGUMENT',
        message: `Must supply a 'string' or 'enum' schema type when using the enum parser format.`,
      });
    }

    let instructions: string | undefined;
    if (schema?.enum) {
      instructions = `Output should be ONLY one of the following enum values. Do not output any additional information or add quotes.\n\n${schema.enum.map((v) => v.toString()).join('\n')}`;
    }

    return {
      parseMessage: (message) => {
        return message.text.replace(/['"]/g, '').trim();
      },
      instructions,
    };
  },
};
