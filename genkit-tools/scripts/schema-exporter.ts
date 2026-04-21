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

import * as fs from 'fs';
import type { JSONSchema7 } from 'json-schema';
import * as path from 'path';
import * as z from 'zod';
import { zodToJsonSchema } from 'zod-to-json-schema';

/** List of files that contain types to be exported. */
const EXPORTED_TYPE_MODULES = [
  '../common/src/types/document.ts',
  '../common/src/types/embedder.ts',
  '../common/src/types/evaluator.ts',
  '../common/src/types/error.ts',
  '../common/src/types/middleware.ts',
  '../common/src/types/model.ts',
  '../common/src/types/parts.ts',
  '../common/src/types/reranker.ts',
  '../common/src/types/reflection.ts',
  '../common/src/types/retriever.ts',
  '../common/src/types/trace.ts',
];

/** Types that may appear that do not need to be included. */
const IGNORED_TYPES = ['NestedSpanDataSchema'];

/**
 * Generates a JSON schema for Zod schemas exported from the modules.
 * @param modulePaths paths to Node modules to parse for Zod schemas
 * @returns JSON schema for extracted Zod schemas
 */
function generateJsonSchema(modulePaths: string[]): JSONSchema7 {
  const schemas: Record<string, z.ZodType> = {};
  for (const modulePath of modulePaths) {
    const module = require(modulePath);
    Object.keys(module).forEach((key) => {
      if (module[key] instanceof z.ZodType && !IGNORED_TYPES.includes(key)) {
        const strippedKey = key.replace(/Schema$/, '');
        schemas[strippedKey] = module[key];
      }
    });
  }
  // Putting all of the schemas in an object and later extracting out the definitions is the most succinct way.
  const { $defs } = zodToJsonSchema(z.object(schemas), {
    definitions: schemas,
    definitionPath: '$defs',
    target: 'jsonSchema7',
  }) as JSONSchema7;
  return { $schema: 'http://json-schema.org/draft-07/schema#', $defs };
}

if (!process.argv[2]) {
  throw Error(
    'Please provide an absolute path to output the generated schema.'
  );
}

fs.writeFileSync(
  path.join(process.argv[2], 'genkit-schema.json'),
  JSON.stringify(generateJsonSchema(EXPORTED_TYPE_MODULES), null, 2)
);
