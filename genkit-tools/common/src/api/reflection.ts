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

import {
  OpenAPIRegistry,
  OpenApiGeneratorV3,
} from '@asteasolutions/zod-to-openapi';
import * as fs from 'fs';
import * as path from 'path';
import * as yaml from 'yaml';
import * as z from 'zod';
import * as action from '../types/action';
import * as apis from '../types/apis';
import { SpanDataSchema, TraceDataSchema } from '../types/trace';

const registry = new OpenAPIRegistry();
registry.register('CustomAny', action.CustomAnySchema.openapi('CustomAny'));
registry.register(
  'JSONSchema7',
  action.JSONSchema7Schema.openapi('JSONSchema7')
);
registry.register('Action', action.ActionSchema.openapi('Action'));
registry.register('TraceData', TraceDataSchema.openapi('TraceData'));
registry.register('SpanData', SpanDataSchema.openapi('SpanData'));
registry.registerPath({
  method: 'get',
  path: '/api/actions',
  summary: 'Retrieves all runnable actions.',
  responses: {
    '200': {
      description: 'Success',
      content: {
        'application/json': {
          schema: z.record(z.string(), action.ActionSchema),
        },
      },
    },
  },
});
registry.registerPath({
  method: 'post',
  path: '/api/runAction',
  summary: 'Runs an action and returns the result.',
  request: {
    body: {
      content: {
        'application/json': {
          schema: apis.RunActionRequestSchema,
        },
      },
    },
  },
  responses: {
    '200': {
      description: 'Success',
      content: {
        'application/json': {
          schema: action.RunActionResponseSchema,
        },
      },
    },
  },
});

const generator = new OpenApiGeneratorV3(registry.definitions);
const document = generator.generateDocument({
  openapi: '3.0.0',
  info: {
    version: '0.0.1',
    title: 'Genkit Reflection API',
    description:
      'A control API that allows clients to inspect app code to view actions, run them, and view the results.',
  },
});

if (!process.argv[2]) {
  throw Error(
    'Please provide an absolute path to output the generated API spec.'
  );
}

// TODO: Make this file declarative and do the file generation and writing somewhere else.

fs.writeFileSync(
  path.join(process.argv[2], 'reflectionApi.yaml'),
  yaml.stringify(document),
  {
    encoding: 'utf-8',
  }
);
