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

// Child process for genkit_test.ts: constructs genkit() with a reflection
// port the parent is already holding, and reports how the failure surfaced.

import { genkit } from '../../src/index.js';

process.on('unhandledRejection', (reason) => {
  const message = reason instanceof Error ? reason.message : String(reason);
  process.stdout.write(`unhandledRejection: ${message}\n`);
  process.exitCode = 7;
});

genkit({ reflectionPort: Number(process.env.TAKEN_PORT) });
