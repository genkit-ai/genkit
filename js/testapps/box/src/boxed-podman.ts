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

// The boxed side of the container demo: this file runs INSIDE the podman
// container, started by `index-podman.ts`. It defines the real `runShell`
// implementation and nothing else. No model, no API key.
//
// The runner sets GENKIT_REFLECTION_HOST/PORT and a per-container secret, so
// the V1 reflection server binds 0.0.0.0:3100 (locked by the secret) and the
// runner publishes it to the host loopback. This process just has to stay
// alive and serve.

import { z } from 'genkit';
import { genkit } from 'genkit/beta';
import { exec } from 'node:child_process';
import { promisify } from 'node:util';

const run = promisify(exec);

const ai = genkit({});

ai.defineTool(
  {
    name: 'runShell',
    description: 'Run a shell command and return its stdout.',
    inputSchema: z.object({ cmd: z.string() }),
    outputSchema: z.object({
      stdout: z.string().optional(),
      error: z.string().optional(),
    }),
  },
  async ({ cmd }) => {
    try {
      const { stdout } = await run(cmd, { timeout: 10_000 });
      return { stdout };
    } catch (e) {
      return { error: `${e}` };
    }
  }
);
