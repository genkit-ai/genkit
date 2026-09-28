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

// Exercises two-level self-mode nesting in a single entry file.
//
//   outer box 'A' (self)  -> runtime C1
//     inner box 'B' (self) -> runtime C2, runs the real `deep` tool
//
// The manager process (test) calls A's `viaA` tool. That runs in C1, which is
// A's runtime (isSelfRuntime for A) but the MANAGER for B, so it calls B's
// `deep` proxy, spawning C2 where the real tool runs.

import { genkit, z } from 'genkit';
import { BOX_SELF_ID_ENV, box, execRunner } from '../../src/index.js';

const ai = genkit({});

// Inner box: the real `deep` tool lives here; executes in C2.
const innerBox = box(ai, { name: 'B', runner: execRunner({ self: true }) });

const realDeep = ai.defineTool(
  {
    name: 'deep',
    description: 'Echoes at the deepest level.',
    inputSchema: z.object({ msg: z.string() }),
    outputSchema: z.object({ echoed: z.string(), pid: z.number() }),
  },
  async ({ msg }) => ({ echoed: `deep:${msg}`, pid: process.pid })
);

const boxedDeep = innerBox.fromTool<
  { msg: string },
  { echoed: string; pid: number }
>(realDeep);

// `viaA` runs in C1 and calls the inner boxed tool (spawning C2).
const realViaA = ai.defineTool(
  {
    name: 'viaA',
    description: 'Calls the inner boxed tool.',
    inputSchema: z.object({ msg: z.string() }),
    outputSchema: z.object({
      echoed: z.string(),
      innerPid: z.number(),
      viaPid: z.number(),
    }),
  },
  async ({ msg }) => {
    const res = await boxedDeep({ msg });
    return { echoed: res.echoed, innerPid: res.pid, viaPid: process.pid };
  }
);

// Outer box: exposes `viaA` as a proxy; the top manager M calls this.
const outerBox = box(ai, { name: 'A', runner: execRunner({ self: true }) });
const boxedViaA = outerBox.fromTool<
  { msg: string },
  { echoed: string; innerPid: number; viaPid: number }
>(realViaA);

async function main() {
  // Runtimes (SELF_ID set) just stay alive serving actions over reflection.
  if (process.env[BOX_SELF_ID_ENV]) {
    setInterval(() => {}, 1 << 30);
    return;
  }
  // Top manager M drives the call. Distinct pids per level prove real nesting.
  const res = await boxedViaA({ msg: 'hi' });
  process.stdout.write('NESTED_RESULT:' + JSON.stringify(res) + '\n');
  await outerBox.close();
  await innerBox.close();
  process.exit(0);
}

void main();
