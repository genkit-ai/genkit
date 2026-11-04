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

// A boxed agent whose sessions live in a file store shared by every box
// process (BOX_TEST_STORE_DIR), so any box can resume any session. The echo
// model replies with the turn count and this process's pid.

import { FileSessionStore, genkit } from 'genkit/beta';

const ai = genkit({});

const echoModel = ai.defineModel({ name: 'echo' }, async (req) => {
  const userTurns = req.messages.filter((m) => m.role === 'user').length;
  return {
    finishReason: 'stop',
    message: {
      role: 'model',
      content: [{ text: `turns=${userTurns} pid=${process.pid}` }],
    },
  };
});

ai.defineAgent({
  name: 'storedAgent',
  model: echoModel,
  store: new FileSessionStore(process.env.BOX_TEST_STORE_DIR!),
});

setInterval(() => {}, 1 << 30);
