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

/**
 * Serves every directory agent under `agents/` for interactive chat.
 *
 * - Dev UI: `GCLOUD_PROJECT=<project> pnpm genkit:dev` - agents appear as
 *   actions and can be chatted with directly.
 * - HTTP: `GCLOUD_PROJECT=<project> pnpm server` - each agent is exposed at
 *   `/api/<name>` (plus `/getSnapshot` and `/abort`), consumable with
 *   `remoteAgent` from `genkit/beta/client`.
 *
 * Uses Vertex AI via Application Default Credentials (`gcloud auth
 * application-default login`).
 */

import { agentDirs, listAgents } from '@genkit-ai/agent-dirs';
import { expressHandler } from '@genkit-ai/express';
import { vertexAI } from '@genkit-ai/google-genai';
import express from 'express';
import { genkit } from 'genkit/beta';

const ai = genkit({
  plugins: [vertexAI(), agentDirs({ dir: './agents' })],
});

const app = express();
app.use(express.json());

const agents = await listAgents(ai);
for (const [name, agent] of Object.entries(agents)) {
  app.post(`/api/${name}`, expressHandler(agent));
  app.post(
    `/api/${name}/getSnapshot`,
    expressHandler(agent.getSnapshotDataAction)
  );
  app.post(`/api/${name}/abort`, expressHandler(agent.abortAgentAction));
}

const port = Number(process.env.PORT ?? 8080);
app.listen(port, () => {
  console.log(
    `agent-dirs demo server on :${port} - agents: ` +
      Object.keys(agents)
        .map((n) => `/api/${n}`)
        .join(', ')
  );
});
