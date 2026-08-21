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
 * This file is the optional host-code path. It is not required to run the
 * agents: `agent-dirs serve ./agents` (`pnpm server:zero`) serves the same
 * folder with zero host code and is the intended default. Write an entry
 * point like this one only to make choices the zero-code runner doesn't
 * expose - a different model provider, mounting on your own express app,
 * a stream manager for durable reconnects.
 *
 * - Dev UI: `GCLOUD_PROJECT=<project> pnpm genkit:dev` - chat with agents
 *   directly.
 * - HTTP: `GCLOUD_PROJECT=<project> pnpm server` - each agent at
 *   `/api/<name>` (plus `/getSnapshot`, `/abort`), consumable with
 *   `remoteAgent` from `genkit/beta/client`.
 *
 * Uses Vertex AI via Application Default Credentials.
 */

import { agentDirs, serveAgents } from '@genkit-ai/agent-dirs';
import { vertexAI } from '@genkit-ai/google-genai';
import { genkit } from 'genkit/beta';

const ai = genkit({
  plugins: [vertexAI(), agentDirs({ dir: './agents' })],
});

await serveAgents(ai);
