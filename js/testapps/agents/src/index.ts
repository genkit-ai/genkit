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

import { expressHandler } from '@genkit-ai/express';
import express from 'express';
import { existsSync } from 'node:fs';
import { join } from 'node:path';

import { branchingAgent, demonstrateBranching } from './branching-agent.js';
import { researchAgent, testResearchAgent } from './research-agent.js';
import {
  testWeatherAgentStateless,
  weatherAgentStateless,
} from './weather-agent-stateless.js';
import {
  testWeatherAgent,
  testWeatherAgentStream,
  weatherAgent,
} from './weather-agent.js';
import { testWorkspaceAgent, workspaceAgent } from './workspace-agent.js';

import {
  fileStoreAgent,
  pruningAgent,
  testFileStoreAgent,
  testFileStoreChainPruningAgent,
} from './file-store-agent.js';

import { backgroundAgent, testBackgroundAgent } from './background-agent.js';
import { bankingAgent, testBankingAgent } from './banking-agent.js';
import {
  codingAgent,
  listWorkspaceFiles,
  readWorkspaceFile,
  testCodingAgent,
} from './coding-agent.js';
import {
  orchestratorAgent,
  testOrchestratorAgent,
  testOrchestratorAgentSimple,
} from './orchestrator-agent.js';
import { taskAgent, testTaskAgent } from './task-agent.js';
import {
  testTripPlannerAgent,
  tripPlannerAgent,
} from './trip-planner-agent.js';

// Force-reference all agents/flows so they register with Genkit.
// (Side-effect imports would also work, but explicit references
// make it clear which actions are available.)
void [
  researchAgent,
  testResearchAgent,
  weatherAgent,
  testWeatherAgent,
  testWeatherAgentStream,
  branchingAgent,
  demonstrateBranching,
  weatherAgentStateless,
  testWeatherAgentStateless,
  workspaceAgent,
  testWorkspaceAgent,
  fileStoreAgent,
  testFileStoreAgent,
  pruningAgent,
  testFileStoreChainPruningAgent,
  bankingAgent,
  testBankingAgent,
  backgroundAgent,
  testBackgroundAgent,
  taskAgent,
  testTaskAgent,
  orchestratorAgent,
  testOrchestratorAgent,
  testOrchestratorAgentSimple,
  tripPlannerAgent,
  testTripPlannerAgent,
  codingAgent,
  testCodingAgent,
  listWorkspaceFiles,
  readWorkspaceFile,
];

export * from './background-agent.js';
export * from './banking-agent.js';
export * from './orchestrator-agent.js';

// ---------------------------------------------------------------------------
// Express server — exposes agents for the web UI
// ---------------------------------------------------------------------------
const app = express();
app.use(express.json());

// CORS for Vite dev server
app.use((_req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*');
  res.header(
    'Access-Control-Allow-Headers',
    'Content-Type, Accept, X-Genkit-Stream-Id'
  );
  res.header('Access-Control-Expose-Headers', 'X-Genkit-Stream-Id');
  res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
  if (_req.method === 'OPTIONS') {
    res.sendStatus(204);
    return;
  }
  next();
});

// Expose agents
app.post('/api/researchAgent', expressHandler(researchAgent));
app.post('/api/weatherAgent', expressHandler(weatherAgent));
app.post(
  '/api/weatherAgent/state',
  expressHandler(weatherAgent.getSnapshotDataAction)
);
app.post('/api/weatherAgentStateless', expressHandler(weatherAgentStateless));
app.post('/api/bankingAgent', expressHandler(bankingAgent));
app.post('/api/workspaceAgent', expressHandler(workspaceAgent));
app.post('/api/backgroundAgent', expressHandler(backgroundAgent));
app.post(
  '/api/backgroundAgent/state',
  expressHandler(backgroundAgent.getSnapshotDataAction)
);
app.post(
  '/api/backgroundAgent/abort',
  expressHandler(backgroundAgent.abortAgentAction)
);
app.post('/api/branchingAgent', expressHandler(branchingAgent));
app.post(
  '/api/branchingAgent/state',
  expressHandler(branchingAgent.getSnapshotDataAction)
);
app.post('/api/taskAgent', expressHandler(taskAgent));
app.post('/api/orchestratorAgent', expressHandler(orchestratorAgent));
app.post('/api/tripPlannerAgent', expressHandler(tripPlannerAgent));
app.post(
  '/api/tripPlannerAgent/state',
  expressHandler(tripPlannerAgent.getSnapshotDataAction)
);
app.post('/api/codingAgent', expressHandler(codingAgent));
app.post(
  '/api/codingAgent/state',
  expressHandler(codingAgent.getSnapshotDataAction)
);
app.post('/api/testCodingAgent', expressHandler(testCodingAgent));

// Workspace browser — exposed as Genkit flows via expressHandler
app.post('/api/workspace/files', expressHandler(listWorkspaceFiles));
app.post('/api/workspace/file', expressHandler(readWorkspaceFile));

// Also expose the test flows for programmatic testing
app.post('/api/testResearchAgent', expressHandler(testResearchAgent));
app.post('/api/testWeatherAgent', expressHandler(testWeatherAgent));
app.post(
  '/api/testWeatherAgentStateless',
  expressHandler(testWeatherAgentStateless)
);
app.post('/api/testBankingAgent', expressHandler(testBankingAgent));
app.post('/api/testWorkspaceAgent', expressHandler(testWorkspaceAgent));
app.post('/api/testBackgroundAgent', expressHandler(testBackgroundAgent));
app.post('/api/testTaskAgent', expressHandler(testTaskAgent));
app.post('/api/testTripPlannerAgent', expressHandler(testTripPlannerAgent));

// ---------------------------------------------------------------------------
// Static web UI — serve the compiled Vite app (web/dist) so the whole demo
// can run from a single server. Run "pnpm run build:web" to generate it.
// (For development with hot-reload, use the Vite dev server instead.)
// ---------------------------------------------------------------------------
const webDist = join(__dirname, '..', 'web', 'dist');
if (existsSync(webDist)) {
  app.use(express.static(webDist));
  // SPA fallback — send index.html for any non-API GET request so client-side
  // routing (react-router) works on page reload / deep links.
  app.get(/^\/(?!api\/).*/, (_req, res) => {
    res.sendFile(join(webDist, 'index.html'));
  });
}

const PORT = process.env.PORT ? parseInt(process.env.PORT) : 8080;
app.listen(PORT, () => {
  console.log(`\n🚀 Express server running on http://localhost:${PORT}`);
  if (existsSync(webDist)) {
    console.log(`   Web UI: open http://localhost:${PORT}\n`);
  } else {
    console.log(
      `   Web UI not built. Run "pnpm run build:web" to serve it here,`
    );
    console.log(
      `   or "pnpm run web:dev" for the Vite dev server (http://localhost:5173).\n`
    );
  }
});
