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

import { demonstrateBranching, nameAgent } from './branching-agent.js';
import {
  clientStateAgent,
  testClientStateAgent,
} from './client-state-agent.js';
import { customAgent, testCustomAgent } from './custom-agent.js';
import {
  testWeatherAgent,
  testWeatherAgentStream,
  weatherAgent,
} from './tool-agent.js';
import { testWorkspaceAgent, workspaceAgent } from './workspace-builder.js';

import {
  fileStoreAgent,
  pruningAgent,
  testFileStoreAgent,
  testFileStoreChainPruningAgent,
} from './file-store.js';

import { backgroundAgent, testBackgroundAgent } from './background-agent.js';
import {
  codingAgent,
  listWorkspaceFiles,
  readWorkspaceFile,
  testCodingAgent,
} from './coding-agent.js';
import { taskAgent, testTaskAgent } from './custom-state-agent.js';
import { bankingAgent, testBankingAgent } from './interrupt-agent.js';
import { testPromptFileAgent, tripPlannerAgent } from './prompt-file-agent.js';
import {
  orchestratorAgent,
  testSubAgentDemo,
  testSubAgentSimple,
} from './subagent-demo.js';

// Force-reference all agents/flows so they register with Genkit.
// (Side-effect imports would also work, but explicit references
// make it clear which actions are available.)
void [
  customAgent,
  testCustomAgent,
  weatherAgent,
  testWeatherAgent,
  testWeatherAgentStream,
  nameAgent,
  demonstrateBranching,
  clientStateAgent,
  testClientStateAgent,
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
  testSubAgentDemo,
  testSubAgentSimple,
  tripPlannerAgent,
  testPromptFileAgent,
  codingAgent,
  testCodingAgent,
  listWorkspaceFiles,
  readWorkspaceFile,
];

export * from './background-agent.js';
export * from './interrupt-agent.js';
export * from './subagent-demo.js';

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
app.post('/api/customAgent', expressHandler(customAgent));
app.post('/api/weatherAgent', expressHandler(weatherAgent));
app.post(
  '/api/weatherAgent/state',
  expressHandler(weatherAgent.getSnapshotDataAction)
);
app.post('/api/clientStateAgent', expressHandler(clientStateAgent));
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
app.post('/api/branchingAgent', expressHandler(nameAgent));
app.post(
  '/api/branchingAgent/state',
  expressHandler(nameAgent.getSnapshotDataAction)
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
app.post('/api/testCustomAgent', expressHandler(testCustomAgent));
app.post('/api/testWeatherAgent', expressHandler(testWeatherAgent));
app.post('/api/testClientStateAgent', expressHandler(testClientStateAgent));
app.post('/api/testBankingAgent', expressHandler(testBankingAgent));
app.post('/api/testWorkspaceAgent', expressHandler(testWorkspaceAgent));
app.post('/api/testBackgroundAgent', expressHandler(testBackgroundAgent));
app.post('/api/testTaskAgent', expressHandler(testTaskAgent));
app.post('/api/testPromptFileAgent', expressHandler(testPromptFileAgent));

const PORT = process.env.PORT ? parseInt(process.env.PORT) : 8080;
app.listen(PORT, () => {
  console.log(`\n🚀 Express server running on http://localhost:${PORT}`);
  console.log(
    `   Web UI: run "cd web && npm run dev" then open http://localhost:5173\n`
  );
});
