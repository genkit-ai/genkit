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
import { testTranslatorAgent, translatorAgent } from './prompt-agent.js';
import { simpleAgent, testSimpleAgent } from './simple-agent.js';
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

import { bankingAgent, testBankingAgent } from './interrupt-agent.js';
import { backgroundAgent, testBackgroundAgent } from './background-agent.js';

// Log loaded agents/flows (existing behavior)
console.log('Loaded agent:', simpleAgent.__action.name);
console.log('Loaded flow:', testSimpleAgent.__action.name);
console.log('Loaded prompt agent:', translatorAgent.__action.name);
console.log('Loaded prompt flow:', testTranslatorAgent.__action.name);
console.log('Loaded tool agent:', weatherAgent.__action.name);
console.log('Loaded tool flow:', testWeatherAgent.__action.name);
console.log('Loaded tool stream flow:', testWeatherAgentStream.__action.name);
console.log('Loaded branching agent:', nameAgent.__action.name);
console.log('Loaded branching flow:', demonstrateBranching.__action.name);
console.log('Loaded client state agent:', clientStateAgent.__action.name);
console.log('Loaded client state flow:', testClientStateAgent.__action.name);
console.log('Loaded workspace agent:', workspaceAgent.__action.name);
console.log('Loaded workspace flow:', testWorkspaceAgent.__action.name);
console.log('Loaded file store agent:', fileStoreAgent.__action.name);
console.log('Loaded file store flow:', testFileStoreAgent.__action.name);
console.log('Loaded pruning agent:', pruningAgent.__action.name);
console.log(
  'Loaded pruning flow:',
  testFileStoreChainPruningAgent.__action.name
);
console.log('Loaded interrupt flow:', testBankingAgent.__action.name);
console.log('Loaded interrupt agent:', bankingAgent.__action.name);
console.log('Loaded background agent:', backgroundAgent.__action.name);
console.log('Loaded background flow:', testBackgroundAgent.__action.name);

export * from './background-agent.js';
export * from './interrupt-agent.js';

// ---------------------------------------------------------------------------
// Express server — exposes session flows for the web UI
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

// Expose session flows
app.post('/api/simpleAgent', expressHandler(simpleAgent as any));
app.post('/api/translatorAgent', expressHandler(translatorAgent as any));
app.post('/api/weatherAgent', expressHandler(weatherAgent as any));
app.post('/api/weatherAgent/state', expressHandler(weatherAgent.getSnapshotDataAction));
app.post('/api/clientStateAgent', expressHandler(clientStateAgent as any));
app.post('/api/bankingAgent', expressHandler(bankingAgent as any));
app.post('/api/workspaceAgent', expressHandler(workspaceAgent as any));
app.post('/api/backgroundAgent', expressHandler(backgroundAgent as any));
app.post('/api/backgroundAgent/state', expressHandler(backgroundAgent.getSnapshotDataAction));
app.post('/api/backgroundAgent/abort', expressHandler(backgroundAgent.abortSessionFlowAction));
app.post('/api/branchingAgent', expressHandler(nameAgent as any));
app.post('/api/branchingAgent/state', expressHandler(nameAgent.getSnapshotDataAction));

// Also expose the test flows for programmatic testing
app.post('/api/testSimpleAgent', expressHandler(testSimpleAgent));
app.post('/api/testTranslatorAgent', expressHandler(testTranslatorAgent));
app.post('/api/testWeatherAgent', expressHandler(testWeatherAgent));
app.post('/api/testClientStateAgent', expressHandler(testClientStateAgent));
app.post('/api/testBankingAgent', expressHandler(testBankingAgent));
app.post('/api/testWorkspaceAgent', expressHandler(testWorkspaceAgent));
app.post('/api/testBackgroundAgent', expressHandler(testBackgroundAgent));

const PORT = process.env.PORT ? parseInt(process.env.PORT) : 8080;
app.listen(PORT, () => {
  console.log(`\n🚀 Express server running on http://localhost:${PORT}`);
  console.log(`   Web UI: run "cd web && npm run dev" then open http://localhost:5173\n`);
});
