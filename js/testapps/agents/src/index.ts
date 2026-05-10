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
import { codingAgent, testCodingAgent } from './coding-agent.js';
import { taskAgent, testTaskAgent } from './custom-state-agent.js';
import { bankingAgent, testBankingAgent } from './interrupt-agent.js';
import { testPromptFileAgent, tripPlannerAgent } from './prompt-file-agent.js';
import {
  orchestratorAgent,
  testSubAgentDemo,
  testSubAgentSimple,
} from './subagent-demo.js';

// Log loaded agents/flows (existing behavior)
console.log('Loaded custom agent:', customAgent.__action.name);
console.log('Loaded custom flow:', testCustomAgent.__action.name);
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
console.log('Loaded task agent:', taskAgent.__action.name);
console.log('Loaded task flow:', testTaskAgent.__action.name);
console.log('Loaded orchestrator agent:', orchestratorAgent.__action.name);
console.log('Loaded sub-agent demo flow:', testSubAgentDemo.__action.name);
console.log('Loaded sub-agent simple flow:', testSubAgentSimple.__action.name);
console.log('Loaded prompt-file agent:', tripPlannerAgent.__action.name);
console.log('Loaded prompt-file flow:', testPromptFileAgent.__action.name);
console.log('Loaded coding agent:', codingAgent.__action.name);
console.log('Loaded coding flow:', testCodingAgent.__action.name);

export * from './background-agent.js';
export * from './interrupt-agent.js';
export * from './subagent-demo.js';

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
app.post('/api/testCodingAgent', expressHandler(testCodingAgent));

// Workspace file browser API — serves the coding agent's workspace contents
app.get('/api/workspace/files', async (_req, res) => {
  try {
    const workspaceDir = require('path').resolve(__dirname, '..', 'workspace');
    const files = await listWorkspaceFiles(workspaceDir, workspaceDir);
    res.json({ files });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/api/workspace/file', async (req, res) => {
  try {
    const workspaceDir = require('path').resolve(__dirname, '..', 'workspace');
    const filePath = req.query.path as string;
    if (!filePath) {
      res.status(400).json({ error: 'Missing path query parameter' });
      return;
    }
    const fullPath = require('path').resolve(workspaceDir, filePath);
    if (!fullPath.startsWith(workspaceDir)) {
      res.status(403).json({ error: 'Path outside workspace' });
      return;
    }
    const content = require('fs').readFileSync(fullPath, 'utf8');
    res.json({ path: filePath, content });
  } catch (err: any) {
    res.status(500).json({ error: err.message });
  }
});

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

// ---------------------------------------------------------------------------
// Helper: recursively list workspace files
// ---------------------------------------------------------------------------

interface WorkspaceFile {
  name: string;
  path: string;
  type: 'file' | 'directory';
  children?: WorkspaceFile[];
}

async function listWorkspaceFiles(
  dir: string,
  rootDir: string
): Promise<WorkspaceFile[]> {
  const fs = require('fs');
  const pathMod = require('path');
  const entries = await fs.promises.readdir(dir, { withFileTypes: true });
  const result: WorkspaceFile[] = [];

  for (const entry of entries) {
    if (entry.name.startsWith('.')) continue;
    const fullPath = pathMod.join(dir, entry.name);
    const relativePath = pathMod.relative(rootDir, fullPath);

    if (entry.isDirectory()) {
      const children = await listWorkspaceFiles(fullPath, rootDir);
      result.push({
        name: entry.name,
        path: relativePath,
        type: 'directory',
        children,
      });
    } else {
      result.push({
        name: entry.name,
        path: relativePath,
        type: 'file',
      });
    }
  }

  return result.sort((a, b) => {
    // Directories first, then alphabetical
    if (a.type !== b.type) return a.type === 'directory' ? -1 : 1;
    return a.name.localeCompare(b.name);
  });
}
