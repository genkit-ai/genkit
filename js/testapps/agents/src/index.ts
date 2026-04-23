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
export * from './abort-agent.js';
