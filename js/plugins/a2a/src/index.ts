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

export {
  GenkitA2ARequestHandler,
  type GenkitA2ARequestHandlerOptions,
  type GenkitAgent,
} from './request-handler.js';

export {
  A2A_PROTOCOL_VERSION,
  deriveAgentCard,
  getAgentDescription,
  getAgentName,
  type AgentLike,
  type DeriveAgentCardOptions,
} from './agent-card.js';

export {
  A2A_METADATA,
  GenkitPartType,
  a2aMessageToGenkit,
  a2aMessageToResumeInput,
  a2aPartToGenkit,
  a2aPartsToGenkit,
  genkitMessageToA2AParts,
  genkitPartToA2A,
  genkitPartsToA2A,
  genkitRoleToA2A,
} from './mapping.js';
