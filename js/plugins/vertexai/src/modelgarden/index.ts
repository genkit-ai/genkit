/**
 * Copyright 2025 Google LLC
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
  claude35Sonnet,
  claude35SonnetV2,
  claude3Haiku,
  claude3Opus,
  claude3Sonnet,
  claudeOpus4,
  claudeOpus41,
  claudeSonnet4,
  codestral,
  llama3,
  llama31,
  llama32,
  mistralLarge,
  mistralNemo,
  vertexAIModelGarden,
} from './legacy/index.js';
export {
  vertexModelGarden,
  type PluginOptions,
  type VertexModelGardenPlugin,
} from './v2/index.js';
