/**
 * Copyright 2024 Google LLC
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
 * Public surface of the OpenAI Responses API integration.
 *
 * Imported by the existing `openai/index.ts` entry to register a
 * dedicated `'openai-responses'` namespace alongside the Chat Completions
 * `'openai/...'` namespace. The two namespaces never share schemas or
 * runners — see {@link OpenAIResponsesConfigSchema} and
 * {@link openAIResponsesModelRef}.
 */

export {
  chatMessagesToResponsesInput,
  toResponsesRequestBody,
} from './request';
export { fromResponsesResponse } from './response';
export {
  defineCompatOpenAIResponsesModel,
  openAIResponsesModelRunner,
} from './runner';
export {
  BuiltInToolSchema,
  CitationSchema,
  OpenAIResponsesConfigSchema,
  SUPPORTED_RESPONSES_MODELS,
  openAIResponsesModelRef,
  type BuiltInToolSpec,
  type Citation,
  type OpenAIResponsesConfig,
} from './types';
