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

import { Runtime } from '@genkit-ai/tools-common/manager';

/** Return type of a configuration result, denoting the set of files processed.
 * */
export interface AIToolConfigResult {
  files: Array<{
    path: string;
    updated: boolean;
  }>;
}

/** `init:ai-tools` config options. */
export interface InitConfigOptions {
  // yes (non-interactive) mode.
  yesMode?: boolean;
}

/** Interface for supported AI tools */
export interface AIToolModule {
  name: string;
  displayName: string;

  /**
   * Configure the AI tool with Genkit context
   * @param configOptions Any user-specified config options
   * @returns Result object with update status and list of files created/updated
   */
  configure(
    runtime: Runtime,
    configOptions?: InitConfigOptions
  ): Promise<AIToolConfigResult>;
}

/** Type to denote user's selection from interactive menu */
export interface AIToolChoice {
  value: string;
  name: string;
  checked: boolean;
}
