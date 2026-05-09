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
import { AIToolConfigResult, AIToolModule, InitConfigOptions } from '../types';
import { getGenkitContext, updateContentInPlace } from '../utils';

const RULES_PATH = '.idx/airules.md';

export const studio: AIToolModule = {
  name: 'studio',
  displayName: 'Firebase Studio',

  /**
   * Configures Firebase Studio (Project IDX) with Genkit context.
   *
   * - .idx/airules.md: Updates Firebase section only (preserves user content)
   *
   * Interactive prompts are shown since this file may contain user-defined
   * AI rules and instructions that we must preserve. We only manage the
   * Genkit-specific section marked with our XML tags.
   */
  async configure(
    runtime: Runtime,
    options?: InitConfigOptions
  ): Promise<AIToolConfigResult> {
    const files: AIToolConfigResult['files'] = [];
    const content = getGenkitContext(runtime);
    const { updated } = await updateContentInPlace(RULES_PATH, content);
    files.push({ path: RULES_PATH, updated });
    return { files };
  },
};
