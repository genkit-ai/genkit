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

import { logger } from '@genkit-ai/tools-common/utils';
import { existsSync } from 'fs';
import { readFile } from 'fs/promises';
import * as path from 'path';

const LEGACY_GENKIT_MD_FILENAME = 'GENKIT.md';
const LEGACY_BAD_INSTRUCTION_MARKER = 'Do not run Genkit commands';

/**
 * Logs a read-only warning if the workspace has a legacy `GENKIT.md` file
 * containing outdated instructions from `genkit init:ai-tools`.
 */
export async function warnIfLegacyGenkitContext(
  projectRoot: string
): Promise<void> {
  const genkitMdPath = path.join(projectRoot, LEGACY_GENKIT_MD_FILENAME);
  if (!existsSync(genkitMdPath)) {
    return;
  }

  try {
    const content = await readFile(genkitMdPath, 'utf-8');
    if (content.includes(LEGACY_BAD_INSTRUCTION_MARKER)) {
      logger.warn(
        'Legacy `GENKIT.md` detected with outdated instructions from deprecated `genkit init:ai-tools`.\n' +
          'You can delete `GENKIT.md` (and any references to it) and install Genkit Agent Skills instead:\n' +
          '  npx skills add genkit-ai/skills'
      );
    }
  } catch (e) {
    logger.debug('Failed to inspect GENKIT.md for legacy instructions', e);
  }
}
