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

import { logger } from '@genkit-ai/tools-common/utils';
import { Command } from 'commander';

const DEVELOP_WITH_AI_DOCS_URL = 'https://genkit.dev/docs/develop-with-ai';

/** Deprecated stub retained for backwards compatibility with existing scripts. */
export const initAiTools = new Command('init:ai-tools')
  .description(
    'deprecated: use Genkit Agent Skills (`npx skills add genkit-ai/skills`) and `genkit mcp` instead'
  )
  .option('-y, --yes', 'Run in non-interactive mode (no-op)')
  .action(() => {
    logger.warn(
      '`genkit init:ai-tools` is deprecated and no longer generates static context files.\n' +
        'Use Genkit Agent Skills (`npx skills add genkit-ai/skills`) and the Genkit MCP server instead:\n' +
        `${DEVELOP_WITH_AI_DOCS_URL}\n` +
        'If this workspace has a legacy `GENKIT.md` file from an earlier run, you can safely delete it.'
    );
  });
