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

/**
 * Deprecated command retained for compatibility with existing scripts.
 */
export const initAiTools = new Command('init:ai-tools')
  .description(
    'deprecated; legacy generated context is cleaned up automatically'
  )
  .action(() => {
    logger.warn(
      'The init:ai-tools command is deprecated and no longer installs generated context files.'
    );
  });
