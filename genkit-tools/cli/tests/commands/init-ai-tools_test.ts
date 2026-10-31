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
import { expect, it, jest } from '@jest/globals';
import { initAiTools } from '../../src/commands/init-ai-tools/command';

it('prints a deprecation notice instead of installing generated context', async () => {
  const warn = jest.spyOn(logger, 'warn').mockImplementation(() => logger);

  await initAiTools.parseAsync(['node', 'genkit', 'init:ai-tools']);

  expect(warn).toHaveBeenCalledWith(
    expect.stringContaining('is deprecated and no longer installs')
  );
  warn.mockRestore();
});
