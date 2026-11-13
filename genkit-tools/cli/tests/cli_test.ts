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

import * as commonUtils from '@genkit-ai/tools-common/utils';
import { Command } from 'commander';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

let mockProgram: Command;

vi.mock('commander', async (importOriginal) => {
  const actual = await importOriginal<typeof import('commander')>();
  return {
    ...actual,
    get program() {
      return mockProgram;
    },
  };
});

vi.mock('@genkit-ai/tools-common/utils', async (importOriginal) => {
  const actual =
    await importOriginal<typeof import('@genkit-ai/tools-common/utils')>();
  return {
    ...actual,
    findProjectRoot: vi.fn(),
    detectRuntime: vi.fn(),
    notifyAnalyticsIfFirstRun: vi.fn(),
    record: vi.fn(),
  };
});

vi.mock('../src/utils/updates.js', () => ({
  showUpdateNotification: vi.fn(),
}));

vi.mock('../src/utils/docs.js', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../src/utils/docs.js')>();
  return {
    ...actual,
    loadDocs: vi.fn().mockResolvedValue({}),
    detectProjectLanguage: vi.fn().mockReturnValue('go'),
  };
});

vi.mock('../src/commands/plugins.js', () => ({
  getPluginCommands: vi.fn().mockResolvedValue([]),
  getPluginSubCommand: vi.fn().mockResolvedValue(undefined),
}));

vi.mock('../src/commands/server-harness.js', async () => {
  const { Command } = await import('commander');
  return {
    SERVER_HARNESS_COMMAND: '__server-harness',
    serverHarness: new Command('__server-harness')
      .argument('<port>')
      .argument('<logFile>')
      .action(() => {}),
  };
});

describe('startCLI analytics preAction hook', () => {
  const originalArgv = process.argv;

  beforeEach(() => {
    vi.resetModules();
    vi.clearAllMocks();
    mockProgram = new Command();
    vi.mocked(commonUtils.findProjectRoot).mockResolvedValue('/mock/go-app');
    vi.mocked(commonUtils.detectRuntime).mockResolvedValue('go');
    vi.mocked(commonUtils.notifyAnalyticsIfFirstRun).mockResolvedValue();
    vi.mocked(commonUtils.record).mockResolvedValue();
    vi.spyOn(commonUtils.logger, 'info').mockImplementation((() => {}) as any);
  });

  afterEach(() => {
    process.argv = originalArgv;
    vi.restoreAllMocks();
  });

  it('passes detected projectRuntime to notifyAnalyticsIfFirstRun before recording RunCommandEvent', async () => {
    process.argv = ['node', 'genkit', '--no-update-notification', 'docs:list'];
    const { startCLI } = await import('../src/cli.js');

    await startCLI();

    expect(commonUtils.notifyAnalyticsIfFirstRun).toHaveBeenCalledWith('go');
    expect(commonUtils.record).toHaveBeenCalledTimes(1);
    const event = vi.mocked(commonUtils.record).mock.calls[0][0];
    expect(event.stickyParameters?.project_runtime).toBe('go');
  });

  it('skips notifyAnalyticsIfFirstRun when --non-interactive is passed', async () => {
    process.argv = [
      'node',
      'genkit',
      '--non-interactive',
      '--no-update-notification',
      'docs:list',
    ];
    const { startCLI } = await import('../src/cli.js');

    await startCLI();

    expect(commonUtils.notifyAnalyticsIfFirstRun).not.toHaveBeenCalled();
  });

  it('skips notifyAnalyticsIfFirstRun on __server-harness daemon process', async () => {
    process.argv = [
      'node',
      'genkit',
      '__server-harness',
      '4000',
      '/tmp/devui.log',
    ];
    const { startCLI } = await import('../src/cli.js');

    await startCLI();

    expect(commonUtils.notifyAnalyticsIfFirstRun).not.toHaveBeenCalled();
  });
});
