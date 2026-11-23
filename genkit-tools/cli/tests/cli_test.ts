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

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

const {
  mockNotifyAnalyticsIfFirstRun,
  mockRecord,
  mockFindProjectRoot,
  mockDetectRuntime,
  mockForceStderr,
  mockShowUpdateNotification,
  mockStartMcpServer,
  mockWarnIfLegacyGenkitContext,
  mockLoadDocs,
} = vi.hoisted(() => ({
  mockNotifyAnalyticsIfFirstRun: vi.fn(() => Promise.resolve()),
  mockRecord: vi.fn(() => Promise.resolve()),
  mockFindProjectRoot: vi.fn(() => Promise.resolve('/mock/root')),
  mockDetectRuntime: vi.fn(() => Promise.resolve('nodejs')),
  mockForceStderr: vi.fn(),
  mockShowUpdateNotification: vi.fn(() => Promise.resolve()),
  mockStartMcpServer: vi.fn(() => Promise.resolve()),
  mockWarnIfLegacyGenkitContext: vi.fn(() => Promise.resolve()),
  mockLoadDocs: vi.fn(() => Promise.resolve({})),
}));

vi.mock('@genkit-ai/tools-common/utils', async (importOriginal) => {
  const actual =
    await importOriginal<typeof import('@genkit-ai/tools-common/utils')>();
  return {
    ...actual,
    notifyAnalyticsIfFirstRun: mockNotifyAnalyticsIfFirstRun,
    record: mockRecord,
    findProjectRoot: mockFindProjectRoot,
    detectRuntime: mockDetectRuntime,
    forceStderr: mockForceStderr,
    logger: {
      debug: vi.fn(),
      info: vi.fn(),
      warn: vi.fn(),
      error: vi.fn(),
    },
  };
});

vi.mock('../src/utils/updates.js', () => ({
  showUpdateNotification: mockShowUpdateNotification,
}));

vi.mock('../src/mcp/server.js', () => ({
  startMcpServer: mockStartMcpServer,
}));

vi.mock('../src/utils/legacy-context-warning.js', () => ({
  warnIfLegacyGenkitContext: mockWarnIfLegacyGenkitContext,
}));

vi.mock('../src/utils/docs.js', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../src/utils/docs.js')>();
  return {
    ...actual,
    loadDocs: mockLoadDocs,
  };
});

vi.mock('../src/commands/plugins.js', () => ({
  getPluginCommands: vi.fn(() => Promise.resolve([])),
  getPluginSubCommand: vi.fn(() => Promise.resolve(undefined)),
}));

describe('startCLI', () => {
  const originalArgv = process.argv;
  let startCLI: typeof import('../src/cli.js').startCLI;

  beforeEach(async () => {
    vi.resetModules();
    vi.clearAllMocks();
    const actualCommander =
      await vi.importActual<typeof import('commander')>('commander');
    vi.doMock('commander', () => ({
      ...actualCommander,
      program: new actualCommander.Command(),
    }));
    ({ startCLI } = await import('../src/cli.js'));
  });

  afterEach(() => {
    process.argv = originalArgv;
  });

  it('skips first-run analytics prompt and update notification for mcp command', async () => {
    process.argv = ['node', 'genkit', 'mcp'];

    await startCLI();

    expect(mockNotifyAnalyticsIfFirstRun).not.toHaveBeenCalled();
    expect(mockShowUpdateNotification).not.toHaveBeenCalled();
    expect(mockForceStderr).toHaveBeenCalled();
    expect(mockStartMcpServer).toHaveBeenCalled();
  });

  it('skips update notification when global options precede mcp command', async () => {
    process.argv = ['node', 'genkit', '--non-interactive', 'mcp'];

    await startCLI();

    expect(mockNotifyAnalyticsIfFirstRun).not.toHaveBeenCalled();
    expect(mockShowUpdateNotification).not.toHaveBeenCalled();
    expect(mockStartMcpServer).toHaveBeenCalled();
  });

  it('runs first-run analytics prompt and update notification for standard commands', async () => {
    process.argv = ['node', 'genkit', 'docs:list', 'js'];

    await startCLI();

    expect(mockNotifyAnalyticsIfFirstRun).toHaveBeenCalledOnce();
    expect(mockShowUpdateNotification).toHaveBeenCalledOnce();
    expect(mockLoadDocs).toHaveBeenCalledOnce();
  });
});
