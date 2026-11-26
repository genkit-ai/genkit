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

import * as readlinePromises from 'readline/promises';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('readline/promises', () => ({
  createInterface: vi.fn(),
}));

const UUID_V4_REGEX =
  /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

describe('analytics', () => {
  const store = new Map<string, unknown>();
  let analyticsModule: typeof import('../../src/utils/analytics.js');
  let getAnalyticsSettings: typeof import('../../src/utils/analytics.js').getAnalyticsSettings;
  let configstoreModule: typeof import('../../src/utils/configstore.js');

  beforeEach(async () => {
    vi.resetModules();
    configstoreModule = await import('../../src/utils/configstore.js');
    analyticsModule = await import('../../src/utils/analytics.js');
    ({ getAnalyticsSettings } = analyticsModule);

    store.clear();
    vi.spyOn(configstoreModule.configstore, 'get').mockImplementation(
      (key: string) => store.get(key)
    );
    vi.spyOn(configstoreModule.configstore, 'set').mockImplementation(((
      key: string,
      value?: unknown
    ) => {
      store.set(key, value);
    }) as typeof configstoreModule.configstore.set);
    vi.spyOn(configstoreModule, 'getUserSettings').mockReturnValue({});
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('getAnalyticsSettings', () => {
    it('returns disabled when analytics notification is not acknowledged', () => {
      expect(getAnalyticsSettings()).toEqual({ enabled: false });
    });

    it('generates and persists a valid v4 UUID clientId when none exists', () => {
      store.set('analytics_notification', true);

      const settings = getAnalyticsSettings();

      expect(settings).toMatchObject({
        enabled: true,
        clientId: expect.stringMatching(UUID_V4_REGEX),
      });
      expect(store.get('genkit-tools-ga-id')).toMatch(UUID_V4_REGEX);
    });

    it('reuses existing clientId from configstore when present', () => {
      store.set('analytics_notification', true);
      store.set('genkit-tools-ga-id', 'existing-client-id');

      const settings = getAnalyticsSettings();

      expect(settings).toMatchObject({
        enabled: true,
        clientId: 'existing-client-id',
      });
      expect(configstoreModule.configstore.set).not.toHaveBeenCalled();
    });
  });

  describe('FirstUsageEvent', () => {
    it('sets project_runtime when provided', () => {
      const event = new analyticsModule.FirstUsageEvent('go');

      expect(event.parameters).toEqual({ project_runtime: 'go' });
    });

    it('falls back to unknown when project_runtime is omitted', () => {
      const event = new analyticsModule.FirstUsageEvent();

      expect(event.parameters).toEqual({ project_runtime: 'unknown' });
    });
  });

  describe('notifyAnalyticsIfFirstRun', () => {
    it.each([
      ['go', 'go'],
      [undefined, 'unknown'],
    ])(
      'acknowledges notification and sends first_visit with project_runtime=%s as %s',
      async (runtime, expected) => {
        vi.mocked(readlinePromises.createInterface).mockReturnValue({
          question: vi.fn().mockResolvedValue(''),
          close: vi.fn(),
        } as unknown as readlinePromises.Interface);
        const fetchSpy = vi
          .spyOn(globalThis, 'fetch')
          .mockResolvedValue(new Response(null, { status: 204 }));
        const stderrSpy = vi
          .spyOn(process.stderr, 'write')
          .mockImplementation(() => true);
        const stdoutSpy = vi
          .spyOn(process.stdout, 'write')
          .mockImplementation(() => true);

        await analyticsModule.notifyAnalyticsIfFirstRun(runtime);

        expect(stdoutSpy).not.toHaveBeenCalled();
        expect(stderrSpy).toHaveBeenCalledWith(
          expect.stringContaining('Genkit CLI and Developer UI use cookies')
        );
        expect(readlinePromises.createInterface).toHaveBeenCalledWith({
          input: process.stdin,
          output: process.stderr,
        });
        expect(store.get('analytics_notification')).toBe(true);
        expect(fetchSpy).toHaveBeenCalledOnce();
        const [, init] = fetchSpy.mock.calls[0];
        expect(JSON.parse(String(init?.body))).toMatchObject({
          events: [
            {
              name: 'first_visit',
              params: { project_runtime: expected },
            },
          ],
        });
      }
    );

    it('skips prompt and event when notification is already acknowledged', async () => {
      store.set('analytics_notification', true);
      const fetchSpy = vi.spyOn(globalThis, 'fetch');

      await analyticsModule.notifyAnalyticsIfFirstRun('go');

      expect(readlinePromises.createInterface).not.toHaveBeenCalled();
      expect(fetchSpy).not.toHaveBeenCalled();
    });

    it('skips prompt and event when analyticsOptOut is enabled', async () => {
      vi.spyOn(configstoreModule, 'getUserSettings').mockReturnValue({
        analyticsOptOut: true,
      });
      const fetchSpy = vi.spyOn(globalThis, 'fetch');

      await analyticsModule.notifyAnalyticsIfFirstRun('go');

      expect(readlinePromises.createInterface).not.toHaveBeenCalled();
      expect(fetchSpy).not.toHaveBeenCalled();
    });
  });
});
