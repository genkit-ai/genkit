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
  const store = new Map<string, any>();
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
      value?: any
    ) => {
      store.set(key, value);
    }) as any);
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

      expect(settings.enabled).toBe(true);
      if (!settings.enabled) {
        throw new Error('Expected analytics settings to be enabled');
      }
      expect(settings.clientId).toMatch(UUID_V4_REGEX);
      expect(configstoreModule.configstore.set).toHaveBeenCalledWith(
        'genkit-tools-ga-id',
        settings.clientId
      );
      expect(store.get('genkit-tools-ga-id')).toBe(settings.clientId);
    });

    it('reuses existing clientId from configstore when present', () => {
      store.set('analytics_notification', true);
      store.set('genkit-tools-ga-id', 'existing-client-id');

      const settings = getAnalyticsSettings();

      expect(settings.enabled).toBe(true);
      if (!settings.enabled) {
        throw new Error('Expected analytics settings to be enabled');
      }
      expect(settings.clientId).toBe('existing-client-id');
      expect(configstoreModule.configstore.set).not.toHaveBeenCalled();
    });
  });

  describe('FirstUsageEvent', () => {
    it('sets project_runtime and falls back to unknown', () => {
      const goEvent = new analyticsModule.FirstUsageEvent('go');
      expect(goEvent.parameters).toEqual({ project_runtime: 'go' });

      const unknownEvent = new analyticsModule.FirstUsageEvent();
      expect(unknownEvent.parameters).toEqual({ project_runtime: 'unknown' });
    });
  });

  describe('notifyAnalyticsIfFirstRun', () => {
    it('records first_visit with the detected project_runtime instead of leaving it unspecified', async () => {
      const question = vi.fn().mockResolvedValue('');
      const close = vi.fn();
      vi.mocked(readlinePromises.createInterface).mockReturnValue({
        question,
        close,
      } as any);
      const fetchSpy = vi
        .spyOn(globalThis, 'fetch')
        .mockResolvedValue(new Response(null, { status: 204 }));
      vi.spyOn(console, 'log').mockImplementation(() => {});

      await analyticsModule.notifyAnalyticsIfFirstRun('go');

      expect(store.get('analytics_notification')).toBe(true);
      expect(fetchSpy).toHaveBeenCalledTimes(1);
      const [, init] = fetchSpy.mock.calls[0];
      const payload = JSON.parse(String(init?.body));
      expect(payload.events[0].name).toBe('first_visit');
      expect(payload.events[0].params.project_runtime).toBe('go');
    });
  });
});
