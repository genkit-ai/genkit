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

import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  jest,
} from '@jest/globals';

const UUID_V4_REGEX =
  /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

describe('analytics', () => {
  const store = new Map<string, any>();
  let getAnalyticsSettings: typeof import('../../src/utils/analytics').getAnalyticsSettings;
  let configstoreModule: typeof import('../../src/utils/configstore');

  beforeEach(() => {
    jest.resetModules();
    configstoreModule = require('../../src/utils/configstore');
    getAnalyticsSettings =
      require('../../src/utils/analytics').getAnalyticsSettings;

    store.clear();
    jest
      .spyOn(configstoreModule.configstore, 'get')
      .mockImplementation((key: string) => store.get(key));
    jest.spyOn(configstoreModule.configstore, 'set').mockImplementation(((
      key: string,
      value?: any
    ) => {
      store.set(key, value);
    }) as any);
    jest.spyOn(configstoreModule, 'getUserSettings').mockReturnValue({});
  });

  afterEach(() => {
    jest.restoreAllMocks();
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
});
