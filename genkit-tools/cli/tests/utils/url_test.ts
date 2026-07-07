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

import { describe, expect, it } from '@jest/globals';
import { formatHostForUrl, httpUrl } from '../../src/utils/url';

describe('url utils', () => {
  it('formats IPv6 hosts for URLs', () => {
    expect(formatHostForUrl('::1')).toBe('[::1]');
    expect(httpUrl('::1', 4000)).toBe('http://[::1]:4000');
  });

  it('maps wildcard bind hosts to localhost URLs', () => {
    expect(formatHostForUrl('0.0.0.0')).toBe('localhost');
    expect(formatHostForUrl('::')).toBe('localhost');
    expect(httpUrl('0.0.0.0', 4000)).toBe('http://localhost:4000');
    expect(httpUrl('::', 4000)).toBe('http://localhost:4000');
  });
});
