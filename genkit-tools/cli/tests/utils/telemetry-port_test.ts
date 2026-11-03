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

import { startTelemetryServer } from '@genkit-ai/telemetry-server';
import { expect, it, jest } from '@jest/globals';
import getPort from 'get-port';
import { resolveTelemetryServer } from '../../src/utils/manager-utils';

jest.mock('@genkit-ai/telemetry-server', () => ({
  LocalFileLogStore: jest.fn(),
  LocalFileTraceStore: jest.fn(),
  startTelemetryServer: jest.fn(() => Promise.resolve()),
}));
jest.mock('get-port', () => ({
  __esModule: true,
  default: jest.fn(() => Promise.resolve(4034)),
  makeRange: jest.fn(() => [4033, 4034]),
}));

it('probes and starts telemetry on the same IPv4 loopback interface', async () => {
  const previous = process.env.GENKIT_TELEMETRY_SERVER;
  delete process.env.GENKIT_TELEMETRY_SERVER;
  try {
    const url = await resolveTelemetryServer({ projectRoot: '/root' });

    expect(getPort).toHaveBeenCalledWith({
      port: [4033, 4034],
      host: '127.0.0.1',
    });
    expect(startTelemetryServer).toHaveBeenCalledWith(
      expect.objectContaining({ port: 4034, host: '127.0.0.1' })
    );
    expect(url).toBe('http://127.0.0.1:4034');
  } finally {
    if (previous === undefined) {
      delete process.env.GENKIT_TELEMETRY_SERVER;
    } else {
      process.env.GENKIT_TELEMETRY_SERVER = previous;
    }
  }
});
