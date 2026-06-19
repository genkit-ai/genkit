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

import { afterEach, describe, expect, it, jest } from '@jest/globals';
import {
  DEV_SERVER_HOST_ENV_VAR,
  getDevServerHost,
  isLoopbackHost,
  rejectNonLoopbackHost,
} from '../../src/utils/dev-server-security';

describe('dev-server-security', () => {
  const originalHost = process.env[DEV_SERVER_HOST_ENV_VAR];

  afterEach(() => {
    if (originalHost === undefined) {
      delete process.env[DEV_SERVER_HOST_ENV_VAR];
    } else {
      process.env[DEV_SERVER_HOST_ENV_VAR] = originalHost;
    }
  });

  describe('isLoopbackHost', () => {
    it('accepts loopback hosts', () => {
      expect(isLoopbackHost('localhost')).toBe(true);
      expect(isLoopbackHost('localhost:4000')).toBe(true);
      expect(isLoopbackHost('127.0.0.1')).toBe(true);
      expect(isLoopbackHost('127.0.0.1:4033')).toBe(true);
      expect(isLoopbackHost('[::1]')).toBe(true);
      expect(isLoopbackHost('[::1]:3100')).toBe(true);
      expect(isLoopbackHost('app.localhost:4000')).toBe(true);
    });

    it('rejects non-loopback and missing hosts', () => {
      expect(isLoopbackHost(undefined)).toBe(false);
      expect(isLoopbackHost('')).toBe(false);
      expect(isLoopbackHost('attacker.evil.com')).toBe(false);
      expect(isLoopbackHost('attacker.evil.com:4000')).toBe(false);
      expect(isLoopbackHost('192.168.0.6:4000')).toBe(false);
      expect(isLoopbackHost('localhost.evil.com')).toBe(false);
    });
  });

  describe('getDevServerHost', () => {
    it('defaults to the loopback interface', () => {
      delete process.env[DEV_SERVER_HOST_ENV_VAR];
      expect(getDevServerHost()).toBe('localhost');
    });

    it('honors the override env var', () => {
      process.env[DEV_SERVER_HOST_ENV_VAR] = '0.0.0.0';
      expect(getDevServerHost()).toBe('0.0.0.0');
    });
  });

  describe('rejectNonLoopbackHost', () => {
    function mockRes() {
      const res: any = {};
      res.status = jest.fn(() => res);
      res.send = jest.fn(() => res);
      return res;
    }

    it('passes loopback requests through', () => {
      delete process.env[DEV_SERVER_HOST_ENV_VAR];
      const res = mockRes();
      const next = jest.fn();
      rejectNonLoopbackHost(
        { headers: { host: 'localhost:4000' } } as any,
        res,
        next
      );
      expect(next).toHaveBeenCalled();
      expect(res.status).not.toHaveBeenCalled();
    });

    it('rejects non-loopback Host headers with 403', () => {
      delete process.env[DEV_SERVER_HOST_ENV_VAR];
      const res = mockRes();
      const next = jest.fn();
      rejectNonLoopbackHost(
        { headers: { host: 'attacker.evil.com' } } as any,
        res,
        next
      );
      expect(next).not.toHaveBeenCalled();
      expect(res.status).toHaveBeenCalledWith(403);
    });

    it('allows any Host when the override env var is set', () => {
      process.env[DEV_SERVER_HOST_ENV_VAR] = '0.0.0.0';
      const res = mockRes();
      const next = jest.fn();
      rejectNonLoopbackHost(
        { headers: { host: 'my-dev-box.internal:4000' } } as any,
        res,
        next
      );
      expect(next).toHaveBeenCalled();
      expect(res.status).not.toHaveBeenCalled();
    });
  });
});
