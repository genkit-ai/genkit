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

import * as assert from 'assert';
import { describe, it } from 'node:test';
import {
  isLoopbackHost,
  resolveReflectionConfig,
  secretsEqual,
} from '../src/reflection-config.js';

describe('resolveReflectionConfig', () => {
  it('is off when nothing is set', () => {
    assert.deepStrictEqual(resolveReflectionConfig({}), { kind: 'off' });
  });

  it('is disabled when GENKIT_REFLECTION_DISABLED is exactly "true"', () => {
    assert.deepStrictEqual(
      resolveReflectionConfig({
        GENKIT_REFLECTION_DISABLED: 'true',
        GENKIT_ENV: 'dev',
        GENKIT_REFLECTION_PORT: '3100',
        GENKIT_REFLECTION_V2_SERVER: 'ws://127.0.0.1:3200',
      }),
      { kind: 'disabled' }
    );
  });

  it('ignores other truthy spellings of disabled', () => {
    for (const value of ['1', 'yes', 'on', 'TRUE']) {
      const config = resolveReflectionConfig({
        GENKIT_REFLECTION_DISABLED: value,
        GENKIT_ENV: 'dev',
      });
      assert.strictEqual(config.kind, 'v1', `for ${value}`);
    }
  });

  it('dials out when a v2 server is set', () => {
    assert.deepStrictEqual(
      resolveReflectionConfig({
        GENKIT_REFLECTION_V2_SERVER: 'ws://127.0.0.1:3200',
        GENKIT_REFLECTION_SECRET_TOKEN: 's3cret',
      }),
      { kind: 'v2', url: 'ws://127.0.0.1:3200', secret: 's3cret' }
    );
  });

  it('prefers v2 over a configured v1 port', () => {
    const config = resolveReflectionConfig({
      GENKIT_REFLECTION_V2_SERVER: 'ws://127.0.0.1:3200',
      GENKIT_REFLECTION_PORT: '3100',
    });
    assert.strictEqual(config.kind, 'v2');
  });

  it('treats a host on its own as the on-switch', () => {
    assert.deepStrictEqual(
      resolveReflectionConfig({ GENKIT_REFLECTION_HOST: '0.0.0.0' }),
      {
        kind: 'v1',
        host: '0.0.0.0',
        port: { kind: 'probeFrom', port: 3100 },
        secret: undefined,
      }
    );
  });

  it('treats a port on its own as the on-switch, and pins it', () => {
    assert.deepStrictEqual(
      resolveReflectionConfig({ GENKIT_REFLECTION_PORT: '4200' }),
      {
        kind: 'v1',
        host: '127.0.0.1',
        port: { kind: 'pinned', port: 4200 },
        secret: undefined,
      }
    );
  });

  it('probes from 3100 in dev', () => {
    assert.deepStrictEqual(resolveReflectionConfig({ GENKIT_ENV: 'dev' }), {
      kind: 'v1',
      host: '127.0.0.1',
      port: { kind: 'probeFrom', port: 3100 },
      secret: undefined,
    });
  });

  it('lets the environment beat the programmatic port', () => {
    const config = resolveReflectionConfig(
      { GENKIT_REFLECTION_PORT: '4200' },
      { port: 9999 }
    );
    assert.deepStrictEqual(config, {
      kind: 'v1',
      host: '127.0.0.1',
      port: { kind: 'pinned', port: 4200 },
      secret: undefined,
    });
  });

  it('uses the programmatic port as the probe start when the env has none', () => {
    const config = resolveReflectionConfig(
      { GENKIT_ENV: 'dev' },
      { port: 9999 }
    );
    assert.deepStrictEqual(config, {
      kind: 'v1',
      host: '127.0.0.1',
      port: { kind: 'probeFrom', port: 9999 },
      secret: undefined,
    });
  });

  it('rejects an invalid port rather than falling back', () => {
    for (const port of ['abc', '-1', '70000', '3100.5']) {
      assert.throws(
        () => resolveReflectionConfig({ GENKIT_REFLECTION_PORT: port }),
        /GENKIT_REFLECTION_PORT/,
        `for ${port}`
      );
    }
  });

  it('accepts port 0', () => {
    const config = resolveReflectionConfig({ GENKIT_REFLECTION_PORT: '0' });
    assert.deepStrictEqual(config, {
      kind: 'v1',
      host: '127.0.0.1',
      port: { kind: 'pinned', port: 0 },
      secret: undefined,
    });
  });
});

describe('isLoopbackHost', () => {
  it('recognizes loopback addresses', () => {
    for (const host of [
      '127.0.0.1',
      '127.1.2.3',
      'localhost',
      '::1',
      '[::1]',
    ]) {
      assert.strictEqual(isLoopbackHost(host), true, host);
    }
  });

  it('rejects routable addresses', () => {
    for (const host of ['0.0.0.0', '192.168.1.5', '10.0.0.1', 'example.com']) {
      assert.strictEqual(isLoopbackHost(host), false, host);
    }
  });
});

describe('secretsEqual', () => {
  it('compares by value, including different lengths', () => {
    assert.strictEqual(secretsEqual('abc', 'abc'), true);
    assert.strictEqual(secretsEqual('abc', 'abd'), false);
    assert.strictEqual(secretsEqual('abc', 'much-longer-secret'), false);
    assert.strictEqual(secretsEqual('', ''), true);
  });
});
