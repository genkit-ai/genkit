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

import { readFileSync } from 'fs';
import * as yaml from 'yaml';
import {
  diffJSON,
  retriable,
  runTestsForApp,
  setupNodeTestApp,
} from './utils.js';

(async () => {
  // Run for nodejs test app
  const testAppRoot = await setupNodeTestApp('test_js_app');
  await runTestsForApp(testAppRoot, 'node lib/index.js', async () => {
    await testReflectionApi();
  });
  // Run same tests for go test app
  await runTestsForApp('../go/tests/test_app', 'go run main.go', async () => {
    await testReflectionApi();
  });
})();

async function testReflectionApi() {
  const url = 'http://localhost:3100';
  await retriable(
    async () => {
      const res = await fetch(`${url}/api/__health`);
      if (res.status != 200) {
        throw new Error(`timed out waiting for code to become helthy`);
      }
    },
    {
      maxRetries: 30,
      delayMs: 1000,
    }
  );

  const t = yaml.parse(readFileSync('specs/reflection_api.yaml', 'utf8'));
  for (const test of t.tests) {
    console.log('path', test.path);
    const fetchopts = {
      method: 'GET',
    } as RequestInit;
    if (test.hasOwnProperty('post')) {
      fetchopts.method = 'POST';
      fetchopts.headers = { 'Content-Type': 'application/json' };
      fetchopts.body = JSON.stringify(test.post);
    }
    const res = await fetch(url + test.path, fetchopts);
    if (res.status != 200) {
      throw new Error(`${test.path}: got status ${res.status}`);
    }
    const gotBody = await res.json();
    const diff = diffJSON(gotBody, test.body, {
      sort: true,
      excludeKeys: [
        // TODO: Go and JS JSON schema generation is very different.
        // Maybe figure out some kind of schema compatibility test?
        'outputSchema',
        'inputSchema',
        // FIXME: Go does not set description field
        'description',
        // FIXME: Go does not set telemetry/latencyMs fields.
        'telemetry',
        'latencyMs',
        'usage',
      ],
    });
    if (diff != '') {
      console.log('test failed:', diff);
      process.exitCode = 1;
      throw new Error('test failed');
    }
  }
  console.log('Done! \\o/');
}
