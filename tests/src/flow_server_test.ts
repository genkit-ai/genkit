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
import { streamFlow } from 'genkit/beta/client';
import * as yaml from 'yaml';
import { retriable, runTestsForApp } from './utils.js';

(async () => {
  // TODO: Add NodeJS tests
  // Run the tests for go test app
  await runTestsForApp('../go/tests/test_app', 'go run main.go', async () => {
    await testFlowServer();
    console.log('Flow server tests done! \\o/');
  });
})();

type TestResults = {
  message: string;
  result: string;
};

async function testFlowServer() {
  const url = 'http://localhost:3400';
  await retriable(
    async () => {
      const res = await fetch(`${url}/streamy`, {
        method: 'POST',
        body: JSON.stringify({
          data: 1,
        }),
      });
      if (res.status != 200) {
        throw new Error(`timed out waiting for flow server to become healthy`);
      }
    },
    {
      maxRetries: 30,
      delayMs: 2000,
    }
  );

  const t = yaml.parse(readFileSync('flow_server_tests.yaml', 'utf8'));
  for (const test of t.tests) {
    let chunkCount = 0;
    let expected = '';
    const want: TestResults = {
      message: test.response.message,
      result: test.response.result,
    };
    console.log(`checking stream for: ${test.path}`);
    (async () => {
      const response = await streamFlow({
        url: `${url}/${test.path}`,
        input: test.post.data,
      });

      for await (const chunk of response.stream) {
        expected = want.message.replace('{count}', chunkCount.toString());
        const chunkJSON = JSON.stringify(await chunk);
        if (chunkJSON != expected) {
          throw new Error(
            `unexpected chunk data received, got: ${chunkJSON}, want: ${want.message}`
          );
        }
        chunkCount++;
      }
      if (chunkCount != test.post.data) {
        throw new Error(
          `unexpected number of stream chunks received: got ${chunkCount}, want: ${test.post.data}`
        );
      }
      const out = await response.output;
      want.result = want.result.replace(/\{count\}/g, chunkCount.toString());
      if (out != want.result) {
        throw new Error(
          `unexpected output received, got: ${out}, want: ${want.result}`
        );
      }
    })();
  }
}
