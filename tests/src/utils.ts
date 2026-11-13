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

import { execSync, spawn, type ChildProcess } from 'child_process';
import fs from 'fs';
import { diffString, type DiffStringOptions } from 'json-diff';
import os from 'os';
import path from 'path';
import puppeteer, { type Page } from 'puppeteer';
import { PuppeteerScreenRecorder } from 'puppeteer-screen-recorder';
import terminate from 'terminate';

export async function runDevUiTest(
  testAppName: string,
  appCmd: string,
  testFn: (page: Page, devUiUrl: string) => Promise<void>
) {
  await runTestsForApp(testAppName, appCmd, async (devUiUrl: string) => {
    const browser = await puppeteer.launch({
      slowMo: 50,
    });
    const page = await browser.newPage();
    const recorder = new PuppeteerScreenRecorder(page);
    const savePath = './last_recording.mp4';
    await recorder.start(savePath);

    try {
      await testFn(page, devUiUrl);
      console.log('Test passed');
    } finally {
      await recorder.stop();
    }
  });
}

export async function runTestsForApp(
  testAppPath: string,
  appCmd: string,
  testFn: (devUiUrl: string) => Promise<void>
) {
  return new Promise(async (resolver, reject) => {
    var gsProcess;
    try {
      const { url, process } = await genkitStart(testAppPath, appCmd);
      gsProcess = process;
      await testFn(url);
      console.log('Test passed');
    } catch (e) {
      reject(e);
    } finally {
      if (gsProcess) {
        terminate(gsProcess.pid!, (error) => {
          console.log('terminate done', error);
          resolver(undefined);
        });
      }
    }
  });
}

export async function setupNodeTestApp(testAppPath: string): Promise<string> {
  const testRoot = path.resolve(os.tmpdir(), `./e2e-run-${Date.now()}`);
  console.log(`testRoot=${testRoot} pwd=${process.cwd()}`);
  fs.mkdirSync(testRoot, { recursive: true });
  fs.cpSync(testAppPath, testRoot, { recursive: true });
  const distDir = path.resolve(process.cwd(), '../dist');
  execSync(`npm i --save ${distDir}/*.tgz`, {
    stdio: 'inherit',
    cwd: testRoot,
  });
  execSync(`npm run build`, { stdio: 'inherit', cwd: testRoot });
  return testRoot;
}

export async function genkitStart(
  testRoot: string,
  appCmd: string
): Promise<{ url: string; process: ChildProcess }> {
  const cliInstallRoot = path.resolve(os.tmpdir(), `./test-cli-${Date.now()}`);
  console.log(`cliInstallRoot=${cliInstallRoot} pwd=${process.cwd()}`);
  fs.mkdirSync(cliInstallRoot, { recursive: true });

  execSync(`npm init -y`, {
    stdio: 'inherit',
    cwd: cliInstallRoot,
  });
  const distDir = path.resolve(process.cwd(), '../dist');
  execSync(
    `npm i --save ${distDir}/genkit-cli-?.?*.?*.tgz ${distDir}/genkit-ai-tools-common-*.tgz ${distDir}/genkit-ai-telemetry-server-*.tgz`,
    {
      stdio: 'inherit',
      cwd: cliInstallRoot,
    }
  );

  return new Promise((urlResolver, reject) => {
    const appProcess = spawn(
      'npm',
      [
        'exec',
        '--prefix',
        cliInstallRoot,
        'genkit',
        'start',
        '--',
        ...appCmd.split(' '),
      ],
      {
        cwd: testRoot,
      }
    );

    // Press enter in case of cookie ack prompt.
    appProcess.stdin.write('\n');

    var done = false;
    setTimeout(() => {
      if (!done) {
        done = true;
        reject(new Error('timeout waiting for genkit start to start'));
      }
    }, 30000);

    appProcess.stdout?.on('data', (data) => {
      console.log('stdout: ' + data.toString());
      const match = data.toString().match(/Genkit Developer UI:\w*([^ ]*)/);
      if (match && match.length > 1) {
        console.log('Developer UI ready, launching test ' + match[1]);
        if (done) {
          return;
        }
        done = true;
        urlResolver({
          url: match[1],
          process: appProcess,
        });
      }
    });
    appProcess.stderr?.on('data', (data) => {
      console.log(data.toString());
    });
    appProcess.on('error', (error): void => {
      console.log(`Error in app process: ${error}`);
      process.exitCode = 22;
    });
    appProcess.on('exit', (code) => {
      console.log(`Developer UI exited with code ${code}`);
    });
  });
}

export function diffJSON(
  lhs: any,
  rhs: any,
  options?: DiffStringOptions
): string {
  return diffString(
    normalizeForComparison(lhs),
    normalizeForComparison(rhs),
    options
  );
}

/**
 * Normalized JSON blob before comparison. Removes fields that contain default
 * values, for example: false booleans and empty string.
 */
function normalizeForComparison(body: any): any {
  const out = {} as any;
  for (const key in body) {
    if (typeof body[key] === 'boolean' && body[key] === false) {
      continue;
    }
    if (typeof body[key] === 'string' && body[key] === '') {
      continue;
    }
    if (typeof body[key] === 'object') {
      const normalizedObject = normalizeForComparison(body[key]);
      if (Object.keys(normalizedObject).length === 0) {
        continue;
      }
      out[key] = normalizedObject;
      continue;
    }
    out[key] = body[key];
  }
  return out;
}

export async function retriable<T>(
  fn: () => Promise<T>,
  opts: { maxRetries?: number; delayMs?: number }
): Promise<T> {
  const maxRetries = opts.maxRetries ?? 3;
  const delayMs = opts.delayMs ?? 0;

  let attempt = 0;
  while (true) {
    try {
      return await fn();
    } catch (e) {
      if (attempt >= maxRetries - 1) {
        throw e;
      }
      if (delayMs > 0) {
        await new Promise((r) => setTimeout(r, delayMs));
      }
    }
    attempt++;
  }
}
