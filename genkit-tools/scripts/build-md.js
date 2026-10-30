/**
 * Copyright 2025 Google LLC
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

const fs = require('fs');
const path = require('path');

const CONTEXT_DIR = path.resolve(__dirname, '..', 'cli', 'context');
const TARGET_DIR = path.resolve(
  __dirname,
  '..',
  'cli',
  'src',
  'commands',
  'init-ai-tools',
  'context'
);

const jsContextPath = path.resolve(CONTEXT_DIR, 'GENKIT.js.md');
const goContextPath = path.resolve(CONTEXT_DIR, 'GENKIT.go.md');

const AUTO_GEN_HEADER = '// Auto-generated, do not edit';

// Ensure output directory exists
if (!fs.existsSync(CONTEXT_DIR)) {
  throw new Error('Context dir is missing.');
}

// Ensure context files exists
if (!fs.existsSync(jsContextPath) || !fs.existsSync(goContextPath)) {
  throw new Error('JS/Go context files missing.');
}

async function mdToTs(filePath, target) {
  try {
    const data = await fs.promises.readFile(filePath, 'utf8');
    const cleaned = JSON.stringify(data.replace(/\r\n/g, '\n').trim());
    const final = `${AUTO_GEN_HEADER}\nexport const GENKIT_CONTEXT = ${cleaned}`;
    fs.writeFileSync(target, final);
  } catch (err) {
    console.error('Error reading file:', err);
  }
}

fs.mkdirSync(TARGET_DIR, { recursive: true });
mdToTs(jsContextPath, path.join(TARGET_DIR, 'nodejs.ts'));
mdToTs(goContextPath, path.join(TARGET_DIR, 'go.ts'));
