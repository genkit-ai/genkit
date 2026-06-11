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

import { execSync } from 'child_process';
import { readFile, writeFile } from 'fs/promises';

interface FormatType {
  regex: RegExp;
  header?: string;
  body: string;
  footer?: string;
}

const FILE_OPTS: { encoding: 'utf-8' } = { encoding: 'utf-8' };
const FORMAT_TYPES: FormatType[] = [
  {
    regex: /\.((tsx?)|(s?css)|(jsx?))$/,
    header: '/**',
    body: ' *',
    footer: ' */',
  },
  { regex: /\.html$/, header: '<!--', body: '', footer: '-->' },
  { regex: /\.go$/, body: '//' },
];
const COPYRIGHT = ` Copyright ${new Date().getFullYear()} Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.`;

async function fileExists(path: string): Promise<boolean> {
  try {
    await readFile(path);
    return true;
  } catch {
    return false;
  }
}

async function getSourceFilesToUpdate() {
  const paths = (
    execSync('git ls-files', FILE_OPTS) +
    '\n' +
    execSync('git ls-files -o --exclude-standard', FILE_OPTS)
  )
    .split('\n')
    .filter((p) => !!p);

  const existingPaths = (
    await Promise.all(
      paths.map(async (path) => ((await fileExists(path)) ? path : null))
    )
  ).filter((path) => path !== null) as string[];

  const fileContents = await Promise.all(
    existingPaths.map((path) => readFile(path, { encoding: 'utf-8' }))
  );

  return fileContents
    .map((contents, idx) => ({
      contents,
      path: existingPaths[idx],
      format: FORMAT_TYPES.find(({ regex }) => regex.test(existingPaths[idx])),
    }))
    .filter(
      ({ contents, format }) => format && !/Copyright \d\d\d\d/.test(contents)
    );
}

function updateContent(content: string, format: FormatType): string {
  const header = format.header ? `${format.header}\n` : '';
  const body = COPYRIGHT.split('\n')
    .map((l) => format.body + l)
    .join('\n');
  const footer = format.footer ? `\n${format.footer}` : '';

  return `${header}${body}${footer}\n\n${content}`;
}

console.log('Checking copyright in sources...');

(async () => {
  const missing = await getSourceFilesToUpdate();
  if (process.argv[2] === '--check') {
    if (missing.length) {
      console.error(
        `Copyright header missing in ${missing.map(({ path }) => path).join(', ')}`
      );
      console.error('Run `npm run format` at root to update');
      process.exit(1);
    }
    console.log('Copyright headers okay');
    process.exit(0);
  }

  const updated = missing.map((m) => {
    m.contents = updateContent(m.contents, m.format!);
    return m;
  });

  await Promise.all(
    updated.map(({ path, contents }) =>
      writeFile(path, contents, { encoding: 'utf-8' })
    )
  );
  console.log(`Updated copyright headers in ${updated.length} files`);
})();
