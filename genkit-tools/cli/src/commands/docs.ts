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

import { findProjectRoot, logger } from '@genkit-ai/tools-common/utils';
import * as clc from 'colorette';
import { Command } from 'commander';
import {
  SUPPORTED_DOC_LANGUAGES,
  detectProjectLanguage,
  loadDocs,
  searchDocs,
} from '../utils/docs.js';
import { writeStderr, writeStdout } from '../utils/output.js';

export const docsList = new Command('docs:list')
  .description('list available Genkit documentation files')
  .argument(
    '[language]',
    `language to list docs for (${SUPPORTED_DOC_LANGUAGES.join(', ')}). Defaults to the auto-detected project language`
  )
  .action(async (language) => {
    try {
      const documents = await loadDocs();
      const lang = language || detectProjectLanguage(await findProjectRoot());
      const fileList = Object.keys(documents)
        .filter((file) => file.startsWith(lang))
        .sort();

      if (fileList.length === 0) {
        writeStderr(`No documentation found for language: ${lang}`);
        return;
      }

      writeStderr(`Genkit Documentation Index (${lang}):\n`);
      fileList.forEach((file) => {
        const doc = documents[file];
        writeStdout(`${clc.bold(doc.title)}`);
        writeStdout(`  Path: ${file}`);
        if (doc.description) {
          writeStdout(`  ${clc.italic(doc.description)}`);
        }
        writeStdout('');
      });
      writeStderr(`Use 'genkit docs:read <path>' to read a document.`);
    } catch (e: unknown) {
      logger.error(
        `Failed to load documentation: ${e instanceof Error ? e.message : String(e)}`
      );
      process.exitCode = 1;
    }
  });

export const docsSearch = new Command('docs:search')
  .description('search Genkit documentation')
  .argument(
    '<query>',
    'keywords to search for. For multiple keywords, enclose in quotes. E.g. "stream flows"'
  )
  .argument(
    '[language]',
    `language to search docs for (${SUPPORTED_DOC_LANGUAGES.join(', ')}). Defaults to the auto-detected project language`
  )
  .action(async (query, language) => {
    try {
      const documents = await loadDocs();
      const lang = language || detectProjectLanguage(await findProjectRoot());
      const results = searchDocs(documents, query, lang).slice(0, 10);

      if (results.length === 0) {
        writeStderr(`No results found for "${query}" in ${lang} docs.`);
        return;
      }

      writeStderr(
        `Found ${results.length} matching documents for "${query}":\n`
      );
      results.forEach((r) => {
        writeStdout(`${clc.bold(r.doc.title)}`);
        writeStdout(`  Path: ${r.file}`);
        if (r.doc.description) {
          writeStdout(`  ${clc.italic(r.doc.description)}`);
        }
        writeStdout('');
      });
    } catch (e: unknown) {
      logger.error(
        `Failed to load documentation: ${e instanceof Error ? e.message : String(e)}`
      );
      process.exitCode = 1;
    }
  });

export const docsRead = new Command('docs:read')
  .description('read a Genkit documentation file')
  .argument('<filePath>', 'path of the document to read')
  .action(async (filePath) => {
    try {
      const documents = await loadDocs();
      const doc = documents[filePath];
      if (!doc) {
        logger.error(`Document not found: ${filePath}`);
        process.exitCode = 1;
        return;
      }

      writeStdout(clc.bold(doc.title));
      writeStdout('='.repeat(doc.title.length));
      writeStdout('');
      writeStdout(doc.text);
    } catch (e: unknown) {
      logger.error(
        `Failed to load documentation: ${e instanceof Error ? e.message : String(e)}`
      );
      process.exitCode = 1;
    }
  });
