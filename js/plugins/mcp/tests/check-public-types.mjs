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

import { realpathSync } from 'node:fs';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const ts = require('typescript');
const pkgDir = realpathSync(fileURLToPath(new URL('..', import.meta.url)));
const configPath = resolve(pkgDir, 'tsconfig.public-types.json');

const formatHost = {
  getCanonicalFileName: (fileName) => fileName,
  getCurrentDirectory: () => pkgDir,
  getNewLine: () => ts.sys.newLine,
};

function reportDiagnostics(diagnostics) {
  const formatted = process.stderr.isTTY
    ? ts.formatDiagnosticsWithColorAndContext(diagnostics, formatHost)
    : ts.formatDiagnostics(diagnostics, formatHost);
  process.stderr.write(formatted);
}

function getModuleSpecifiers(sourceFile) {
  const specifiers = [];
  for (const statement of sourceFile.statements) {
    if (
      (ts.isImportDeclaration(statement) ||
        ts.isExportDeclaration(statement)) &&
      statement.moduleSpecifier &&
      ts.isStringLiteral(statement.moduleSpecifier)
    ) {
      specifiers.push(statement.moduleSpecifier);
    }
  }
  return specifiers;
}

function main() {
  const readResult = ts.readConfigFile(configPath, ts.sys.readFile);
  if (readResult.error) {
    reportDiagnostics([readResult.error]);
    process.exitCode = 1;
    return;
  }

  const parsed = ts.parseJsonConfigFileContent(
    readResult.config,
    ts.sys,
    pkgDir,
    undefined,
    configPath
  );
  if (parsed.errors.length > 0) {
    reportDiagnostics(parsed.errors);
    process.exitCode = 1;
    return;
  }

  const host = ts.createCompilerHost(parsed.options);
  host.getCurrentDirectory = () => pkgDir;
  const program = ts.createProgram({
    rootNames: parsed.fileNames,
    options: parsed.options,
    host,
  });

  const diagnostics = ts.getPreEmitDiagnostics(program);
  if (diagnostics.length > 0) {
    reportDiagnostics(diagnostics);
    process.exitCode = 1;
    return;
  }

  const checks = [
    {
      fixture: 'tests/public-types.mts',
      expectedDts: 'lib/index.d.mts',
    },
    {
      fixture: 'tests/public-types.cts',
      expectedDts: 'lib/index.d.ts',
    },
  ];

  const checker = program.getTypeChecker();
  const failures = [];

  for (const { fixture, expectedDts } of checks) {
    const fixturePath = resolve(pkgDir, fixture);
    const expectedPath = resolve(pkgDir, expectedDts);
    const sourceFile = program.getSourceFile(fixturePath);
    if (!sourceFile) {
      failures.push(`program did not include fixture ${fixture}`);
      continue;
    }

    const specifiers = getModuleSpecifiers(sourceFile);
    for (const specifier of specifiers) {
      if (specifier.text.startsWith('.')) {
        failures.push(
          `${fixture} uses relative specifier '${specifier.text}' instead of package exports`
        );
      }
    }

    const mcpSpecifiers = specifiers.filter(
      (specifier) => specifier.text === '@genkit-ai/mcp'
    );
    if (mcpSpecifiers.length === 0) {
      failures.push(`${fixture} does not import '@genkit-ai/mcp'`);
      continue;
    }

    for (const specifier of mcpSpecifiers) {
      const symbol = checker.getSymbolAtLocation(specifier);
      const resolvedFiles = [
        ...new Set(
          (symbol?.declarations ?? []).map((decl) =>
            resolve(decl.getSourceFile().fileName)
          )
        ),
      ];
      if (resolvedFiles.length !== 1 || resolvedFiles[0] !== expectedPath) {
        failures.push(
          `${fixture} resolved '@genkit-ai/mcp' to ${resolvedFiles.join(', ') || '<unresolved>'} instead of ${expectedPath}`
        );
      }
    }
  }

  if (failures.length > 0) {
    console.error(
      `Public declaration verification failed:\n- ${failures.join('\n- ')}`
    );
    process.exitCode = 1;
  }
}

main();
