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

// @ts-check
import vitestPlugin from '@vitest/eslint-plugin';
import eslintConfigPrettier from 'eslint-config-prettier';
import { defineConfig } from 'eslint/config';
import tseslint from 'typescript-eslint';

export default defineConfig(
  {
    ignores: ['**/dist/**', '**/lib/**', '**/node_modules/**', 'scripts/**'],
  },
  {
    files: ['**/*.ts'],
    extends: [tseslint.configs.base, eslintConfigPrettier],
    rules: {
      'no-unsafe-finally': 'error',
      'no-async-promise-executor': 'error',
      'preserve-caught-error': 'error',
      'no-case-declarations': 'error',
      'prefer-rest-params': 'error',
      'no-var': 'error',
      'prefer-const': 'error',
      '@typescript-eslint/no-non-null-asserted-optional-chain': 'error',
      'no-extra-boolean-cast': 'error',
      'no-prototype-builtins': 'error',
      'no-useless-assignment': 'error',
      'no-useless-escape': 'error',
    },
  },
  {
    files: ['**/*.test.ts'],
    plugins: {
      vitest: vitestPlugin,
    },
    rules: {
      'vitest/no-focused-tests': 'error',
      'vitest/consistent-test-it': [
        'error',
        { fn: 'it', withinDescribe: 'it' },
      ],
      'vitest/require-top-level-describe': 'error',
      'vitest/no-identical-title': 'error',
      'vitest/expect-expect': [
        'error',
        { assertFunctionNames: ['expect', 'assert*'] },
      ],
      'vitest/prefer-to-be': 'error',
      'vitest/prefer-to-have-length': 'error',
    },
  }
);
