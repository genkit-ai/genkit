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
import eslint from '@eslint/js';
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
    extends: [
      eslint.configs.recommended,
      ...tseslint.configs.recommended,
      eslintConfigPrettier,
    ],
    rules: {
      eqeqeq: ['error', 'always', { null: 'ignore' }],
      '@typescript-eslint/consistent-type-imports': 'error',
      '@typescript-eslint/no-import-type-side-effects': 'error',
      '@typescript-eslint/no-explicit-any': 'off',
      '@typescript-eslint/no-unused-vars': [
        'error',
        {
          argsIgnorePattern: '^_',
          varsIgnorePattern: '^_',
          caughtErrorsIgnorePattern: '^_',
        },
      ],
    },
  },
  {
    files: ['*/src/**/*.ts', '*/tests/**/*.ts'],
    languageOptions: {
      parserOptions: {
        project: [
          './cli/tsconfig.test.json',
          './common/tsconfig.test.json',
          './telemetry-server/tsconfig.test.json',
        ],
        tsconfigRootDir: import.meta.dirname,
      },
    },
    rules: {
      '@typescript-eslint/no-floating-promises': 'error',
      '@typescript-eslint/no-misused-promises': 'error',
      '@typescript-eslint/await-thenable': 'error',
      '@typescript-eslint/switch-exhaustiveness-check': 'error',
      '@typescript-eslint/return-await': [
        'error',
        'error-handling-correctness-only',
      ],
      '@typescript-eslint/only-throw-error': 'error',
    },
  },
  {
    files: ['*/src/**/*.ts'],
    rules: {
      'no-console': 'error',
    },
  },
  {
    files: ['**/*.test.ts'],
    plugins: {
      vitest: vitestPlugin,
    },
    rules: {
      // Allow `importOriginal<typeof import('...')>()` in Vitest `vi.mock()` factories.
      '@typescript-eslint/consistent-type-imports': [
        'error',
        { disallowTypeAnnotations: false },
      ],
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
