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

import type { Doc } from '../../src/utils/docs';

export const SAMPLE_DOCS: Record<string, Doc> = {
  'js/flows.md': {
    title: 'Defining Flows (JS)',
    description: 'Define flows in JavaScript with defineFlow.',
    text: '# JS Flows',
    lang: 'js',
    headers: 'defineFlow',
  },
  'go/flows.md': {
    title: 'Defining Flows (Go)',
    description: 'Define flows in Go with genkit.DefineFlow.',
    text: '# Go Flows',
    lang: 'go',
    headers: 'DefineFlow',
  },
  'python/flows.md': {
    title: 'Defining Flows (Python)',
    description: 'Define flows in Python with @ai.flow.',
    text: '# Python Flows',
    lang: 'python',
    headers: 'flow decorator',
  },
  'dart/flows.md': {
    title: 'Defining Flows (Dart)',
    description: 'Define flows in Dart with ai.defineFlow.',
    text: '# Dart Flows',
    lang: 'dart',
    headers: 'defineFlow schemantic',
  },
};

export const WORKSPACE_MARKERS = [
  { marker: 'go.mod', expectedLang: 'go', expectedPath: 'go/flows.md' },
  {
    marker: 'pyproject.toml',
    expectedLang: 'python',
    expectedPath: 'python/flows.md',
  },
  {
    marker: 'pubspec.yaml',
    expectedLang: 'dart',
    expectedPath: 'dart/flows.md',
  },
  {
    marker: 'package.json',
    expectedLang: 'js',
    expectedPath: 'js/flows.md',
  },
] as const;
