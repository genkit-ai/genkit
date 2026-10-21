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
 */

import * as assert from 'assert';
import { readFileSync } from 'fs';
import { describe, it } from 'node:test';

interface PackageManifest {
  optionalDependencies?: Record<string, string>;
  peerDependencies?: Record<string, string>;
  peerDependenciesMeta?: Record<string, { optional?: boolean }>;
}

const packageManifest = JSON.parse(
  readFileSync('package.json', 'utf8')
) as PackageManifest;

describe('core package metadata', () => {
  it('keeps Firebase monitoring as an optional peer integration', () => {
    assert.strictEqual(
      packageManifest.optionalDependencies?.['@genkit-ai/firebase'],
      undefined
    );
    assert.strictEqual(
      packageManifest.peerDependencies?.['@genkit-ai/firebase'],
      '^1.16.1'
    );
    assert.strictEqual(
      packageManifest.peerDependenciesMeta?.['@genkit-ai/firebase']?.optional,
      true
    );
  });
});
