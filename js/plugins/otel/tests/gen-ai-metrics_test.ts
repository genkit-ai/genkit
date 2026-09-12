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

import type { Attributes, Histogram, Meter } from '@opentelemetry/api';
import * as assert from 'assert';
import { describe, it } from 'node:test';
import { GenAiMetrics } from '../src/genai/gen-ai-metrics.js';

interface RecordedPoint {
  name: string;
  value: number;
  attributes?: Attributes;
}

/** A minimal Meter that captures histogram records for assertions. */
function fakeMeter(points: RecordedPoint[]): Meter {
  const makeHistogram = (name: string): Histogram => ({
    record(value: number, attributes?: Attributes) {
      points.push({ name, value, attributes });
    },
  });
  return {
    createHistogram: (name: string) => makeHistogram(name),
  } as unknown as Meter;
}

describe('GenAiMetrics', () => {
  it('records input and output token points tagged by type', () => {
    const points: RecordedPoint[] = [];
    const metrics = new GenAiMetrics(fakeMeter(points));
    metrics.recordTokenUsage({ 'gen_ai.request.model': 'x' }, 10, 20);

    const tokenPoints = points.filter(
      (p) => p.name === 'gen_ai.client.token.usage'
    );
    assert.strictEqual(tokenPoints.length, 2);
    assert.deepStrictEqual(tokenPoints[0], {
      name: 'gen_ai.client.token.usage',
      value: 10,
      attributes: {
        'gen_ai.request.model': 'x',
        'gen_ai.token.type': 'input',
      },
    });
    assert.strictEqual(tokenPoints[1].value, 20);
    assert.strictEqual(
      tokenPoints[1].attributes?.['gen_ai.token.type'],
      'output'
    );
  });

  it('omits token points that are undefined', () => {
    const points: RecordedPoint[] = [];
    const metrics = new GenAiMetrics(fakeMeter(points));
    metrics.recordTokenUsage({}, undefined, 5);
    const tokenPoints = points.filter(
      (p) => p.name === 'gen_ai.client.token.usage'
    );
    assert.strictEqual(tokenPoints.length, 1);
    assert.strictEqual(
      tokenPoints[0].attributes?.['gen_ai.token.type'],
      'output'
    );
  });

  it('records operation duration', () => {
    const points: RecordedPoint[] = [];
    const metrics = new GenAiMetrics(fakeMeter(points));
    metrics.recordDuration(1.5, { 'error.type': 'Error' });
    const durationPoints = points.filter(
      (p) => p.name === 'gen_ai.client.operation.duration'
    );
    assert.strictEqual(durationPoints.length, 1);
    assert.strictEqual(durationPoints[0].value, 1.5);
    assert.strictEqual(durationPoints[0].attributes?.['error.type'], 'Error');
  });
});
