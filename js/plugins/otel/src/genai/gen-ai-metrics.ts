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
import { GenAiAttr, GenAiMetric } from './gen-ai-attributes.js';

// Explicit token-count buckets recommended by the spec for the token-usage
// histogram; the duration histogram uses the default seconds buckets.
const TOKEN_BUCKETS = [
  1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304,
  16777216, 67108864,
];

/**
 * The two spec-defined GenAI client metrics, recorded per model operation.
 *
 * See the spec:
 * https://github.com/open-telemetry/semantic-conventions-genai
 *
 * Instruments are created up front from the meter. When no MeterProvider is
 * configured, `@opentelemetry/api` returns no-op instruments, so this stays a
 * no-op until the app wires up metrics collection.
 */
export class GenAiMetrics {
  private readonly tokenUsage: Histogram;
  private readonly operationDuration: Histogram;

  constructor(meter: Meter) {
    this.tokenUsage = meter.createHistogram(GenAiMetric.tokenUsage, {
      unit: '{token}',
      description: 'Number of input and output tokens used by the model.',
      advice: { explicitBucketBoundaries: TOKEN_BUCKETS },
    });
    this.operationDuration = meter.createHistogram(
      GenAiMetric.operationDuration,
      {
        unit: 's',
        description: 'Duration of a GenAI model operation.',
      }
    );
  }

  /**
   * Records input/output token counts, one point per non-null count, tagged
   * with `gen_ai.token.type`.
   */
  recordTokenUsage(
    baseAttributes: Attributes,
    inputTokens?: number,
    outputTokens?: number
  ): void {
    if (inputTokens != null) {
      this.tokenUsage.record(inputTokens, {
        ...baseAttributes,
        [GenAiAttr.tokenType]: 'input',
      });
    }
    if (outputTokens != null) {
      this.tokenUsage.record(outputTokens, {
        ...baseAttributes,
        [GenAiAttr.tokenType]: 'output',
      });
    }
  }

  /**
   * Records the operation duration in seconds. Recorded for both successful
   * and failed operations (failures carry `error.type` in `attributes`).
   */
  recordDuration(seconds: number, attributes: Attributes): void {
    this.operationDuration.record(seconds, attributes);
  }
}
