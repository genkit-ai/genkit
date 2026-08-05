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

import { ValueType } from '@opentelemetry/api';
import { hrTimeDuration, hrTimeToMilliseconds } from '@opentelemetry/core';
import type { ReadableSpan } from '@opentelemetry/sdk-trace-base';
import { GENKIT_VERSION } from 'genkit';
import { logger } from 'genkit/logging';
import { toDisplayPath } from 'genkit/tracing';
import {
  MetricCounter,
  MetricHistogram,
  internalMetricNamespaceWrap,
  type Telemetry,
} from '../metrics.js';
import {
  createCommonLogAttributes,
  extractErrorMessage,
  extractErrorName,
  extractErrorStack,
  extractOuterFeatureNameFromPath,
  truncatePath,
} from '../utils.js';

class PathsTelemetry implements Telemetry {
  /**
   * Wraps the declared metrics in a Genkit-specific, internal namespace.
   */
  private _N = internalMetricNamespaceWrap.bind(null, 'feature');

  private pathCounter = new MetricCounter(this._N('path/requests'), {
    description: 'Tracks unique flow paths per flow.',
    valueType: ValueType.INT,
  });

  private pathLatencies = new MetricHistogram(this._N('path/latency'), {
    description: 'Latencies per flow path.',
    ValueType: ValueType.DOUBLE,
    unit: 'ms',
  });

  tick(
    span: ReadableSpan,
    logInputAndOutput: boolean,
    projectId?: string
  ): void {
    const attributes = span.attributes;

    const path = attributes['genkit:path'] as string;

    const isFailureSource = !!span.attributes['genkit:isFailureSource'];
    const state = attributes['genkit:state'] as string;

    if (!path || !isFailureSource || state !== 'error') {
      // Only tick metrics for failing, leaf spans.
      return;
    }

    const sessionId = attributes['genkit:sessionId'] as string;
    const threadName = attributes['genkit:threadName'] as string;

    const errorName = extractErrorName(span.events) || '<unknown>';
    const errorMessage = extractErrorMessage(span.events) || '<unknown>';
    const errorStack = extractErrorStack(span.events) || '';

    const latency = hrTimeToMilliseconds(
      hrTimeDuration(span.startTime, span.endTime)
    );

    const pathDimensions = {
      featureName: extractOuterFeatureNameFromPath(path),
      status: 'failure',
      error: errorName,
      path: path,
      source: 'ts',
      sourceVersion: GENKIT_VERSION,
    };
    this.pathCounter.add(1, pathDimensions);
    this.pathLatencies.record(latency, pathDimensions);

    const displayPath = truncatePath(toDisplayPath(path));
    logger.error(`Error[${displayPath}, ${errorName}]`, {
      ...createCommonLogAttributes(span, projectId),
      path: displayPath,
      qualifiedPath: path,
      name: errorName,
      message: errorMessage,
      stack: errorStack,
      source: 'ts',
      sourceVersion: GENKIT_VERSION,
      sessionId,
      threadName,
    });
  }
}

const pathsTelemetry = new PathsTelemetry();
export { pathsTelemetry };
