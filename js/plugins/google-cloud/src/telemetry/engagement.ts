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

import { ValueType, type Attributes } from '@opentelemetry/api';
import type { ReadableSpan } from '@opentelemetry/sdk-trace-base';
import { GENKIT_VERSION } from 'genkit';
import { logger } from 'genkit/logging';
import {
  MetricCounter,
  internalMetricNamespaceWrap,
  type Telemetry,
} from '../metrics.js';
import { createCommonLogAttributes, truncate } from '../utils.js';

class EngagementTelemetry implements Telemetry {
  /**
   * Wraps the declared metrics in a Genkit-specific, internal namespace.
   */
  private _N = internalMetricNamespaceWrap.bind(null, 'engagement');

  private feedbackCounter = new MetricCounter(this._N('feedback'), {
    description: 'Counts calls to genkit flows.',
    valueType: ValueType.INT,
  });

  private acceptanceCounter = new MetricCounter(this._N('acceptance'), {
    description: 'Tracks unique flow paths per flow.',
    valueType: ValueType.INT,
  });

  tick(
    span: ReadableSpan,
    logInputAndOutput: boolean,
    projectId?: string
  ): void {
    const subtype = span.attributes['genkit:metadata:subtype'] as string;

    if (subtype === 'userFeedback') {
      this.writeUserFeedback(span, projectId);
      return;
    }

    if (subtype === 'userAcceptance') {
      this.writeUserAcceptance(span, projectId);
      return;
    }

    logger.warn(`Unknown user engagement subtype: ${subtype}`);
  }

  private writeUserFeedback(span: ReadableSpan, projectId?: string) {
    const attributes = span.attributes;
    const name = this.extractTraceName(attributes);

    const dimensions = {
      name,
      value: attributes['genkit:metadata:feedbackValue'],
      hasText: !!attributes['genkit:metadata:textFeedback'],
      source: 'ts',
      sourceVersion: GENKIT_VERSION,
    };
    this.feedbackCounter.add(1, dimensions);

    const metadata = {
      ...createCommonLogAttributes(span, projectId),
      feedbackValue: attributes['genkit:metadata:feedbackValue'],
    };
    if (attributes['genkit:metadata:textFeedback']) {
      metadata['textFeedback'] = truncate(
        attributes['genkit:metadata:textFeedback'] as string
      );
    }
    logger.info(`UserFeedback[${name}]`, metadata);
  }

  private writeUserAcceptance(span: ReadableSpan, projectId?: string) {
    const attributes = span.attributes;
    const name = this.extractTraceName(attributes);

    const dimensions = {
      name,
      value: attributes['genkit:metadata:acceptanceValue'],
      source: 'ts',
      sourceVersion: GENKIT_VERSION,
    };
    this.acceptanceCounter.add(1, dimensions);

    const metadata = {
      ...createCommonLogAttributes(span, projectId),
      acceptanceValue: attributes['genkit:metadata:acceptanceValue'],
    };
    logger.info(`UserAcceptance[${name}]`, metadata);
  }

  private extractTraceName(attributes: Attributes) {
    const path = attributes['genkit:path'] as string;
    if (!path || path === '<unknown>') {
      return '<unknown>';
    }

    const name = path.match('/{(.+)}+');
    return name ? name[1] : '<unknown>';
  }
}

const engagementTelemetry = new EngagementTelemetry();
export { engagementTelemetry };
