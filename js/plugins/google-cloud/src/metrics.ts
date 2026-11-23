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

import {
  metrics,
  type Counter,
  type Histogram,
  type Meter,
} from '@opentelemetry/api';
import type { ReadableSpan } from '@opentelemetry/sdk-trace-base';

export const METER_NAME = 'genkit';
export const METRIC_NAME_PREFIX = 'genkit';
const METRIC_DIMENSION_MAX_CHARS = 256;

export function internalMetricNamespaceWrap(...namespaces: string[]) {
  return [METRIC_NAME_PREFIX, ...namespaces].join('/');
}

type MetricCreateFn<T> = (meter: Meter) => T;

/**
 * Wrapper for OpenTelemetry metrics.
 *
 * The OpenTelemetry {MeterProvider} can only be accessed through the metrics
 * API after the NodeSDK library has been initialized. To prevent race
 * conditions we defer the instantiation of the metric to when it is first
 * ticked.
 */
class Metric<T> {
  readonly createFn: MetricCreateFn<T>;
  readonly meterName: string;
  metric?: T;

  constructor(createFn: MetricCreateFn<T>, meterName: string = METER_NAME) {
    this.meterName = meterName;
    this.createFn = createFn;
  }

  get(): T {
    if (!this.metric) {
      this.metric = this.createFn(
        metrics.getMeterProvider().getMeter(this.meterName)
      );
    }

    return this.metric;
  }
}

/**
 * Wrapper for an OpenTelemetry Counter.
 *
 * By using this wrapper, we defer initialization of the counter until it is
 * need, which ensures that the OpenTelemetry SDK has been initialized before
 * the metric has been defined.
 */
export class MetricCounter extends Metric<Counter> {
  constructor(name: string, options: any) {
    super((meter) => meter.createCounter(name, options));
  }

  add(val?: number, opts?: any) {
    if (val) {
      truncateDimensions(opts);
      this.get().add(val, opts);
    }
  }
}

/**
 * Wrapper for an OpenTelemetry Histogram.
 *
 * By using this wrapper, we defer initialization of the counter until it is
 * need, which ensures that the OpenTelemetry SDK has been initialized before
 * the metric has been defined.
 */
export class MetricHistogram extends Metric<Histogram> {
  constructor(name: string, options: any) {
    super((meter) => meter.createHistogram(name, options));
  }

  record(val?: number, opts?: any) {
    if (val) {
      truncateDimensions(opts);
      this.get().record(val, opts);
    }
  }
}

/**
 * Truncates all of the metric dimensions to ensure they are below the trace
 * attribute max. Labels should be consistent between metrics and traces so that
 * data can be properly correlated. This will truncate all dimensions for
 * metrics such that they will be consistent with data included in traces.
 */
function truncateDimensions(opts?: any) {
  if (opts) {
    Object.keys(opts).forEach((k) => {
      if (typeof opts[k] === 'string') {
        opts[k] = opts[k].substring(0, METRIC_DIMENSION_MAX_CHARS);
      }
    });
  }
}

export interface Telemetry {
  tick(span: ReadableSpan, logIO: boolean, projectId?: string): void;
}
