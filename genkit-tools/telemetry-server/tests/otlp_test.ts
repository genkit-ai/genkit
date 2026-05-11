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

import * as assert from 'assert';
import { describe, it } from 'node:test';
import { logDataFromOtlp, traceDataFromOtlp } from '../src/utils/otlp';

describe('otlp-traces', () => {
  it('should transform OTLP payload to TraceData', () => {
    const otlpPayload = {
      resourceSpans: [
        {
          resource: {
            attributes: [],
            droppedAttributesCount: 0,
          },
          scopeSpans: [
            {
              scope: { name: 'genkit-tracer', version: 'v1' },
              spans: [
                {
                  traceId: 'c5892692eb25cce482eb13587b73c425',
                  spanId: '86dc3d35cc11e336',
                  parentSpanId: 'd05557675cb95b72',
                  name: 'generateContentStream',
                  kind: 1,
                  startTimeUnixNano: '1760827335359000000',
                  endTimeUnixNano: '1760827336695073000',
                  attributes: [
                    {
                      key: 'genkit:name',
                      value: { stringValue: 'generateContentStream' },
                    },
                    {
                      key: 'genkit:path',
                      value: {
                        stringValue:
                          '/{geminiStream_submitQuery}/{generateContentStream}',
                      },
                    },
                    {
                      key: 'genkit:input',
                      value: {
                        stringValue: '.....',
                      },
                    },
                    {
                      key: 'genkit:state',
                      value: { stringValue: 'success' },
                    },
                  ],
                  droppedAttributesCount: 0,
                  events: [],
                  droppedEventsCount: 0,
                  status: { code: 0 },
                  links: [],
                  droppedLinksCount: 0,
                },
                {
                  traceId: 'c5892692eb25cce482eb13587b73c425',
                  spanId: 'd05557675cb95b72',
                  name: 'geminiStream_submitQuery',
                  kind: 2,
                  startTimeUnixNano: '1760827334493000000',
                  endTimeUnixNano: '1760827336711390583',
                  attributes: [
                    {
                      key: 'genkit:name',
                      value: { stringValue: 'geminiStream_submitQuery' },
                    },
                    {
                      key: 'genkit:isRoot',
                      value: { boolValue: true },
                    },
                  ],
                  droppedAttributesCount: 0,
                  events: [],
                  droppedEventsCount: 0,
                  status: { code: 0 },
                  links: [],
                  droppedLinksCount: 0,
                },
              ],
            },
          ],
        },
      ],
    };

    const expectedTraceData = [
      {
        traceId: 'c5892692eb25cce482eb13587b73c425',
        spans: {
          '86dc3d35cc11e336': {
            traceId: 'c5892692eb25cce482eb13587b73c425',
            spanId: '86dc3d35cc11e336',
            parentSpanId: 'd05557675cb95b72',
            startTime: 1760827335359,
            endTime: 1760827336695,
            displayName: 'generateContentStream',
            attributes: {
              'genkit:name': 'generateContentStream',
              'genkit:path':
                '/{geminiStream_submitQuery}/{generateContentStream}',
              'genkit:input': '.....',
              'genkit:state': 'success',
            },
            instrumentationLibrary: {
              name: 'genkit-tracer',
              version: 'v1',
            },
            spanKind: 'INTERNAL',
          },
          d05557675cb95b72: {
            traceId: 'c5892692eb25cce482eb13587b73c425',
            spanId: 'd05557675cb95b72',
            parentSpanId: undefined,
            startTime: 1760827334493,
            endTime: 1760827336711,
            displayName: 'geminiStream_submitQuery',
            attributes: {
              'genkit:name': 'geminiStream_submitQuery',
              'genkit:isRoot': true,
            },
            instrumentationLibrary: {
              name: 'genkit-tracer',
              version: 'v1',
            },
            spanKind: 'SERVER',
          },
        },
      },
    ];

    const result = traceDataFromOtlp(otlpPayload as any);
    assert.deepStrictEqual(result, expectedTraceData);
  });

  it('should transform OTLP payload with non-zero status to TraceData', () => {
    const otlpPayload = {
      resourceSpans: [
        {
          resource: {
            attributes: [],
            droppedAttributesCount: 0,
          },
          scopeSpans: [
            {
              scope: { name: 'genkit-tracer', version: 'v1' },
              spans: [
                {
                  traceId: 'c5892692eb25cce482eb13587b73c425',
                  spanId: '86dc3d35cc11e336',
                  parentSpanId: 'd05557675cb95b72',
                  name: 'generateContentStream',
                  kind: 1,
                  startTimeUnixNano: '1760827335359000000',
                  endTimeUnixNano: '1760827336695073000',
                  attributes: [],
                  droppedAttributesCount: 0,
                  events: [],
                  droppedEventsCount: 0,
                  status: { code: 2, message: 'An error occurred' },
                  links: [],
                  droppedLinksCount: 0,
                },
              ],
            },
          ],
        },
      ],
    };

    const expectedTraceData = [
      {
        traceId: 'c5892692eb25cce482eb13587b73c425',
        spans: {
          '86dc3d35cc11e336': {
            traceId: 'c5892692eb25cce482eb13587b73c425',
            spanId: '86dc3d35cc11e336',
            parentSpanId: 'd05557675cb95b72',
            startTime: 1760827335359,
            endTime: 1760827336695,
            displayName: 'generateContentStream',
            attributes: {},
            instrumentationLibrary: {
              name: 'genkit-tracer',
              version: 'v1',
            },
            spanKind: 'INTERNAL',
            status: {
              code: 2,
              message: 'An error occurred',
            },
          },
        },
      },
    ];

    const result = traceDataFromOtlp(otlpPayload as any);
    assert.deepStrictEqual(result, expectedTraceData);
  });
});

describe('otlp-logs', () => {
  it('should transform OTLP log payload to LogRecordData', () => {
    const otlpPayload = {
      resourceLogs: [
        {
          resource: {
            attributes: [],
            droppedAttributesCount: 0,
          },
          scopeLogs: [
            {
              scope: { name: 'genkit-tracer', version: 'v1' },
              logRecords: [
                {
                  timeUnixNano: '1760827335359000000',
                  severityText: 'INFO',
                  severityNumber: 9,
                  body: { stringValue: 'This is a test log message' },
                  traceId: 'c5892692eb25cce482eb13587b73c425',
                  spanId: '86dc3d35cc11e336',
                  attributes: [
                    {
                      key: 'genkit:name',
                      value: { stringValue: 'generateContentStream' },
                    },
                  ],
                },
                {
                  timeUnixNano: '1760827336695073000',
                  severityText: 'ERROR',
                  severityNumber: 17,
                  body: { stringValue: 'An error occurred' },
                  traceId: 'c5892692eb25cce482eb13587b73c425',
                },
              ],
            },
          ],
        },
      ],
    };

    const expectedLogData = [
      {
        logId: '',
        traceId: 'c5892692eb25cce482eb13587b73c425',
        spanId: '86dc3d35cc11e336',
        timestamp: 1760827335359,
        severityNumber: 9,
        severityText: 'INFO',
        body: 'This is a test log message',
        attributes: {
          'genkit:name': 'generateContentStream',
        },
        instrumentationLibrary: {
          name: 'genkit-tracer',
          version: 'v1',
        },
      },
      {
        logId: '',
        traceId: 'c5892692eb25cce482eb13587b73c425',
        spanId: undefined,
        timestamp: 1760827336695,
        severityNumber: 17,
        severityText: 'ERROR',
        body: 'An error occurred',
        attributes: {},
        instrumentationLibrary: {
          name: 'genkit-tracer',
          version: 'v1',
        },
      },
    ];

    const result = logDataFromOtlp(otlpPayload);
    // Overwrite the random logId generator for deep strict equal
    result.forEach((log) => (log.logId = ''));
    assert.deepStrictEqual(result, expectedLogData);
  });
});
