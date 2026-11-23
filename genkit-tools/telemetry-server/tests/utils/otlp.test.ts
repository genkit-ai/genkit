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

import { describe, expect, it } from 'vitest';
import {
  decodeOtlpProtobuf,
  logDataFromOtlp,
  traceDataFromOtlp,
} from '../../src/utils/otlp.js';

function pbVarint(val: bigint | number): Buffer {
  let v = BigInt.asUintN(64, BigInt(val));
  const bytes: number[] = [];
  do {
    let b = Number(v & 0x7fn);
    v >>= 7n;
    if (v !== 0n) b |= 0x80;
    bytes.push(b);
  } while (v !== 0n);
  return Buffer.from(bytes);
}

function pbTag(fieldNumber: number, wireType: number): Buffer {
  return pbVarint((fieldNumber << 3) | wireType);
}

function pbBytes(fieldNumber: number, data: Buffer | string): Buffer {
  const buf = typeof data === 'string' ? Buffer.from(data, 'utf8') : data;
  return Buffer.concat([pbTag(fieldNumber, 2), pbVarint(buf.length), buf]);
}

function pbVarintField(fieldNumber: number, val: bigint | number): Buffer {
  return Buffer.concat([pbTag(fieldNumber, 0), pbVarint(val)]);
}

function pbFixed64(fieldNumber: number, val: bigint): Buffer {
  const buf = Buffer.alloc(8);
  buf.writeBigUInt64LE(val, 0);
  return Buffer.concat([pbTag(fieldNumber, 1), buf]);
}

function pbDouble(fieldNumber: number, val: number): Buffer {
  const buf = Buffer.alloc(8);
  buf.writeDoubleLE(val, 0);
  return Buffer.concat([pbTag(fieldNumber, 1), buf]);
}

function pbKeyValue(key: string, anyValueBuf: Buffer): Buffer {
  return Buffer.concat([pbBytes(1, key), pbBytes(2, anyValueBuf)]);
}

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
    expect(result).toEqual(expectedTraceData);
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
    expect(result).toEqual(expectedTraceData);
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
                    {
                      key: 'confidence',
                      value: { doubleValue: 0.87 },
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
          confidence: 0.87,
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
    expect(result).toEqual(expectedLogData);
  });
});

describe('otlp-protobuf', () => {
  it('decodes ExportTraceServiceRequest protobuf payload to TraceData', () => {
    const traceIdHex = 'c5892692eb25cce482eb13587b73c425';
    const childSpanIdHex = '86dc3d35cc11e336';
    const rootSpanIdHex = 'd05557675cb95b72';

    const childSpanBuf = Buffer.concat([
      pbBytes(1, Buffer.from(traceIdHex, 'hex')),
      pbBytes(2, Buffer.from(childSpanIdHex, 'hex')),
      pbBytes(4, Buffer.from(rootSpanIdHex, 'hex')),
      pbBytes(5, 'generateContentStream'),
      pbVarintField(6, 1), // INTERNAL
      pbFixed64(7, 1760827335359000000n),
      pbFixed64(8, 1760827336695073000n),
      pbBytes(
        9,
        pbKeyValue('genkit:name', pbBytes(1, 'generateContentStream'))
      ),
      pbBytes(9, pbKeyValue('count', pbVarintField(3, 42))),
      pbBytes(9, pbKeyValue('score', pbDouble(4, 0.95))),
      pbBytes(
        9,
        pbKeyValue(
          'tags',
          pbBytes(
            5,
            Buffer.concat([
              pbBytes(1, pbBytes(1, 'a')),
              pbBytes(1, pbBytes(1, 'b')),
            ])
          )
        )
      ),
      pbBytes(
        15,
        Buffer.concat([pbBytes(2, 'An error occurred'), pbVarintField(3, 2)])
      ),
    ]);

    const rootSpanBuf = Buffer.concat([
      pbBytes(1, Buffer.from(traceIdHex, 'hex')),
      pbBytes(2, Buffer.from(rootSpanIdHex, 'hex')),
      pbBytes(4, Buffer.alloc(8, 0)), // all-zero parent_span_id -> undefined
      pbBytes(5, 'geminiStream_submitQuery'),
      pbVarintField(6, 2), // SERVER
      pbFixed64(7, 1760827334493000000n),
      // omitted end_time_unix_nano (in-progress span -> endTime 0)
      pbBytes(9, pbKeyValue('genkit:isRoot', pbVarintField(2, 1))),
    ]);

    const scopeBuf = Buffer.concat([
      pbBytes(1, 'genkit-tracer'),
      pbBytes(2, 'v1'),
    ]);
    const scopeSpansBuf = Buffer.concat([
      pbBytes(1, scopeBuf),
      pbBytes(2, childSpanBuf),
      pbBytes(2, rootSpanBuf),
    ]);
    const resourceSpansBuf = Buffer.concat([pbBytes(2, scopeSpansBuf)]);
    const exportTraceReqBuf = pbBytes(1, resourceSpansBuf);

    const decoded = decodeOtlpProtobuf(exportTraceReqBuf, 'traces');
    const result = traceDataFromOtlp(decoded);

    expect(result).toEqual([
      {
        traceId: traceIdHex,
        spans: {
          [childSpanIdHex]: {
            traceId: traceIdHex,
            spanId: childSpanIdHex,
            parentSpanId: rootSpanIdHex,
            startTime: 1760827335359,
            endTime: 1760827336695,
            displayName: 'generateContentStream',
            attributes: {
              'genkit:name': 'generateContentStream',
              count: 42,
              score: 0.95,
              tags: ['a', 'b'],
            },
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
          [rootSpanIdHex]: {
            traceId: traceIdHex,
            spanId: rootSpanIdHex,
            parentSpanId: undefined,
            startTime: 1760827334493,
            endTime: 0,
            displayName: 'geminiStream_submitQuery',
            attributes: {
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
    ]);
  });

  it('decodes ExportLogsServiceRequest protobuf payload to LogRecordData', () => {
    const traceIdHex = 'c5892692eb25cce482eb13587b73c425';
    const spanIdHex = '86dc3d35cc11e336';

    const log1Buf = Buffer.concat([
      pbFixed64(1, 1760827335359000000n),
      pbVarintField(2, 9),
      pbBytes(3, 'INFO'),
      pbBytes(5, pbBytes(1, 'This is a test log message')),
      pbBytes(
        6,
        pbKeyValue('genkit:name', pbBytes(1, 'generateContentStream'))
      ),
      pbBytes(6, pbKeyValue('confidence', pbDouble(4, 0.87))),
      pbBytes(9, Buffer.from(traceIdHex, 'hex')),
      pbBytes(10, Buffer.from(spanIdHex, 'hex')),
    ]);

    // Second log uses observed_time_unix_nano (field 11) and omits span_id
    const log2Buf = Buffer.concat([
      pbVarintField(2, 17),
      pbBytes(3, 'ERROR'),
      pbBytes(5, pbBytes(1, 'An error occurred')),
      pbBytes(9, Buffer.from(traceIdHex, 'hex')),
      pbFixed64(11, 1760827336695073000n),
    ]);

    const scopeBuf = Buffer.concat([
      pbBytes(1, 'genkit-tracer'),
      pbBytes(2, 'v1'),
    ]);
    const scopeLogsBuf = Buffer.concat([
      pbBytes(1, scopeBuf),
      pbBytes(2, log1Buf),
      pbBytes(2, log2Buf),
    ]);
    const resourceLogsBuf = Buffer.concat([pbBytes(2, scopeLogsBuf)]);
    const exportLogsReqBuf = pbBytes(1, resourceLogsBuf);

    const decoded = decodeOtlpProtobuf(exportLogsReqBuf, 'logs');
    const result = logDataFromOtlp(decoded);
    result.forEach((log) => (log.logId = ''));

    expect(result).toEqual([
      {
        logId: '',
        traceId: traceIdHex,
        spanId: spanIdHex,
        timestamp: 1760827335359,
        severityNumber: 9,
        severityText: 'INFO',
        body: 'This is a test log message',
        attributes: {
          'genkit:name': 'generateContentStream',
          confidence: 0.87,
        },
        instrumentationLibrary: {
          name: 'genkit-tracer',
          version: 'v1',
        },
      },
      {
        logId: '',
        traceId: traceIdHex,
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
    ]);
  });

  it('decodes kvlistValue and bytesValue attributes', () => {
    const traceIdHex = 'c5892692eb25cce482eb13587b73c425';
    const spanIdHex = '86dc3d35cc11e336';
    const rawBytes = Buffer.from([0xde, 0xad, 0xbe, 0xef]);

    const spanBuf = Buffer.concat([
      pbBytes(1, Buffer.from(traceIdHex, 'hex')),
      pbBytes(2, Buffer.from(spanIdHex, 'hex')),
      pbBytes(5, 'spanWithNestedValues'),
      pbVarintField(6, 1),
      pbFixed64(7, 1760827335359000000n),
      pbFixed64(8, 1760827336695073000n),
      pbBytes(
        9,
        pbKeyValue(
          'nested',
          pbBytes(
            6,
            Buffer.concat([
              pbBytes(1, pbKeyValue('innerStr', pbBytes(1, 'hello'))),
              pbBytes(1, pbKeyValue('innerInt', pbVarintField(3, 7))),
            ])
          )
        )
      ),
      pbBytes(9, pbKeyValue('raw', pbBytes(7, rawBytes))),
    ]);
    const traceReqBuf = pbBytes(1, pbBytes(2, pbBytes(2, spanBuf)));

    const decodedTrace = decodeOtlpProtobuf(traceReqBuf, 'traces');
    const traceResult = traceDataFromOtlp(decodedTrace);
    expect(traceResult).toHaveLength(1);
    expect(traceResult[0].spans[spanIdHex].attributes['nested']).toEqual({
      innerStr: 'hello',
      innerInt: 7,
    });
    expect(traceResult[0].spans[spanIdHex].attributes['raw']).toBe(
      rawBytes.toString('base64')
    );
  });

  it('throws on truncated or malformed protobuf payloads', () => {
    expect(() =>
      decodeOtlpProtobuf(Buffer.from([0x0a, 0x10, 0x01]), 'traces')
    ).toThrow();
    // Unclosed group (wire type 3 without matching wire type 4)
    expect(() =>
      decodeOtlpProtobuf(
        Buffer.concat([pbTag(2, 3), pbVarintField(1, 1)]),
        'traces'
      )
    ).toThrow(/unexpected EOF in group/);
    // Mismatched end-group field number
    expect(() =>
      decodeOtlpProtobuf(
        Buffer.concat([pbTag(2, 3), pbVarintField(1, 1), pbTag(3, 4)]),
        'traces'
      )
    ).toThrow(/mismatched end-group field number/);
  });
});
