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

import type {
  LogRecordData,
  SpanData,
  TraceData,
} from '@genkit-ai/tools-common';

// These interfaces are based on the OTLP JSON format.
// A full definition can be found at:
// https://github.com/open-telemetry/opentelemetry-proto/blob/main/opentelemetry/proto

interface OtlpValue {
  stringValue?: string;
  intValue?: number;
  boolValue?: boolean;
  doubleValue?: number;
  arrayValue?: {
    values: OtlpValue[];
  };
  kvlistValue?: {
    values: OtlpAttribute[];
  };
  bytesValue?: string;
}

interface OtlpAttribute {
  key: string;
  value: OtlpValue;
}

interface OtlpSpan {
  traceId: string;
  spanId: string;
  parentSpanId?: string;
  name: string;
  kind: number;
  startTimeUnixNano: string;
  endTimeUnixNano: string;
  attributes: OtlpAttribute[];
  droppedAttributesCount: number;
  events: any[];
  droppedEventsCount: number;
  status?: {
    code: number;
    message?: string;
  };
  links: any[];
  droppedLinksCount: number;
}

interface OtlpScopeSpan {
  scope: {
    name: string;
    version?: string;
  };
  spans: OtlpSpan[];
}

interface OtlpResourceSpan {
  resource: {
    attributes: OtlpAttribute[];
    droppedAttributesCount: number;
  };
  scopeSpans: OtlpScopeSpan[];
}

interface OtlpLogRecord {
  timeUnixNano: string;
  observedTimeUnixNano?: string;
  severityNumber?: number;
  severityText?: string;
  body?: OtlpValue;
  attributes?: OtlpAttribute[];
  traceId?: string;
  spanId?: string;
}

interface OtlpScopeLog {
  scope: {
    name: string;
    version?: string;
  };
  logRecords: OtlpLogRecord[];
}

interface OtlpResourceLog {
  resource: {
    attributes: OtlpAttribute[];
    droppedAttributesCount: number;
  };
  scopeLogs: OtlpScopeLog[];
}

export interface OtlpPayload {
  resourceSpans?: OtlpResourceSpan[];
  resourceLogs?: OtlpResourceLog[];
}

const textDecoder = new TextDecoder('utf-8');

class ProtoReader {
  private readonly view: DataView;
  private pos = 0;
  private readonly len: number;

  constructor(private readonly buf: Uint8Array) {
    this.view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    this.len = buf.byteLength;
  }

  eof(): boolean {
    return this.pos >= this.len;
  }

  readVarint(): bigint {
    let result = 0n;
    let shift = 0n;
    while (this.pos < this.len) {
      const byte = this.buf[this.pos++];
      result |= BigInt(byte & 0x7f) << shift;
      if ((byte & 0x80) === 0) {
        return result;
      }
      shift += 7n;
      if (shift >= 70n) {
        throw new Error('Invalid protobuf payload: varint too long');
      }
    }
    throw new Error('Invalid protobuf payload: unexpected EOF in varint');
  }

  readTag(): { fieldNumber: number; wireType: number } {
    const tag = this.readVarint();
    const fieldNumber = Number(tag >> 3n);
    const wireType = Number(tag & 0x07n);
    if (fieldNumber <= 0) {
      throw new Error(
        `Invalid protobuf payload: invalid field number ${fieldNumber}`
      );
    }
    return { fieldNumber, wireType };
  }

  readUint32(wireType: number): number {
    if (wireType === 0) {
      return Number(BigInt.asUintN(32, this.readVarint()));
    }
    if (wireType === 5) {
      if (this.pos + 4 > this.len) {
        throw new Error('Invalid protobuf payload: unexpected EOF in fixed32');
      }
      const val = this.view.getUint32(this.pos, true);
      this.pos += 4;
      return val;
    }
    throw new Error(
      `Invalid protobuf payload: expected uint32 wire type, got ${wireType}`
    );
  }

  readInt64Number(wireType: number): number {
    if (wireType !== 0) {
      throw new Error(
        `Invalid protobuf payload: expected varint wire type, got ${wireType}`
      );
    }
    return Number(BigInt.asIntN(64, this.readVarint()));
  }

  readFixed64String(wireType: number): string {
    if (wireType === 1) {
      if (this.pos + 8 > this.len) {
        throw new Error('Invalid protobuf payload: unexpected EOF in fixed64');
      }
      const val = this.view.getBigUint64(this.pos, true);
      this.pos += 8;
      return val.toString();
    }
    if (wireType === 0) {
      return BigInt.asUintN(64, this.readVarint()).toString();
    }
    throw new Error(
      `Invalid protobuf payload: expected fixed64 wire type, got ${wireType}`
    );
  }

  readDouble(wireType: number): number {
    if (wireType !== 1) {
      throw new Error(
        `Invalid protobuf payload: expected 64-bit wire type for double, got ${wireType}`
      );
    }
    if (this.pos + 8 > this.len) {
      throw new Error('Invalid protobuf payload: unexpected EOF in double');
    }
    const val = this.view.getFloat64(this.pos, true);
    this.pos += 8;
    return val;
  }

  readBytes(wireType: number): Uint8Array {
    if (wireType !== 2) {
      throw new Error(
        `Invalid protobuf payload: expected length-delimited wire type, got ${wireType}`
      );
    }
    const length = Number(this.readVarint());
    if (length < 0 || this.pos + length > this.len) {
      throw new Error('Invalid protobuf payload: length out of bounds');
    }
    const slice = this.buf.subarray(this.pos, this.pos + length);
    this.pos += length;
    return slice;
  }

  readString(wireType: number): string {
    return textDecoder.decode(this.readBytes(wireType));
  }

  skipField(wireType: number, fieldNumber?: number): void {
    switch (wireType) {
      case 0:
        this.readVarint();
        break;
      case 1:
        if (this.pos + 8 > this.len) {
          throw new Error(
            'Invalid protobuf payload: unexpected EOF in 64-bit field'
          );
        }
        this.pos += 8;
        break;
      case 2:
        this.readBytes(2);
        break;
      case 3:
        while (!this.eof()) {
          const { fieldNumber: innerField, wireType: innerWire } =
            this.readTag();
          if (innerWire === 4) {
            if (fieldNumber !== undefined && innerField !== fieldNumber) {
              throw new Error(
                `Invalid protobuf payload: mismatched end-group field number ${innerField}, expected ${fieldNumber}`
              );
            }
            return;
          }
          this.skipField(innerWire, innerField);
        }
        throw new Error('Invalid protobuf payload: unexpected EOF in group');
      case 5:
        if (this.pos + 4 > this.len) {
          throw new Error(
            'Invalid protobuf payload: unexpected EOF in 32-bit field'
          );
        }
        this.pos += 4;
        break;
      default:
        throw new Error(
          `Invalid protobuf payload: unknown wire type ${wireType}`
        );
    }
  }
}

function decodeAnyValue(bytes: Uint8Array): OtlpValue {
  const reader = new ProtoReader(bytes);
  const value: OtlpValue = {};
  while (!reader.eof()) {
    const { fieldNumber, wireType } = reader.readTag();
    switch (fieldNumber) {
      case 1:
        value.stringValue = reader.readString(wireType);
        break;
      case 2:
        if (wireType !== 0) {
          throw new Error(
            `Invalid protobuf payload: expected varint for bool_value, got ${wireType}`
          );
        }
        value.boolValue = reader.readVarint() !== 0n;
        break;
      case 3:
        value.intValue = reader.readInt64Number(wireType);
        break;
      case 4:
        value.doubleValue = reader.readDouble(wireType);
        break;
      case 5:
        value.arrayValue = decodeArrayValue(reader.readBytes(wireType));
        break;
      case 6:
        value.kvlistValue = decodeKeyValueList(reader.readBytes(wireType));
        break;
      case 7:
        value.bytesValue = Buffer.from(reader.readBytes(wireType)).toString(
          'base64'
        );
        break;
      default:
        reader.skipField(wireType, fieldNumber);
        break;
    }
  }
  return value;
}

function decodeArrayValue(bytes: Uint8Array): { values: OtlpValue[] } {
  const reader = new ProtoReader(bytes);
  const values: OtlpValue[] = [];
  while (!reader.eof()) {
    const { fieldNumber, wireType } = reader.readTag();
    if (fieldNumber === 1) {
      values.push(decodeAnyValue(reader.readBytes(wireType)));
    } else {
      reader.skipField(wireType, fieldNumber);
    }
  }
  return { values };
}

function decodeKeyValueList(bytes: Uint8Array): { values: OtlpAttribute[] } {
  const reader = new ProtoReader(bytes);
  const values: OtlpAttribute[] = [];
  while (!reader.eof()) {
    const { fieldNumber, wireType } = reader.readTag();
    if (fieldNumber === 1) {
      values.push(decodeKeyValue(reader.readBytes(wireType)));
    } else {
      reader.skipField(wireType, fieldNumber);
    }
  }
  return { values };
}

function decodeKeyValue(bytes: Uint8Array): OtlpAttribute {
  const reader = new ProtoReader(bytes);
  let key = '';
  let value: OtlpValue = {};
  while (!reader.eof()) {
    const { fieldNumber, wireType } = reader.readTag();
    switch (fieldNumber) {
      case 1:
        key = reader.readString(wireType);
        break;
      case 2:
        value = decodeAnyValue(reader.readBytes(wireType));
        break;
      default:
        reader.skipField(wireType, fieldNumber);
        break;
    }
  }
  return { key, value };
}

function decodeResource(bytes: Uint8Array): OtlpResourceSpan['resource'] {
  const reader = new ProtoReader(bytes);
  const attributes: OtlpAttribute[] = [];
  let droppedAttributesCount = 0;
  while (!reader.eof()) {
    const { fieldNumber, wireType } = reader.readTag();
    switch (fieldNumber) {
      case 1:
        attributes.push(decodeKeyValue(reader.readBytes(wireType)));
        break;
      case 2:
        droppedAttributesCount = reader.readUint32(wireType);
        break;
      default:
        reader.skipField(wireType, fieldNumber);
        break;
    }
  }
  return { attributes, droppedAttributesCount };
}

function decodeScope(bytes: Uint8Array): OtlpScopeSpan['scope'] {
  const reader = new ProtoReader(bytes);
  let name = '';
  let version: string | undefined;
  while (!reader.eof()) {
    const { fieldNumber, wireType } = reader.readTag();
    switch (fieldNumber) {
      case 1:
        name = reader.readString(wireType);
        break;
      case 2:
        version = reader.readString(wireType);
        break;
      default:
        reader.skipField(wireType, fieldNumber);
        break;
    }
  }
  return { name, version };
}

function decodeStatus(bytes: Uint8Array): NonNullable<OtlpSpan['status']> {
  const reader = new ProtoReader(bytes);
  let code = 0;
  let message: string | undefined;
  while (!reader.eof()) {
    const { fieldNumber, wireType } = reader.readTag();
    switch (fieldNumber) {
      case 2: {
        const msg = reader.readString(wireType);
        if (msg) message = msg;
        break;
      }
      case 3:
        code = reader.readUint32(wireType);
        break;
      default:
        reader.skipField(wireType, fieldNumber);
        break;
    }
  }
  return message !== undefined ? { code, message } : { code };
}

function decodeSpan(bytes: Uint8Array): OtlpSpan {
  const reader = new ProtoReader(bytes);
  let traceId = '';
  let spanId = '';
  let parentSpanId: string | undefined;
  let name = '';
  let kind = 0;
  let startTimeUnixNano = '0';
  let endTimeUnixNano = '0';
  const attributes: OtlpAttribute[] = [];
  let droppedAttributesCount = 0;
  let droppedEventsCount = 0;
  let status: OtlpSpan['status'];
  let droppedLinksCount = 0;

  while (!reader.eof()) {
    const { fieldNumber, wireType } = reader.readTag();
    switch (fieldNumber) {
      case 1:
        traceId = Buffer.from(reader.readBytes(wireType)).toString('hex');
        break;
      case 2:
        spanId = Buffer.from(reader.readBytes(wireType)).toString('hex');
        break;
      case 4: {
        const parentBytes = reader.readBytes(wireType);
        if (parentBytes.length > 0 && parentBytes.some((b) => b !== 0)) {
          parentSpanId = Buffer.from(parentBytes).toString('hex');
        }
        break;
      }
      case 5:
        name = reader.readString(wireType);
        break;
      case 6:
        kind = reader.readUint32(wireType);
        break;
      case 7:
        startTimeUnixNano = reader.readFixed64String(wireType);
        break;
      case 8:
        endTimeUnixNano = reader.readFixed64String(wireType);
        break;
      case 9:
        attributes.push(decodeKeyValue(reader.readBytes(wireType)));
        break;
      case 10:
        droppedAttributesCount = reader.readUint32(wireType);
        break;
      case 12:
        droppedEventsCount = reader.readUint32(wireType);
        break;
      case 14:
        droppedLinksCount = reader.readUint32(wireType);
        break;
      case 15:
        status = decodeStatus(reader.readBytes(wireType));
        break;
      default:
        reader.skipField(wireType, fieldNumber);
        break;
    }
  }

  return {
    traceId,
    spanId,
    parentSpanId,
    name,
    kind,
    startTimeUnixNano,
    endTimeUnixNano,
    attributes,
    droppedAttributesCount,
    events: [],
    droppedEventsCount,
    status,
    links: [],
    droppedLinksCount,
  };
}

function decodeScopeSpan(bytes: Uint8Array): OtlpScopeSpan {
  const reader = new ProtoReader(bytes);
  let scope: OtlpScopeSpan['scope'] = { name: '', version: undefined };
  const spans: OtlpSpan[] = [];
  while (!reader.eof()) {
    const { fieldNumber, wireType } = reader.readTag();
    switch (fieldNumber) {
      case 1:
        scope = decodeScope(reader.readBytes(wireType));
        break;
      case 2:
        spans.push(decodeSpan(reader.readBytes(wireType)));
        break;
      default:
        reader.skipField(wireType, fieldNumber);
        break;
    }
  }
  return { scope, spans };
}

function decodeResourceSpan(bytes: Uint8Array): OtlpResourceSpan {
  const reader = new ProtoReader(bytes);
  let resource: OtlpResourceSpan['resource'] = {
    attributes: [],
    droppedAttributesCount: 0,
  };
  const scopeSpans: OtlpScopeSpan[] = [];
  while (!reader.eof()) {
    const { fieldNumber, wireType } = reader.readTag();
    switch (fieldNumber) {
      case 1:
        resource = decodeResource(reader.readBytes(wireType));
        break;
      case 2:
        scopeSpans.push(decodeScopeSpan(reader.readBytes(wireType)));
        break;
      default:
        reader.skipField(wireType, fieldNumber);
        break;
    }
  }
  return { resource, scopeSpans };
}

function decodeLogRecord(bytes: Uint8Array): OtlpLogRecord {
  const reader = new ProtoReader(bytes);
  let timeUnixNano: string | undefined;
  let observedTimeUnixNano: string | undefined;
  let severityNumber: number | undefined;
  let severityText: string | undefined;
  let body: OtlpValue | undefined;
  const attributes: OtlpAttribute[] = [];
  let traceId: string | undefined;
  let spanId: string | undefined;

  while (!reader.eof()) {
    const { fieldNumber, wireType } = reader.readTag();
    switch (fieldNumber) {
      case 1:
        timeUnixNano = reader.readFixed64String(wireType);
        break;
      case 2:
        severityNumber = reader.readUint32(wireType);
        break;
      case 3:
        severityText = reader.readString(wireType);
        break;
      case 5:
        body = decodeAnyValue(reader.readBytes(wireType));
        break;
      case 6:
        attributes.push(decodeKeyValue(reader.readBytes(wireType)));
        break;
      case 9: {
        const traceBytes = reader.readBytes(wireType);
        if (traceBytes.length > 0 && traceBytes.some((b) => b !== 0)) {
          traceId = Buffer.from(traceBytes).toString('hex');
        }
        break;
      }
      case 10: {
        const spanBytes = reader.readBytes(wireType);
        if (spanBytes.length > 0 && spanBytes.some((b) => b !== 0)) {
          spanId = Buffer.from(spanBytes).toString('hex');
        }
        break;
      }
      case 11:
        observedTimeUnixNano = reader.readFixed64String(wireType);
        break;
      default:
        reader.skipField(wireType, fieldNumber);
        break;
    }
  }

  return {
    timeUnixNano: timeUnixNano ?? '0',
    observedTimeUnixNano,
    severityNumber,
    severityText,
    body,
    attributes,
    traceId,
    spanId,
  };
}

function decodeScopeLog(bytes: Uint8Array): OtlpScopeLog {
  const reader = new ProtoReader(bytes);
  let scope: OtlpScopeLog['scope'] = { name: '', version: undefined };
  const logRecords: OtlpLogRecord[] = [];
  while (!reader.eof()) {
    const { fieldNumber, wireType } = reader.readTag();
    switch (fieldNumber) {
      case 1:
        scope = decodeScope(reader.readBytes(wireType));
        break;
      case 2:
        logRecords.push(decodeLogRecord(reader.readBytes(wireType)));
        break;
      default:
        reader.skipField(wireType, fieldNumber);
        break;
    }
  }
  return { scope, logRecords };
}

function decodeResourceLog(bytes: Uint8Array): OtlpResourceLog {
  const reader = new ProtoReader(bytes);
  let resource: OtlpResourceLog['resource'] = {
    attributes: [],
    droppedAttributesCount: 0,
  };
  const scopeLogs: OtlpScopeLog[] = [];
  while (!reader.eof()) {
    const { fieldNumber, wireType } = reader.readTag();
    switch (fieldNumber) {
      case 1:
        resource = decodeResource(reader.readBytes(wireType));
        break;
      case 2:
        scopeLogs.push(decodeScopeLog(reader.readBytes(wireType)));
        break;
      default:
        reader.skipField(wireType, fieldNumber);
        break;
    }
  }
  return { resource, scopeLogs };
}

/**
 * Decodes an OTLP protobuf wire payload (`ExportTraceServiceRequest` or
 * `ExportLogsServiceRequest`) into an {@link OtlpPayload}.
 */
export function decodeOtlpProtobuf(
  buf: Uint8Array,
  signal: 'traces' | 'logs'
): OtlpPayload {
  const reader = new ProtoReader(buf);
  const resourceSpans: OtlpResourceSpan[] = [];
  const resourceLogs: OtlpResourceLog[] = [];

  while (!reader.eof()) {
    const { fieldNumber, wireType } = reader.readTag();
    if (fieldNumber === 1 && wireType === 2) {
      const entryBytes = reader.readBytes(wireType);
      if (signal === 'logs') {
        resourceLogs.push(decodeResourceLog(entryBytes));
      } else {
        resourceSpans.push(decodeResourceSpan(entryBytes));
      }
    } else {
      reader.skipField(wireType, fieldNumber);
    }
  }

  const payload: OtlpPayload = {};
  if (resourceSpans.length > 0) {
    payload.resourceSpans = resourceSpans;
  }
  if (resourceLogs.length > 0) {
    payload.resourceLogs = resourceLogs;
  }
  return payload;
}

export function fromOtlpValue(value: OtlpValue): any {
  if (!value) return undefined;
  if (value.stringValue !== undefined) return value.stringValue;
  if (value.intValue !== undefined) return value.intValue;
  if (value.boolValue !== undefined) return value.boolValue;
  if (value.doubleValue !== undefined) return value.doubleValue;
  if (value.arrayValue !== undefined)
    return (value.arrayValue.values ?? []).map(fromOtlpValue);
  if (value.kvlistValue !== undefined) {
    const obj: Record<string, any> = {};
    for (const kv of value.kvlistValue.values ?? []) {
      const val = fromOtlpValue(kv.value);
      if (val !== undefined) {
        obj[kv.key] = val;
      }
    }
    return obj;
  }
  if (value.bytesValue !== undefined) return value.bytesValue;
  return undefined;
}

function toMillis(nano: string | undefined): number {
  if (!nano) return 0;
  const parsed = parseInt(nano, 10);
  return Number.isNaN(parsed) ? 0 : Math.round(parsed / 1_000_000);
}

function toSpanData(span: OtlpSpan, scope?: OtlpScopeSpan['scope']): SpanData {
  const attributes: Record<string, any> = {};
  (span.attributes ?? []).forEach((attr) => {
    const val = fromOtlpValue(attr.value);
    if (val !== undefined) {
      attributes[attr.key] = val;
    }
  });

  let spanKind: string;
  switch (span.kind) {
    case 1:
      spanKind = 'INTERNAL';
      break;
    case 2:
      spanKind = 'SERVER';
      break;
    case 3:
      spanKind = 'CLIENT';
      break;
    case 4:
      spanKind = 'PRODUCER';
      break;
    case 5:
      spanKind = 'CONSUMER';
      break;
    default:
      spanKind = 'UNSPECIFIED';
      break;
  }

  const spanData: SpanData = {
    traceId: span.traceId,
    spanId: span.spanId,
    parentSpanId: span.parentSpanId,
    startTime: toMillis(span.startTimeUnixNano),
    endTime: toMillis(span.endTimeUnixNano),
    displayName: span.name,
    attributes,
    instrumentationLibrary: {
      name: scope?.name ?? '',
      version: scope?.version,
    },
    spanKind,
  };
  if (span.status && span.status.code !== 0) {
    const status: { code: number; message?: string } = {
      code: span.status.code,
    };
    if (span.status.message) {
      status.message = span.status.message;
    }
    spanData.status = status;
  }
  return spanData;
}

export function traceDataFromOtlp(otlpData: OtlpPayload): TraceData[] {
  const traces: Record<string, TraceData> = {};

  if (otlpData.resourceSpans) {
    otlpData.resourceSpans.forEach((resourceSpan) => {
      (resourceSpan.scopeSpans ?? []).forEach((scopeSpan) => {
        (scopeSpan.spans ?? []).forEach((span) => {
          if (!traces[span.traceId]) {
            traces[span.traceId] = {
              traceId: span.traceId,
              spans: {},
            };
          }
          traces[span.traceId].spans[span.spanId] = toSpanData(
            span,
            scopeSpan.scope
          );
        });
      });
    });
  }

  return Object.values(traces);
}

function toLogRecordData(
  log: OtlpLogRecord,
  scope?: OtlpScopeLog['scope']
): LogRecordData {
  const attributes: Record<string, any> = {};
  if (log.attributes) {
    log.attributes.forEach((attr) => {
      const val = fromOtlpValue(attr.value);
      if (val !== undefined) {
        attributes[attr.key] = val;
      }
    });
  }

  let body: any = undefined;
  if (log.body) {
    body = fromOtlpValue(log.body);
  }

  return {
    logId: '', // Server will populate this if empty
    traceId: log.traceId,
    spanId: log.spanId,
    timestamp: toMillis(
      log.timeUnixNano && log.timeUnixNano !== '0'
        ? log.timeUnixNano
        : log.observedTimeUnixNano
    ),
    severityNumber: log.severityNumber,
    severityText: log.severityText,
    body,
    attributes,
    instrumentationLibrary: {
      name: scope?.name ?? '',
      version: scope?.version,
    },
  };
}

export function logDataFromOtlp(otlpData: OtlpPayload): LogRecordData[] {
  const logs: LogRecordData[] = [];
  if (!otlpData.resourceLogs) return logs;

  otlpData.resourceLogs.forEach((resourceLog) => {
    (resourceLog.scopeLogs ?? []).forEach((scopeLog) => {
      (scopeLog.logRecords ?? []).forEach((log) => {
        logs.push(toLogRecordData(log, scopeLog.scope));
      });
    });
  });

  return logs;
}
