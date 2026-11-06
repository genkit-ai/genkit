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

import type { NestedSpanData, Part, TraceData } from '@genkit-ai/tools-common';
import { MessageSchema } from '@genkit-ai/tools-common';
import {
  formatDuration,
  getSpanStatus,
  getSpanType,
  parseAndSanitizeJson,
  stackTraceSpans,
} from '@genkit-ai/tools-common/utils';
import YAML from 'yaml';
import { z } from 'zod';

type FormattedPart = {
  type: 'toolCall' | 'toolResponse' | 'text' | 'media' | 'other';
  text: string;
};

const COMPACT_VALUE = Symbol('compactValue');

type CompactFormattedValue = {
  [COMPACT_VALUE]: string | string[];
};

function compactValue(value: string | string[]): CompactFormattedValue {
  return { [COMPACT_VALUE]: value };
}

function isCompactValue(val: unknown): val is CompactFormattedValue {
  return Boolean(val && typeof val === 'object' && COMPACT_VALUE in val);
}

function isEmptyPlainObject(val: unknown): boolean {
  return (
    Boolean(val) &&
    typeof val === 'object' &&
    !Array.isArray(val) &&
    !isCompactValue(val) &&
    Object.keys(val as Record<string, unknown>).length === 0
  );
}

const USAGE_NON_ZERO_FIELDS = [
  'inputTokens',
  'outputTokens',
  'totalTokens',
  'thoughtsTokens',
  'cachedContentTokens',
  'cacheWriteTokens',
  'inputImages',
  'outputImages',
  'inputVideos',
  'outputVideos',
  'inputAudioFiles',
  'outputAudioFiles',
] as const;

function formatCompactUsage(usage: Record<string, unknown>): string | null {
  const parts: string[] = [];
  for (const field of USAGE_NON_ZERO_FIELDS) {
    const val = usage[field];
    if (typeof val === 'number' && val > 0) {
      parts.push(`${field}: ${val}`);
    }
  }
  if (parts.length === 0) {
    for (const charField of ['inputCharacters', 'outputCharacters'] as const) {
      const val = usage[charField];
      if (typeof val === 'number' && val > 0) {
        parts.push(`${charField}: ${val}`);
      }
    }
  }
  return parts.length > 0 ? parts.join(', ') : null;
}

/**
 * Formats a single Part object into a human-readable FormattedPart.
 */
function formatPart(part: Part): FormattedPart | null {
  if (part.toolRequest) {
    const tr = part.toolRequest;
    const argsStr = JSON.stringify(tr.input || {});
    return { type: 'toolCall', text: `Tool Call: ${tr.name}(${argsStr})` };
  }
  if (part.toolResponse) {
    const tr = part.toolResponse;
    const outStr =
      typeof tr.output === 'string'
        ? tr.output
        : JSON.stringify(tr.output || {});
    return {
      type: 'toolResponse',
      text: `Tool Output [${tr.name}]: ${outStr}`,
    };
  }
  if (part.text !== undefined && part.text !== null) {
    return { type: 'text', text: part.text };
  }
  if (part.media) {
    return {
      type: 'media',
      text: `[Media: ${part.media.contentType || 'file'} ${part.media.url}]`,
    };
  }
  if (part.reasoning) {
    return { type: 'text', text: `Reasoning: ${part.reasoning}` };
  }
  if (part.resource) {
    return { type: 'other', text: `[Resource: ${part.resource.uri}]` };
  }
  return null;
}

/**
 * Formats an entire message's content array into an array of strings,
 * including its role (User, System, Tool, Model).
 */
function formatMessageContent(
  msg: z.infer<typeof MessageSchema>
): string[] | null {
  const role = msg.role;

  const formattedParts = msg.content
    .map(formatPart)
    .filter((p): p is FormattedPart => p !== null);

  const rolePrefix =
    role === 'user'
      ? 'User: '
      : role === 'system'
        ? 'System: '
        : role === 'tool'
          ? 'Tool: '
          : 'Model: ';

  if (formattedParts.length === 0) {
    return [`${rolePrefix}(empty)`];
  }

  // Check for tool call(s) first
  const toolCalls = formattedParts.filter((p) => p.type === 'toolCall');
  if (toolCalls.length > 0) {
    return toolCalls.map((p) => p.text);
  }

  // Check for tool response(s) next
  const toolResps = formattedParts.filter((p) => p.type === 'toolResponse');
  if (toolResps.length > 0) {
    return toolResps.map((p) => p.text);
  }

  // Text / media content
  const texts = formattedParts.map((p) => p.text);
  const fullText = texts.join('\n');

  const lines = fullText.split(/\r?\n/);
  if (lines.length === 1) {
    return [`${rolePrefix}"${lines[0]}"`];
  } else {
    const res = [`${rolePrefix}`];
    lines.forEach((l: string) => res.push(`  ${l}`));
    return res;
  }
}

function hasSpecialFormatting(val: unknown): boolean {
  if (!val || typeof val !== 'object') return false;
  if (isCompactValue(val)) return true;

  // Try to parse array elements directly
  if (Array.isArray(val)) {
    if (
      val.some(
        (item) => isCompactValue(item) || MessageSchema.safeParse(item).success
      )
    ) {
      return true;
    }
  }

  if (MessageSchema.safeParse(val).success) {
    return true;
  }
  return Object.values(val).some(hasSpecialFormatting);
}

/**
 * Recursively formats unknown nested values (arrays, objects, primitives)
 * into an array of compact text lines for tree rendering.
 */
function formatCompactValue(
  val: unknown,
  childPrefix: string = '',
  keyName?: string
): string[] {
  if (val === undefined || val === null) return [];

  if (isCompactValue(val)) {
    const inner = val[COMPACT_VALUE];
    if (typeof inner === 'string') {
      return keyName
        ? [`${childPrefix}${keyName}: ${inner}`]
        : [`${childPrefix}${inner}`];
    }
    if (keyName) {
      return [
        `${childPrefix}${keyName}:`,
        ...inner.map((l) => `${childPrefix}  ${l}`),
      ];
    }
    return inner.map((l) => `${childPrefix}${l}`);
  }

  // 1. Matches MessageSchema -> format as Message
  const msgRes = MessageSchema.safeParse(val);
  if (msgRes.success) {
    const msgLines = formatMessageContent(msgRes.data);
    if (msgLines) {
      if (keyName) {
        return [
          `${childPrefix}${keyName}:`,
          ...msgLines.map((l) => `${childPrefix}  ${l}`),
        ];
      }
      return msgLines.map((l) => `${childPrefix}${l}`);
    }
  }

  // 2. Primitive (string, number, boolean)
  if (typeof val !== 'object') {
    const formatted = typeof val === 'string' ? val : String(val);
    if (keyName) {
      return [`${childPrefix}${keyName}: ${formatted}`];
    }
    return [`${childPrefix}${formatted}`];
  }

  // 3. Array -> Recurse on items
  if (Array.isArray(val)) {
    if (val.length === 0) {
      if (keyName) return [`${childPrefix}${keyName}: []`];
      return [];
    }
    const lines: string[] = [];
    if (keyName) {
      lines.push(`${childPrefix}${keyName}:`);
    }
    const itemPrefix = keyName ? `${childPrefix}  ` : childPrefix;

    const isAllMessages = val.every(
      (item) => MessageSchema.safeParse(item).success
    );
    if (isAllMessages) {
      val.forEach((item) => {
        lines.push(...formatCompactValue(item, itemPrefix));
      });
      return lines;
    }

    if (!hasSpecialFormatting(val)) {
      const yamlStr = YAML.stringify(val, {
        indent: 2,
        lineWidth: 0,
      }).trimEnd();
      yamlStr.split(/\r?\n/).forEach((l) => lines.push(`${itemPrefix}${l}`));
      return lines;
    }

    val.forEach((item) => {
      const itemLines = formatCompactValue(item, itemPrefix + '  ');
      if (itemLines.length > 0) {
        itemLines[0] =
          itemPrefix + '- ' + itemLines[0].slice(itemPrefix.length + 2);
        lines.push(...itemLines);
      }
    });
    return lines;
  }

  // 4. Object -> Recurse on properties if it contains messages or compact values
  const obj = val as Record<string, unknown>;
  const keys = Object.keys(obj);
  if (keys.length === 0) {
    if (keyName) return [`${childPrefix}${keyName}: {}`];
    return [];
  }

  const containsSpecial = hasSpecialFormatting(obj);

  const lines: string[] = [];
  if (keyName) {
    lines.push(`${childPrefix}${keyName}:`);
  }
  const propPrefix = keyName ? `${childPrefix}  ` : childPrefix;

  if (containsSpecial) {
    for (const [k, v] of Object.entries(obj)) {
      if (v === undefined || v === null) continue;
      lines.push(...formatCompactValue(v, propPrefix, k));
    }
  } else {
    const yamlStr = YAML.stringify(val, { indent: 2, lineWidth: 0 }).trimEnd();
    yamlStr.split(/\r?\n/).forEach((l) => lines.push(`${propPrefix}${l}`));
  }

  return lines;
}

/**
 * Checks whether a span represents a `generate` util span or a `model` action span,
 * where verbose tool schemas, echoed requests, and zero-valued usage counters
 * should be compacted in tree view.
 */
function isGenerateOrModelSpan(
  span: NestedSpanData,
  sanitizedInput: unknown
): boolean {
  const type = getSpanType(span);
  if (type === 'model' || type === 'background-model' || type === 'generate') {
    return true;
  }
  if (type === 'util') {
    const attrs = span.attributes || {};
    const name = span.displayName || (attrs['genkit:name'] as string) || '';
    if (name === 'generate') {
      return true;
    }
    if (
      sanitizedInput &&
      typeof sanitizedInput === 'object' &&
      !Array.isArray(sanitizedInput) &&
      Array.isArray((sanitizedInput as Record<string, unknown>).messages)
    ) {
      return true;
    }
  }
  return false;
}

function summarizeToolItem(item: unknown): string | null {
  if (typeof item === 'string') {
    return item.startsWith('/tool/') ? item.slice('/tool/'.length) : item;
  }
  if (
    item &&
    typeof item === 'object' &&
    'name' in item &&
    typeof (item as { name: unknown }).name === 'string'
  ) {
    const name = (item as { name: string }).name;
    return name.startsWith('/tool/') ? name.slice('/tool/'.length) : name;
  }
  return null;
}

type MessageHistoryContext = {
  messages: unknown[];
  fromParent: boolean;
};

function compactGenerateOrModelInput(
  input: unknown,
  priorContext?: MessageHistoryContext
): unknown {
  if (!input || typeof input !== 'object' || Array.isArray(input)) {
    return input;
  }
  const obj = input as Record<string, unknown>;
  const result: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(obj)) {
    if (value === undefined || value === null) continue;
    if ((key === 'config' || key === 'output') && isEmptyPlainObject(value)) {
      continue;
    }
    if (
      (key === 'use' || key === 'docs' || key === 'resources') &&
      Array.isArray(value) &&
      value.length === 0
    ) {
      continue;
    }
    if (key === 'tools' && Array.isArray(value)) {
      if (value.length === 0) continue;
      const names = value.map(summarizeToolItem);
      if (names.every((n): n is string => n !== null)) {
        result[key] = compactValue(`[${names.join(', ')}]`);
        continue;
      }
    }
    if (
      key === 'messages' &&
      Array.isArray(value) &&
      priorContext &&
      priorContext.messages.length > 0
    ) {
      const priorMessages = priorContext.messages;
      let shared = 0;
      const maxLen = Math.min(value.length, priorMessages.length);
      while (
        shared < maxLen &&
        JSON.stringify(value[shared]) === JSON.stringify(priorMessages[shared])
      ) {
        shared++;
      }
      if (shared > 0) {
        const noun = shared === 1 ? 'message' : 'messages';
        if (shared === value.length) {
          const source = priorContext.fromParent ? 'parent' : 'previous turn';
          result[key] = compactValue(`[same ${shared} ${noun} as ${source}]`);
        } else {
          const deltaLines: string[] = [`... (${shared} earlier ${noun})`];
          for (const msg of value.slice(shared)) {
            deltaLines.push(...formatCompactValue(msg, ''));
          }
          result[key] = compactValue(deltaLines);
        }
        continue;
      }
    }
    result[key] = value;
  }

  return result;
}

function compactGenerateOrModelOutput(output: unknown): unknown {
  if (!output || typeof output !== 'object' || Array.isArray(output)) {
    return output;
  }
  const obj = output as Record<string, unknown>;
  const hasMessageOrCandidates = Boolean(
    (obj.message && typeof obj.message === 'object') ||
      (Array.isArray(obj.candidates) && obj.candidates.length > 0)
  );

  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(obj)) {
    if (value === undefined || value === null) continue;
    if (
      hasMessageOrCandidates &&
      (key === 'request' || key === 'custom' || key === 'raw')
    ) {
      continue;
    }
    if (
      (key === 'config' ||
        key === 'output' ||
        key === 'custom' ||
        key === 'raw') &&
      isEmptyPlainObject(value)
    ) {
      continue;
    }
    if (
      (key === 'usage' || key === 'totalUsage') &&
      value &&
      typeof value === 'object' &&
      !Array.isArray(value)
    ) {
      const summary = formatCompactUsage(value as Record<string, unknown>);
      if (summary) {
        result[key] = compactValue(summary);
      }
      continue;
    }
    if (key === 'candidates' && Array.isArray(value)) {
      result[key] = value.map((cand) => compactGenerateOrModelOutput(cand));
      continue;
    }
    result[key] = value;
  }

  return result;
}

/**
 * Recursively renders a span and its nested children into a hierarchical string
 * array, dynamically appending the span's Input and Output payloads alongside
 * its child spans, drawing appropriate box-drawing tree connectors.
 */
function renderSpanTree(
  span: NestedSpanData,
  prefix: string = '',
  isLast: boolean = true,
  priorHistory?: MessageHistoryContext
): string[] {
  const lines: string[] = [];
  const connector = isLast ? '└─ ' : '├─ ';
  const childPrefix = prefix + (isLast ? '   ' : '│  ');

  const type = getSpanType(span);
  const duration = formatDuration(span.startTime, span.endTime);
  const status = getSpanStatus(span);
  const name =
    span.displayName ||
    (span.attributes?.['genkit:name'] as string) ||
    span.spanId;

  let header = `${prefix}${connector}${name}`;
  if (type) header += ` (${type})`;
  if (duration) header += ` [${duration}]`;
  if (status) header += ` ${status}`;

  lines.push(header);

  const attrs = span.attributes || {};
  const rawInput = attrs['genkit:input'];
  const rawOutput = attrs['genkit:output'];
  const errorMessage =
    status === '✖' || span.status?.code === 2
      ? span.status?.message
      : undefined;
  const children = span.spans || [];

  const sanitizedInput =
    rawInput !== undefined
      ? parseAndSanitizeJson(rawInput, true /* keepBase64 */)
      : undefined;
  const isGenOrModel = isGenerateOrModelSpan(span, sanitizedInput);

  let childHistory: MessageHistoryContext | undefined = priorHistory
    ? { messages: priorHistory.messages, fromParent: false }
    : undefined;
  if (
    isGenOrModel &&
    sanitizedInput &&
    typeof sanitizedInput === 'object' &&
    !Array.isArray(sanitizedInput) &&
    Array.isArray((sanitizedInput as Record<string, unknown>).messages)
  ) {
    childHistory = {
      messages: (sanitizedInput as Record<string, unknown>)
        .messages as unknown[],
      fromParent: true,
    };
  }

  const items: {
    type: 'input' | 'output' | 'error' | 'child';
    data: any;
  }[] = [];
  if (rawInput !== undefined)
    items.push({ type: 'input', data: sanitizedInput });
  if (rawOutput !== undefined) items.push({ type: 'output', data: rawOutput });
  if (errorMessage) items.push({ type: 'error', data: errorMessage });
  children.forEach((c) => items.push({ type: 'child', data: c }));

  items.forEach((item, index) => {
    const itemIsLast = index === items.length - 1;
    const itemConnector = itemIsLast ? '└─ ' : '├─ ';
    const itemPrefix = childPrefix + (itemIsLast ? '   ' : '│  ');

    if (item.type === 'input') {
      const displayInput = isGenOrModel
        ? compactGenerateOrModelInput(item.data, priorHistory)
        : item.data;
      const compactInput = formatCompactValue(displayInput, '', 'Input');
      if (compactInput.length > 0) {
        lines.push(`${childPrefix}${itemConnector}${compactInput[0]}`);
        for (let i = 1; i < compactInput.length; i++) {
          lines.push(`${itemPrefix}${compactInput[i]}`);
        }
      }
    } else if (item.type === 'output') {
      const sanitizedOutput = parseAndSanitizeJson(
        item.data,
        true /* keepBase64 */
      );
      const displayOutput = isGenOrModel
        ? compactGenerateOrModelOutput(sanitizedOutput)
        : sanitizedOutput;
      const compactOutput = formatCompactValue(displayOutput, '', 'Output');
      if (compactOutput.length > 0) {
        lines.push(`${childPrefix}${itemConnector}${compactOutput[0]}`);
        for (let i = 1; i < compactOutput.length; i++) {
          lines.push(`${itemPrefix}${compactOutput[i]}`);
        }
      }
    } else if (item.type === 'error') {
      const compactError = formatCompactValue(item.data, '', 'Error');
      if (compactError.length > 0) {
        lines.push(`${childPrefix}${itemConnector}${compactError[0]}`);
        for (let i = 1; i < compactError.length; i++) {
          lines.push(`${itemPrefix}${compactError[i]}`);
        }
      }
    } else if (item.type === 'child') {
      const childSpan = item.data as NestedSpanData;
      lines.push(
        ...renderSpanTree(childSpan, childPrefix, itemIsLast, childHistory)
      );
      // Support flat `generate` trees where sequential `model` spans are siblings
      const childRawIn = childSpan.attributes?.['genkit:input'];
      if (childRawIn !== undefined) {
        const childSanitizedIn = parseAndSanitizeJson(childRawIn, true);
        if (
          isGenerateOrModelSpan(childSpan, childSanitizedIn) &&
          childSanitizedIn &&
          typeof childSanitizedIn === 'object' &&
          !Array.isArray(childSanitizedIn) &&
          Array.isArray((childSanitizedIn as Record<string, unknown>).messages)
        ) {
          childHistory = {
            messages: (childSanitizedIn as Record<string, unknown>)
              .messages as unknown[],
            fromParent: false,
          };
        }
      }
    }
  });

  return lines;
}

/**
 * Formats a complete TraceData payload into a full execution tree string.
 * This sets up the trace metadata header and orchestrates rendering the root span tree.
 */
export function formatTraceTree(trace: TraceData): string {
  const lines: string[] = [];
  lines.push(`Trace ID: ${trace.traceId}`);
  if (trace.displayName) lines.push(`Name:     ${trace.displayName}`);
  if (trace.startTime) {
    lines.push(`Time:     ${new Date(trace.startTime).toLocaleString()}`);
    if (trace.endTime) {
      lines.push(`Duration: ${formatDuration(trace.startTime, trace.endTime)}`);
    }
  }

  const rootSpan = stackTraceSpans(trace);
  if (rootSpan) {
    lines.push('\nExecution Tree:');
    lines.push(...renderSpanTree(rootSpan, '', true));
  } else {
    lines.push('\nNo spans found in trace.');
  }

  return lines.join('\n');
}

/**
 * Creates a cloned, sanitized version of a trace for pure JSON output. Base64
 * media data inside span attributes can optionally be stripped for brevity.
 */
export function cleanTraceJson(
  trace: TraceData,
  keepBase64: boolean
): TraceData {
  const result: TraceData = JSON.parse(JSON.stringify(trace));
  if (result.spans) {
    for (const spanId of Object.keys(result.spans)) {
      const span = result.spans[spanId];
      if (span && span.attributes) {
        for (const [key, value] of Object.entries(span.attributes)) {
          span.attributes[key] = parseAndSanitizeJson(value, keepBase64);
        }
      }
    }
  }
  return result;
}
