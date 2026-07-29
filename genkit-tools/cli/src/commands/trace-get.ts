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
import { MessageSchema, PartSchema } from '@genkit-ai/tools-common';
import type { BaseRuntimeManager } from '@genkit-ai/tools-common/manager';
import {
  findProjectRoot,
  formatDuration,
  getSpanStatus,
  getSpanType,
  logger,
  parseAndSanitizeJson,
  stackTraceSpans,
} from '@genkit-ai/tools-common/utils';
import { yellow } from 'colorette';
import { Command } from 'commander';
import YAML from 'yaml';
import { z } from 'zod';
import { runWithManager } from '../utils/manager-utils';

/**
 * Options for the `trace:get` CLI command.
 */
export interface TraceGetOptions {
  /** Output format: 'tree' (default execution tree) or 'json' (parsed nested telemetry JSON). */
  format: 'tree' | 'json';
  /** If true, preserves raw base64 media data URLs instead of sanitizing them with placeholders. */
  keepMedia?: boolean;
}

type FormattedPart = {
  type: 'toolCall' | 'toolResponse' | 'text' | 'media' | 'other';
  text: string;
};

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
function formatMessageContent(msgData: unknown): string[] | null {
  const parseRes = MessageSchema.safeParse(msgData);
  if (!parseRes.success) return null;

  const msg = parseRes.data;
  const role = msg.role;

  const formattedParts = msg.content
    .map(formatPart)
    .filter((p): p is FormattedPart => p !== null);
  if (formattedParts.length === 0) return null;

  // Check for tool call first
  const toolCall = formattedParts.find((p) => p.type === 'toolCall');
  if (toolCall) {
    return [toolCall.text];
  }

  // Check for tool response next
  const toolResp = formattedParts.find((p) => p.type === 'toolResponse');
  if (toolResp) {
    return [toolResp.text];
  }

  // Text / media content
  const texts = formattedParts.map((p) => p.text);
  const fullText = texts.join('\n');
  const rolePrefix =
    role === 'user'
      ? 'User: '
      : role === 'system'
        ? 'System: '
        : role === 'tool'
          ? 'Tool: '
          : 'Model: ';

  const lines = fullText.split(/\r?\n/);
  if (lines.length === 1) {
    return [`${rolePrefix}"${lines[0]}"`];
  } else {
    const res = [`${rolePrefix}`];
    lines.forEach((l: string) => res.push(`  ${l}`));
    return res;
  }
}

/**
 * Formats an array of Parts or a single Part into an array of string
 * descriptions.
 */
function formatPartsList(partsData: unknown): string[] | null {
  const parseRes = z.array(PartSchema).safeParse(partsData);
  if (parseRes.success && parseRes.data.length > 0) {
    const formatted = parseRes.data
      .map(formatPart)
      .filter((p): p is FormattedPart => p !== null);
    if (formatted.length > 0) {
      return formatted.map((f) => f.text);
    }
  }
  const singlePartRes = PartSchema.safeParse(partsData);
  if (singlePartRes.success) {
    const formatted = formatPart(singlePartRes.data);
    if (formatted) return [formatted.text];
  }
  return null;
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

  // 2. Matches PartSchema or array of PartSchema -> format as Parts
  const partLines = formatPartsList(val);
  if (partLines) {
    if (keyName) {
      return [
        `${childPrefix}${keyName}:`,
        ...partLines.map((l) => `${childPrefix}  ${l}`),
      ];
    }
    return partLines.map((l) => `${childPrefix}${l}`);
  }

  // 3. Primitive (string, number, boolean)
  if (typeof val !== 'object') {
    const formatted = typeof val === 'string' ? val : String(val);
    if (keyName) {
      return [`${childPrefix}${keyName}: ${formatted}`];
    }
    return [`${childPrefix}${formatted}`];
  }

  // 4. Array -> Recurse on items
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

    const isAllPrimitives = val.every(
      (item) => typeof item !== 'object' || item === null
    );
    if (isAllPrimitives) {
      const yamlStr = YAML.stringify(val, {
        indent: 2,
        lineWidth: 0,
      }).trimEnd();
      yamlStr.split(/\r?\n/).forEach((l) => lines.push(`${itemPrefix}${l}`));
      return lines;
    }

    val.forEach((item) => {
      lines.push(...formatCompactValue(item, itemPrefix));
    });
    return lines;
  }

  // 5. Object -> Recurse on properties if it contains messages/parts
  const obj = val as Record<string, unknown>;
  const keys = Object.keys(obj);
  if (keys.length === 0) {
    if (keyName) return [`${childPrefix}${keyName}: {}`];
    return [];
  }

  const hasMessagesOrParts = keys.some(
    (k) =>
      k === 'messages' ||
      k === 'message' ||
      k === 'content' ||
      k === 'prompt' ||
      k === 'history' ||
      MessageSchema.safeParse(obj[k]).success ||
      PartSchema.safeParse(obj[k]).success
  );

  const lines: string[] = [];
  if (keyName) {
    lines.push(`${childPrefix}${keyName}:`);
  }
  const propPrefix = keyName ? `${childPrefix}  ` : childPrefix;

  if (hasMessagesOrParts) {
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
 * Recursively renders a span and its nested children into a hierarchical string
 * array, dynamically appending the span's Input and Output payloads alongside
 * its child spans, drawing appropriate box-drawing tree connectors.
 */
function renderSpanTree(
  span: NestedSpanData,
  prefix: string = '',
  isLast: boolean = true,
  keepMedia: boolean = false
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
  const children = span.spans || [];

  const items: { type: 'input' | 'output' | 'child'; data: any }[] = [];
  if (rawInput !== undefined) items.push({ type: 'input', data: rawInput });
  if (rawOutput !== undefined) items.push({ type: 'output', data: rawOutput });
  children.forEach((c) => items.push({ type: 'child', data: c }));

  items.forEach((item, index) => {
    const itemIsLast = index === items.length - 1;
    const itemConnector = itemIsLast ? '└─ ' : '├─ ';
    const itemPrefix = childPrefix + (itemIsLast ? '   ' : '│  ');

    if (item.type === 'input') {
      const sanitizedInput = parseAndSanitizeJson(item.data, keepMedia);
      const compactInput = formatCompactValue(sanitizedInput, '', 'Input');
      if (compactInput.length > 0) {
        lines.push(`${childPrefix}${itemConnector}${compactInput[0]}`);
        for (let i = 1; i < compactInput.length; i++) {
          lines.push(`${itemPrefix}${compactInput[i]}`);
        }
      }
    } else if (item.type === 'output') {
      const sanitizedOutput = parseAndSanitizeJson(item.data, keepMedia);
      const compactOutput = formatCompactValue(sanitizedOutput, '', 'Output');
      if (compactOutput.length > 0) {
        lines.push(`${childPrefix}${itemConnector}${compactOutput[0]}`);
        for (let i = 1; i < compactOutput.length; i++) {
          lines.push(`${itemPrefix}${compactOutput[i]}`);
        }
      }
    } else if (item.type === 'child') {
      lines.push(
        ...renderSpanTree(item.data, childPrefix, itemIsLast, keepMedia)
      );
    }
  });

  return lines;
}

/**
 * Formats a complete TraceData payload into a full execution tree string.
 * This sets up the trace metadata header and orchestrates rendering the root span tree.
 */
function formatTraceTree(trace: TraceData, keepMedia: boolean = false): string {
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
    lines.push(...renderSpanTree(rootSpan, '', true, keepMedia));
  } else {
    lines.push('\nNo spans found in trace.');
  }

  return lines.join('\n');
}

/**
 * Creates a cloned, sanitized version of a trace for pure JSON output. Base64
 * media data inside span attributes can optionally be stripped for brevity.
 */
function cleanTraceJson(trace: TraceData, keepMedia: boolean): any {
  const result: any = JSON.parse(JSON.stringify(trace));
  if (result.spans) {
    for (const spanId of Object.keys(result.spans)) {
      const span = result.spans[spanId];
      if (span.attributes) {
        for (const [key, value] of Object.entries(span.attributes)) {
          span.attributes[key] = parseAndSanitizeJson(value, keepMedia);
        }
      }
    }
  }
  return result;
}

/** Command to get a trace. */
export const traceGet = new Command('trace:get')
  .description('get a trace by id')
  .argument('<traceId>', 'id of the trace to get')
  .option('-f, --format <format>', 'output format (tree, json)', 'tree')
  .option('--keep-media', 'do not strip base64 data URLs in output', false)
  .action(async (traceId: string, options: TraceGetOptions) => {
    const projectRoot = await findProjectRoot();

    const runAction = async (manager: BaseRuntimeManager) => {
      try {
        const response = await manager.getTrace({ traceId });
        if (!response) {
          logger.error(`Trace with ID '${traceId}' not found.`);
          return;
        }

        const format = (options.format || 'tree').toLowerCase();
        const keepMedia = !!options.keepMedia;

        if (format === 'tree') {
          console.log(
            yellow(
              'Hint: pass `--format json` flag to get trace data in JSON format\n'
            )
          );
          console.log(formatTraceTree(response, keepMedia));
        } else if (format === 'json') {
          console.log(
            JSON.stringify(cleanTraceJson(response, keepMedia), undefined, 2)
          );
        } else {
          logger.error(
            `Unknown format '${options.format}'. Supported formats: tree, json.`
          );
        }
      } catch (e) {
        logger.error(`Error retrieving trace: ${e}`);
      }
    };

    await runWithManager(projectRoot, runAction);
  });
