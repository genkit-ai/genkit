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

import type {
  GenerationUsage,
  MessageData,
  NestedSpanData,
  Part,
  SpanData,
  ToolDefinition,
  ToolRequestPart,
  ToolResponsePart,
} from '../../types';
import {
  cleanActionName,
  getSpanName,
  getSpanStatus,
  isTruthyAttr,
  parseSpanJson,
  shortToolName,
} from '../trace';

const PROMPT_SPAN_TYPES: ReadonlySet<unknown> = new Set([
  'prompt',
  'promptTemplate',
]);

const PART_SIGNATURE_KEYS = [
  'text',
  'reasoning',
  'media',
  'toolRequest',
  'toolResponse',
  'data',
  'resource',
  'custom',
] as const;

/**
 * Partitioned spans and subagent mappings within a single trajectory scope
 * (the slice of the span tree rooted at `scopeRoot` down to, but excluding,
 * any nested child subagent spans, which each form their own `Trajectory`).
 */
export interface ScopePartition {
  parentMap: Map<string, NestedSpanData>;
  modelSpans: NestedSpanData[];
  toolSpans: NestedSpanData[];
  scopeSpans: NestedSpanData[];
  subagentByToolSpanId: Map<string, string>;
}

/** Parsed structure of a span's `genkit:input` JSON attribute. */
export interface ParsedSpanInput {
  tools?: Partial<ToolDefinition>[];
  messages?: MessageData[];
  message?: MessageData;
  resume?: {
    respond?: ToolResponsePart[];
    restart?: ToolRequestPart[];
  };
}

/** Parsed structure of a span's `genkit:output` JSON attribute. */
export interface ParsedSpanOutput {
  message?: MessageData;
  finishReason?: string;
  finishMessage?: string;
  usage?: GenerationUsage;
  candidates?: Array<{
    message?: MessageData;
    finishReason?: string;
    usage?: GenerationUsage;
  }>;
  error?: {
    message?: string;
    details?: { stack?: string };
  };
}

/** Normalized output extracted from a model execution span. */
export interface ModelOutputData {
  message?: MessageData;
  finishReason?: string;
  usage?: GenerationUsage;
  error?: { message: string; stack?: string };
}

/** Extracts a tool call reference ID from span attributes, if present. */
export function getToolCallId(span: SpanData): string | undefined {
  const id =
    span.attributes?.['genkit:metadata:tool:callId'] ??
    span.attributes?.['genkit:tool:callId'];
  return typeof id === 'string' && id.length > 0 ? id : undefined;
}

/**
 * Returns true if neither side specifies a conflicting tool call reference ID,
 * preventing name-based fallbacks from pairing mismatched parallel calls.
 */
export function hasCompatibleRef(
  reqRef?: string,
  candidateRef?: string
): boolean {
  return !reqRef || !candidateRef || reqRef === candidateRef;
}

/**
 * Removes and returns the first item matching `reqRef` by exact ref, or falling back
 * to a compatible ref with matching short tool name.
 */
export function takeMatchingByRefOrName<T>(
  items: T[],
  reqRef: string | undefined,
  reqShortName: string,
  getRef: (item: T) => string | undefined,
  getName: (item: T) => string
): T | undefined {
  let idx = reqRef ? items.findIndex((item) => getRef(item) === reqRef) : -1;
  if (idx < 0) {
    idx = items.findIndex(
      (item) =>
        hasCompatibleRef(reqRef, getRef(item)) &&
        shortToolName(getName(item)) === reqShortName
    );
  }
  return idx >= 0 ? items.splice(idx, 1)[0] : undefined;
}

/** Resolves the 0-based `turnIndex` and enclosing `runTurn-<N>` span for a span. */
export function getTurnContext(
  span: NestedSpanData,
  parentMap: ReadonlyMap<string, NestedSpanData>
): { turnIndex: number; turnSpan?: NestedSpanData } {
  let curr: NestedSpanData | undefined = span;
  while (curr) {
    const match = /^runTurn-(\d+)$/.exec(cleanActionName(getSpanName(curr)));
    if (match) {
      return {
        turnIndex: Math.max(0, Number.parseInt(match[1], 10) - 1),
        turnSpan: curr,
      };
    }
    curr = parentMap.get(curr.spanId);
  }
  return { turnIndex: 0, turnSpan: undefined };
}

function extractModelError(
  modelSpan: SpanData,
  out?: ParsedSpanOutput
): { message: string; stack?: string } | undefined {
  if (out?.error && typeof out.error.message === 'string') {
    const stack = out.error.details?.stack;
    return {
      message: out.error.message,
      ...(typeof stack === 'string' ? { stack } : {}),
    };
  }
  if (getSpanStatus(modelSpan) === '✖') {
    return {
      message:
        modelSpan.status?.message ||
        (modelSpan.attributes?.['genkit:error'] as string | undefined) ||
        out?.finishMessage ||
        'Model execution failed',
    };
  }
  return undefined;
}

/**
 * Extracts the output message, finish reason, token usage, and normalized error from a model span.
 */
export function getModelOutputData(modelSpan: SpanData): ModelOutputData {
  const out = parseSpanJson<ParsedSpanOutput>(
    modelSpan.attributes?.['genkit:output']
  );
  const candidate = out?.candidates?.[0];
  return {
    message: out?.message ?? candidate?.message,
    finishReason: out?.finishReason ?? candidate?.finishReason,
    usage: out?.usage ?? candidate?.usage,
    error: extractModelError(modelSpan, out),
  };
}

/**
 * Checks whether a span and its entire descendant subtree should be excluded from trajectory projection.
 */
export function isExcludedFromTrajectory(span: SpanData): boolean {
  const attrs = span.attributes;
  return (
    isTruthyAttr(attrs?.['genkit:metadata:genkit-dev-internal']) ||
    isTruthyAttr(attrs?.['genkit:metadata:trajectory:exclude']) ||
    PROMPT_SPAN_TYPES.has(attrs?.['genkit:metadata:subtype']) ||
    PROMPT_SPAN_TYPES.has(attrs?.['genkit:type'])
  );
}

/** Extracts text from an array of Parts. */
export function extractTextFromParts(parts: Part[] | undefined): string {
  if (!Array.isArray(parts)) return '';
  return parts
    .map((p) => (typeof p?.text === 'string' ? p.text : ''))
    .filter(Boolean)
    .join('\n')
    .trim();
}

/** Sums GenerationUsage records across multiple steps. */
export function aggregateUsage(
  usages: Array<GenerationUsage | undefined>
): GenerationUsage | undefined {
  const result: Record<string, unknown> = {};
  const custom: Record<string, number> = {};
  for (const u of usages) {
    if (!u) continue;
    for (const [k, v] of Object.entries(u)) {
      if (typeof v === 'number') {
        result[k] = ((result[k] as number) ?? 0) + v;
      } else if (k === 'custom' && v && typeof v === 'object') {
        for (const [ck, cv] of Object.entries(v as Record<string, unknown>)) {
          if (typeof cv === 'number') custom[ck] = (custom[ck] ?? 0) + cv;
        }
      }
    }
  }
  if (Object.keys(custom).length > 0) result.custom = custom;
  return Object.keys(result).length > 0
    ? (result as GenerationUsage)
    : undefined;
}

/** Normalizes a candidate message content array into valid Part objects. */
export function normalizeParts(content: unknown): Part[] {
  if (!Array.isArray(content)) {
    return typeof content === 'string' ? [{ text: content }] : [];
  }
  return content.filter((p): p is Part => Boolean(p && typeof p === 'object'));
}

/**
 * Computes a deterministic signature for a message's role and non-metadata content
 * so repeated history messages can be deduplicated.
 */
export function messageSignature(msg: MessageData): string {
  const simplifiedParts = normalizeParts(msg.content).map((p) => {
    const key = PART_SIGNATURE_KEYS.find((k) => p[k] !== undefined);
    return key ? { [key]: p[key] } : p;
  });
  return `${msg.role}:${JSON.stringify(simplifiedParts)}`;
}

/** Checks if a message is explicitly tagged as history in its metadata. */
export function isExplicitHistoryMessage(msg: MessageData): boolean {
  const meta = msg.metadata;
  return Boolean(meta?.['_genkit_history'] || meta?.['purpose'] === 'history');
}
