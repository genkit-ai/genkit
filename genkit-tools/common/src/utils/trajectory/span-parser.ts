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
  AgentFinishReason,
  GenerationUsage,
  MessageData,
  NestedSpanData,
  Part,
  SpanData,
  ToolDefinition,
  TrajectoryStep,
} from '../../types/index.js';
import {
  cleanActionName,
  getSpanName,
  getSpanStatus,
  isTruthyAttr,
  parseSpanJson,
  shortToolName,
} from '../trace.js';

type FinishReason = NonNullable<TrajectoryStep['finishReason']>;
type RuntimeError = NonNullable<TrajectoryStep['error']>;

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
 * Partitioned spans within a single trajectory scope.
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
}

/** Parsed structure of a span's `genkit:output` JSON attribute. */
export interface ParsedSpanOutput {
  message?: MessageData;
  finishReason?: FinishReason | AgentFinishReason | string;
  finishMessage?: string;
  usage?: GenerationUsage;
  candidates?: Array<{
    message?: MessageData;
    finishReason?: FinishReason | string;
    usage?: GenerationUsage;
  }>;
  error?: {
    status?: string;
    message?: string;
    details?: unknown;
  };
}

/** Normalized output extracted from a model execution span. */
export interface ModelOutputData {
  message?: MessageData;
  finishReason?: string;
  usage?: GenerationUsage;
  error?: RuntimeError;
}

/** Parses a span JSON attribute and narrows it to a non-null, non-array object. */
export function parseSpanObject<T extends object>(val: unknown): T | undefined {
  const parsed = parseSpanJson<T>(val);
  return parsed && typeof parsed === 'object' && !Array.isArray(parsed)
    ? (parsed as T)
    : undefined;
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
  const visited = new Set<string>();
  while (curr) {
    if (visited.has(curr.spanId)) {
      break;
    }
    visited.add(curr.spanId);
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

/** Extracts a `RuntimeError` from a span's structured `genkit:output.error` or failure status. */
export function extractSpanRuntimeError(
  span: SpanData,
  fallbackMessage: string,
  out?: ParsedSpanOutput
): RuntimeError | undefined {
  if (out?.error && typeof out.error.message === 'string') {
    return {
      ...(typeof out.error.status === 'string'
        ? { status: out.error.status }
        : {}),
      message: out.error.message,
      ...(out.error.details !== undefined
        ? { details: out.error.details }
        : {}),
    };
  }
  if (getSpanStatus(span) === '✖') {
    const rawErrAttr = parseSpanJson<unknown>(
      span.attributes?.['genkit:error']
    );
    if (
      rawErrAttr &&
      typeof rawErrAttr === 'object' &&
      'message' in rawErrAttr &&
      typeof (rawErrAttr as { message?: unknown }).message === 'string'
    ) {
      const errObj = rawErrAttr as RuntimeError;
      return {
        ...(typeof errObj.status === 'string' ? { status: errObj.status } : {}),
        message: errObj.message,
        ...(errObj.details !== undefined ? { details: errObj.details } : {}),
      };
    }
    return {
      message:
        span.status?.message ||
        (typeof rawErrAttr === 'string' ? rawErrAttr : undefined) ||
        out?.finishMessage ||
        fallbackMessage,
    };
  }
  return undefined;
}

/**
 * Extracts the output message, finish reason, token usage, and normalized error from a model span.
 */
export function getModelOutputData(modelSpan: SpanData): ModelOutputData {
  const out = parseSpanObject<ParsedSpanOutput>(
    modelSpan.attributes?.['genkit:output']
  );
  const candidate = out?.candidates?.[0];
  return {
    message: out?.message ?? candidate?.message,
    finishReason: out?.finishReason ?? candidate?.finishReason,
    usage: out?.usage ?? candidate?.usage,
    error: extractSpanRuntimeError(modelSpan, 'Model execution failed', out),
  };
}

/**
 * Checks whether a span and its entire descendant subtree should be excluded from trajectory projection.
 */
export function isExcludedFromTrajectory(span: SpanData): boolean {
  const attrs = span.attributes;
  return (
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
  return content
    .map((p) => (typeof p === 'string' ? { text: p } : p))
    .filter((p): p is Part => Boolean(p && typeof p === 'object'));
}

/**
 * Computes a deterministic signature for a message's role and non-metadata content
 * so repeated history messages can be deduplicated.
 */
export function messageSignature(msg?: MessageData): string {
  if (!msg) return '';
  const simplifiedParts = normalizeParts(msg.content).map((p) => {
    const key = PART_SIGNATURE_KEYS.find((k) => p[k] !== undefined);
    return key ? { [key]: p[key] } : p;
  });
  return `${msg.role ?? ''}:${JSON.stringify(simplifiedParts)}`;
}

/** Checks if a message is explicitly tagged as history in its metadata. */
export function isExplicitHistoryMessage(msg?: MessageData): boolean {
  const meta = msg?.metadata;
  return Boolean(meta?.['_genkit_history'] || meta?.['purpose'] === 'history');
}
