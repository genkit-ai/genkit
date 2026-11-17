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

import type { RuntimeError } from '../../types/error.js';
import type {
  AgentInput,
  GenerationUsage,
  MessageData,
  NestedSpanData,
  Part,
  SpanData,
  ToolDefinition,
} from '../../types/index.js';
import {
  cleanActionName,
  getSpanName,
  getSpanStatus,
  isTruthyAttr,
  parseSpanJson,
  shortToolName,
} from '../trace.js';

const PROMPT_SPAN_TYPES: ReadonlySet<unknown> = new Set([
  'prompt',
  'promptTemplate',
]);

/**
 * Spans belonging to a single `Trajectory` (either the root trace/agent or an
 * individual nested `/agent/...` subagent), grouped by role during tree
 * traversal. Traversal stops at nested `/agent/...` boundaries so child
 * subagents are extracted into their own `Trajectory` entries.
 */
export interface TrajectorySpans {
  /** Maps each `spanId` in this trajectory to its parent span for ancestor/turn lookups. */
  parentMap: Map<string, NestedSpanData>;
  /** Model execution spans (`model` / `background-model`) in this trajectory. */
  modelSpans: NestedSpanData[];
  /** Tool execution spans (`tool` / `tool.v2`) in this trajectory. */
  toolSpans: NestedSpanData[];
  /** All spans belonging to this trajectory in traversal order. */
  scopeSpans: NestedSpanData[];
  subagentByToolSpanId: Map<string, string>;
}

/** Parsed structure of a model span's `genkit:input` JSON attribute. */
export interface ParsedModelInput {
  tools?: Partial<ToolDefinition>[];
  messages?: MessageData[];
}

/** Parsed structure of a span's `genkit:input` JSON attribute. */
export type ParsedSpanInput = ParsedModelInput | AgentInput;

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
  error?: RuntimeError;
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

function toRuntimeError(err: {
  status?: unknown;
  message: string;
  details?: unknown;
}): RuntimeError {
  return {
    ...(typeof err.status === 'string' ? { status: err.status } : {}),
    message: err.message,
    ...(err.details !== undefined ? { details: err.details } : {}),
  };
}

/** Extracts a `RuntimeError` from a span's structured `genkit:output.error` or failure status. */
export function extractSpanRuntimeError(
  span: SpanData,
  fallbackMessage: string,
  out?: ParsedSpanOutput
): RuntimeError | undefined {
  if (out?.error && typeof out.error.message === 'string') {
    return toRuntimeError(out.error);
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
      return toRuntimeError(rawErrAttr as RuntimeError);
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
    .filter((p): p is Part =>
      Boolean(p && typeof p === 'object' && !Array.isArray(p))
    );
}

/**
 * Computes a deterministic signature for a message's role and non-metadata content
 * so repeated history messages can be deduplicated.
 */
export function messageSignature(msg?: MessageData): string {
  if (!msg) return '';
  const simplifiedParts = normalizeParts(msg.content).map(
    ({ metadata: _metadata, ...rest }) => rest
  );
  return `${msg.role ?? ''}:${JSON.stringify(simplifiedParts)}`;
}

/** Checks if a message is explicitly tagged as history in its metadata. */
export function isExplicitHistoryMessage(msg?: MessageData): boolean {
  const meta = msg?.metadata;
  return Boolean(meta?.['_genkit_history'] || meta?.['purpose'] === 'history');
}
