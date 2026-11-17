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
  NestedSpanData,
  Part,
  ToolRequestPart,
  ToolResponsePart,
  TrajectoryAction,
  TrajectoryStatus,
} from '../../types/index.js';
import {
  getSpanName,
  getSpanStatus,
  parseSpanJson,
  shortToolName,
} from '../trace.js';
import {
  extractSpanRuntimeError,
  getTurnContext,
  normalizeParts,
  takeMatchingByRefOrName,
  type ParsedSpanInput,
  type TrajectorySpans,
} from './span-parser.js';

/**
 * A `toolResponse` part extracted from `role: 'tool'` messages in model inputs,
 * used as a fallback output source when pairing an earlier `toolRequest`.
 */
export interface ToolResponseEntry {
  part: ToolResponsePart;
}

/** Shared mutable pairing state while projecting a trajectory. */
export interface ToolPairingContext {
  trajectorySpans: TrajectorySpans;
  claimedToolSpanIds: Set<string>;
  toolResponses: ToolResponseEntry[];
}

/**
 * Pre-collects unique `toolResponse` parts across span `input.messages` in a trajectory
 * (deduplicating repeated copies across cumulative multi-turn message histories)
 * so earlier `toolRequest` steps can look ahead and pair with their responses.
 */
export function collectToolResponses(
  inputs: ParsedSpanInput[]
): ToolResponseEntry[] {
  const responses: ToolResponseEntry[] = [];
  const seenKeys = new Set<string>();

  for (const input of inputs) {
    if (!input) continue;
    const messages =
      'messages' in input && input.messages
        ? input.messages
        : 'message' in input && input.message
          ? [input.message]
          : [];
    for (const msg of messages) {
      if (msg?.role !== 'tool' || !Array.isArray(msg.content)) continue;
      for (const part of msg.content) {
        if (!part?.toolResponse) continue;
        const key = `${part.toolResponse.ref || ''}:${part.toolResponse.name}:${JSON.stringify(part.toolResponse.output)}`;
        if (seenKeys.has(key)) continue;
        seenKeys.add(key);
        responses.push({ part: part as ToolResponsePart });
      }
    }
  }
  return responses;
}

function isToolV2Span(span: NestedSpanData): boolean {
  const attrs = span.attributes;
  return (
    attrs?.['genkit:metadata:subtype'] === 'tool.v2' ||
    attrs?.['genkit:type'] === 'tool.v2' ||
    getSpanName(span).startsWith('/tool.v2/')
  );
}

function parseToolSpanResult(span?: NestedSpanData): {
  output?: unknown;
  content?: Part[];
} {
  const parsed = parseSpanJson(span?.attributes?.['genkit:output']);
  if (
    span &&
    isToolV2Span(span) &&
    parsed &&
    typeof parsed === 'object' &&
    !Array.isArray(parsed) &&
    ('output' in parsed || 'content' in parsed || 'metadata' in parsed)
  ) {
    const envelope = parsed as { output?: unknown; content?: unknown };
    const content = Array.isArray(envelope.content)
      ? normalizeParts(envelope.content)
      : undefined;
    return {
      output: envelope.output,
      ...(content && content.length > 0 ? { content } : {}),
    };
  }
  return parsed !== undefined ? { output: parsed } : {};
}

function claimMatchingToolSpan(
  reqShortName: string,
  modelSpan: NestedSpanData,
  nextModelSpan: NestedSpanData | undefined,
  turnIndex: number,
  ctx: ToolPairingContext
): NestedSpanData | undefined {
  const { toolSpans, parentMap } = ctx.trajectorySpans;
  const nameMatched = toolSpans.filter(
    (ts) =>
      !ctx.claimedToolSpanIds.has(ts.spanId) &&
      shortToolName(getSpanName(ts)) === reqShortName
  );

  const matched =
    nameMatched.find(
      (ts) =>
        ts.startTime >= modelSpan.startTime &&
        (!nextModelSpan || ts.startTime <= nextModelSpan.startTime)
    ) ??
    nameMatched.find(
      (ts) => getTurnContext(ts, parentMap).turnIndex === turnIndex
    );

  if (matched) ctx.claimedToolSpanIds.add(matched.spanId);
  return matched;
}

function classifyToolAction(
  matchedSpan: NestedSpanData | undefined,
  matchedResp: ToolResponseEntry | undefined,
  finishReason: string | undefined
): {
  type: TrajectoryAction['type'];
  status: TrajectoryStatus;
  error?: RuntimeError;
} {
  if (matchedSpan && getSpanStatus(matchedSpan) === '✖') {
    return {
      type: 'tool',
      status: 'error',
      error: extractSpanRuntimeError(matchedSpan, 'Tool execution failed'),
    };
  }
  if (
    (matchedSpan && !matchedSpan.endTime) ||
    (!matchedSpan && !matchedResp && finishReason !== 'stop')
  ) {
    return { type: 'tool', status: 'in-progress' };
  }
  return { type: 'tool', status: 'success' };
}

/** Pairs all `toolRequest` parts of a model step into `TrajectoryAction` records. */
export function pairStepToolActions(
  toolRequestParts: ToolRequestPart[],
  modelSpan: NestedSpanData,
  nextModelSpan: NestedSpanData | undefined,
  turnIndex: number,
  finishReason: string | undefined,
  ctx: ToolPairingContext
): TrajectoryAction[] {
  if (!Array.isArray(toolRequestParts)) return [];
  return toolRequestParts
    .map((part, idx) => {
      if (!part?.toolRequest) return null;
      return buildSingleToolAction(
        part,
        idx,
        modelSpan,
        nextModelSpan,
        turnIndex,
        finishReason,
        ctx
      );
    })
    .filter((action): action is TrajectoryAction => action !== null);
}

function buildSingleToolAction(
  reqPart: ToolRequestPart,
  reqIdx: number,
  modelSpan: NestedSpanData,
  nextModelSpan: NestedSpanData | undefined,
  turnIndex: number,
  finishReason: string | undefined,
  ctx: ToolPairingContext
): TrajectoryAction {
  const req = reqPart.toolRequest;
  const reqShortName = shortToolName(req.name);
  const matchedSpan = claimMatchingToolSpan(
    reqShortName,
    modelSpan,
    nextModelSpan,
    turnIndex,
    ctx
  );
  const matchedResp = takeMatchingByRefOrName(
    ctx.toolResponses,
    req.ref,
    reqShortName,
    (r) => r.part.toolResponse.ref,
    (r) => r.part.toolResponse.name
  );

  const spanInput = parseSpanJson(matchedSpan?.attributes?.['genkit:input']);
  const spanResult = parseToolSpanResult(matchedSpan);
  const input = req.input !== undefined ? req.input : spanInput;
  const output =
    spanResult.output !== undefined
      ? spanResult.output
      : matchedResp?.part.toolResponse.output;
  const respContent = Array.isArray(matchedResp?.part.toolResponse.content)
    ? normalizeParts(matchedResp.part.toolResponse.content)
    : undefined;
  const content =
    spanResult.content ??
    (respContent && respContent.length > 0 ? respContent : undefined);

  const { type, status, error } = classifyToolAction(
    matchedSpan,
    matchedResp,
    finishReason
  );

  const startTime = matchedSpan?.startTime;
  const endTime = matchedSpan?.endTime || undefined;

  return {
    actionId: req.ref || `${modelSpan.spanId}-action-${reqIdx + 1}`,
    ...(matchedSpan ? { spanId: matchedSpan.spanId } : {}),
    name: req.name,
    type,
    status,
    ...(input !== undefined ? { input } : {}),
    ...(output !== undefined ? { output } : {}),
    ...(content ? { content } : {}),
    ...(error ? { error } : {}),
    ...(startTime !== undefined ? { startTime } : {}),
    ...(endTime !== undefined ? { endTime } : {}),
  };
}
