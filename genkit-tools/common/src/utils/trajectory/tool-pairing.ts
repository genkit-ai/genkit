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
  resolveInTraceInterruptResume,
  resolveToolInterrupt,
  type ResumeDecision,
} from './interrupts.js';
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
 * used as a fallback output source and interrupt-resume signal when pairing an earlier `toolRequest`.
 */
export interface ToolResponseEntry {
  part: ToolResponsePart;
  /** True if metadata marks this response as fulfilling an interrupted tool request. */
  resumed?: boolean;
}

/** Shared mutable pairing state while projecting a trajectory. */
export interface ToolPairingContext {
  trajectorySpans: TrajectorySpans;
  claimedToolSpanIds: Set<string>;
  toolResponses: ToolResponseEntry[];
  resumeDecisions: ResumeDecision[];
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
        responses.push({
          part: part as ToolResponsePart,
          resumed: Boolean(
            part.metadata?.['interruptResponse'] || msg.metadata?.['resumed']
          ),
        });
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

function matchesToolSpanName(
  span: NestedSpanData,
  reqShortName: string,
  subagentByToolSpanId: ReadonlyMap<string, string>
): boolean {
  const tsName = shortToolName(getSpanName(span));
  if (tsName === reqShortName) return true;
  if (subagentByToolSpanId.has(span.spanId)) {
    const normTs = tsName.replace(/^delegate_to_/, '');
    const normReq = reqShortName.replace(/^delegate_to_/, '');
    return Boolean(normTs && normTs === normReq);
  }
  return false;
}

function claimMatchingToolSpan(
  reqShortName: string,
  modelSpan: NestedSpanData,
  nextModelSpan: NestedSpanData | undefined,
  turnIndex: number,
  ctx: ToolPairingContext
): NestedSpanData | undefined {
  const { toolSpans, subagentByToolSpanId, parentMap } = ctx.trajectorySpans;
  const nameMatched = toolSpans.filter(
    (ts) =>
      !ctx.claimedToolSpanIds.has(ts.spanId) &&
      ts.attributes?.['genkit:metadata:resumed'] === undefined &&
      matchesToolSpanName(ts, reqShortName, subagentByToolSpanId)
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
  subtrajectoryId: string | undefined,
  isInterrupted: boolean,
  matchedSpan: NestedSpanData | undefined,
  matchedResp: ToolResponseEntry | undefined,
  finishReason: string | undefined
): {
  type: TrajectoryAction['type'];
  status: TrajectoryStatus;
  error?: RuntimeError;
} {
  const type: TrajectoryAction['type'] = subtrajectoryId ? 'subagent' : 'tool';

  if (isInterrupted) return { type, status: 'interrupted' };
  if (matchedSpan && getSpanStatus(matchedSpan) === '✖') {
    return {
      type,
      status: 'error',
      error: extractSpanRuntimeError(matchedSpan, 'Tool execution failed'),
    };
  }
  if (
    (matchedSpan && !matchedSpan.endTime) ||
    (!matchedSpan && !matchedResp && finishReason !== 'stop')
  ) {
    return { type, status: 'in-progress' };
  }
  return { type, status: 'success' };
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
  let output =
    spanResult.output !== undefined
      ? spanResult.output
      : matchedResp?.part.toolResponse.output;
  const respContent = Array.isArray(matchedResp?.part.toolResponse.content)
    ? normalizeParts(matchedResp.part.toolResponse.content)
    : undefined;
  let content =
    spanResult.content ??
    (respContent && respContent.length > 0 ? respContent : undefined);

  const errAttr =
    (matchedSpan?.attributes?.['genkit:error'] as string | undefined) ||
    matchedSpan?.status?.message;
  const { isInterrupted, interrupt } = resolveToolInterrupt({
    reqPart,
    modelSpan,
    parentMap: ctx.trajectorySpans.parentMap,
    matchedToolSpan: matchedSpan,
    toolSpanErrAttr: errAttr,
    finishReason,
    actionOutput: matchedResp?.resumed ? undefined : output,
  });

  const subtrajectoryId = matchedSpan
    ? ctx.trajectorySpans.subagentByToolSpanId.get(matchedSpan.spanId)
    : undefined;
  const { type, status, error } = classifyToolAction(
    subtrajectoryId,
    isInterrupted,
    matchedSpan,
    matchedResp,
    finishReason
  );

  let resume: TrajectoryAction['resume'] | undefined;
  if (isInterrupted) {
    const resolved = resolveInTraceInterruptResume({
      reqRef: req.ref,
      reqShortName,
      resumeDecisions: ctx.resumeDecisions,
      toolSpans: ctx.trajectorySpans.toolSpans,
      claimedToolSpanIds: ctx.claimedToolSpanIds,
      actionOutput: output,
      actionContent: content,
      parseToolResult: parseToolSpanResult,
    });
    resume = resolved.resumeInfo;
    output = resolved.resolvedOutput;
    content = resolved.resolvedContent;
  }

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
    ...(subtrajectoryId ? { subtrajectoryId } : {}),
    ...(interrupt !== undefined ? { interrupt } : {}),
    ...(resume ? { resume } : {}),
  };
}
