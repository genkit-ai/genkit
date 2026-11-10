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
  NestedSpanData,
  ToolRequestPart,
  ToolResponsePart,
  TrajectoryAction,
} from '../../types';
import {
  getSpanName,
  getSpanStatus,
  parseSpanJson,
  shortToolName,
} from '../trace';
import {
  isToolRequestInterrupted,
  resolveInTraceInterruptResume,
  type ResumeDecision,
} from './interrupts';
import {
  getToolCallId,
  getTurnContext,
  hasCompatibleRef,
  takeMatchingByRefOrName,
  type ParsedSpanInput,
  type ScopePartition,
} from './span-parser';

/**
 * A `toolResponse` part extracted from `role: 'tool'` messages in model inputs,
 * used as a fallback output source and interrupt-resume signal when pairing an earlier `toolRequest`.
 */
export interface ToolResponseEntry {
  part: ToolResponsePart;
  /** True if metadata marks this response as fulfilling an interrupted tool request. */
  resumed?: boolean;
}

/** Shared mutable pairing state while projecting a trajectory scope. */
export interface ToolPairingContext {
  partition: ScopePartition;
  claimedToolSpanIds: Set<string>;
  toolResponses: ToolResponseEntry[];
  resumeDecisions: ResumeDecision[];
}

/**
 * Pre-collects unique `toolResponse` parts across span `input.messages` in a scope
 * (deduplicating repeated copies across cumulative multi-turn message histories)
 * so earlier `toolRequest` steps can look ahead and pair with their responses.
 */
export function collectToolResponses(
  inputs: ParsedSpanInput[]
): ToolResponseEntry[] {
  const responses: ToolResponseEntry[] = [];
  const seenKeys = new Set<string>();

  for (const input of inputs) {
    for (const msg of input.messages ?? []) {
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
  reqRef: string | undefined,
  reqShortName: string,
  modelSpan: NestedSpanData,
  nextModelSpan: NestedSpanData | undefined,
  turnIndex: number,
  ctx: ToolPairingContext
): NestedSpanData | undefined {
  const { toolSpans, subagentByToolSpanId, parentMap } = ctx.partition;
  const candidates = toolSpans.filter(
    (ts) =>
      !ctx.claimedToolSpanIds.has(ts.spanId) &&
      ts.attributes?.['genkit:metadata:resumed'] === undefined
  );
  const byRef = reqRef
    ? candidates.find((ts) => getToolCallId(ts) === reqRef)
    : undefined;
  const nameMatched = byRef
    ? []
    : candidates.filter(
        (ts) =>
          hasCompatibleRef(reqRef, getToolCallId(ts)) &&
          matchesToolSpanName(ts, reqShortName, subagentByToolSpanId)
      );

  const matched =
    byRef ??
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
  toolSpanErrAttr: string | undefined,
  finishReason: string | undefined
): {
  type: TrajectoryAction['type'];
  status: TrajectoryAction['status'];
  error?: string;
} {
  const type: TrajectoryAction['type'] = subtrajectoryId
    ? 'subagent'
    : isInterrupted
      ? 'interrupt'
      : 'tool';

  if (isInterrupted) return { type, status: 'interrupted' };
  if (matchedSpan && getSpanStatus(matchedSpan) === '✖') {
    return {
      type,
      status: 'error',
      error: toolSpanErrAttr || 'Tool execution failed',
    };
  }
  if (
    (matchedSpan && !matchedSpan.endTime) ||
    (!matchedSpan && !matchedResp && finishReason !== 'stop')
  ) {
    return { type, status: 'pending' };
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
  return toolRequestParts.map((part, idx) =>
    buildSingleToolAction(
      part,
      idx,
      modelSpan,
      nextModelSpan,
      turnIndex,
      finishReason,
      ctx
    )
  );
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
    req.ref,
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
  const spanOutput = parseSpanJson(matchedSpan?.attributes?.['genkit:output']);
  const input = req.input !== undefined ? req.input : spanInput;
  let output =
    spanOutput !== undefined
      ? spanOutput
      : matchedResp?.part.toolResponse.output;

  const errAttr =
    (matchedSpan?.attributes?.['genkit:error'] as string | undefined) ||
    matchedSpan?.status?.message;
  const isInterrupted = isToolRequestInterrupted({
    reqPart,
    modelSpan,
    parentMap: ctx.partition.parentMap,
    matchedToolSpan: matchedSpan,
    toolSpanErrAttr: errAttr,
    finishReason,
    actionOutput: matchedResp?.resumed ? undefined : output,
  });

  const subtrajectoryId = matchedSpan
    ? ctx.partition.subagentByToolSpanId.get(matchedSpan.spanId)
    : undefined;
  const { type, status, error } = classifyToolAction(
    subtrajectoryId,
    isInterrupted,
    matchedSpan,
    matchedResp,
    errAttr,
    finishReason
  );

  let resume: TrajectoryAction['resume'] | undefined;
  if (isInterrupted) {
    const resolved = resolveInTraceInterruptResume({
      reqRef: req.ref,
      reqShortName,
      resumeDecisions: ctx.resumeDecisions,
      toolSpans: ctx.partition.toolSpans,
      claimedToolSpanIds: ctx.claimedToolSpanIds,
      actionOutput: output,
    });
    resume = resolved.resumeInfo;
    output = resolved.resolvedOutput;
  }

  const startTime = matchedSpan?.startTime;
  const endTime = matchedSpan?.endTime || undefined;
  const durationMs =
    startTime !== undefined && endTime !== undefined && endTime >= startTime
      ? endTime - startTime
      : undefined;

  return {
    actionId: req.ref || `${modelSpan.spanId}-action-${reqIdx + 1}`,
    ...(matchedSpan ? { spanId: matchedSpan.spanId } : {}),
    name: req.name,
    type,
    status,
    ...(input !== undefined ? { input } : {}),
    ...(output !== undefined ? { output } : {}),
    ...(error ? { error } : {}),
    ...(startTime !== undefined ? { startTime } : {}),
    ...(endTime !== undefined ? { endTime } : {}),
    ...(durationMs !== undefined ? { durationMs } : {}),
    ...(subtrajectoryId ? { subtrajectoryId } : {}),
    ...(resume ? { resume } : {}),
  };
}
