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
  MessageData,
  NestedSpanData,
  Part,
  ToolRequestPart,
  TrajectoryAction,
} from '../../types';
import {
  getSpanName,
  getSpanStatus,
  parseSpanJson,
  shortToolName,
} from '../trace';
import {
  getToolCallId,
  takeMatchingByRefOrName,
  type ParsedSpanInput,
  type ParsedSpanOutput,
} from './span-parser';

/** A user/caller decision resuming an interrupted tool request (`respond` or `restart`). */
export interface ResumeDecision {
  name: string;
  ref?: string;
  strategy: 'respond' | 'restart';
  payload?: unknown;
  output?: unknown;
}

function toRespondDecision(part?: Part): ResumeDecision | undefined {
  const tr = part?.toolResponse;
  if (!tr || typeof tr.name !== 'string') return undefined;
  return {
    name: tr.name,
    ref: tr.ref,
    strategy: 'respond',
    payload: tr.output,
    output: tr.output,
  };
}

function toRestartDecision(
  part?: Part,
  fallbackToInput = false
): ResumeDecision | undefined {
  const tr = part?.toolRequest;
  if (!tr || typeof tr.name !== 'string') return undefined;
  const resumedMeta = part?.metadata?.['resumed'];
  if (!fallbackToInput && resumedMeta === undefined) return undefined;
  return {
    name: tr.name,
    ref: tr.ref,
    strategy: 'restart',
    payload: resumedMeta ?? tr.input,
  };
}

/** Extracts resume decisions from an `AgentInput` or `GenerateActionOptions` input object. */
export function extractResumeDecisionsFromInput(
  input?: ParsedSpanInput
): ResumeDecision[] {
  const resume = input?.resume;
  if (!resume || typeof resume !== 'object') return [];
  const responds = (resume.respond ?? []).map(toRespondDecision);
  const restarts = (resume.restart ?? []).map((p) =>
    toRestartDecision(p, true)
  );
  return [...responds, ...restarts].filter(
    (d): d is ResumeDecision => d !== undefined
  );
}

function extractMessageResumeDecisions(
  messages?: MessageData[]
): ResumeDecision[] {
  if (!Array.isArray(messages)) return [];
  const decisions: ResumeDecision[] = [];
  for (const msg of messages) {
    if (!Array.isArray(msg?.content)) continue;
    for (const part of msg.content) {
      const dec =
        msg.role === 'tool' && part?.metadata?.['interruptResponse']
          ? toRespondDecision(part)
          : msg.role === 'model'
            ? toRestartDecision(part)
            : undefined;
      if (dec) decisions.push(dec);
    }
  }
  return decisions;
}

/** Collects deduplicated HITL resume decisions across parsed span inputs in a scope. */
export function collectScopeResumeDecisions(
  inputs: ParsedSpanInput[]
): ResumeDecision[] {
  const decisions: ResumeDecision[] = [];
  const seenKeys = new Set<string>();

  for (const input of inputs) {
    const candidates = [
      ...extractResumeDecisionsFromInput(input),
      ...extractMessageResumeDecisions(input.messages),
    ];
    for (const dec of candidates) {
      const key = `${dec.strategy}:${dec.ref || ''}:${shortToolName(dec.name)}:${JSON.stringify(dec.payload)}`;
      if (!seenKeys.has(key)) {
        seenKeys.add(key);
        decisions.push(dec);
      }
    }
  }
  return decisions;
}

function hasMatchingAncestorOutput(
  span: NestedSpanData,
  parentMap: ReadonlyMap<string, NestedSpanData>,
  predicate: (out: ParsedSpanOutput) => boolean
): boolean {
  let curr = parentMap.get(span.spanId);
  while (curr) {
    const out = parseSpanJson<ParsedSpanOutput>(
      curr.attributes?.['genkit:output']
    );
    if (out && predicate(out)) return true;
    curr = parentMap.get(curr.spanId);
  }
  return false;
}

/** Checks whether an enclosing generate/turn span reported `finishReason === 'interrupted'`. */
export function isEnclosingTurnInterrupted(
  span: NestedSpanData,
  parentMap: ReadonlyMap<string, NestedSpanData>
): boolean {
  return hasMatchingAncestorOutput(
    span,
    parentMap,
    (out) => out.finishReason === 'interrupted'
  );
}

function hasEnclosingInterruptMetadata(
  span: NestedSpanData,
  parentMap: ReadonlyMap<string, NestedSpanData>,
  toolName: string,
  ref?: string
): boolean {
  return hasMatchingAncestorOutput(span, parentMap, (out) => {
    const msg = out.message ?? out.candidates?.[0]?.message;
    return Boolean(
      msg?.content?.some(
        (p) =>
          p?.toolRequest?.name === toolName &&
          (ref === undefined || p.toolRequest.ref === ref) &&
          p.metadata?.['interrupt'] !== undefined
      )
    );
  });
}

/** Determines whether a tool request triggered a human-in-the-loop (HITL) interrupt. */
export function isToolRequestInterrupted(params: {
  reqPart: ToolRequestPart;
  modelSpan: NestedSpanData;
  parentMap: ReadonlyMap<string, NestedSpanData>;
  matchedToolSpan: NestedSpanData | undefined;
  toolSpanErrAttr: string | undefined;
  finishReason: string | undefined;
  actionOutput: unknown;
}): boolean {
  const {
    reqPart,
    modelSpan,
    parentMap,
    matchedToolSpan,
    toolSpanErrAttr,
    finishReason,
    actionOutput,
  } = params;
  const req = reqPart.toolRequest;

  if (
    reqPart.metadata?.['interrupt'] !== undefined ||
    reqPart.metadata?.['resolvedInterrupt'] !== undefined ||
    matchedToolSpan?.attributes?.['genkit:metadata:interrupt'] !== undefined ||
    toolSpanErrAttr?.includes('ToolInterruptError') ||
    hasEnclosingInterruptMetadata(modelSpan, parentMap, req.name, req.ref)
  ) {
    return true;
  }

  const turnInterrupted =
    finishReason === 'interrupted' ||
    isEnclosingTurnInterrupted(modelSpan, parentMap);
  return (
    turnInterrupted &&
    (!matchedToolSpan ||
      getSpanStatus(matchedToolSpan) === '✖' ||
      actionOutput === undefined)
  );
}

function claimRestartedToolOutput(
  reqRef: string | undefined,
  reqShortName: string,
  toolSpans: NestedSpanData[],
  claimedToolSpanIds: Set<string>
): unknown {
  const candidates = toolSpans.filter(
    (ts) =>
      !claimedToolSpanIds.has(ts.spanId) &&
      ts.attributes?.['genkit:metadata:resumed'] !== undefined
  );
  const restartedSpan = takeMatchingByRefOrName(
    candidates,
    reqRef,
    reqShortName,
    getToolCallId,
    getSpanName
  );
  if (!restartedSpan) return undefined;
  claimedToolSpanIds.add(restartedSpan.spanId);
  return parseSpanJson(restartedSpan.attributes?.['genkit:output']);
}

/** Pairs an interrupted action with an in-trace `ResumeDecision` (and restarted tool span if applicable). */
export function resolveInTraceInterruptResume(params: {
  reqRef: string | undefined;
  reqShortName: string;
  resumeDecisions: ResumeDecision[];
  toolSpans: NestedSpanData[];
  claimedToolSpanIds: Set<string>;
  actionOutput: unknown;
}): {
  resumeInfo?: TrajectoryAction['resume'];
  resolvedOutput: unknown;
} {
  const {
    reqRef,
    reqShortName,
    resumeDecisions,
    toolSpans,
    claimedToolSpanIds,
  } = params;
  let resolvedOutput = params.actionOutput;

  const decision = takeMatchingByRefOrName(
    resumeDecisions,
    reqRef,
    reqShortName,
    (d) => d.ref,
    (d) => d.name
  );
  if (!decision) return { resolvedOutput };

  resolvedOutput ??= decision.output;
  if (decision.strategy === 'restart') {
    resolvedOutput ??= claimRestartedToolOutput(
      reqRef,
      reqShortName,
      toolSpans,
      claimedToolSpanIds
    );
  }

  return {
    resolvedOutput,
    resumeInfo: {
      strategy: decision.strategy,
      ...(decision.payload !== undefined ? { payload: decision.payload } : {}),
    },
  };
}
