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
} from '../../types/index.js';
import {
  getSpanName,
  getSpanStatus,
  parseSpanJson,
  shortToolName,
} from '../trace.js';
import {
  parseSpanObject,
  takeMatchingByRefOrName,
  type ParsedSpanInput,
  type ParsedSpanOutput,
} from './span-parser.js';

/** A user/caller decision resuming an interrupted tool request (`respond` or `restart`). */
export interface ResumeDecision {
  name: string;
  ref?: string;
  strategy: 'respond' | 'restart';
  payload?: unknown;
  output?: unknown;
  content?: Part[];
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
    ...(Array.isArray(tr.content) ? { content: tr.content } : {}),
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
  const resume = input && 'resume' in input ? input.resume : undefined;
  if (!resume || typeof resume !== 'object') return [];
  const responds = Array.isArray(resume.respond)
    ? resume.respond.map(toRespondDecision)
    : [];
  const restarts = Array.isArray(resume.restart)
    ? resume.restart.map((p) => toRestartDecision(p, true))
    : [];
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
    if (!input) continue;
    const candidates = [
      ...extractResumeDecisionsFromInput(input),
      ...extractMessageResumeDecisions(
        'messages' in input ? input.messages : undefined
      ),
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
  const visited = new Set<string>();
  while (curr && !visited.has(curr.spanId)) {
    visited.add(curr.spanId);
    const out = parseSpanObject<ParsedSpanOutput>(
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

function findEnclosingInterruptMetadata(
  span: NestedSpanData,
  parentMap: ReadonlyMap<string, NestedSpanData>,
  toolName: string,
  ref?: string
): { found: boolean; metadata?: unknown } {
  let found = false;
  let metadata: unknown;
  hasMatchingAncestorOutput(span, parentMap, (out) => {
    const msg = out.message ?? out.candidates?.[0]?.message;
    const matchedPart = msg?.content?.find(
      (p) =>
        p?.toolRequest?.name === toolName &&
        (ref === undefined || p.toolRequest.ref === ref) &&
        p.metadata?.['interrupt'] !== undefined
    );
    if (matchedPart) {
      found = true;
      metadata = matchedPart.metadata?.['interrupt'];
      return true;
    }
    return false;
  });
  return { found, metadata };
}

/**
 * Determines whether a tool request triggered a human-in-the-loop (HITL) interrupt
 * and resolves the interrupt metadata payload when present.
 */
export function resolveToolInterrupt(params: {
  reqPart: ToolRequestPart;
  modelSpan: NestedSpanData;
  parentMap: ReadonlyMap<string, NestedSpanData>;
  matchedToolSpan: NestedSpanData | undefined;
  toolSpanErrAttr: string | undefined;
  finishReason: string | undefined;
  actionOutput: unknown;
}): { isInterrupted: boolean; interrupt?: unknown } {
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

  const partMeta =
    reqPart.metadata?.['interrupt'] ?? reqPart.metadata?.['resolvedInterrupt'];
  if (partMeta !== undefined) {
    return { isInterrupted: true, interrupt: partMeta };
  }

  const spanInterruptAttr = parseSpanJson(
    matchedToolSpan?.attributes?.['genkit:metadata:interrupt']
  );
  if (spanInterruptAttr !== undefined) {
    return { isInterrupted: true, interrupt: spanInterruptAttr };
  }

  if (toolSpanErrAttr?.includes('ToolInterruptError')) {
    return { isInterrupted: true, interrupt: true };
  }

  const enclosing = findEnclosingInterruptMetadata(
    modelSpan,
    parentMap,
    req.name,
    req.ref
  );
  if (enclosing.found) {
    return { isInterrupted: true, interrupt: enclosing.metadata ?? true };
  }

  const turnInterrupted =
    finishReason === 'interrupted' ||
    isEnclosingTurnInterrupted(modelSpan, parentMap);
  const inferred =
    turnInterrupted &&
    (!matchedToolSpan ||
      getSpanStatus(matchedToolSpan) === '✖' ||
      actionOutput === undefined);
  return inferred
    ? { isInterrupted: true, interrupt: true }
    : { isInterrupted: false };
}

function claimRestartedToolResult(
  reqShortName: string,
  toolSpans: NestedSpanData[],
  claimedToolSpanIds: Set<string>,
  parseToolResult: (span?: NestedSpanData) => {
    output?: unknown;
    content?: Part[];
  }
): { output?: unknown; content?: Part[] } {
  const restartedSpan = toolSpans.find(
    (ts) =>
      !claimedToolSpanIds.has(ts.spanId) &&
      ts.attributes?.['genkit:metadata:resumed'] !== undefined &&
      shortToolName(getSpanName(ts)) === reqShortName
  );
  if (!restartedSpan) return {};
  claimedToolSpanIds.add(restartedSpan.spanId);
  return parseToolResult(restartedSpan);
}

/** Pairs an interrupted action with an in-trace `ResumeDecision` (and restarted tool span if applicable). */
export function resolveInTraceInterruptResume(params: {
  reqRef: string | undefined;
  reqShortName: string;
  resumeDecisions: ResumeDecision[];
  toolSpans: NestedSpanData[];
  claimedToolSpanIds: Set<string>;
  actionOutput: unknown;
  actionContent: Part[] | undefined;
  parseToolResult: (span?: NestedSpanData) => {
    output?: unknown;
    content?: Part[];
  };
}): {
  resumeInfo?: TrajectoryAction['resume'];
  resolvedOutput: unknown;
  resolvedContent?: Part[];
} {
  const {
    reqRef,
    reqShortName,
    resumeDecisions,
    toolSpans,
    claimedToolSpanIds,
    parseToolResult,
  } = params;
  let resolvedOutput = params.actionOutput;
  let resolvedContent = params.actionContent;

  const decision = takeMatchingByRefOrName(
    resumeDecisions,
    reqRef,
    reqShortName,
    (d) => d.ref,
    (d) => d.name
  );
  if (!decision) return { resolvedOutput, resolvedContent };

  resolvedOutput ??= decision.output;
  resolvedContent ??= decision.content;
  if (decision.strategy === 'restart') {
    const restarted = claimRestartedToolResult(
      reqShortName,
      toolSpans,
      claimedToolSpanIds,
      parseToolResult
    );
    resolvedOutput ??= restarted.output;
    resolvedContent ??= restarted.content;
  }

  return {
    resolvedOutput,
    ...(resolvedContent ? { resolvedContent } : {}),
    resumeInfo: {
      strategy: decision.strategy,
      ...(decision.payload !== undefined ? { payload: decision.payload } : {}),
    },
  };
}
