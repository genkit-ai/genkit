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
  const decisions: ResumeDecision[] = [];
  if (Array.isArray(resume.respond)) {
    for (const part of resume.respond) {
      const dec = toRespondDecision(part);
      if (dec) decisions.push(dec);
    }
  }
  if (Array.isArray(resume.restart)) {
    for (const part of resume.restart) {
      const dec = toRestartDecision(part, true);
      if (dec) decisions.push(dec);
    }
  }
  return decisions;
}

function extractMessageResumeDecisions(
  messages?: MessageData[]
): ResumeDecision[] {
  if (!Array.isArray(messages)) return [];
  const decisions: ResumeDecision[] = [];
  for (const msg of messages) {
    if (!Array.isArray(msg?.content)) continue;
    const isResumedToolMsg =
      msg.role === 'tool' && Boolean(msg.metadata?.['resumed']);
    for (const part of msg.content) {
      const dec =
        msg.role === 'tool' &&
        (isResumedToolMsg || part?.metadata?.['interruptResponse'])
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

function findInAncestors<T>(
  span: NestedSpanData,
  parentMap: ReadonlyMap<string, NestedSpanData>,
  mapper: (out: ParsedSpanOutput) => T | undefined
): T | undefined {
  let curr = parentMap.get(span.spanId);
  const visited = new Set<string>();
  while (curr && !visited.has(curr.spanId)) {
    visited.add(curr.spanId);
    const out = parseSpanObject<ParsedSpanOutput>(
      curr.attributes?.['genkit:output']
    );
    if (out) {
      const match = mapper(out);
      if (match !== undefined) return match;
    }
    curr = parentMap.get(curr.spanId);
  }
  return undefined;
}

/** Checks whether an enclosing generate/turn span reported `finishReason === 'interrupted'`. */
export function isEnclosingTurnInterrupted(
  span: NestedSpanData,
  parentMap: ReadonlyMap<string, NestedSpanData>
): boolean {
  return (
    findInAncestors(span, parentMap, (out) =>
      out.finishReason === 'interrupted' ? true : undefined
    ) ?? false
  );
}

function findEnclosingInterruptMetadata(
  span: NestedSpanData,
  parentMap: ReadonlyMap<string, NestedSpanData>,
  toolName: string,
  ref?: string
): unknown {
  return findInAncestors(span, parentMap, (out) => {
    const msg = out.message ?? out.candidates?.[0]?.message;
    if (!Array.isArray(msg?.content)) return undefined;
    const matchedPart = msg.content.find(
      (p) =>
        p?.toolRequest?.name === toolName &&
        (ref === undefined || p.toolRequest.ref === ref) &&
        p.metadata?.['interrupt'] !== undefined
    );
    return matchedPart?.metadata?.['interrupt'];
  });
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

  const enclosingMeta = findEnclosingInterruptMetadata(
    modelSpan,
    parentMap,
    req.name,
    req.ref
  );
  if (enclosingMeta !== undefined) {
    return { isInterrupted: true, interrupt: enclosingMeta };
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

function claimRestartedToolSpan(
  reqRef: string | undefined,
  reqShortName: string,
  toolSpans: NestedSpanData[],
  claimedToolSpanIds: Set<string>
): NestedSpanData | undefined {
  const candidates = toolSpans.filter(
    (ts) =>
      !claimedToolSpanIds.has(ts.spanId) &&
      ts.attributes?.['genkit:metadata:resumed'] !== undefined
  );
  const restartedSpan = takeMatchingByRefOrName(
    candidates,
    reqRef,
    reqShortName,
    (ts) => {
      const callId = ts.attributes?.['genkit:metadata:tool:callId'];
      return typeof callId === 'string' && callId.length > 0
        ? callId
        : undefined;
    },
    getSpanName
  );
  if (restartedSpan) {
    claimedToolSpanIds.add(restartedSpan.spanId);
  }
  return restartedSpan;
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
}): {
  resumeInfo?: TrajectoryAction['resume'];
  resolvedOutput: unknown;
  resolvedContent?: Part[];
  restartedSpan?: NestedSpanData;
} {
  const {
    reqRef,
    reqShortName,
    resumeDecisions,
    toolSpans,
    claimedToolSpanIds,
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
  const restartedSpan =
    decision.strategy === 'restart'
      ? claimRestartedToolSpan(
          reqRef,
          reqShortName,
          toolSpans,
          claimedToolSpanIds
        )
      : undefined;

  return {
    resolvedOutput,
    ...(resolvedContent ? { resolvedContent } : {}),
    ...(restartedSpan ? { restartedSpan } : {}),
    resumeInfo: {
      strategy: decision.strategy,
      ...(decision.payload !== undefined ? { payload: decision.payload } : {}),
    },
  };
}
