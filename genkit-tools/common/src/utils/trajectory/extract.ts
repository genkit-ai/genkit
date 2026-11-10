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
  ToolDefinition,
  ToolRequestPart,
  TraceData,
  Trajectory,
  TrajectoryAction,
  TrajectoryBundle,
  TrajectoryStep,
} from '../../types';
import {
  cleanActionName,
  extractSessionIdFromTrace,
  getSpanName,
  isAgentSpan,
  isModelSpan,
  isToolSpan,
  parseSpanJson,
  sortSpansChronologically,
  stackTraceSpans,
} from '../trace';
import {
  collectScopeResumeDecisions,
  isEnclosingTurnInterrupted,
} from './interrupts';
import {
  aggregateUsage,
  extractTextFromParts,
  getModelOutputData,
  getTurnContext,
  isExcludedFromTrajectory,
  isExplicitHistoryMessage,
  messageSignature,
  normalizeParts,
  type ParsedSpanInput,
  type ScopePartition,
} from './span-parser';
import {
  collectToolResponses,
  pairStepToolActions,
  type ToolPairingContext,
} from './tool-pairing';

/**
 * A "trajectory scope" is the contiguous region of a trace's span tree that
 * projects into a single `Trajectory`. It starts at `scopeRoot` (either the
 * unwrapped trace root span or a nested subagent span) and includes all
 * descendant turn, model, and tool spans down to—but excluding—any nested
 * agent spans, each of which starts its own child trajectory scope in
 * `TrajectoryBundle.trajectories`.
 */
interface ScopeExtractionState extends ToolPairingContext {
  traceId: string;
  emittedMessageSigs: Set<string>;
  steps: TrajectoryStep[];
  systemInstructions?: string;
}

function findEnclosingToolSpan(
  span: NestedSpanData,
  scopeRoot: NestedSpanData,
  parentMap: ReadonlyMap<string, NestedSpanData>
): NestedSpanData | undefined {
  let curr = parentMap.get(span.spanId);
  while (curr && curr.spanId !== scopeRoot.spanId) {
    if (isToolSpan(curr)) return curr;
    curr = parentMap.get(curr.spanId);
  }
  return undefined;
}

interface ScopeWalkContext {
  trace: TraceData;
  scopeRoot: NestedSpanData;
  trajectoryId: string;
  trajectories: Map<string, Trajectory>;
  partition: ScopePartition;
}

function walkScopeSpan(node: NestedSpanData, ctx: ScopeWalkContext): void {
  const { trace, scopeRoot, trajectoryId, trajectories, partition } = ctx;
  const isRootNode = node.spanId === scopeRoot.spanId;

  if (!isRootNode && isAgentSpan(node)) {
    const enclosingTool = findEnclosingToolSpan(
      node,
      scopeRoot,
      partition.parentMap
    );
    if (enclosingTool) {
      partition.subagentByToolSpanId.set(enclosingTool.spanId, node.spanId);
    } else {
      partition.toolSpans.push(node);
      partition.scopeSpans.push(node);
      partition.subagentByToolSpanId.set(node.spanId, node.spanId);
    }
    trajectories.set(
      node.spanId,
      buildTrajectoryScope(trace, node, node.spanId, trajectories, trajectoryId)
    );
    return;
  }

  partition.scopeSpans.push(node);
  if (isModelSpan(node)) {
    partition.modelSpans.push(node);
  } else if (isToolSpan(node) && !isRootNode) {
    partition.toolSpans.push(node);
  }

  for (const child of node.spans || []) {
    partition.parentMap.set(child.spanId, node);
    walkScopeSpan(child, ctx);
  }
}

/**
 * Walks `scopeRoot` to collect model/tool spans in this scope while splitting
 * nested subagent boundaries into child trajectories.
 */
function partitionTrajectoryScope(
  trace: TraceData,
  scopeRoot: NestedSpanData,
  trajectoryId: string,
  trajectories: Map<string, Trajectory>
): ScopePartition {
  const partition: ScopePartition = {
    parentMap: new Map<string, NestedSpanData>(),
    modelSpans: [],
    toolSpans: [],
    scopeSpans: [],
    subagentByToolSpanId: new Map<string, string>(),
  };

  walkScopeSpan(scopeRoot, {
    trace,
    scopeRoot,
    trajectoryId,
    trajectories,
    partition,
  });
  sortSpansChronologically(partition.modelSpans);
  sortSpansChronologically(partition.toolSpans);
  sortSpansChronologically(partition.scopeSpans);

  return partition;
}

function resolveAgentName(scopeRoot: NestedSpanData): string | undefined {
  if (isModelSpan(scopeRoot) && !isAgentSpan(scopeRoot)) return undefined;
  return cleanActionName(getSpanName(scopeRoot)) || undefined;
}

function parseScopeInputs(scopeSpans: NestedSpanData[]): ParsedSpanInput[] {
  return scopeSpans
    .map((s) => parseSpanJson<ParsedSpanInput>(s.attributes?.['genkit:input']))
    .filter((i): i is ParsedSpanInput => Boolean(i && typeof i === 'object'));
}

function collectScopeTools(inputs: ParsedSpanInput[]): ToolDefinition[] {
  const toolsByName = new Map<string, ToolDefinition>();
  for (const input of inputs) {
    for (const t of input.tools ?? []) {
      if (!t || typeof t !== 'object' || typeof t.name !== 'string') continue;
      if (!toolsByName.has(t.name)) {
        toolsByName.set(t.name, {
          name: t.name,
          description: typeof t.description === 'string' ? t.description : '',
          ...(t.key ? { key: t.key } : {}),
          ...(t.inputSchema ? { inputSchema: t.inputSchema } : {}),
          ...(t.outputSchema ? { outputSchema: t.outputSchema } : {}),
          ...(t.metadata ? { metadata: t.metadata } : {}),
        });
      }
    }
  }
  return Array.from(toolsByName.values());
}

function extractInitialSystemInstructions(
  messages: MessageData[]
): string | undefined {
  for (const msg of messages) {
    if (!Array.isArray(msg?.content)) continue;
    if (
      msg.role === 'system' ||
      (msg.metadata?.['agentPreamble'] && msg.role !== 'user')
    ) {
      const text = extractTextFromParts(msg.content);
      if (text) return text;
    }
  }
  return undefined;
}

function tryAppendMessageStep(
  msg: MessageData,
  turnIndex: number,
  spanId: string,
  timestamp: number,
  state: ScopeExtractionState
): void {
  if (!msg || (msg.role !== 'user' && msg.role !== 'system')) return;
  if (isExplicitHistoryMessage(msg)) return;

  const parts = normalizeParts(msg.content).filter(
    (p) => !p.toolRequest && !p.toolResponse
  );
  if (parts.length === 0) return;

  const sig = `${turnIndex}:${messageSignature(msg)}`;
  if (state.emittedMessageSigs.has(sig)) return;
  state.emittedMessageSigs.add(sig);

  const stepNumber = state.steps.length + 1;
  state.steps.push({
    stepId: `${state.traceId}-step-${stepNumber}-${spanId}`,
    stepNumber,
    traceId: state.traceId,
    spanId,
    turnIndex,
    source: msg.role === 'system' ? 'system' : 'user',
    startTime: timestamp,
    endTime: timestamp,
    latencyMs: 0,
    content: parts,
    actions: [],
  });
}

function appendNewInputSteps(
  messages: MessageData[],
  turnIndex: number,
  sourceSpanId: string,
  timestamp: number,
  state: ScopeExtractionState
): void {
  let lastModelOrToolIdx = -1;
  for (let i = 0; i < messages.length; i++) {
    const role = messages[i]?.role;
    if (role === 'model' || role === 'tool') lastModelOrToolIdx = i;
  }

  const newMessages = messages.slice(lastModelOrToolIdx + 1);
  const hasUserMsg = newMessages.some(
    (m) => m?.role === 'user' && !isExplicitHistoryMessage(m)
  );

  for (const msg of newMessages) {
    if (msg?.metadata?.['agentPreamble'] && msg.role === 'system') continue;
    if (msg?.role === 'system' && state.steps.length === 0 && hasUserMsg) {
      continue;
    }
    tryAppendMessageStep(msg, turnIndex, sourceSpanId, timestamp, state);
  }
}

function appendFallbackInputSteps(state: ScopeExtractionState): void {
  const { partition } = state;
  for (const span of partition.scopeSpans) {
    const { turnIndex } = getTurnContext(span, partition.parentMap);
    const input = parseSpanJson<ParsedSpanInput>(
      span.attributes?.['genkit:input']
    );
    const messages = input?.message
      ? [input.message]
      : Array.isArray(input?.messages)
        ? input.messages
        : [];
    for (const msg of messages) {
      tryAppendMessageStep(msg, turnIndex, span.spanId, span.startTime, state);
    }
  }
}

function resolveStopReason(
  actions: TrajectoryAction[],
  hasToolRequests: boolean,
  modelSpan: NestedSpanData,
  state: ScopeExtractionState,
  finishReason?: string,
  modelError?: { message: string; stack?: string }
): TrajectoryStep['stopReason'] {
  if (
    actions.some((a) => a.type === 'interrupt' || a.status === 'interrupted') ||
    finishReason === 'interrupted' ||
    (hasToolRequests &&
      isEnclosingTurnInterrupted(modelSpan, state.partition.parentMap))
  ) {
    return 'interrupted';
  }
  if (modelError || finishReason === 'failed') return 'errored';
  if (finishReason === 'aborted' || finishReason === 'cancelled') {
    return 'cancelled';
  }
  if (finishReason === 'length') return 'max_tokens';
  return actions.length > 0 ? 'tools' : 'completed';
}

function splitModelOutputParts(content: unknown): {
  contentParts: Part[];
  toolRequestParts: ToolRequestPart[];
} {
  const contentParts: Part[] = [];
  const toolRequestParts: ToolRequestPart[] = [];
  for (const part of normalizeParts(content)) {
    if (part.toolRequest) {
      toolRequestParts.push(part as ToolRequestPart);
    } else if (!part.toolResponse) {
      contentParts.push(part);
    }
  }
  return { contentParts, toolRequestParts };
}

function appendModelStep(
  modelSpan: NestedSpanData,
  nextModelSpan: NestedSpanData | undefined,
  state: ScopeExtractionState
): void {
  const { turnIndex, turnSpan } = getTurnContext(
    modelSpan,
    state.partition.parentMap
  );
  const input = parseSpanJson<ParsedSpanInput>(
    modelSpan.attributes?.['genkit:input']
  );
  const messages = Array.isArray(input?.messages) ? input.messages : [];

  state.systemInstructions ??= extractInitialSystemInstructions(messages);
  appendNewInputSteps(
    messages,
    turnIndex,
    turnSpan?.spanId || modelSpan.spanId,
    modelSpan.startTime,
    state
  );

  const { message, finishReason, usage, error } = getModelOutputData(modelSpan);
  const { contentParts, toolRequestParts } = splitModelOutputParts(
    message?.content
  );
  const actions = pairStepToolActions(
    toolRequestParts,
    modelSpan,
    nextModelSpan,
    turnIndex,
    finishReason,
    state
  );

  const startTime = modelSpan.startTime;
  const endTime = Math.max(
    modelSpan.endTime || startTime,
    ...actions.map((a) => a.endTime ?? 0)
  );
  const model = cleanActionName(getSpanName(modelSpan));
  const stepNumber = state.steps.length + 1;

  state.steps.push({
    stepId: `${state.traceId}-step-${stepNumber}-${modelSpan.spanId}`,
    stepNumber,
    traceId: state.traceId,
    spanId: modelSpan.spanId,
    turnIndex,
    source: 'agent',
    startTime,
    endTime,
    latencyMs: Math.max(0, endTime - startTime),
    ...(model ? { model } : {}),
    content: contentParts,
    actions,
    stopReason: resolveStopReason(
      actions,
      toolRequestParts.length > 0,
      modelSpan,
      state,
      finishReason,
      error
    ),
    ...(error ? { error } : {}),
    ...(usage ? { usage } : {}),
  });
}

function buildTrajectoryScope(
  trace: TraceData,
  scopeRoot: NestedSpanData,
  trajectoryId: string,
  trajectories: Map<string, Trajectory>,
  parentTrajectoryId?: string
): Trajectory {
  const partition = partitionTrajectoryScope(
    trace,
    scopeRoot,
    trajectoryId,
    trajectories
  );
  const parsedInputs = parseScopeInputs(partition.scopeSpans);
  const state: ScopeExtractionState = {
    traceId: trace.traceId,
    partition,
    claimedToolSpanIds: new Set<string>(),
    toolResponses: collectToolResponses(parsedInputs),
    resumeDecisions: collectScopeResumeDecisions(parsedInputs),
    emittedMessageSigs: new Set<string>(),
    steps: [],
  };

  for (let i = 0; i < partition.modelSpans.length; i++) {
    appendModelStep(
      partition.modelSpans[i],
      partition.modelSpans[i + 1],
      state
    );
  }
  if (partition.modelSpans.length === 0) appendFallbackInputSteps(state);

  const agentName = resolveAgentName(scopeRoot);
  const usage = aggregateUsage(state.steps.map((s) => s.usage));
  const startTime =
    scopeRoot.startTime ||
    (state.steps.length > 0
      ? state.steps[0].startTime
      : (trace.startTime ?? 0));
  const endTime = Math.max(
    scopeRoot.endTime || 0,
    ...state.steps.map((s) => s.endTime),
    startTime
  );

  return {
    trajectoryId,
    ...(parentTrajectoryId ? { parentTrajectoryId } : {}),
    ...(agentName ? { agentName } : {}),
    ...(state.systemInstructions
      ? { systemInstructions: state.systemInstructions }
      : {}),
    tools: collectScopeTools(parsedInputs),
    steps: state.steps,
    ...(usage ? { usage } : {}),
    startTime,
    endTime,
  };
}

/** Extracts a TrajectoryBundle from a single TraceData object. */
export function extractTrajectoryFromTrace(trace: TraceData): TrajectoryBundle {
  const rootId = trace.traceId || 'root';
  const sessionId = extractSessionIdFromTrace(trace);
  const prunedRoot = stackTraceSpans(trace, {
    exclude: isExcludedFromTrajectory,
  });

  const trajectories = new Map<string, Trajectory>();
  if (prunedRoot) {
    trajectories.set(
      rootId,
      buildTrajectoryScope(trace, prunedRoot, rootId, trajectories)
    );
  } else {
    trajectories.set(rootId, {
      trajectoryId: rootId,
      tools: [],
      steps: [],
      startTime: trace.startTime ?? 0,
      endTime: trace.endTime ?? 0,
    });
  }

  return {
    rootTrajectoryId: rootId,
    ...(sessionId ? { sessionId } : {}),
    trajectories: Object.fromEntries(trajectories),
  };
}
