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
  AgentFinishReason,
  MessageData,
  NestedSpanData,
  Part,
  ToolDefinition,
  ToolRequestPart,
  TraceData,
  Trajectory,
  TrajectoryBundle,
  TrajectoryStatus,
  TrajectoryStep,
} from '../../types/index.js';
import {
  cleanActionName,
  extractSessionIdFromTrace,
  getSpanName,
  isAgentSpan,
  isModelSpan,
  isToolSpan,
  sortSpansChronologically,
  stackTraceSpans,
} from '../trace.js';
import {
  aggregateUsage,
  extractSpanRuntimeError,
  extractTextFromParts,
  getModelOutputData,
  getTurnContext,
  isExcludedFromTrajectory,
  isExplicitHistoryMessage,
  messageSignature,
  normalizeParts,
  parseSpanObject,
  type ParsedSpanInput,
  type ParsedSpanOutput,
  type TrajectorySpans,
} from './span-parser.js';
import {
  collectToolResponses,
  pairStepToolActions,
  type ToolPairingContext,
} from './tool-pairing.js';

type FinishReason = NonNullable<TrajectoryStep['finishReason']>;

const AGENT_FINISH_REASONS: ReadonlySet<string> = new Set([
  'stop',
  'length',
  'blocked',
  'interrupted',
  'other',
  'unknown',
  'aborted',
  'detached',
  'failed',
]);

/**
 * State for projecting a contiguous region of a trace's span tree into a
 * single `Trajectory` (starting at `rootSpan` and stopping at nested `/agent/`
 * boundaries, which become child entries in `TrajectoryBundle.trajectories`).
 */
interface TrajectoryExtractionState extends ToolPairingContext {
  traceId: string;
  emittedMessageSigs: Set<string>;
  spanStepCounts: Map<string, number>;
  steps: TrajectoryStep[];
  systemInstructions?: string;
}

function nextStepId(spanId: string, state: TrajectoryExtractionState): string {
  const count = (state.spanStepCounts.get(spanId) ?? 0) + 1;
  state.spanStepCounts.set(spanId, count);
  return `${spanId}-${count}`;
}

function findEnclosingToolSpan(
  span: NestedSpanData,
  rootSpan: NestedSpanData,
  parentMap: ReadonlyMap<string, NestedSpanData>
): NestedSpanData | undefined {
  let curr = parentMap.get(span.spanId);
  while (curr && curr.spanId !== rootSpan.spanId) {
    if (isToolSpan(curr)) return curr;
    curr = parentMap.get(curr.spanId);
  }
  return undefined;
}

interface TrajectoryWalkContext {
  trace: TraceData;
  rootSpan: NestedSpanData;
  trajectoryId: string;
  trajectories: Map<string, Trajectory>;
  trajectorySpans: TrajectorySpans;
}

function collectSpanIntoTrajectory(
  node: NestedSpanData,
  ctx: TrajectoryWalkContext
): void {
  const { trace, rootSpan, trajectoryId, trajectories, trajectorySpans } = ctx;
  const isRootNode = node.spanId === rootSpan.spanId;

  if (!isRootNode && isAgentSpan(node)) {
    const enclosingTool = findEnclosingToolSpan(
      node,
      rootSpan,
      trajectorySpans.parentMap
    );
    if (enclosingTool) {
      trajectorySpans.subagentByToolSpanId.set(
        enclosingTool.spanId,
        node.spanId
      );
    } else {
      trajectorySpans.toolSpans.push(node);
      trajectorySpans.scopeSpans.push(node);
      trajectorySpans.subagentByToolSpanId.set(node.spanId, node.spanId);
    }
    trajectories.set(
      node.spanId,
      buildTrajectory(trace, node, node.spanId, trajectories, trajectoryId)
    );
    return;
  }

  trajectorySpans.scopeSpans.push(node);
  if (isModelSpan(node)) {
    trajectorySpans.modelSpans.push(node);
  } else if (isToolSpan(node) && !isRootNode) {
    trajectorySpans.toolSpans.push(node);
  }

  for (const child of node.spans || []) {
    trajectorySpans.parentMap.set(child.spanId, node);
    collectSpanIntoTrajectory(child, ctx);
  }
}

/**
 * Walks `rootSpan` to collect model/tool spans for this trajectory while
 * splitting nested subagent boundaries into child trajectories.
 */
function collectTrajectorySpans(
  trace: TraceData,
  rootSpan: NestedSpanData,
  trajectoryId: string,
  trajectories: Map<string, Trajectory>
): TrajectorySpans {
  const trajectorySpans: TrajectorySpans = {
    parentMap: new Map<string, NestedSpanData>(),
    modelSpans: [],
    toolSpans: [],
    scopeSpans: [],
    subagentByToolSpanId: new Map<string, string>(),
  };

  collectSpanIntoTrajectory(rootSpan, {
    trace,
    rootSpan,
    trajectoryId,
    trajectories,
    trajectorySpans,
  });
  sortSpansChronologically(trajectorySpans.modelSpans);
  sortSpansChronologically(trajectorySpans.toolSpans);
  sortSpansChronologically(trajectorySpans.scopeSpans);

  return trajectorySpans;
}

function resolveAgentName(rootSpan: NestedSpanData): string | undefined {
  if (isModelSpan(rootSpan) && !isAgentSpan(rootSpan)) return undefined;
  return cleanActionName(getSpanName(rootSpan)) || undefined;
}

function parseTrajectoryInputs(spans: NestedSpanData[]): ParsedSpanInput[] {
  return spans
    .map((s) =>
      parseSpanObject<ParsedSpanInput>(s.attributes?.['genkit:input'])
    )
    .filter((i): i is ParsedSpanInput => Boolean(i && typeof i === 'object'));
}

function collectTrajectoryTools(inputs: ParsedSpanInput[]): ToolDefinition[] {
  const toolsByName = new Map<string, ToolDefinition>();
  for (const input of inputs) {
    if (!('tools' in input) || !Array.isArray(input.tools)) continue;
    for (const t of input.tools) {
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
  state: TrajectoryExtractionState
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
    stepId: nextStepId(spanId, state),
    stepNumber,
    traceId: state.traceId,
    spanId,
    turnIndex,
    source: msg.role === 'system' ? 'system' : 'user',
    startTime: timestamp,
    endTime: timestamp,
    content: parts,
    actions: [],
  });
}

function appendNewInputSteps(
  messages: MessageData[],
  turnIndex: number,
  sourceSpanId: string,
  timestamp: number,
  state: TrajectoryExtractionState
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
    if (msg?.metadata?.['agentPreamble'] && msg?.role === 'system') continue;
    if (msg?.role === 'system' && state.steps.length === 0 && hasUserMsg) {
      continue;
    }
    tryAppendMessageStep(msg, turnIndex, sourceSpanId, timestamp, state);
  }
}

function appendFallbackInputSteps(state: TrajectoryExtractionState): void {
  const { trajectorySpans } = state;
  for (const span of trajectorySpans.scopeSpans) {
    const { turnIndex } = getTurnContext(span, trajectorySpans.parentMap);
    const input = parseSpanObject<ParsedSpanInput>(
      span.attributes?.['genkit:input']
    );
    const messages =
      input && 'message' in input && input.message
        ? [input.message]
        : input && 'messages' in input && Array.isArray(input.messages)
          ? input.messages
          : [];
    for (const msg of messages) {
      tryAppendMessageStep(msg, turnIndex, span.spanId, span.startTime, state);
    }
  }
}

function resolveStepFinishReason(
  finishReason?: string,
  modelError?: RuntimeError
): FinishReason {
  if (modelError || finishReason === 'failed') return 'failed';
  if (finishReason === 'aborted' || finishReason === 'cancelled') {
    return 'aborted';
  }
  if (finishReason === 'length') return 'length';
  if (finishReason === 'blocked') return 'blocked';
  if (finishReason === 'interrupted') return 'interrupted';
  if (finishReason === 'other' || finishReason === 'unknown') {
    return finishReason;
  }
  return 'stop';
}

function resolveTrajectoryOutcome(
  scopeRoot: NestedSpanData,
  steps: TrajectoryStep[]
): {
  status: TrajectoryStatus;
  finishReason?: AgentFinishReason;
  error?: RuntimeError;
} {
  const scopeOut = parseSpanObject<ParsedSpanOutput>(
    scopeRoot.attributes?.['genkit:output']
  );
  const scopeError = extractSpanRuntimeError(
    scopeRoot,
    'Trajectory execution failed',
    scopeOut
  );
  const scopeFinish =
    typeof scopeOut?.finishReason === 'string' &&
    AGENT_FINISH_REASONS.has(scopeOut.finishReason)
      ? (scopeOut.finishReason as AgentFinishReason)
      : undefined;

  if (scopeError || scopeFinish === 'failed') {
    return {
      status: 'error',
      finishReason: scopeFinish ?? 'failed',
      ...(scopeError ? { error: scopeError } : {}),
    };
  }
  if (scopeFinish === 'aborted') {
    return { status: 'error', finishReason: 'aborted' };
  }
  if (
    scopeFinish === 'interrupted' ||
    steps.some(
      (s) =>
        s.finishReason === 'interrupted' ||
        s.actions.some((a) => a.status === 'interrupted')
    )
  ) {
    return { status: 'interrupted', finishReason: 'interrupted' };
  }
  if (!scopeRoot.endTime || scopeFinish === 'detached') {
    return {
      status: 'in-progress',
      ...(scopeFinish ? { finishReason: scopeFinish } : {}),
    };
  }

  const lastStep = steps[steps.length - 1];
  if (lastStep?.finishReason === 'failed' || lastStep?.error) {
    return {
      status: 'error',
      finishReason: 'failed',
      ...(lastStep.error ? { error: lastStep.error } : {}),
    };
  }
  if (lastStep?.finishReason === 'aborted') {
    return { status: 'error', finishReason: 'aborted' };
  }
  if (steps.some((s) => s.actions.some((a) => a.status === 'in-progress'))) {
    return {
      status: 'in-progress',
      ...(scopeFinish ? { finishReason: scopeFinish } : {}),
    };
  }

  const resolvedFinish =
    scopeFinish ?? (lastStep?.finishReason as AgentFinishReason | undefined);
  return {
    status: 'success',
    ...(resolvedFinish ? { finishReason: resolvedFinish } : {}),
  };
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
  state: TrajectoryExtractionState
): void {
  const { turnIndex, turnSpan } = getTurnContext(
    modelSpan,
    state.trajectorySpans.parentMap
  );
  const input = parseSpanObject<ParsedSpanInput>(
    modelSpan.attributes?.['genkit:input']
  );
  const messages =
    input && 'messages' in input && Array.isArray(input.messages)
      ? input.messages
      : [];

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
    stepId: nextStepId(modelSpan.spanId, state),
    stepNumber,
    traceId: state.traceId,
    spanId: modelSpan.spanId,
    turnIndex,
    source: 'agent',
    startTime,
    endTime,
    ...(model ? { model } : {}),
    content: contentParts,
    actions,
    finishReason: resolveStepFinishReason(finishReason, error),
    ...(error ? { error } : {}),
    ...(usage ? { usage } : {}),
  });
}

function buildTrajectory(
  trace: TraceData,
  rootSpan: NestedSpanData,
  trajectoryId: string,
  trajectories: Map<string, Trajectory>,
  parentTrajectoryId?: string
): Trajectory {
  const trajectorySpans = collectTrajectorySpans(
    trace,
    rootSpan,
    trajectoryId,
    trajectories
  );
  const parsedInputs = parseTrajectoryInputs(trajectorySpans.scopeSpans);
  const state: TrajectoryExtractionState = {
    traceId: trace.traceId,
    trajectorySpans,
    claimedToolSpanIds: new Set<string>(),
    toolResponses: collectToolResponses(parsedInputs),
    emittedMessageSigs: new Set<string>(),
    spanStepCounts: new Map<string, number>(),
    steps: [],
  };

  for (let i = 0; i < trajectorySpans.modelSpans.length; i++) {
    appendModelStep(
      trajectorySpans.modelSpans[i],
      trajectorySpans.modelSpans[i + 1],
      state
    );
  }
  if (trajectorySpans.modelSpans.length === 0) appendFallbackInputSteps(state);

  const agentName = resolveAgentName(rootSpan);
  const usage = aggregateUsage(state.steps.map((s) => s.usage));
  const startTime =
    rootSpan.startTime ||
    (state.steps.length > 0
      ? state.steps[0].startTime
      : (trace.startTime ?? 0));
  const endTime = Math.max(
    rootSpan.endTime || 0,
    ...state.steps.map((s) => s.endTime),
    startTime
  );
  const { status, finishReason, error } = resolveTrajectoryOutcome(
    rootSpan,
    state.steps
  );

  return {
    trajectoryId,
    ...(parentTrajectoryId ? { parentTrajectoryId } : {}),
    ...(agentName ? { agentName } : {}),
    status,
    ...(finishReason ? { finishReason } : {}),
    ...(error ? { error } : {}),
    ...(state.systemInstructions
      ? { systemInstructions: state.systemInstructions }
      : {}),
    tools: collectTrajectoryTools(parsedInputs),
    steps: state.steps,
    ...(usage ? { usage } : {}),
    startTime,
    endTime,
  };
}

/** Extracts a TrajectoryBundle from a single TraceData object. */
export function extractTrajectoryFromTrace(trace: TraceData): TrajectoryBundle {
  if (!trace) {
    return {
      rootTrajectoryId: 'root',
      trajectories: {
        root: {
          trajectoryId: 'root',
          status: 'success',
          tools: [],
          steps: [],
          startTime: 0,
          endTime: 0,
        },
      },
    };
  }
  const rootId = trace.traceId || 'root';
  const sessionId = extractSessionIdFromTrace(trace);
  const prunedRoot = stackTraceSpans(trace, {
    exclude: isExcludedFromTrajectory,
  });

  const trajectories = new Map<string, Trajectory>();
  if (prunedRoot) {
    trajectories.set(
      rootId,
      buildTrajectory(trace, prunedRoot, rootId, trajectories)
    );
  } else {
    trajectories.set(rootId, {
      trajectoryId: rootId,
      status: 'success',
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
