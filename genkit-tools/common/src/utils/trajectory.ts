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
  GenerationUsage,
  MessageData,
  NestedSpanData,
  Part,
  SpanData,
  ToolDefinition,
  ToolRequestPart,
  ToolResponsePart,
  TraceData,
  Trajectory,
  TrajectoryAction,
  TrajectoryBundle,
  TrajectoryStep,
} from '../types';
import { getSpanStatus, stackTraceSpans } from './trace';

interface ResumeDecision {
  name: string;
  ref?: string;
  strategy: 'respond' | 'restart';
  payload?: unknown;
  output?: unknown;
}

/**
 * Safely parses a JSON string attribute from a span, or returns the value as-is if already parsed.
 */
function parseSpanJson<T = any>(val: unknown): T | undefined {
  if (val === undefined || val === null) return undefined;
  if (typeof val === 'string') {
    const trimmed = val.trim();
    if (!trimmed) return undefined;
    try {
      return JSON.parse(trimmed) as T;
    } catch {
      return val as unknown as T;
    }
  }
  return val as T;
}

/**
 * Checks whether a span and its entire descendant subtree should be excluded from trajectory projection.
 */
function isExcludedSpan(span: NestedSpanData): boolean {
  const attrs = span.attributes || {};
  const devInternal = attrs['genkit:metadata:genkit-dev-internal'];
  if (devInternal === true || String(devInternal).toLowerCase() === 'true') {
    return true;
  }
  const trajectoryExclude = attrs['genkit:metadata:trajectory:exclude'];
  if (
    trajectoryExclude === true ||
    String(trajectoryExclude).toLowerCase() === 'true'
  ) {
    return true;
  }
  const subtype = attrs['genkit:metadata:subtype'] as string | undefined;
  const type = attrs['genkit:type'] as string | undefined;
  if (
    subtype === 'promptTemplate' ||
    subtype === 'prompt' ||
    type === 'promptTemplate' ||
    type === 'prompt'
  ) {
    return true;
  }
  return false;
}

/**
 * Checks whether a span represents an agent boundary.
 */
function isAgentSpan(span: NestedSpanData): boolean {
  const attrs = span.attributes || {};
  const subtype = attrs['genkit:metadata:subtype'] as string | undefined;
  const type = attrs['genkit:type'] as string | undefined;
  if (subtype === 'agent' || type === 'agent') {
    return true;
  }
  const name =
    span.displayName || (attrs['genkit:name'] as string | undefined) || '';
  if (name.startsWith('/agent/')) {
    return true;
  }
  return false;
}

/**
 * Checks whether a span represents a model execution.
 */
function isModelSpan(span: NestedSpanData): boolean {
  const attrs = span.attributes || {};
  const subtype = attrs['genkit:metadata:subtype'] as string | undefined;
  const type = attrs['genkit:type'] as string | undefined;
  const name =
    span.displayName || (attrs['genkit:name'] as string | undefined) || '';
  return (
    subtype === 'model' || type === 'model' || name.startsWith('/model/')
  );
}

/**
 * Checks whether a span represents a tool execution.
 */
function isToolSpan(span: NestedSpanData): boolean {
  const attrs = span.attributes || {};
  const subtype = attrs['genkit:metadata:subtype'] as string | undefined;
  const type = attrs['genkit:type'] as string | undefined;
  const name =
    span.displayName || (attrs['genkit:name'] as string | undefined) || '';
  return (
    subtype === 'tool' ||
    subtype === 'tool.v2' ||
    type === 'tool' ||
    type === 'tool.v2' ||
    name.startsWith('/tool/') ||
    name.startsWith('/tool.v2/')
  );
}

/**
 * Extracts a clean display name for an agent, model, or tool span.
 */
function cleanActionName(rawName?: string): string {
  if (!rawName) return '';
  const stripped = rawName.replace(
    /^\/(agent|flow|tool\.v2|tool|prompt|model|background-model|util)\//,
    ''
  );
  return stripped;
}

/**
 * Extracts the short tool name (stripping any leading slash-delimited namespace).
 */
function shortToolName(rawName?: string): string {
  if (!rawName) return '';
  const cleaned = cleanActionName(rawName);
  const lastSlash = cleaned.lastIndexOf('/');
  return lastSlash >= 0 ? cleaned.substring(lastSlash + 1) : cleaned;
}

/**
 * Recursively clones and prunes excluded spans from a NestedSpanData tree.
 */
function pruneSpanTree(node: NestedSpanData): NestedSpanData | undefined {
  if (isExcludedSpan(node)) {
    return undefined;
  }
  const children = (node.spans || [])
    .map((child) => pruneSpanTree(child as NestedSpanData))
    .filter((c): c is NestedSpanData => !!c)
    .sort((a, b) => a.startTime - b.startTime || a.spanId.localeCompare(b.spanId));
  return {
    ...node,
    spans: children,
  };
}

/**
 * Checks whether a subtree contains any model spans.
 */
function hasDescendantModelSpan(node: NestedSpanData): boolean {
  for (const child of node.spans || []) {
    const nested = child as NestedSpanData;
    if (isModelSpan(nested) || hasDescendantModelSpan(nested)) {
      return true;
    }
  }
  return false;
}

/**
 * Extracts text from an array of Parts.
 */
function extractTextFromParts(parts: Part[] | undefined): string {
  if (!parts || !Array.isArray(parts)) return '';
  return parts
    .map((p) => (typeof p?.text === 'string' ? p.text : ''))
    .filter(Boolean)
    .join('\n')
    .trim();
}

/**
 * Sums GenerationUsage records across multiple steps.
 */
function aggregateUsage(
  usages: (GenerationUsage | undefined)[]
): GenerationUsage | undefined {
  const valid = usages.filter((u): u is GenerationUsage => !!u);
  if (valid.length === 0) return undefined;

  const numericKeys: (keyof GenerationUsage)[] = [
    'inputTokens',
    'outputTokens',
    'totalTokens',
    'inputCharacters',
    'outputCharacters',
    'inputImages',
    'outputImages',
    'inputVideos',
    'outputVideos',
    'inputAudioFiles',
    'outputAudioFiles',
    'thoughtsTokens',
    'cachedContentTokens',
  ];

  const agg: GenerationUsage = {};
  let hasAny = false;

  for (const key of numericKeys) {
    let sum = 0;
    let present = false;
    for (const u of valid) {
      const val = u[key];
      if (typeof val === 'number') {
        sum += val;
        present = true;
      }
    }
    if (present) {
      (agg as any)[key] = sum;
      hasAny = true;
    }
  }

  const customAgg: Record<string, number> = {};
  let hasCustom = false;
  for (const u of valid) {
    if (u.custom && typeof u.custom === 'object') {
      for (const [k, v] of Object.entries(u.custom)) {
        if (typeof v === 'number') {
          customAgg[k] = (customAgg[k] ?? 0) + v;
          hasCustom = true;
        }
      }
    }
  }
  if (hasCustom) {
    agg.custom = customAgg;
    hasAny = true;
  }

  return hasAny ? agg : undefined;
}

/**
 * Normalizes a candidate message content array into valid Part objects.
 */
function normalizeParts(content: unknown): Part[] {
  if (!Array.isArray(content)) {
    if (typeof content === 'string') {
      return [{ text: content }];
    }
    return [];
  }
  return content.filter((p): p is Part => !!p && typeof p === 'object');
}

/**
 * Computes a deterministic signature for a message's role and non-metadata content
 * so repeated history messages can be deduplicated.
 */
function messageSignature(msg: MessageData): string {
  const simplifiedParts = (msg.content || []).map((p) => {
    if (p.text !== undefined) return { text: p.text };
    if (p.reasoning !== undefined) return { reasoning: p.reasoning };
    if (p.media !== undefined) return { media: p.media };
    if (p.toolRequest !== undefined) return { toolRequest: p.toolRequest };
    if (p.toolResponse !== undefined) return { toolResponse: p.toolResponse };
    if (p.data !== undefined) return { data: p.data };
    if (p.resource !== undefined) return { resource: p.resource };
    if (p.custom !== undefined) return { custom: p.custom };
    return p;
  });
  return `${msg.role}:${JSON.stringify(simplifiedParts)}`;
}

/**
 * Checks if a message is explicitly tagged as history in its metadata.
 */
function isExplicitHistoryMessage(msg: MessageData): boolean {
  const meta = msg.metadata;
  if (!meta) return false;
  return Boolean(meta['_genkit_history'] || meta['purpose'] === 'history');
}

/**
 * Extracts resume decisions from an AgentInput or GenerateActionOptions input object.
 */
function extractResumeDecisionsFromInput(inputObj: any): ResumeDecision[] {
  const decisions: ResumeDecision[] = [];
  const resume = inputObj?.resume;
  if (!resume || typeof resume !== 'object') return decisions;

  if (Array.isArray(resume.respond)) {
    for (const part of resume.respond as ToolResponsePart[]) {
      const tr = part?.toolResponse;
      if (tr && typeof tr.name === 'string') {
        decisions.push({
          name: tr.name,
          ref: tr.ref,
          strategy: 'respond',
          payload: tr.output,
          output: tr.output,
        });
      }
    }
  }

  if (Array.isArray(resume.restart)) {
    for (const part of resume.restart as ToolRequestPart[]) {
      const tr = part?.toolRequest;
      if (tr && typeof tr.name === 'string') {
        decisions.push({
          name: tr.name,
          ref: tr.ref,
          strategy: 'restart',
          payload: part.metadata?.['resumed'] ?? tr.input,
        });
      }
    }
  }

  return decisions;
}

/**
 * Extracts a TrajectoryBundle from a single TraceData object.
 */
export function extractTrajectoryFromTrace(trace: TraceData): TrajectoryBundle {
  const rootId = trace.traceId || 'root';
  const rawSpans = Object.values(trace.spans || {});

  // Discover sessionId from any span in the trace
  let sessionId: string | undefined;
  for (const s of rawSpans) {
    const sid =
      s.attributes?.['genkit:metadata:agent:sessionId'] ??
      s.attributes?.['session.id'];
    if (typeof sid === 'string' && sid.length > 0) {
      sessionId = sid;
      break;
    }
  }

  const stackedRoot = stackTraceSpans(trace);
  const prunedRoot = stackedRoot ? pruneSpanTree(stackedRoot) : undefined;

  if (!prunedRoot) {
    return {
      rootTrajectoryId: rootId,
      ...(sessionId ? { sessionId } : {}),
      trajectories: {
        [rootId]: {
          trajectoryId: rootId,
          tools: [],
          steps: [],
          startTime: trace.startTime ?? 0,
          endTime: trace.endTime ?? 0,
        },
      },
    };
  }

  const trajectories: Record<string, Trajectory> = {};

  function buildTrajectoryScope(
    scopeRoot: NestedSpanData,
    trajectoryId: string,
    parentTrajectoryId?: string
  ): Trajectory {
    const parentMap = new Map<string, NestedSpanData>();
    const modelSpans: NestedSpanData[] = [];
    const toolSpans: NestedSpanData[] = [];
    const scopeSpans: NestedSpanData[] = [];
    const subagentByToolSpanId = new Map<string, string>();
    const directSubagentSpans = new Map<string, string>();

    function findEnclosingToolSpan(
      span: NestedSpanData
    ): NestedSpanData | undefined {
      let curr = parentMap.get(span.spanId);
      while (curr && curr.spanId !== scopeRoot.spanId) {
        if (isToolSpan(curr)) return curr;
        curr = parentMap.get(curr.spanId);
      }
      return undefined;
    }

    function walkScope(node: NestedSpanData, isRootNode: boolean): void {
      if (!isRootNode) {
        // Check if this node is a subagent boundary
        if (isAgentSpan(node)) {
          const subId = node.spanId;
          const enclosingTool = findEnclosingToolSpan(node);
          if (enclosingTool) {
            subagentByToolSpanId.set(enclosingTool.spanId, subId);
          } else {
            directSubagentSpans.set(node.spanId, subId);
          }
          trajectories[subId] = buildTrajectoryScope(
            node,
            subId,
            trajectoryId
          );
          return;
        }

        // Also handle delegate_to_<agent> tool spans that directly contain model spans without a child agent span
        const rawName =
          node.displayName ||
          (node.attributes?.['genkit:name'] as string | undefined) ||
          '';
        const shortName = shortToolName(rawName);
        const hasChildAgent = (node.spans || []).some((c) =>
          isAgentSpan(c as NestedSpanData)
        );
        if (
          isToolSpan(node) &&
          shortName.startsWith('delegate_to_') &&
          !hasChildAgent &&
          hasDescendantModelSpan(node)
        ) {
          toolSpans.push(node);
          scopeSpans.push(node);
          const subId = node.spanId;
          subagentByToolSpanId.set(node.spanId, subId);
          trajectories[subId] = buildTrajectoryScope(
            node,
            subId,
            trajectoryId
          );
          return;
        }
      }

      scopeSpans.push(node);
      if (isModelSpan(node)) {
        modelSpans.push(node);
      } else if (isToolSpan(node) && !isRootNode) {
        toolSpans.push(node);
      }

      for (const child of node.spans || []) {
        const nestedChild = child as NestedSpanData;
        parentMap.set(nestedChild.spanId, node);
        walkScope(nestedChild, false);
      }
    }

    walkScope(scopeRoot, true);

    modelSpans.sort(
      (a, b) => a.startTime - b.startTime || a.spanId.localeCompare(b.spanId)
    );
    toolSpans.sort(
      (a, b) => a.startTime - b.startTime || a.spanId.localeCompare(b.spanId)
    );
    scopeSpans.sort(
      (a, b) => a.startTime - b.startTime || a.spanId.localeCompare(b.spanId)
    );

    // Helper to resolve turnIndex and enclosing turn span for any span in scope
    function getTurnContext(span: NestedSpanData): {
      turnIndex: number;
      turnSpan?: NestedSpanData;
    } {
      let curr: NestedSpanData | undefined = span;
      while (curr) {
        const name =
          curr.displayName ||
          (curr.attributes?.['genkit:name'] as string | undefined) ||
          '';
        const match = /^runTurn-(\d+)$/.exec(cleanActionName(name));
        if (match) {
          return {
            turnIndex: Math.max(0, Number.parseInt(match[1], 10) - 1),
            turnSpan: curr,
          };
        }
        if (curr.spanId === scopeRoot.spanId) break;
        curr = parentMap.get(curr.spanId);
      }
      return { turnIndex: 0, turnSpan: undefined };
    }

    // Determine agentName
    let agentName: string | undefined;
    const rootRawName =
      scopeRoot.displayName ||
      (scopeRoot.attributes?.['genkit:name'] as string | undefined);
    const cleanedRootName = cleanActionName(rootRawName);
    if (cleanedRootName.startsWith('delegate_to_')) {
      agentName = cleanedRootName.replace(/^delegate_to_/, '');
    } else if (
      isAgentSpan(scopeRoot) ||
      scopeRoot.attributes?.['genkit:type'] === 'flow' ||
      scopeRoot.attributes?.['genkit:metadata:subtype'] === 'flow' ||
      parentTrajectoryId !== undefined
    ) {
      agentName = cleanedRootName || undefined;
    } else if (cleanedRootName && !isModelSpan(scopeRoot)) {
      agentName = cleanedRootName;
    }

    // Extract tools and systemInstructions
    const toolsByName = new Map<string, ToolDefinition>();
    let systemInstructions: string | undefined;

    for (const s of scopeSpans) {
      const input = parseSpanJson<any>(s.attributes?.['genkit:input']);
      if (Array.isArray(input?.tools)) {
        for (const t of input.tools) {
          if (t && typeof t === 'object' && typeof t.name === 'string') {
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
      }
    }

    // Collect all toolResponse parts from model input messages and generate outputs for fallback pairing
    const toolResponsesFromMessages: {
      part: ToolResponsePart;
      claimed: boolean;
      resumed?: boolean;
    }[] = [];
    const seenToolRespKeys = new Set<string>();

    // Collect resume decisions across scope spans
    const resumeDecisions: ResumeDecision[] = [];

    for (const s of scopeSpans) {
      const input = parseSpanJson<any>(s.attributes?.['genkit:input']);
      for (const dec of extractResumeDecisionsFromInput(input)) {
        resumeDecisions.push(dec);
      }
      if (Array.isArray(input?.messages)) {
        for (const msg of input.messages as MessageData[]) {
          if (msg?.role === 'tool' && Array.isArray(msg.content)) {
            for (const part of msg.content) {
              if (part?.toolResponse) {
                const key = `${part.toolResponse.ref || ''}:${part.toolResponse.name}:${JSON.stringify(part.toolResponse.output)}`;
                if (!seenToolRespKeys.has(key)) {
                  seenToolRespKeys.add(key);
                  const isInterruptResp = Boolean(
                    part.metadata?.['interruptResponse'] ||
                      msg.metadata?.['resumed']
                  );
                  toolResponsesFromMessages.push({
                    part: part as ToolResponsePart,
                    claimed: false,
                    resumed: isInterruptResp,
                  });
                  if (part.metadata?.['interruptResponse']) {
                    resumeDecisions.push({
                      name: part.toolResponse.name,
                      ref: part.toolResponse.ref,
                      strategy: 'respond',
                      payload: part.toolResponse.output,
                      output: part.toolResponse.output,
                    });
                  }
                }
              }
            }
          } else if (msg?.role === 'model' && Array.isArray(msg.content)) {
            for (const part of msg.content) {
              if (
                part?.toolRequest &&
                part.metadata?.['resumed'] !== undefined
              ) {
                resumeDecisions.push({
                  name: part.toolRequest.name,
                  ref: part.toolRequest.ref,
                  strategy: 'restart',
                  payload: part.metadata['resumed'],
                });
              }
            }
          }
        }
      }
    }

    const claimedToolSpanIds = new Set<string>();
    const steps: TrajectoryStep[] = [];
    const emittedMessageSigs = new Set<string>();

    // Helper to extract output message and metadata from a model span
    function getModelOutputData(modelSpan: NestedSpanData): {
      message?: MessageData;
      finishReason?: string;
      usage?: GenerationUsage;
      error?: { message: string; stack?: string };
    } {
      const out = parseSpanJson<any>(modelSpan.attributes?.['genkit:output']);
      const candidate = Array.isArray(out?.candidates)
        ? out.candidates[0]
        : undefined;
      const message: MessageData | undefined =
        out?.message ?? candidate?.message;
      const finishReason: string | undefined =
        out?.finishReason ?? candidate?.finishReason;
      const usage: GenerationUsage | undefined =
        out?.usage ?? candidate?.usage;

      let error: { message: string; stack?: string } | undefined;
      if (out?.error && typeof out.error.message === 'string') {
        error = {
          message: out.error.message,
          ...(typeof out.error.details?.stack === 'string'
            ? { stack: out.error.details.stack }
            : {}),
        };
      } else if (
        getSpanStatus(modelSpan) === '✖' ||
        modelSpan.status?.code === 2
      ) {
        const errMsg =
          modelSpan.status?.message ||
          (modelSpan.attributes?.['genkit:error'] as string | undefined) ||
          out?.finishMessage ||
          'Model execution failed';
        error = { message: errMsg };
      }

      return { message, finishReason, usage, error };
    }

    // Check if an enclosing generate/turn span reported finishReason === 'interrupted'
    function isEnclosingTurnInterrupted(span: NestedSpanData): boolean {
      let curr = parentMap.get(span.spanId);
      while (curr) {
        const out = parseSpanJson<any>(curr.attributes?.['genkit:output']);
        if (out?.finishReason === 'interrupted') return true;
        if (curr.spanId === scopeRoot.spanId) break;
        curr = parentMap.get(curr.spanId);
      }
      return false;
    }

    // Check if an enclosing generate/turn span has revised interrupt metadata on a toolRequest
    function getEnclosingInterruptMetadata(
      span: NestedSpanData,
      toolName: string,
      ref?: string
    ): unknown {
      let curr = parentMap.get(span.spanId);
      while (curr) {
        const out = parseSpanJson<any>(curr.attributes?.['genkit:output']);
        const msg: MessageData | undefined =
          out?.message ?? out?.candidates?.[0]?.message;
        if (msg && Array.isArray(msg.content)) {
          for (const p of msg.content) {
            if (
              p?.toolRequest &&
              p.toolRequest.name === toolName &&
              (ref === undefined || p.toolRequest.ref === ref) &&
              p.metadata?.['interrupt'] !== undefined
            ) {
              return p.metadata['interrupt'];
            }
          }
        }
        if (curr.spanId === scopeRoot.spanId) break;
        curr = parentMap.get(curr.spanId);
      }
      return undefined;
    }

    // Process each model span in chronological order
    for (let mIdx = 0; mIdx < modelSpans.length; mIdx++) {
      const modelSpan = modelSpans[mIdx];
      const nextModelSpan = modelSpans[mIdx + 1];
      const { turnIndex, turnSpan } = getTurnContext(modelSpan);
      const input = parseSpanJson<any>(modelSpan.attributes?.['genkit:input']);
      const messages: MessageData[] = Array.isArray(input?.messages)
        ? input.messages
        : [];

      // 1. Extract systemInstructions from initial system or agentPreamble messages
      for (const msg of messages) {
        if (!msg || !Array.isArray(msg.content)) continue;
        if (
          msg.role === 'system' ||
          (msg.metadata?.['agentPreamble'] && msg.role !== 'user')
        ) {
          const sysText = extractTextFromParts(msg.content);
          if (sysText && !systemInstructions) {
            systemInstructions = sysText;
          }
        }
      }

      // 2. Find new user / system messages introduced before this model call.
      // Any message appearing at or before the last 'model' or 'tool' message in `messages`
      // belongs to an earlier step or turn in the conversation history.
      let lastModelOrToolIdx = -1;
      for (let i = 0; i < messages.length; i++) {
        const r = messages[i]?.role;
        if (r === 'model' || r === 'tool') {
          lastModelOrToolIdx = i;
        }
      }

      const candidateNewMessages = messages.slice(lastModelOrToolIdx + 1);
      const hasUserInCandidate = candidateNewMessages.some(
        (m) => m?.role === 'user' && !isExplicitHistoryMessage(m)
      );

      for (const msg of candidateNewMessages) {
        if (!msg || (msg.role !== 'user' && msg.role !== 'system')) continue;
        if (isExplicitHistoryMessage(msg)) continue;
        if (msg.metadata?.['agentPreamble'] && msg.role === 'system') continue;

        // If this is the initial system message and there is also a user prompt,
        // keep it in systemInstructions rather than duplicating it as Step 1.
        if (
          msg.role === 'system' &&
          steps.length === 0 &&
          hasUserInCandidate
        ) {
          continue;
        }

        const parts = normalizeParts(msg.content).filter(
          (p) => !p.toolRequest && !p.toolResponse
        );
        if (parts.length === 0) continue;

        const sig = `${turnIndex}:${messageSignature(msg)}`;
        if (emittedMessageSigs.has(sig)) continue;
        emittedMessageSigs.add(sig);

        const stepNum = steps.length + 1;
        const sourceSpanId = turnSpan?.spanId || modelSpan.spanId;
        steps.push({
          stepId: `${trace.traceId}-step-${stepNum}-${sourceSpanId}`,
          stepNumber: stepNum,
          traceId: trace.traceId,
          spanId: sourceSpanId,
          turnIndex,
          source: msg.role === 'system' ? 'system' : 'user',
          startTime: modelSpan.startTime,
          endTime: modelSpan.startTime,
          latencyMs: 0,
          content: parts,
          actions: [],
        });
      }

      // 3. Build the Agent step for this model span
      const {
        message: outMsg,
        finishReason,
        usage,
        error: modelError,
      } = getModelOutputData(modelSpan);

      const rawOutParts = normalizeParts(outMsg?.content);
      const contentParts: Part[] = [];
      const toolRequestParts: ToolRequestPart[] = [];

      for (const p of rawOutParts) {
        if (p.toolRequest) {
          toolRequestParts.push(p as ToolRequestPart);
        } else if (!p.toolResponse) {
          contentParts.push(p);
        }
      }

      const actions: TrajectoryAction[] = [];
      const pairedToolEndTimes: number[] = [];

      for (let reqIdx = 0; reqIdx < toolRequestParts.length; reqIdx++) {
        const reqPart = toolRequestParts[reqIdx];
        const req = reqPart.toolRequest;
        const reqRef = req.ref;
        const reqShortName = shortToolName(req.name);

        // Pair with a tool span:
        // (a) Match by explicit genkit:metadata:tool:callId === reqRef
        let matchedToolSpan: NestedSpanData | undefined;
        if (reqRef) {
          matchedToolSpan = toolSpans.find(
            (ts) =>
              !claimedToolSpanIds.has(ts.spanId) &&
              (ts.attributes?.['genkit:metadata:tool:callId'] === reqRef ||
                ts.attributes?.['genkit:tool:callId'] === reqRef)
          );
        }

        // (b) Fallback: match by (shortToolName, callOrder) within the current model step window
        if (!matchedToolSpan) {
          matchedToolSpan = toolSpans.find((ts) => {
            if (claimedToolSpanIds.has(ts.spanId)) return false;
            if (ts.attributes?.['genkit:metadata:resumed'] !== undefined) {
              return false;
            }
            const tsName = shortToolName(
              ts.displayName ||
                (ts.attributes?.['genkit:name'] as string | undefined)
            );
            if (tsName !== reqShortName) return false;
            const afterCurrentModel = ts.startTime >= modelSpan.startTime;
            const beforeNextModel =
              !nextModelSpan || ts.startTime <= nextModelSpan.startTime;
            return afterCurrentModel && beforeNextModel;
          });
        }

        // (c) Relaxed fallback within same turn
        if (!matchedToolSpan) {
          matchedToolSpan = toolSpans.find((ts) => {
            if (claimedToolSpanIds.has(ts.spanId)) return false;
            if (ts.attributes?.['genkit:metadata:resumed'] !== undefined) {
              return false;
            }
            const tsName = shortToolName(
              ts.displayName ||
                (ts.attributes?.['genkit:name'] as string | undefined)
            );
            return (
              tsName === reqShortName &&
              getTurnContext(ts).turnIndex === turnIndex
            );
          });
        }

        if (matchedToolSpan) {
          claimedToolSpanIds.add(matchedToolSpan.spanId);
          if (matchedToolSpan.endTime) {
            pairedToolEndTimes.push(matchedToolSpan.endTime);
          }
        }

        // Also look for a matching toolResponse in subsequent messages
        let matchedRespEntry = reqRef
          ? toolResponsesFromMessages.find(
              (r) => !r.claimed && r.part.toolResponse.ref === reqRef
            )
          : undefined;
        if (!matchedRespEntry) {
          matchedRespEntry = toolResponsesFromMessages.find(
            (r) =>
              !r.claimed &&
              shortToolName(r.part.toolResponse.name) === reqShortName
          );
        }
        if (matchedRespEntry) {
          matchedRespEntry.claimed = true;
        }

        const toolSpanInput = matchedToolSpan
          ? parseSpanJson(matchedToolSpan.attributes?.['genkit:input'])
          : undefined;
        const rawToolSpanOutput = matchedToolSpan
          ? parseSpanJson(matchedToolSpan.attributes?.['genkit:output'])
          : undefined;

        const actionInput =
          req.input !== undefined ? req.input : toolSpanInput;
        let actionOutput =
          rawToolSpanOutput !== undefined
            ? rawToolSpanOutput
            : matchedRespEntry?.part.toolResponse.output;

        const toolSpanErrAttr =
          (matchedToolSpan?.attributes?.['genkit:error'] as
            | string
            | undefined) || matchedToolSpan?.status?.message;
        const enclosingInterruptMeta = getEnclosingInterruptMetadata(
          modelSpan,
          req.name,
          reqRef
        );

        const isInterrupted = Boolean(
          reqPart.metadata?.['interrupt'] !== undefined ||
            reqPart.metadata?.['resolvedInterrupt'] !== undefined ||
            enclosingInterruptMeta !== undefined ||
            matchedToolSpan?.attributes?.['genkit:metadata:interrupt'] !==
              undefined ||
            (toolSpanErrAttr &&
              toolSpanErrAttr.includes('ToolInterruptError')) ||
            ((finishReason === 'interrupted' ||
              isEnclosingTurnInterrupted(modelSpan)) &&
              (!matchedToolSpan ||
                getSpanStatus(matchedToolSpan) === '✖' ||
                actionOutput === undefined))
        );

        const subtrajectoryId = matchedToolSpan
          ? subagentByToolSpanId.get(matchedToolSpan.spanId)
          : undefined;

        let actionType: TrajectoryAction['type'] = 'tool';
        if (subtrajectoryId) {
          actionType = 'subagent';
        } else if (isInterrupted) {
          actionType = 'interrupt';
        }

        let actionStatus: TrajectoryAction['status'] = 'success';
        let actionError: string | undefined;

        if (isInterrupted) {
          actionStatus = 'interrupted';
        } else if (
          matchedToolSpan &&
          (getSpanStatus(matchedToolSpan) === '✖' ||
            matchedToolSpan.status?.code === 2)
        ) {
          actionStatus = 'error';
          actionError = toolSpanErrAttr || 'Tool execution failed';
        } else if (matchedToolSpan && !matchedToolSpan.endTime) {
          actionStatus = 'pending';
        } else if (
          !matchedToolSpan &&
          !matchedRespEntry &&
          finishReason !== 'stop'
        ) {
          actionStatus = 'pending';
        }

        // Check if this interrupted action was resumed in the same trace
        let resumeInfo: TrajectoryAction['resume'] | undefined;
        if (isInterrupted) {
          const decisionIdx = resumeDecisions.findIndex(
            (d) =>
              (reqRef && d.ref === reqRef) ||
              shortToolName(d.name) === reqShortName
          );
          if (decisionIdx >= 0) {
            const decision = resumeDecisions.splice(decisionIdx, 1)[0];
            resumeInfo = {
              strategy: decision.strategy,
              ...(decision.payload !== undefined
                ? { payload: decision.payload }
                : {}),
            };
            if (decision.output !== undefined && actionOutput === undefined) {
              actionOutput = decision.output;
            }
            if (decision.strategy === 'restart') {
              // Check if there is a resumed tool span that executed the restart
              const restartedSpan = toolSpans.find(
                (ts) =>
                  !claimedToolSpanIds.has(ts.spanId) &&
                  ts.attributes?.['genkit:metadata:resumed'] !== undefined &&
                  shortToolName(
                    ts.displayName ||
                      (ts.attributes?.['genkit:name'] as string | undefined)
                  ) === reqShortName
              );
              if (restartedSpan) {
                claimedToolSpanIds.add(restartedSpan.spanId);
                const restartedOut = parseSpanJson(
                  restartedSpan.attributes?.['genkit:output']
                );
                if (restartedOut !== undefined && actionOutput === undefined) {
                  actionOutput = restartedOut;
                }
              }
            }
          }
        }

        const startTime = matchedToolSpan?.startTime;
        const endTime = matchedToolSpan?.endTime;
        const durationMs =
          startTime !== undefined && endTime !== undefined && endTime >= startTime
            ? endTime - startTime
            : undefined;

        actions.push({
          actionId: reqRef || `${modelSpan.spanId}-action-${reqIdx + 1}`,
          ...(matchedToolSpan ? { spanId: matchedToolSpan.spanId } : {}),
          name: req.name,
          type: actionType,
          status: actionStatus,
          ...(actionInput !== undefined ? { input: actionInput } : {}),
          ...(actionOutput !== undefined ? { output: actionOutput } : {}),
          ...(actionError ? { error: actionError } : {}),
          ...(startTime !== undefined ? { startTime } : {}),
          ...(endTime !== undefined ? { endTime } : {}),
          ...(durationMs !== undefined ? { durationMs } : {}),
          ...(subtrajectoryId ? { subtrajectoryId } : {}),
          ...(resumeInfo ? { resume: resumeInfo } : {}),
        });
      }

      const stepStartTime = modelSpan.startTime;
      const stepEndTime = Math.max(
        modelSpan.endTime || modelSpan.startTime,
        ...pairedToolEndTimes
      );
      const latencyMs = Math.max(0, stepEndTime - stepStartTime);

      let stopReason: TrajectoryStep['stopReason'] = 'completed';
      if (
        actions.some(
          (a) => a.type === 'interrupt' || a.status === 'interrupted'
        ) ||
        finishReason === 'interrupted' ||
        (toolRequestParts.length > 0 && isEnclosingTurnInterrupted(modelSpan))
      ) {
        stopReason = 'interrupted';
      } else if (modelError || finishReason === 'failed') {
        stopReason = 'errored';
      } else if (finishReason === 'aborted' || finishReason === 'cancelled') {
        stopReason = 'cancelled';
      } else if (finishReason === 'length') {
        stopReason = 'max_tokens';
      } else if (actions.length > 0) {
        stopReason = 'tools';
      }

      const modelName = cleanActionName(
        (modelSpan.attributes?.['genkit:name'] as string | undefined) ||
          modelSpan.displayName
      );

      const stepNum = steps.length + 1;
      steps.push({
        stepId: `${trace.traceId}-step-${stepNum}-${modelSpan.spanId}`,
        stepNumber: stepNum,
        traceId: trace.traceId,
        spanId: modelSpan.spanId,
        turnIndex,
        source: 'agent',
        startTime: stepStartTime,
        endTime: stepEndTime,
        latencyMs,
        ...(modelName ? { model: modelName } : {}),
        content: contentParts,
        actions,
        stopReason,
        ...(modelError ? { error: modelError } : {}),
        ...(usage ? { usage } : {}),
      });
    }

    // Fallback: if no model spans exist in a turn (e.g. live streaming in progress or direct agent input),
    // extract any user/system message from runTurn-<N> or scopeRoot input.
    if (modelSpans.length === 0) {
      for (const s of scopeSpans) {
        const { turnIndex } = getTurnContext(s);
        const input = parseSpanJson<any>(s.attributes?.['genkit:input']);
        const candidateMessages: MessageData[] = [];
        if (
          input?.message &&
          typeof input.message === 'object' &&
          Array.isArray(input.message.content)
        ) {
          candidateMessages.push(input.message as MessageData);
        } else if (Array.isArray(input?.messages)) {
          candidateMessages.push(...(input.messages as MessageData[]));
        }

        for (const msg of candidateMessages) {
          if (!msg || (msg.role !== 'user' && msg.role !== 'system')) continue;
          if (isExplicitHistoryMessage(msg)) continue;
          const parts = normalizeParts(msg.content).filter(
            (p) => !p.toolRequest && !p.toolResponse
          );
          if (parts.length === 0) continue;

          const sig = `${turnIndex}:${messageSignature(msg)}`;
          if (emittedMessageSigs.has(sig)) continue;
          emittedMessageSigs.add(sig);

          const stepNum = steps.length + 1;
          steps.push({
            stepId: `${trace.traceId}-step-${stepNum}-${s.spanId}`,
            stepNumber: stepNum,
            traceId: trace.traceId,
            spanId: s.spanId,
            turnIndex,
            source: msg.role === 'system' ? 'system' : 'user',
            startTime: s.startTime,
            endTime: s.startTime,
            latencyMs: 0,
            content: parts,
            actions: [],
          });
        }
      }
    }

    const startTime =
      scopeRoot.startTime ||
      (steps.length > 0 ? steps[0].startTime : trace.startTime ?? 0);
    const endTime = Math.max(
      scopeRoot.endTime || 0,
      ...steps.map((s) => s.endTime),
      startTime
    );

    const trajectoryUsage = aggregateUsage(steps.map((s) => s.usage));

    return {
      trajectoryId,
      ...(parentTrajectoryId ? { parentTrajectoryId } : {}),
      ...(agentName ? { agentName } : {}),
      ...(systemInstructions ? { systemInstructions } : {}),
      tools: Array.from(toolsByName.values()),
      steps,
      ...(trajectoryUsage ? { usage: trajectoryUsage } : {}),
      startTime,
      endTime,
    };
  }

  trajectories[rootId] = buildTrajectoryScope(prunedRoot, rootId, undefined);

  return {
    rootTrajectoryId: rootId,
    ...(sessionId ? { sessionId } : {}),
    trajectories,
  };
}

/**
 * Merges multiple TraceData records sharing an agent session into a single continuous multi-turn TrajectoryBundle.
 */
export function mergeSessionTrajectories(
  traces: TraceData[]
): TrajectoryBundle {
  if (!traces || traces.length === 0) {
    return {
      rootTrajectoryId: 'root',
      trajectories: {
        root: {
          trajectoryId: 'root',
          tools: [],
          steps: [],
          startTime: 0,
          endTime: 0,
        },
      },
    };
  }

  const getTraceStartTime = (t: TraceData): number => {
    if (typeof t.startTime === 'number' && t.startTime > 0) return t.startTime;
    const spans = Object.values(t.spans || {});
    if (spans.length === 0) return 0;
    return Math.min(...spans.map((s) => s.startTime || 0));
  };

  const sortedTraces = [...traces].sort(
    (a, b) =>
      getTraceStartTime(a) - getTraceStartTime(b) ||
      a.traceId.localeCompare(b.traceId)
  );

  if (sortedTraces.length === 1) {
    return extractTrajectoryFromTrace(sortedTraces[0]);
  }

  const bundles = sortedTraces.map((t) => extractTrajectoryFromTrace(t));
  const rootTrajectoryId = bundles[0].rootTrajectoryId;
  const sessionId = bundles.find((b) => b.sessionId)?.sessionId;

  const mergedTrajectories: Record<string, Trajectory> = {};
  const toolsByName = new Map<string, ToolDefinition>();
  let systemInstructions: string | undefined;
  let agentName: string | undefined;
  const mergedSteps: TrajectoryStep[] = [];

  // Track seen user/system message signatures across turns to prevent duplicate history steps
  let turnOffset = 0;

  for (let tIdx = 0; tIdx < bundles.length; tIdx++) {
    const trace = sortedTraces[tIdx];
    const bundle = bundles[tIdx];
    const rootTraj = bundle.trajectories[bundle.rootTrajectoryId];

    // Copy all subagent trajectories into mergedTrajectories
    for (const [key, traj] of Object.entries(bundle.trajectories)) {
      if (key === bundle.rootTrajectoryId) continue;
      mergedTrajectories[key] = {
        ...traj,
        parentTrajectoryId:
          traj.parentTrajectoryId === bundle.rootTrajectoryId
            ? rootTrajectoryId
            : traj.parentTrajectoryId,
      };
    }

    if (!rootTraj) continue;

    if (!agentName && rootTraj.agentName) {
      agentName = rootTraj.agentName;
    }
    if (!systemInstructions && rootTraj.systemInstructions) {
      systemInstructions = rootTraj.systemInstructions;
    }
    for (const tool of rootTraj.tools) {
      if (!toolsByName.has(tool.name)) {
        toolsByName.set(tool.name, tool);
      }
    }

    // Check if this trace contains a resume decision for an interrupted action in an earlier trace
    const traceResumeDecisions: ResumeDecision[] = [];
    for (const s of Object.values(trace.spans || {}) as SpanData[]) {
      const input = parseSpanJson<any>(s.attributes?.['genkit:input']);
      traceResumeDecisions.push(...extractResumeDecisionsFromInput(input));
      if (Array.isArray(input?.messages)) {
        for (const msg of input.messages as MessageData[]) {
          if (msg?.role === 'tool' && Array.isArray(msg.content)) {
            for (const part of msg.content) {
              if (
                part?.toolResponse &&
                part.metadata?.['interruptResponse'] !== undefined
              ) {
                traceResumeDecisions.push({
                  name: part.toolResponse.name,
                  ref: part.toolResponse.ref,
                  strategy: 'respond',
                  payload: part.toolResponse.output,
                  output: part.toolResponse.output,
                });
              }
            }
          }
        }
      }
    }

    for (const decision of traceResumeDecisions) {
      // Find the most recent interrupted action in mergedSteps matching this decision
      for (let sIdx = mergedSteps.length - 1; sIdx >= 0; sIdx--) {
        const prevStep = mergedSteps[sIdx];
        const matchingAction = prevStep.actions.find(
          (a) =>
            (a.type === 'interrupt' || a.status === 'interrupted') &&
            !a.resume &&
            ((decision.ref && a.actionId === decision.ref) ||
              shortToolName(a.name) === shortToolName(decision.name))
        );
        if (matchingAction) {
          matchingAction.resume = {
            strategy: decision.strategy,
            ...(decision.payload !== undefined
              ? { payload: decision.payload }
              : {}),
          };
          if (
            decision.output !== undefined &&
            matchingAction.output === undefined
          ) {
            matchingAction.output = decision.output;
          }
          break;
        }
      }
    }

    let maxLocalTurn = 0;
    for (const step of rootTraj.steps) {
      const localTurn = step.turnIndex ?? 0;
      if (localTurn > maxLocalTurn) maxLocalTurn = localTurn;
      const effectiveTurnIndex = turnOffset + localTurn;
      const stepNumber = mergedSteps.length + 1;
      mergedSteps.push({
        ...step,
        stepNumber,
        turnIndex: effectiveTurnIndex,
      });
    }
    turnOffset += maxLocalTurn + 1;
  }

  const startTime =
    mergedSteps.length > 0
      ? Math.min(...mergedSteps.map((s) => s.startTime))
      : getTraceStartTime(sortedTraces[0]);
  const endTime =
    mergedSteps.length > 0
      ? Math.max(...mergedSteps.map((s) => s.endTime))
      : Math.max(...sortedTraces.map((t) => t.endTime ?? getTraceStartTime(t)));

  const mergedUsage = aggregateUsage(mergedSteps.map((s) => s.usage));

  mergedTrajectories[rootTrajectoryId] = {
    trajectoryId: rootTrajectoryId,
    ...(agentName ? { agentName } : {}),
    ...(systemInstructions ? { systemInstructions } : {}),
    tools: Array.from(toolsByName.values()),
    steps: mergedSteps,
    ...(mergedUsage ? { usage: mergedUsage } : {}),
    startTime,
    endTime,
  };

  return {
    rootTrajectoryId,
    ...(sessionId ? { sessionId } : {}),
    trajectories: mergedTrajectories,
  };
}
