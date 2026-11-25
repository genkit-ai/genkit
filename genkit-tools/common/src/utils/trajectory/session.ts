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
  ToolDefinition,
  TraceData,
  Trajectory,
  TrajectoryAction,
  TrajectoryBundle,
  TrajectoryStep,
} from '../../types/index.js';
import { isToolSpan, shortToolName } from '../trace.js';
import { extractTrajectoryFromTrace } from './extract.js';
import {
  collectScopeResumeDecisions,
  resolveInTraceInterruptResume,
} from './interrupts.js';
import {
  aggregateUsage,
  isExcludedFromTrajectory,
  parseSpanObject,
  type ParsedSpanInput,
} from './span-parser.js';
import { parseToolSpanResult } from './tool-pairing.js';

function deduplicateTools(roots: Trajectory[]): ToolDefinition[] {
  const seen = new Set<string>();
  const merged: ToolDefinition[] = [];
  for (const root of roots) {
    for (const tool of root.tools) {
      if (!seen.has(tool.name)) {
        seen.add(tool.name);
        merged.push(tool);
      }
    }
  }
  return merged;
}

function mergeRootSteps(roots: Trajectory[]): TrajectoryStep[] {
  const mergedSteps: TrajectoryStep[] = [];
  let turnOffset = 0;

  for (const root of roots) {
    let maxTurnInRoot = -1;
    for (const step of root.steps) {
      if (step.turnIndex !== undefined && step.turnIndex > maxTurnInRoot) {
        maxTurnInRoot = step.turnIndex;
      }
      mergedSteps.push({
        ...step,
        stepNumber: mergedSteps.length + 1,
        ...(step.turnIndex !== undefined
          ? { turnIndex: turnOffset + step.turnIndex }
          : {}),
      });
    }
    if (maxTurnInRoot >= 0) {
      turnOffset += maxTurnInRoot + 1;
    }
  }

  return mergedSteps;
}

function resolveMergedOutcome(
  latestRoot: Trajectory,
  mergedSteps: TrajectoryStep[]
): Pick<Trajectory, 'status' | 'finishReason' | 'error'> {
  if (
    latestRoot.status === 'error' ||
    latestRoot.status === 'interrupted' ||
    latestRoot.status === 'in-progress'
  ) {
    return {
      status: latestRoot.status,
      ...(latestRoot.finishReason
        ? { finishReason: latestRoot.finishReason }
        : {}),
      ...(latestRoot.error ? { error: latestRoot.error } : {}),
    };
  }

  const lastStep = mergedSteps.at(-1);
  if (
    lastStep?.finishReason === 'interrupted' ||
    mergedSteps.some((s) =>
      s.actions.some((a) => a.status === 'interrupted' && !a.resume)
    )
  ) {
    return { status: 'interrupted', finishReason: 'interrupted' };
  }

  return {
    status: latestRoot.status,
    ...(latestRoot.finishReason
      ? { finishReason: latestRoot.finishReason }
      : {}),
    ...(latestRoot.error ? { error: latestRoot.error } : {}),
  };
}

/**
 * Merges multiple chronologically ordered (or unordered) `TrajectoryBundle`s from
 * the same session into a single unified `TrajectoryBundle`.
 */
export function mergeTrajectoryBundles(
  bundles: TrajectoryBundle[]
): TrajectoryBundle {
  if (bundles.length === 0) {
    return { rootTrajectoryId: '', trajectories: {} };
  }
  if (bundles.length === 1) {
    return bundles[0];
  }

  const sortedBundles = [...bundles].sort((a, b) => {
    const aStart = a.trajectories[a.rootTrajectoryId]?.startTime ?? 0;
    const bStart = b.trajectories[b.rootTrajectoryId]?.startTime ?? 0;
    return aStart - bStart;
  });

  const roots = sortedBundles
    .map((b) => b.trajectories[b.rootTrajectoryId])
    .filter((t): t is Trajectory => t !== undefined);

  if (roots.length === 0) {
    return sortedBundles[0];
  }

  const firstBundle = sortedBundles[0];
  const rootTrajectoryId = firstBundle.rootTrajectoryId;
  const sessionId = sortedBundles.find((b) => b.sessionId)?.sessionId;
  const mergedSteps = mergeRootSteps(roots);
  const latestRoot = roots[roots.length - 1];
  const agentName = roots.find((r) => r.agentName)?.agentName;
  const systemInstructions = roots.find(
    (r) => r.systemInstructions
  )?.systemInstructions;
  const usage = aggregateUsage(roots.map((r) => r.usage));
  const startTimes = roots.map((r) => r.startTime).filter((t) => t > 0);
  const endTimes = roots.map((r) => r.endTime).filter((t) => t > 0);

  const mergedRoot: Trajectory = {
    trajectoryId: rootTrajectoryId,
    ...(agentName ? { agentName } : {}),
    ...resolveMergedOutcome(latestRoot, mergedSteps),
    ...(systemInstructions ? { systemInstructions } : {}),
    tools: deduplicateTools(roots),
    steps: mergedSteps,
    ...(usage ? { usage } : {}),
    startTime: startTimes.length > 0 ? Math.min(...startTimes) : 0,
    endTime: endTimes.length > 0 ? Math.max(...endTimes) : 0,
  };

  const trajectories: Record<string, Trajectory> = {
    [rootTrajectoryId]: mergedRoot,
  };

  for (const bundle of sortedBundles) {
    for (const [id, traj] of Object.entries(bundle.trajectories)) {
      if (id === bundle.rootTrajectoryId) continue;
      trajectories[id] =
        traj.parentTrajectoryId === bundle.rootTrajectoryId
          ? { ...traj, parentTrajectoryId: rootTrajectoryId }
          : traj;
    }
  }

  return {
    rootTrajectoryId,
    ...(sessionId ? { sessionId } : {}),
    trajectories,
  };
}

function pairCrossTraceInterruptResumes(
  sortedTraces: TraceData[],
  bundle: TrajectoryBundle
): TrajectoryBundle {
  const traceOrder = new Map(sortedTraces.map((t, idx) => [t.traceId, idx]));
  const trajectories = { ...bundle.trajectories };

  for (let traceIdx = 1; traceIdx < sortedTraces.length; traceIdx++) {
    const trace = sortedTraces[traceIdx];
    const spans = Object.values(trace.spans ?? {}).filter(
      (s) => !isExcludedFromTrajectory(s)
    );
    const inputs = spans
      .map((s) =>
        parseSpanObject<ParsedSpanInput>(s.attributes?.['genkit:input'])
      )
      .filter((i): i is ParsedSpanInput => i !== undefined);
    const resumeDecisions = collectScopeResumeDecisions(inputs);
    if (resumeDecisions.length === 0) continue;

    const restartedToolSpans = spans.filter(
      (s) =>
        isToolSpan(s) && s.attributes?.['genkit:metadata:resumed'] !== undefined
    ) as NestedSpanData[];
    const claimedToolSpanIds = new Set<string>();

    for (const [trajId, traj] of Object.entries(trajectories)) {
      let trajChanged = false;
      const updatedSteps = traj.steps.map((step) => {
        const stepOrder = step.traceId
          ? (traceOrder.get(step.traceId) ?? -1)
          : -1;
        if (stepOrder < 0 || stepOrder >= traceIdx) return step;

        let stepChanged = false;
        const updatedActions: TrajectoryAction[] = step.actions.map(
          (action) => {
            if (
              action.status !== 'interrupted' ||
              action.resume !== undefined
            ) {
              return action;
            }
            const resolved = resolveInTraceInterruptResume({
              reqRef: action.actionId,
              reqShortName: shortToolName(action.name),
              resumeDecisions,
              toolSpans: restartedToolSpans,
              claimedToolSpanIds,
              actionOutput: action.output,
              actionContent: action.content,
            });
            if (!resolved.resumeInfo) return action;

            stepChanged = true;
            let resolvedOutput = resolved.resolvedOutput;
            let resolvedContent = resolved.resolvedContent;
            if (resolved.restartedSpan) {
              const restarted = parseToolSpanResult(resolved.restartedSpan);
              resolvedOutput ??= restarted.output;
              resolvedContent ??= restarted.content;
            }
            return {
              ...action,
              ...(resolvedOutput !== undefined
                ? { output: resolvedOutput }
                : {}),
              ...(resolvedContent ? { content: resolvedContent } : {}),
              resume: resolved.resumeInfo,
            };
          }
        );

        if (!stepChanged) return step;
        trajChanged = true;
        return { ...step, actions: updatedActions };
      });

      if (trajChanged) {
        trajectories[trajId] = { ...traj, steps: updatedSteps };
      }
    }
  }

  const root = trajectories[bundle.rootTrajectoryId];
  if (root) {
    const latestTraceId = sortedTraces[sortedTraces.length - 1].traceId;
    const latestStep = root.steps.at(-1);
    const hasUnresumedInterrupt = root.steps.some((s) =>
      s.actions.some((a) => a.status === 'interrupted' && !a.resume)
    );
    if (
      !hasUnresumedInterrupt &&
      root.status === 'interrupted' &&
      latestStep?.traceId === latestTraceId &&
      latestStep.finishReason &&
      latestStep.finishReason !== 'interrupted'
    ) {
      trajectories[bundle.rootTrajectoryId] = {
        ...root,
        status: latestStep.finishReason === 'failed' ? 'error' : 'success',
        finishReason: latestStep.finishReason === 'failed' ? 'failed' : 'stop',
      };
    }
  }

  return { ...bundle, trajectories };
}

/**
 * Extracts and merges multiple `TraceData` objects from the same session into
 * a single chronological `TrajectoryBundle`.
 */
export function extractTrajectoryFromTraces(
  traces: TraceData[]
): TrajectoryBundle {
  if (traces.length === 0) {
    return { rootTrajectoryId: '', trajectories: {} };
  }
  const sortedTraces = [...traces].sort(
    (a, b) => (a.startTime ?? 0) - (b.startTime ?? 0)
  );
  const merged = mergeTrajectoryBundles(
    sortedTraces.map(extractTrajectoryFromTrace)
  );
  return pairCrossTraceInterruptResumes(sortedTraces, merged);
}
