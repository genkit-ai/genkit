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

/**
 * `useGenkitAgent` — the headline React hook for the Agents API.
 *
 * Absorbs what every page in the original testapp manually wires:
 *
 *   - `for await (const chunk of response.stream)` + dispatch on chunk.type
 *   - Continuation-token round-trip (no state vs snapshotId fork)
 *   - Mid-stream streaming text accumulation, separate from committed history
 *   - Tool call lifecycle (call → result → error)
 *   - Interrupt detection (in-stream `interrupt` events) + resumption helpers
 *   - Foreground → background phase transition (detached event), same hook
 *     continues to render in the background phase via polling
 *   - URL-bookmark restoration via `resumeFromContinuation`
 *   - Custom state (typed via the agent's stateSchema)
 *   - Artifact collection
 */
import type { AgentEvent } from 'genkit/beta/client';
import { runFlow, streamFlow, walkAgentEvent } from 'genkit/beta/client';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

export type ToolCallState = 'call' | 'result' | 'error';
export interface ToolCall<I = unknown, O = unknown> {
  id: string;
  name: string;
  input: I;
  output?: O;
  state: ToolCallState;
}

export interface AgentMessage {
  role: 'user' | 'model' | 'tool' | 'system' | string;
  content: Array<{
    text?: string;
    toolRequest?: { name: string; input?: unknown; ref?: string };
    toolResponse?: { name: string; output?: unknown; ref?: string };
    [k: string]: unknown;
  }>;
}

export interface PendingInterrupt {
  toolCallId: string;
  toolName: string;
  input: unknown;
  kind: 'respond' | 'restart';
  metadata?: unknown;
}

export type AgentPhase =
  | 'idle'
  | 'streaming'
  | 'background'
  | 'awaiting-interrupt'
  | 'done'
  | 'error';

export interface UseGenkitAgentOptions {
  url: string;
  stateUrl?: string;
  abortUrl?: string;
  /**
   * If provided on mount, hook fetches the snapshot and rehydrates state.
   * Accepts either a continuationId (opaque, preferred) or a raw snapshotId
   * (convenient when the value came from a URL route param).
   */
  resumeFromContinuation?: string;
  /** Alias for `resumeFromContinuation` when you have a raw snapshotId from the URL. */
  resumeFromSnapshotId?: string;
  headers?: Record<string, string>;
}

export interface AgentVariant<S = unknown> {
  /** Continuation token for this branch — pass to `continueFrom` to pick it. */
  continuationId: string | undefined;
  /** Raw snapshotId convenience (server-stored agents only). */
  snapshotId: string | undefined;
  /** Final assistant message for this branch. */
  message: AgentMessage | undefined;
  /** Read-only state snapshot for this branch. */
  state:
    | { messages?: AgentMessage[]; custom?: S; artifacts?: any[] }
    | undefined;
}

export interface UseGenkitAgentResult<S = unknown> {
  messages: AgentMessage[];
  artifacts: any[];
  customState: S | undefined;
  streamingText: string;
  streamingReasoning: string;
  toolCalls: ToolCall[];
  statusLabel: string | null;
  progress: { current: number; total: number; label?: string } | null;
  phase: AgentPhase;
  error: Error | null;
  /** Opaque token to round-trip; round-tripping is automatic, this is only exposed for debugging. */
  continuationId: string | undefined;
  /**
   * Derived snapshotId for server-stored agents (extracted from the
   * continuation token). Useful for URL bookmarks and for direct calls to
   * `/state` / `/abort` endpoints. Undefined when the underlying agent is
   * stateless (`state:` token).
   */
  snapshotId: string | undefined;
  pendingInterrupt: PendingInterrupt | null;
  respondToInterrupt: (output: unknown) => void;
  restartInterrupt: (metadata?: unknown) => void;
  submit: (input: AgentInputBody) => void;
  /**
   * Cancel the current turn. In the foreground phase, aborts the streaming
   * connection. In the background phase, **also calls the server's abort
   * endpoint** (`{url}/abort`) so the background work actually stops, then
   * clears the local poll. Idempotent if there's nothing in flight.
   */
  abort: () => Promise<void>;
  reset: () => void;

  /**
   * Run the same turn N times in parallel from the current continuation
   * point, returning all variants. Useful for "pick your variant" UIs.
   * The agent state is unchanged by this call — pick a variant with
   * `continueFrom(variant.continuationId)` to advance.
   */
  runVariants: (
    input: AgentInputBody,
    count?: number
  ) => Promise<AgentVariant<S>[]>;

  /**
   * Advance the agent to a specific continuation/snapshot (e.g. one of the
   * variants returned by `runVariants`, or a snapshotId from a URL).
   * Re-fetches the snapshot's state via the `/state` endpoint and updates
   * `messages` / `customState` / `artifacts` accordingly.
   */
  continueFrom: (continuationOrSnapshotId: string) => Promise<void>;
}

export interface AgentInputBody {
  messages?: AgentMessage[];
  resume?: {
    respond?: Array<{
      toolResponse: { name: string; output?: unknown; ref?: string };
    }>;
    restart?: Array<{
      toolRequest: { name: string; input?: unknown; ref?: string };
      metadata?: unknown;
    }>;
  };
  detach?: boolean;
}

interface AgentOutputBody<S = unknown> {
  continuationId?: string;
  message?: AgentMessage;
  artifacts?: any[];
  state?: { messages?: AgentMessage[]; custom?: S; artifacts?: any[] };
}

interface AgentInitBody {
  continuationId?: string;
}

export function useGenkitAgent<S = unknown>(
  options: UseGenkitAgentOptions
): UseGenkitAgentResult<S> {
  const { url, headers } = options;
  const stateUrl = options.stateUrl ?? `${url}/state`;
  // Accept either a full continuation token or a raw snapshotId; coerce
  // to a continuation token internally.
  const resumeFromContinuation = options.resumeFromContinuation
    ? options.resumeFromContinuation
    : options.resumeFromSnapshotId
      ? toContinuationToken(options.resumeFromSnapshotId)
      : undefined;

  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [artifacts, setArtifacts] = useState<any[]>([]);
  const [customState, setCustomState] = useState<S | undefined>(undefined);
  const [streamingText, setStreamingText] = useState('');
  const [streamingReasoning, setStreamingReasoning] = useState('');
  const [toolCalls, setToolCalls] = useState<ToolCall[]>([]);
  const [statusLabel, setStatusLabel] = useState<string | null>(null);
  const [progress, setProgress] = useState<{
    current: number;
    total: number;
    label?: string;
  } | null>(null);
  const [phase, setPhase] = useState<AgentPhase>('idle');
  const [error, setError] = useState<Error | null>(null);

  const [continuationId, setContinuationId] = useState<string | undefined>(
    resumeFromContinuation
  );
  const continuationRef = useRef<string | undefined>(resumeFromContinuation);
  continuationRef.current = continuationId;

  // Live refs for the abort() closure, which needs current phase + snapshotId
  // without re-creating itself on every render.
  const phaseRef = useRef<AgentPhase>('idle');
  phaseRef.current = phase;
  const snapshotIdRef = useRef<string | undefined>(undefined);

  const [pendingInterrupt, setPendingInterrupt] =
    useState<PendingInterrupt | null>(null);

  const abortRef = useRef<AbortController | null>(null);
  // Active background-poll timeout. Cleared on reset/abort/unmount so we
  // don't leak timers or call setState on an unmounted hook.
  const pollTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Generation token bumped on every reset/abort so an in-flight poll cycle
  // started before the reset stops calling setState.
  const pollGenRef = useRef(0);
  // Continuations we've already rehydrated this mount — avoids re-fetching
  // the snapshot every time the URL pushes a new id mid-conversation.
  const rehydratedRef = useRef<Set<string>>(new Set());

  function clearActivePoll() {
    if (pollTimeoutRef.current) {
      clearTimeout(pollTimeoutRef.current);
      pollTimeoutRef.current = null;
    }
    pollGenRef.current += 1;
  }

  // Resume from snapshot on mount (or when a new external `resumeFrom*`
  // arrives). Skips when we've already rehydrated this exact token —
  // important because `agent.snapshotId` driving a URL push would
  // otherwise feed back into this effect and re-fetch.
  useEffect(() => {
    if (!resumeFromContinuation) return;
    if (rehydratedRef.current.has(resumeFromContinuation)) return;
    rehydratedRef.current.add(resumeFromContinuation);
    let cancelled = false;
    (async () => {
      try {
        const sid = extractSnapshotId(resumeFromContinuation);
        if (!sid) return;
        const snapshot = await runFlow<any, AgentInitBody>({
          url: stateUrl,
          input: sid,
        });
        if (cancelled) return;
        const state = snapshot?.state ?? {};
        if (Array.isArray(state.messages)) setMessages(state.messages);
        if (Array.isArray(state.artifacts)) setArtifacts(state.artifacts);
        if (state.custom !== undefined) setCustomState(state.custom as S);
        if (snapshot?.status === 'pending') {
          setPhase('background');
          startBackgroundPoll(sid);
        } else {
          setPhase('done');
        }
      } catch (e) {
        if (cancelled) return;
        setError(e instanceof Error ? e : new Error(String(e)));
      }
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [resumeFromContinuation]);

  function startBackgroundPoll(snapshotId: string) {
    // Capture the current generation; if reset/abort fires we'll bail.
    const gen = pollGenRef.current;
    const tick = async () => {
      pollTimeoutRef.current = null;
      if (gen !== pollGenRef.current) return; // superseded — drop silently.
      try {
        const snap = await runFlow<any, AgentInitBody>({
          url: stateUrl,
          input: snapshotId,
        });
        if (gen !== pollGenRef.current) return;
        const state = snap?.state ?? {};
        if (Array.isArray(state.messages)) setMessages(state.messages);
        if (Array.isArray(state.artifacts)) setArtifacts(state.artifacts);
        if (state.custom !== undefined) setCustomState(state.custom as S);
        if (snap?.status === 'done') {
          setPhase('done');
          return;
        }
        if (snap?.status === 'failed' || snap?.status === 'aborted') {
          setPhase('error');
          setError(new Error(snap.error?.message ?? 'background run failed'));
          return;
        }
        pollTimeoutRef.current = setTimeout(tick, 2000);
      } catch (e) {
        if (gen !== pollGenRef.current) return;
        setError(e instanceof Error ? e : new Error(String(e)));
        setPhase('error');
      }
    };
    pollTimeoutRef.current = setTimeout(tick, 1000);
  }

  const reset = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    clearActivePoll();
    rehydratedRef.current = new Set();
    setMessages([]);
    setArtifacts([]);
    setCustomState(undefined);
    setStreamingText('');
    setStreamingReasoning('');
    setToolCalls([]);
    setStatusLabel(null);
    setProgress(null);
    setPhase('idle');
    setError(null);
    setContinuationId(undefined);
    setPendingInterrupt(null);
  }, []);

  const abortUrl = options.abortUrl ?? `${url}/abort`;

  const abort = useCallback(async (): Promise<void> => {
    // Cancel any in-flight fetch (foreground stream) and stop local polling.
    abortRef.current?.abort();
    abortRef.current = null;
    const wasBackground = phaseRef.current === 'background';
    const sidForAbort = wasBackground ? snapshotIdRef.current : undefined;
    clearActivePoll();
    setPhase((p) => (p === 'streaming' || p === 'background' ? 'idle' : p));
    // Tell the server to stop the background work too — without this the
    // background run keeps consuming model/tool budget after the client
    // gives up. Silently swallow network errors; the user-visible state is
    // already 'idle'.
    if (wasBackground && sidForAbort) {
      try {
        await runFlow({
          url: abortUrl,
          input: sidForAbort,
          headers,
        });
      } catch (_) {
        // ignore — local state already cleared
      }
    }
  }, [abortUrl, headers]);

  const internalSubmit = useCallback(
    (input: AgentInputBody) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      if (input.messages?.length) {
        setMessages((prev) => [...prev, ...(input.messages as AgentMessage[])]);
      }

      setStreamingText('');
      setStreamingReasoning('');
      setToolCalls([]);
      setStatusLabel(null);
      setProgress(null);
      setError(null);
      setPendingInterrupt(null);
      setPhase('streaming');

      const init: AgentInitBody = continuationRef.current
        ? { continuationId: continuationRef.current }
        : {};

      const { output: outputPromise, stream } = streamFlow<
        AgentOutputBody<S>,
        AgentEvent,
        AgentInitBody
      >({
        url,
        input,
        init,
        headers,
        abortSignal: controller.signal,
      });

      (async () => {
        let interruptDetected: PendingInterrupt | null = null;
        let detachedDuringStream = false;
        try {
          for await (const event of stream) {
            walkAgentEvent(event, {
              onText: (delta) => setStreamingText((prev) => prev + delta),
              onReasoning: (delta) =>
                setStreamingReasoning((prev) => prev + delta),
              onToolRequest: ({ toolCallId, toolName, input }) => {
                setToolCalls((prev) => {
                  const next = prev.slice();
                  const idx = next.findIndex((tc) => tc.id === toolCallId);
                  if (idx === -1) {
                    next.push({
                      id: toolCallId,
                      name: toolName,
                      input,
                      state: 'call',
                    });
                  } else {
                    next[idx] = { ...next[idx], input };
                  }
                  return next;
                });
              },
              onToolResponse: ({ toolCallId, toolName, output }) => {
                setToolCalls((prev) => {
                  const next = prev.slice();
                  const idx = next.findIndex((tc) => tc.id === toolCallId);
                  if (idx === -1) {
                    next.push({
                      id: toolCallId,
                      name: toolName,
                      input: undefined,
                      output,
                      state: 'result',
                    });
                  } else {
                    next[idx] = { ...next[idx], output, state: 'result' };
                  }
                  return next;
                });
              },
              onStatus: (s) => setStatusLabel(s.label),
              onProgress: (p) => setProgress(p),
              onPhase: (p) => setStatusLabel(p),
              onArtifact: (artifact) => {
                setArtifacts((prev) => [...prev, artifact]);
              },
              onInterrupt: (irpt) => {
                interruptDetected = irpt;
              },
              onDetached: ({ continuationId: cid }) => {
                detachedDuringStream = true;
                if (cid) setContinuationId(cid);
              },
              onTurnEnd: ({ continuationId: cid }) => {
                if (cid) setContinuationId(cid);
              },
            });
            if (detachedDuringStream) break;
          }

          const result = await outputPromise;

          if (result?.continuationId) {
            setContinuationId(result.continuationId);
          }
          if (result?.message) {
            setMessages((prev) => [...prev, result.message as AgentMessage]);
          }
          if (result?.state?.custom !== undefined) {
            setCustomState(result.state.custom as S);
          }
          if (Array.isArray(result?.state?.messages)) {
            setMessages(result.state.messages as AgentMessage[]);
          }
          if (Array.isArray(result?.state?.artifacts)) {
            setArtifacts(result.state.artifacts as any[]);
          }
          if (Array.isArray(result?.artifacts)) {
            setArtifacts((prev) =>
              mergeArtifacts(prev, result.artifacts as any[])
            );
          }

          if (detachedDuringStream || input.detach) {
            setPhase('background');
            const sid = result?.continuationId
              ? extractSnapshotId(result.continuationId)
              : null;
            if (sid) startBackgroundPoll(sid);
            return;
          }

          if (interruptDetected) {
            setPendingInterrupt(interruptDetected);
            setPhase('awaiting-interrupt');
            return;
          }

          setPhase('done');
        } catch (err) {
          if (controller.signal.aborted) return;
          setError(err instanceof Error ? err : new Error(String(err)));
          setPhase('error');
        }
      })();
    },
    [url, headers]
  );

  const submit = useCallback(
    (input: AgentInputBody) => internalSubmit(input),
    [internalSubmit]
  );

  const respondToInterrupt = useCallback(
    (output: unknown) => {
      if (!pendingInterrupt) return;
      internalSubmit({
        resume: {
          respond: [
            {
              toolResponse: {
                name: pendingInterrupt.toolName,
                ref: pendingInterrupt.toolCallId,
                output,
              },
            },
          ],
        },
      });
    },
    [pendingInterrupt, internalSubmit]
  );

  const restartInterrupt = useCallback(
    (metadata?: unknown) => {
      if (!pendingInterrupt) return;
      internalSubmit({
        resume: {
          restart: [
            {
              toolRequest: {
                name: pendingInterrupt.toolName,
                ref: pendingInterrupt.toolCallId,
                input: pendingInterrupt.input,
              },
              metadata,
            },
          ],
        },
      });
    },
    [pendingInterrupt, internalSubmit]
  );

  useEffect(
    () => () => {
      abortRef.current?.abort();
      if (pollTimeoutRef.current) clearTimeout(pollTimeoutRef.current);
    },
    []
  );

  const snapshotId = useMemo(
    () => extractSnapshotId(continuationId) ?? undefined,
    [continuationId]
  );
  snapshotIdRef.current = snapshotId;

  // Run the same turn N times in parallel from the current continuation.
  // Used for branching UIs ("pick your variant"). Doesn't mutate the
  // hook's state — caller chooses one and calls `continueFrom`.
  const runVariants = useCallback(
    async (input: AgentInputBody, count = 2): Promise<AgentVariant<S>[]> => {
      const init: AgentInitBody = continuationRef.current
        ? { continuationId: continuationRef.current }
        : {};
      const calls = Array.from({ length: count }, () =>
        runFlow<AgentOutputBody<S>, AgentInitBody>({
          url,
          input,
          init,
          headers,
        })
      );
      const outputs = await Promise.all(calls);
      return outputs.map((o) => ({
        continuationId: o?.continuationId,
        snapshotId: o?.continuationId
          ? (extractSnapshotId(o.continuationId) ?? undefined)
          : undefined,
        message: o?.message,
        state: o?.state,
      }));
    },
    [url, headers]
  );

  // Advance to a chosen variant / snapshot. Fetches the snapshot's state
  // from `/state` and rehydrates messages / customState / artifacts.
  const continueFrom = useCallback(
    async (continuationOrSnapshotId: string): Promise<void> => {
      const token = continuationOrSnapshotId.startsWith(SNAP_PREFIX)
        ? continuationOrSnapshotId
        : toContinuationToken(continuationOrSnapshotId);
      const sid = extractSnapshotId(token);
      setContinuationId(token);
      if (!sid) return;
      try {
        const snapshot = await runFlow<any, AgentInitBody>({
          url: stateUrl,
          input: sid,
          headers,
        });
        const state = snapshot?.state ?? {};
        if (Array.isArray(state.messages)) setMessages(state.messages);
        if (Array.isArray(state.artifacts)) setArtifacts(state.artifacts);
        if (state.custom !== undefined) setCustomState(state.custom as S);
        setPhase('done');
      } catch (e) {
        setError(e instanceof Error ? e : new Error(String(e)));
      }
    },
    [stateUrl, headers]
  );

  return useMemo<UseGenkitAgentResult<S>>(
    () => ({
      messages,
      artifacts,
      customState,
      streamingText,
      streamingReasoning,
      toolCalls,
      statusLabel,
      progress,
      phase,
      error,
      continuationId,
      snapshotId,
      pendingInterrupt,
      respondToInterrupt,
      restartInterrupt,
      submit,
      abort,
      reset,
      runVariants,
      continueFrom,
    }),
    [
      messages,
      artifacts,
      customState,
      streamingText,
      streamingReasoning,
      toolCalls,
      statusLabel,
      progress,
      phase,
      error,
      continuationId,
      snapshotId,
      pendingInterrupt,
      respondToInterrupt,
      restartInterrupt,
      submit,
      abort,
      reset,
      runVariants,
      continueFrom,
    ]
  );
}

function mergeArtifacts(a: any[], b: any[]): any[] {
  const seen = new Set(a.map((x) => x?.name).filter(Boolean));
  return [...a, ...b.filter((x) => !x?.name || !seen.has(x.name))];
}

/** Continuation token format — mirrors `genkit/beta`'s prefixes. */
const SNAP_PREFIX = 'snap:';

function extractSnapshotId(continuationId?: string): string | null {
  if (!continuationId) return null;
  if (continuationId.startsWith(SNAP_PREFIX))
    return continuationId.slice(SNAP_PREFIX.length);
  return null;
}

function toContinuationToken(snapshotId: string): string {
  return SNAP_PREFIX + snapshotId;
}
