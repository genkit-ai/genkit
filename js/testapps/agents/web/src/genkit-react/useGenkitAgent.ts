/**
 * `useGenkitAgent` — the headline React hook for the v2 Agents API.
 *
 * Subsumes what every page in the original testapp manually wires:
 *
 *   - `for await (const chunk of response.stream)` + `walkAgentEvent` dispatch
 *   - `stateRef` / `snapshotIdRef` round-tripping (replaced with single
 *     opaque `continuationId`)
 *   - Mid-stream streaming text accumulation, separate from committed history
 *   - Tool call lifecycle (call → result → error)
 *   - Interrupt detection (now in-stream) and resumption helpers
 *   - Foreground → background transition (detached event), with the same
 *     hook continuing to render in the background phase
 *   - Custom state (typed via the agent's stateSchema)
 *   - Artifact collection
 *
 * Returned reactive fields cover the common cases; the `chunks` escape hatch
 * is there for anything custom.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { streamFlow, runFlow, walkAgentEvent } from 'genkit/beta/client';
import type { AgentEvent, AgentStreamChunkV2 } from 'genkit/beta/client';

// Re-exported convenience type covering both Genkit-native shape and what
// the hook builds on top.
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
  | 'streaming' // foreground turn in progress
  | 'background' // detached, server is running, may be live-streaming via subscribe
  | 'awaiting-interrupt'
  | 'done'
  | 'error';

export interface UseGenkitAgentOptions {
  /** Base URL for the agent. The /state path is derived for snapshot reads. */
  url: string;
  /** Optional separate URL for the snapshot data action. Defaults to `${url}/state`. */
  stateUrl?: string;
  /** Optional separate URL for the abort action. Defaults to `${url}/abort`. */
  abortUrl?: string;
  /**
   * If provided on mount, the hook fetches the snapshot and rehydrates
   * messages/state/artifacts from it. Useful for URL bookmarks.
   */
  resumeFromContinuation?: string;
  headers?: Record<string, string>;
}

export interface UseGenkitAgentResult<S = unknown> {
  // History
  messages: AgentMessage[];
  artifacts: any[];
  customState: S | undefined;

  // Streaming-turn state
  streamingText: string;
  streamingReasoning: string;
  toolCalls: ToolCall[];
  statusLabel: string | null;
  progress: { current: number; total: number; label?: string } | null;
  phase: AgentPhase;
  error: Error | null;

  // Continuation
  continuationId: string | undefined;

  // Interrupts
  pendingInterrupt: PendingInterrupt | null;
  respondToInterrupt: (output: unknown) => void;
  restartInterrupt: (metadata?: unknown) => void;

  // Standard control
  submit: (input: AgentInputBody) => void;
  abort: () => void;
  reset: () => void;
}

// Same shape the v2 agent server accepts.
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
  snapshotId?: string;
  state?: { messages?: AgentMessage[]; custom?: S; artifacts?: any[] };
  message?: AgentMessage;
  artifacts?: any[];
}

interface AgentInitBody {
  continuationId?: string;
}

export function useGenkitAgent<S = unknown>(
  options: UseGenkitAgentOptions
): UseGenkitAgentResult<S> {
  const { url, resumeFromContinuation, headers } = options;
  const stateUrl = options.stateUrl ?? `${url}/state`;

  // History (committed)
  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [artifacts, setArtifacts] = useState<any[]>([]);
  const [customState, setCustomState] = useState<S | undefined>(undefined);

  // In-flight turn
  const [streamingText, setStreamingText] = useState('');
  const [streamingReasoning, setStreamingReasoning] = useState('');
  const [toolCalls, setToolCalls] = useState<ToolCall[]>([]);
  const [statusLabel, setStatusLabel] = useState<string | null>(null);
  const [progress, setProgress] = useState<
    { current: number; total: number; label?: string } | null
  >(null);
  const [phase, setPhase] = useState<AgentPhase>('idle');
  const [error, setError] = useState<Error | null>(null);

  // Continuation token round-tripped on every turn
  const [continuationId, setContinuationId] = useState<string | undefined>(
    resumeFromContinuation
  );
  const continuationRef = useRef<string | undefined>(resumeFromContinuation);
  continuationRef.current = continuationId;

  const [pendingInterrupt, setPendingInterrupt] = useState<PendingInterrupt | null>(
    null
  );

  const abortRef = useRef<AbortController | null>(null);

  // Resume from snapshot on mount if requested.
  useEffect(() => {
    if (!resumeFromContinuation) return;
    let cancelled = false;
    (async () => {
      try {
        // The state endpoint accepts a snapshotId string. We decode the
        // continuation token client-side to extract it; if it's not a
        // snapshot-shaped token (i.e. client-managed state), we skip the
        // remote fetch and just rehydrate from the embedded state.
        const sid = extractSnapshotId(resumeFromContinuation);
        if (sid) {
          const snapshot = await runFlow<any, AgentInitBody>({
            url: stateUrl,
            input: sid,
          });
          if (cancelled) return;
          const state = snapshot?.state ?? {};
          if (Array.isArray(state.messages)) setMessages(state.messages);
          if (Array.isArray(state.artifacts)) setArtifacts(state.artifacts);
          if (state.custom !== undefined) setCustomState(state.custom as S);
          // If still running, we'd auto-resubscribe here. For prototype
          // scope: surface the snapshot status and let the consumer decide.
          if (snapshot?.status === 'pending') {
            setPhase('background');
            startBackgroundPoll(sid);
          } else {
            setPhase('done');
          }
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
    // Minimal polling implementation for prototype. Production would prefer
    // subscribeAgent via StreamManager.
    const tick = async () => {
      try {
        const snap = await runFlow<any, AgentInitBody>({
          url: stateUrl,
          input: snapshotId,
        });
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
        setTimeout(tick, 2000);
      } catch (e) {
        setError(e instanceof Error ? e : new Error(String(e)));
        setPhase('error');
      }
    };
    setTimeout(tick, 1000);
  }

  const reset = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
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

  const abort = useCallback(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setPhase((p) => (p === 'streaming' ? 'idle' : p));
  }, []);

  const internalSubmit = useCallback(
    (input: AgentInputBody) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;

      // Optimistically append the user-supplied messages so the UI updates
      // before the first server chunk arrives.
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
        AgentStreamChunkV2,
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
        try {
          for await (const chunk of stream) {
            let detached = false;
            walkAgentEvent(chunk, {
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
              onArtifact: (artifact) => {
                setArtifacts((prev) => [...prev, artifact]);
              },
              onInterrupt: (irpt) => {
                interruptDetected = irpt;
              },
              onDetached: () => {
                detached = true;
              },
            });
            if (detached) break;
          }

          const result = await outputPromise;

          // Update continuation token for next turn.
          if (result?.continuationId) {
            setContinuationId(result.continuationId);
          }

          // Commit the final assistant message into history.
          if (result?.message) {
            setMessages((prev) => [...prev, result.message as AgentMessage]);
          }
          // Pull custom state from state.custom if present.
          if (result?.state?.custom !== undefined) {
            setCustomState(result.state.custom as S);
          }
          if (Array.isArray(result?.state?.messages)) {
            // Server-managed agents: replace history with server's truth.
            setMessages(result.state.messages as AgentMessage[]);
          }
          if (Array.isArray(result?.state?.artifacts)) {
            setArtifacts(result.state.artifacts as any[]);
          }
          if (Array.isArray(result?.artifacts)) {
            setArtifacts((prev) => mergeArtifacts(prev, result.artifacts as any[]));
          }

          // If the server reported detached during streaming OR returned a
          // snapshotId with no completion, transition to background phase.
          if (
            (result?.snapshotId && phaseRefIs('background', input.detach)) ||
            input.detach
          ) {
            setPhase('background');
            if (result?.snapshotId) startBackgroundPoll(result.snapshotId);
            return;
          }

          // Interrupt routing — set pendingInterrupt for the consumer.
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

  // We don't actually have access to a phase ref to look at "are we already
  // in background"; the input.detach is the load-bearing signal. Helper to
  // make that clear.
  function phaseRefIs(_p: AgentPhase, detach?: boolean): boolean {
    return !!detach;
  }

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

  useEffect(() => () => abortRef.current?.abort(), []);

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
      pendingInterrupt,
      respondToInterrupt,
      restartInterrupt,
      submit,
      abort,
      reset,
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
      pendingInterrupt,
      respondToInterrupt,
      restartInterrupt,
      submit,
      abort,
      reset,
    ]
  );
}

function mergeArtifacts(a: any[], b: any[]): any[] {
  const seen = new Set(a.map((x) => x?.name).filter(Boolean));
  return [...a, ...b.filter((x) => !x?.name || !seen.has(x.name))];
}

function extractSnapshotId(continuationId: string): string | null {
  if (continuationId.startsWith('v1:')) return continuationId.slice(3);
  return null;
}
