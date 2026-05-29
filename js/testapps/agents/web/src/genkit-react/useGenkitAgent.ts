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
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { streamFlow, runFlow, walkAgentEvent } from 'genkit/beta/client';
import type { AgentEvent } from 'genkit/beta/client';

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
  /** If provided on mount, hook fetches the snapshot and rehydrates state. */
  resumeFromContinuation?: string;
  headers?: Record<string, string>;
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
  continuationId: string | undefined;
  pendingInterrupt: PendingInterrupt | null;
  respondToInterrupt: (output: unknown) => void;
  restartInterrupt: (metadata?: unknown) => void;
  submit: (input: AgentInputBody) => void;
  abort: () => void;
  reset: () => void;
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
  const { url, resumeFromContinuation, headers } = options;
  const stateUrl = options.stateUrl ?? `${url}/state`;

  const [messages, setMessages] = useState<AgentMessage[]>([]);
  const [artifacts, setArtifacts] = useState<any[]>([]);
  const [customState, setCustomState] = useState<S | undefined>(undefined);
  const [streamingText, setStreamingText] = useState('');
  const [streamingReasoning, setStreamingReasoning] = useState('');
  const [toolCalls, setToolCalls] = useState<ToolCall[]>([]);
  const [statusLabel, setStatusLabel] = useState<string | null>(null);
  const [progress, setProgress] = useState<
    { current: number; total: number; label?: string } | null
  >(null);
  const [phase, setPhase] = useState<AgentPhase>('idle');
  const [error, setError] = useState<Error | null>(null);

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
            setArtifacts((prev) => mergeArtifacts(prev, result.artifacts as any[]));
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
