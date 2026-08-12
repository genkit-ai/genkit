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

import type { MessageData, Part } from 'genkit/beta';
import { remoteAgent, type DetachedTask } from 'genkit/beta/client';
import { useCallback, useMemo, useRef, useState } from 'react';
import Markdown from 'react-markdown';

// ---------------------------------------------------------------------------
// Background Agent — fire-and-forget with polling
//
// Demonstrates:
//   • `chat.detach()` — submit a task to run in the background
//   • Polling the returned `DetachedTask` for status updates
//   • Aborting a background task via `task.abort()`
//   • Non-chat UI: task submission → status polling → result display
//
// The key pattern: `detach()` returns a snapshotId immediately, then the
// returned task's `poll()` yields snapshots until a terminal status.
// ---------------------------------------------------------------------------

const ENDPOINT = '/api/backgroundAgent';

type TaskStatus =
  | 'idle'
  | 'submitting'
  | 'pending'
  | 'done'
  | 'failed'
  | 'aborted'
  | 'expired';

export default function BackgroundAgent() {
  const [topic, setTopic] = useState('');
  const [status, setStatus] = useState<TaskStatus>('idle');
  const [snapshotId, setSnapshotId] = useState<string | null>(null);
  const [report, setReport] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pollCount, setPollCount] = useState(0);

  // Typed HTTP client. `remoteAgent` defaults the state/abort endpoints to
  // `${url}/state` and `${url}/abort`, matching the background agent's routes.
  const agent = useMemo(() => remoteAgent({ url: ENDPOINT }), []);

  // The in-flight detached task, kept so we can abort it.
  const taskRef = useRef<DetachedTask | null>(null);
  // Bumped to cancel an in-progress poll loop (e.g. on abort/reset).
  const pollGenRef = useRef(0);

  // ── Stop polling ─────────────────────────────────────────────────────
  const stopPolling = useCallback(() => {
    pollGenRef.current++;
  }, []);

  // ── Poll the detached task for completion ────────────────────────────
  const startPolling = useCallback((task: DetachedTask) => {
    const gen = ++pollGenRef.current;
    setPollCount(0);

    (async () => {
      try {
        for await (const snapshot of task.poll({ intervalMs: 2000 })) {
          if (gen !== pollGenRef.current) return; // superseded — bail out
          setPollCount((c) => c + 1);

          const s = snapshot.status as TaskStatus;

          if (s === 'done') {
            setStatus('done');
            // Extract the model's response from the snapshot state
            const messages: MessageData[] = snapshot.state?.messages || [];
            const modelMessages = messages.filter(
              (m: MessageData) => m.role === 'model'
            );
            const lastModel = modelMessages[modelMessages.length - 1];
            const text = (lastModel?.content || [])
              .filter((p: Part) => p.text)
              .map((p: Part) => p.text)
              .join('');
            setReport(text || '(empty report)');
          } else if (s === 'failed') {
            setStatus('failed');
            setError('The background task failed on the server.');
          } else if (s === 'aborted') {
            setStatus('aborted');
          } else if (s === 'expired') {
            // The background worker stopped sending heartbeats (e.g. the server
            // was restarted), so the task can never complete. Surface it as a
            // terminal expired state.
            setStatus('expired');
            setError(
              'The background task expired: its worker stopped responding ' +
                '(the server may have been restarted).'
            );
          }
        }
      } catch (err: any) {
        if (gen === pollGenRef.current) {
          console.error('Poll error:', err.message);
        }
      }
    })();
  }, []);

  // ── Submit task to background ────────────────────────────────────────
  const handleSubmit = useCallback(async () => {
    if (!topic.trim() || status === 'submitting' || status === 'pending')
      return;

    setStatus('submitting');
    setReport(null);
    setError(null);
    setSnapshotId(null);

    try {
      // Submit a detached (background) turn. The server starts processing and
      // resolves immediately with a handle exposing the snapshotId.
      const task = await agent.chat().detach(topic.trim());
      taskRef.current = task;

      setSnapshotId(task.snapshotId);
      setStatus('pending');
      startPolling(task);
    } catch (err: any) {
      setStatus('failed');
      setError(err.message);
    }
  }, [agent, topic, status, startPolling]);

  // ── Abort the background task ────────────────────────────────────────
  const handleAbort = useCallback(async () => {
    if (!taskRef.current) return;
    stopPolling();

    try {
      await taskRef.current.abort();
      setStatus('aborted');
    } catch (err: any) {
      setError(`Abort failed: ${err.message}`);
    }
  }, [stopPolling]);

  // ── Reset to submit a new task ───────────────────────────────────────
  const handleReset = useCallback(() => {
    stopPolling();
    taskRef.current = null;
    setStatus('idle');
    setReport(null);
    setError(null);
    setSnapshotId(null);
    setTopic('');
    setPollCount(0);
  }, [stopPolling]);

  return (
    <div className="background-layout">
      <div className="background-panel">
        <div className="chat-header">
          <h2>Background Agent</h2>
          <span className="chat-desc">
            Submit a task to run in the background. The server returns
            immediately while processing continues — poll for the result.
          </span>
        </div>

        {/* ── Input form ──────────────────────────────────────────────── */}
        {(status === 'idle' || status === 'submitting') && (
          <div className="background-form">
            <label className="background-label" htmlFor="topic">
              Research Topic
            </label>
            <textarea
              id="topic"
              className="background-input"
              placeholder="e.g., The impact of quantum computing on cybersecurity"
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              disabled={status === 'submitting'}
              rows={3}
            />
            <button
              className="btn btn-send"
              onClick={handleSubmit}
              disabled={!topic.trim() || status === 'submitting'}>
              {status === 'submitting'
                ? 'Submitting…'
                : '🚀 Generate Report (Background)'}
            </button>
          </div>
        )}

        {/* ── Polling status ──────────────────────────────────────────── */}
        {status === 'pending' && (
          <div className="background-status">
            <div className="background-status-icon">⏳</div>
            <h3>Processing in Background…</h3>
            <p className="background-status-detail">
              The server is generating your report. This page is polling for the
              result every 2 seconds.
            </p>
            <div className="background-meta">
              <code>snapshotId: {snapshotId}</code>
              <span className="background-poll-count">Polls: {pollCount}</span>
            </div>
            <button className="btn btn-deny" onClick={handleAbort}>
              ✋ Abort
            </button>
          </div>
        )}

        {/* ── Completed ───────────────────────────────────────────────── */}
        {status === 'done' && (
          <div className="background-result">
            <div className="background-result-header">
              <span className="background-status-badge done">✅ Complete</span>
              <code className="background-snapshot-id">{snapshotId}</code>
              <button className="btn btn-send" onClick={handleReset}>
                New Report
              </button>
            </div>
            <div className="background-report markdown-body">
              <Markdown>{report ?? ''}</Markdown>
            </div>
          </div>
        )}

        {/* ── Failed / Aborted / Expired ──────────────────────────────── */}
        {(status === 'failed' ||
          status === 'aborted' ||
          status === 'expired') && (
          <div className="background-result">
            <div className="background-result-header">
              <span className={`background-status-badge ${status}`}>
                {status === 'aborted'
                  ? '🛑 Aborted'
                  : status === 'expired'
                    ? '⌛ Expired'
                    : '❌ Failed'}
              </span>

              {snapshotId && (
                <code className="background-snapshot-id">{snapshotId}</code>
              )}
              <button className="btn btn-send" onClick={handleReset}>
                Try Again
              </button>
            </div>
            {error && <p className="background-error">{error}</p>}
          </div>
        )}
      </div>

      {/* ── Info sidebar ────────────────────────────────────────────────── */}
      <aside className="info-sidebar">
        <h3>📋 How It Works</h3>
        <ol>
          <li>
            Client sends <code>{'{ detach: true }'}</code> with the input
            message.
          </li>
          <li>
            Server saves a snapshot with status <code>"pending"</code> and
            returns the <code>snapshotId</code> immediately.
          </li>
          <li>
            The LLM request continues running in the background on the server.
          </li>
          <li>
            Client polls <code>/state</code> endpoint with the snapshotId every
            2 seconds.
          </li>
          <li>
            When <code>status === "done"</code>, the report is extracted from
            the snapshot's message history.
          </li>
        </ol>

        <h4>Status Values</h4>
        <ul className="background-status-list">
          <li>
            <code>pending</code> — still processing
          </li>
          <li>
            <code>done</code> — completed successfully
          </li>
          <li>
            <code>failed</code> — error during processing
          </li>
          <li>
            <code>aborted</code> — cancelled by the client
          </li>
        </ul>

        <h4>Key APIs</h4>
        <pre className="background-code">{`// Submit a detached (background) turn
const agent = remoteAgent({
  url: '/api/backgroundAgent',
});
const task = await agent
  .chat()
  .detach(topic);
// task.snapshotId available now

// Poll until a terminal status
for await (
  const snap of task.poll({ intervalMs: 2000 })
) {
  // snap.status, snap.state
}

// Abort
await task.abort();`}</pre>
      </aside>
    </div>
  );
}
