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
// Demonstrates the ergonomic `remoteAgent` detached-task API:
//   • chat.detach(topic) — submit a task to run in the background; resolves to
//     a `DetachedTask` with the snapshotId
//   • task.poll({ intervalMs }) — an async iterator that yields snapshots until
//     the task reaches a terminal status (done/failed/aborted)
//   • task.abort() — cancel the background task
//   • Non-chat UI: task submission → status polling → result display
// ---------------------------------------------------------------------------

const ENDPOINT = '/api/backgroundAgent';

type TaskStatus =
  | 'idle'
  | 'submitting'
  | 'pending'
  | 'done'
  | 'failed'
  | 'aborted';

export default function BackgroundAgent() {
  const [topic, setTopic] = useState('');
  const [status, setStatus] = useState<TaskStatus>('idle');
  const [snapshotId, setSnapshotId] = useState<string | null>(null);
  const [report, setReport] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pollCount, setPollCount] = useState(0);

  // The agent client and the live detached task we're polling.
  const agent = useMemo(() => remoteAgent({ url: ENDPOINT }), []);
  const taskRef = useRef<DetachedTask | null>(null);
  const cancelledRef = useRef(false);

  // ── Submit task to background ────────────────────────────────────────
  const handleSubmit = useCallback(async () => {
    if (!topic.trim() || status === 'submitting' || status === 'pending')
      return;

    setStatus('submitting');
    setReport(null);
    setError(null);
    setSnapshotId(null);
    setPollCount(0);
    cancelledRef.current = false;

    try {
      // chat.detach() submits the turn in the background. The server starts
      // processing and the promise resolves with a handle as soon as the
      // snapshot is created.
      const chat = agent.chat();
      const task = await chat.detach(topic.trim());
      taskRef.current = task;

      setSnapshotId(task.snapshotId);
      setStatus('pending');

      // task.poll() yields a snapshot every `intervalMs` until the task
      // reaches a terminal status. The iterator completes on its own.
      for await (const snapshot of task.poll({ intervalMs: 2000 })) {
        if (cancelledRef.current) break;
        setPollCount((c) => c + 1);

        const s = snapshot.status as TaskStatus | undefined;
        if (s === 'done') {
          setStatus('done');
          // Extract the model's response from the snapshot's message history.
          const messages: MessageData[] = snapshot.state?.messages || [];
          const modelMessages = messages.filter((m) => m.role === 'model');
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
        }
      }
    } catch (err: any) {
      if (cancelledRef.current) return;
      setStatus('failed');
      setError(err.message);
    }
  }, [topic, status, agent]);

  // ── Abort the background task ────────────────────────────────────────
  const handleAbort = useCallback(async () => {
    const task = taskRef.current;
    if (!task) return;
    cancelledRef.current = true;

    try {
      await task.abort();
      setStatus('aborted');
    } catch (err: any) {
      setError(`Abort failed: ${err.message}`);
    }
  }, []);

  // ── Reset to submit a new task ───────────────────────────────────────
  const handleReset = useCallback(() => {
    cancelledRef.current = true;
    taskRef.current = null;
    setStatus('idle');
    setReport(null);
    setError(null);
    setSnapshotId(null);
    setTopic('');
    setPollCount(0);
  }, []);

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

        {/* ── Failed / Aborted ────────────────────────────────────────── */}
        {(status === 'failed' || status === 'aborted') && (
          <div className="background-result">
            <div className="background-result-header">
              <span className={`background-status-badge ${status}`}>
                {status === 'aborted' ? '🛑 Aborted' : '❌ Failed'}
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
            Client calls <code>chat.detach(topic)</code> — the turn runs in the
            background on the server.
          </li>
          <li>
            The server saves a snapshot with status <code>"pending"</code> and
            the promise resolves with a <code>DetachedTask</code> handle (with
            the <code>snapshotId</code>) immediately.
          </li>
          <li>
            The LLM request continues running in the background on the server.
          </li>
          <li>
            Client iterates <code>task.poll(&#123; intervalMs &#125;)</code>,
            which yields a snapshot every 2 seconds.
          </li>
          <li>
            When <code>status === "done"</code>, the report is extracted from
            the snapshot's message history and the iterator completes.
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
        <pre className="background-code">{`// Submit in the background
const chat = agent.chat();
const task = await chat.detach(topic);
// → task.snapshotId

// Poll for status
for await (const snap of task.poll({
  intervalMs: 2000,
})) {
  // snap.status, snap.state
}

// Abort
await task.abort();`}</pre>
      </aside>
    </div>
  );
}
