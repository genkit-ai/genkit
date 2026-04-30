import { useCallback, useRef, useState } from 'react';
import { runFlow } from 'genkit/beta/client';
import Markdown from 'react-markdown';

// ---------------------------------------------------------------------------
// Background Agent — fire-and-forget with polling
//
// Demonstrates:
//   • `detach: true` — submit a task to run in the background
//   • Polling `getSnapshotDataAction` for status updates
//   • Aborting a background task via `abortSessionFlowAction`
//   • Non-chat UI: task submission → status polling → result display
//
// The key pattern: the server returns a snapshotId immediately, then the
// client polls a /state endpoint until the snapshot status is 'done'.
// ---------------------------------------------------------------------------

const ENDPOINT = '/api/backgroundAgent';
const STATE_ENDPOINT = '/api/backgroundAgent/state';
const ABORT_ENDPOINT = '/api/backgroundAgent/abort';

type TaskStatus = 'idle' | 'submitting' | 'pending' | 'done' | 'failed' | 'aborted';

export default function BackgroundAgent() {
  const [topic, setTopic] = useState('');
  const [status, setStatus] = useState<TaskStatus>('idle');
  const [snapshotId, setSnapshotId] = useState<string | null>(null);
  const [report, setReport] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [pollCount, setPollCount] = useState(0);

  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // ── Stop polling ─────────────────────────────────────────────────────
  const stopPolling = useCallback(() => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }, []);

  // ── Poll for task completion ─────────────────────────────────────────
  const startPolling = useCallback(
    (id: string) => {
      stopPolling();
      setPollCount(0);

      pollTimerRef.current = setInterval(async () => {
        try {
          setPollCount((c) => c + 1);
          const snapshot = (await runFlow({
            url: STATE_ENDPOINT,
            input: id,
          })) as any;

          if (!snapshot) return;

          const s = snapshot.status as TaskStatus;

          if (s === 'done') {
            stopPolling();
            setStatus('done');
            // Extract the model's response from the snapshot state
            const messages = snapshot.state?.messages || [];
            const modelMessages = messages.filter(
              (m: any) => m.role === 'model'
            );
            const lastModel = modelMessages[modelMessages.length - 1];
            const text = lastModel?.content
              ?.filter((p: any) => p.text)
              .map((p: any) => p.text)
              .join('');
            setReport(text || '(empty report)');
          } else if (s === 'failed') {
            stopPolling();
            setStatus('failed');
            setError('The background task failed on the server.');
          } else if (s === 'aborted') {
            stopPolling();
            setStatus('aborted');
          }
        } catch (err: any) {
          // Don't stop polling on transient errors
          console.error('Poll error:', err.message);
        }
      }, 2000);
    },
    [stopPolling]
  );

  // ── Submit task to background ────────────────────────────────────────
  const handleSubmit = useCallback(async () => {
    if (!topic.trim() || status === 'submitting' || status === 'pending')
      return;

    setStatus('submitting');
    setReport(null);
    setError(null);
    setSnapshotId(null);

    try {
      // Send the message with `detach: true`. The server will start
      // processing in the background and return a snapshotId immediately.
      const result = (await runFlow({
        url: ENDPOINT,
        input: {
          messages: [{ role: 'user', content: [{ text: topic.trim() }] }],
          detach: true,
        },
        init: {},
      })) as any;

      const id = result?.snapshotId;
      if (!id) {
        setStatus('failed');
        setError('Server did not return a snapshotId. Detach may not be supported.');
        return;
      }

      setSnapshotId(id);
      setStatus('pending');
      startPolling(id);
    } catch (err: any) {
      setStatus('failed');
      setError(err.message);
    }
  }, [topic, status, startPolling]);

  // ── Abort the background task ────────────────────────────────────────
  const handleAbort = useCallback(async () => {
    if (!snapshotId) return;
    stopPolling();

    try {
      await runFlow({ url: ABORT_ENDPOINT, input: snapshotId });
      setStatus('aborted');
    } catch (err: any) {
      setError(`Abort failed: ${err.message}`);
    }
  }, [snapshotId, stopPolling]);

  // ── Reset to submit a new task ───────────────────────────────────────
  const handleReset = useCallback(() => {
    stopPolling();
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
            Submit a task to run in the background. The server returns immediately
            while processing continues — poll for the result.
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
              disabled={!topic.trim() || status === 'submitting'}
            >
              {status === 'submitting' ? 'Submitting…' : '🚀 Generate Report (Background)'}
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
              <span className="background-poll-count">
                Polls: {pollCount}
              </span>
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
              <code className="background-snapshot-id">
                {snapshotId}
              </code>
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
              <span
                className={`background-status-badge ${status}`}
              >
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
            Client sends <code>{'{ detach: true }'}</code> with the input message.
          </li>
          <li>
            Server saves a snapshot with status <code>"pending"</code> and returns
            the <code>snapshotId</code> immediately.
          </li>
          <li>
            The LLM request continues running in the background on the server.
          </li>
          <li>
            Client polls <code>/state</code> endpoint with the snapshotId every
            2 seconds.
          </li>
          <li>
            When <code>status === "done"</code>, the report is extracted from the
            snapshot's message history.
          </li>
        </ol>

        <h4>Status Values</h4>
        <ul className="background-status-list">
          <li><code>pending</code> — still processing</li>
          <li><code>done</code> — completed successfully</li>
          <li><code>failed</code> — error during processing</li>
          <li><code>aborted</code> — cancelled by the client</li>
        </ul>

        <h4>Key APIs</h4>
        <pre className="background-code">{`// Submit with detach
runFlow({
  url: '/api/backgroundAgent',
  input: {
    messages: [...],
    detach: true,
  },
});
// → { snapshotId: '...' }

// Poll for status
runFlow({
  url: '/api/.../state',
  input: snapshotId,
});
// → { status, state }

// Abort
runFlow({
  url: '/api/.../abort',
  input: snapshotId,
});`}</pre>
      </aside>
    </div>
  );
}
