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

import { useMemo, useState } from 'react';
import Markdown from 'react-markdown';
import { runFlow } from 'genkit/beta/client';
import { useGenkitAgent } from '../genkit-react';

// ---------------------------------------------------------------------------
// Background Agent — MIGRATED TO v2 HOOKS
//
// Before: 342 LOC of manual polling, hardcoded 2s interval, status-string
// switching, message-history extraction, separate abort handler, separate
// reset state machine.
//
// After: the v2 hook auto-detects `detach: true`, transitions to
// `phase: 'background'` on the in-stream `detached` event, and the
// internal poll surfaces `messages` + `customState` + `artifacts`
// reactively. Background lifecycle is just a phase of the same hook,
// not a separate API.
//
// We still call `runFlow` directly for the abort action because that's a
// one-shot non-streaming call against a sibling endpoint. A future
// `agent.abortBackground()` helper could absorb this too.
// ---------------------------------------------------------------------------

const ENDPOINT = '/api/backgroundAgent';
const ABORT_ENDPOINT = '/api/backgroundAgent/abort';

export default function BackgroundAgent() {
  const [topic, setTopic] = useState('');
  const agent = useGenkitAgent({ url: ENDPOINT });

  const handleSubmit = () => {
    if (!topic.trim() || agent.phase === 'streaming' || agent.phase === 'background') return;
    agent.submit({
      messages: [{ role: 'user', content: [{ text: topic.trim() }] }],
      detach: true,
    });
  };

  const handleAbort = async () => {
    if (!agent.continuationId) return;
    const sid = agent.continuationId.startsWith('v1:')
      ? agent.continuationId.slice(3)
      : null;
    if (!sid) return;
    try {
      await runFlow({ url: ABORT_ENDPOINT, input: sid });
    } catch (e) {
      // Visible via agent.error if polling picks it up; ignore here.
    }
    agent.abort();
  };

  const handleReset = () => {
    agent.reset();
    setTopic('');
  };

  // Pull the final report text out of the hook's `messages` array.
  const report = useMemo<string | null>(() => {
    const modelMsgs = agent.messages.filter((m) => m.role === 'model');
    const last = modelMsgs[modelMsgs.length - 1];
    if (!last) return null;
    return (last.content ?? [])
      .filter((p: any) => p.text)
      .map((p: any) => p.text)
      .join('');
  }, [agent.messages]);

  const snapshotId = agent.continuationId?.startsWith('v1:')
    ? agent.continuationId.slice(3)
    : agent.continuationId;

  // Map hook phase to the legacy status names this UI used.
  const status: 'idle' | 'submitting' | 'pending' | 'done' | 'failed' | 'aborted' =
    agent.phase === 'idle'
      ? 'idle'
      : agent.phase === 'streaming'
        ? 'submitting'
        : agent.phase === 'background'
          ? 'pending'
          : agent.phase === 'done'
            ? 'done'
            : agent.phase === 'error'
              ? 'failed'
              : 'idle';

  return (
    <div className="background-layout">
      <div className="background-panel">
        <div className="chat-header">
          <h2>Background Agent — v2</h2>
          <span className="chat-desc">
            Submit a task to run in the background. The v2 hook auto-detects
            detach + drives the polling internally.
          </span>
        </div>

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
              {status === 'submitting'
                ? 'Submitting…'
                : '🚀 Generate Report (Background)'}
            </button>
          </div>
        )}

        {status === 'pending' && (
          <div className="background-status">
            <div className="background-status-icon">⏳</div>
            <h3>Processing in Background…</h3>
            <p className="background-status-detail">
              The hook is polling internally; messages and state update
              reactively as the snapshot evolves.
            </p>
            <div className="background-meta">
              <code>continuationId: {agent.continuationId}</code>
            </div>
            {agent.statusLabel && (
              <div className="background-meta">
                <span>Status: {agent.statusLabel}</span>
              </div>
            )}
            <button className="btn btn-deny" onClick={handleAbort}>
              ✋ Abort
            </button>
          </div>
        )}

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

        {status === 'failed' && (
          <div className="background-result">
            <div className="background-result-header">
              <span className="background-status-badge failed">❌ Failed</span>
              {snapshotId && (
                <code className="background-snapshot-id">{snapshotId}</code>
              )}
              <button className="btn btn-send" onClick={handleReset}>
                Try Again
              </button>
            </div>
            {agent.error && <p className="background-error">{agent.error.message}</p>}
          </div>
        )}
      </div>

      <aside className="info-sidebar">
        <h3>📋 v2 Background Pattern</h3>
        <ol>
          <li>
            User submits a topic with <code>detach: true</code>. The hook calls
            <code>streamFlow</code> with the input; the server runs in the
            background.
          </li>
          <li>
            The server emits an in-stream <code>detached</code> event with the{' '}
            <code>snapshotId</code> + <code>continuationId</code>. The hook
            transitions to <code>phase: 'background'</code> automatically.
          </li>
          <li>
            The hook starts polling the snapshot endpoint internally, updating{' '}
            <code>messages</code>, <code>artifacts</code>, and{' '}
            <code>customState</code> reactively as the snapshot evolves.
          </li>
          <li>
            When the snapshot reports <code>status: 'done'</code>, the hook
            transitions to <code>phase: 'done'</code> with the final
            messages already in state. The UI renders the report from{' '}
            <code>agent.messages</code>.
          </li>
        </ol>

        <h4>Page code</h4>
        <pre className="background-code">{`const agent = useGenkitAgent({ url });

agent.submit({
  messages: [{ role: 'user', content: [{ text }] }],
  detach: true,
});

// agent.phase: 'streaming' → 'background' → 'done'
// agent.messages updates as the snapshot evolves
// agent.continuationId is stable, bookmarkable

// Resume on page reload:
useGenkitAgent({
  url,
  resumeFromContinuation: bookmarkedId,
});`}</pre>
      </aside>
    </div>
  );
}
