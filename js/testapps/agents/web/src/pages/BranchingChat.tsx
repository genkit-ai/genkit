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

import { useCallback, useEffect, useMemo, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { useGenkitAgent, type AgentVariant } from '../genkit-react';

// ---------------------------------------------------------------------------
// Branching Chat — "Pick Your Variant" UI
//
// Demonstrates the v2 hook's branching API: `agent.runVariants(input, n)`
// fires N parallel runs from the same continuation, returning all variants.
// `agent.continueFrom(snapshotId)` advances the agent to a chosen branch.
// No raw `runFlow` imports needed — the hook covers parallel branching,
// snapshot restoration, and continuation round-tripping.
// ---------------------------------------------------------------------------

const ENDPOINT = '/api/branchingAgent';

export default function BranchingChat() {
  const { snapshotId: urlSnapshotId } = useParams<{ snapshotId: string }>();
  const navigate = useNavigate();

  const agent = useGenkitAgent({
    url: ENDPOINT,
    resumeFromSnapshotId: urlSnapshotId,
  });

  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [variants, setVariants] = useState<AgentVariant[] | null>(null);
  const [pendingUserText, setPendingUserText] = useState<string | null>(null);

  // Bookmark the chosen branch into the URL whenever the agent advances.
  useEffect(() => {
    if (!agent.snapshotId || agent.snapshotId === urlSnapshotId) return;
    navigate(`/branching/${agent.snapshotId}`, { replace: true });
  }, [agent.snapshotId, urlSnapshotId, navigate]);

  const handleSend = useCallback(
    async (text: string) => {
      if (loading || variants) return;
      setInput('');
      setLoading(true);
      setPendingUserText(text);
      try {
        const v = await agent.runVariants(
          { messages: [{ role: 'user', content: [{ text }] }] },
          2
        );
        setVariants(v);
      } finally {
        setLoading(false);
      }
    },
    [agent, loading, variants]
  );

  const handlePick = useCallback(
    async (which: number) => {
      if (!variants) return;
      const chosen = variants[which];
      if (!chosen.snapshotId) return;
      await agent.continueFrom(chosen.snapshotId);
      setVariants(null);
      setPendingUserText(null);
    },
    [agent, variants]
  );

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (input.trim()) handleSend(input.trim());
    }
  };

  // Render the conversation from the hook's committed `messages` plus the
  // in-flight user message (which lives outside the agent state until a
  // variant is picked).
  const renderedMessages = useMemo(() => {
    const rows: Array<{ role: string; text: string }> = [];
    for (const m of agent.messages) {
      const text =
        m.content
          ?.map((p: any) => p.text)
          .filter(Boolean)
          .join('') ?? '';
      if (text) rows.push({ role: m.role, text });
    }
    if (pendingUserText) {
      rows.push({ role: 'user', text: pendingUserText });
    }
    return rows;
  }, [agent.messages, pendingUserText]);

  return (
    <div className="page-with-sidebar">
      <div className="chat-panel">
        <div className="chat-header">
          <div className="chat-header-top">
            <h2>🔀 Branching Chat</h2>
            {agent.snapshotId && (
              <Link
                to="/branching"
                className="btn btn-new-session"
                reloadDocument>
                ✨ New Session
              </Link>
            )}
          </div>
          <span className="chat-desc">
            Every response generates two variants from the same snapshot. Pick
            the one you prefer to continue the conversation.
          </span>
        </div>

        <div className="chat-messages">
          {renderedMessages.length === 0 && !loading && !variants && (
            <div className="chat-empty">
              Send a message to start. Each response will show two variants —
              pick your favorite to choose which branch to follow.
            </div>
          )}

          {renderedMessages.map((msg, i) => (
            <div
              key={i}
              className={`message ${msg.role === 'user' ? 'message-user' : ''}`}>
              <div className="message-role">
                {msg.role === 'user' ? 'You' : 'Model'}
              </div>
              <div className="message-text">{msg.text}</div>
            </div>
          ))}

          {loading && (
            <div className="variant-loading">
              <div className="variant-loading-icon">🔀</div>
              Generating two variants…
            </div>
          )}

          {variants && (
            <div className="variant-picker">
              <div className="variant-picker-label">
                Pick a variant to continue:
              </div>
              <div className="variant-cards">
                {variants.map((v, i) => (
                  <button
                    key={i}
                    className="variant-card"
                    onClick={() => handlePick(i)}>
                    <div className="variant-card-badge">
                      {String.fromCharCode(65 + i)}
                    </div>
                    <div className="variant-card-text">{variantText(v)}</div>
                    <div className="variant-card-action">Use this ✓</div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {agent.error && (
            <div className="message message-system">
              <div className="message-role">system</div>
              <div className="message-text">Error: {agent.error.message}</div>
            </div>
          )}
        </div>

        <div className="chat-input-area">
          <input
            className="chat-input"
            type="text"
            placeholder={
              variants ? 'Pick a variant above first…' : 'Type a message…'
            }
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={loading || !!variants}
          />
          <button
            className="btn btn-send"
            onClick={() => input.trim() && handleSend(input.trim())}
            disabled={loading || !!variants || !input.trim()}>
            Send
          </button>
        </div>
      </div>

      <aside className="info-sidebar">
        <h3>📋 v2 Branching API</h3>
        <ol>
          <li>
            <code>agent.runVariants(input, 2)</code> fires two parallel runs
            from the current continuation. Returns an{' '}
            <code>AgentVariant[]</code> with each branch's{' '}
            <code>continuationId</code>, <code>message</code>, and{' '}
            <code>state</code>.
          </li>
          <li>
            User picks one → <code>agent.continueFrom(variant.snapshotId)</code>{' '}
            advances the hook to that branch (fetches its <code>/state</code> +
            updates <code>messages</code>/<code>customState</code>/
            <code>artifacts</code> reactively).
          </li>
          <li>
            Discarded variants stay in the store as immutable snapshots — the
            user could revisit them via their URL.
          </li>
        </ol>

        <h4>Page code</h4>
        <pre>{`const agent = useGenkitAgent({
  url: '/api/branchingAgent',
  resumeFromSnapshotId: urlSnapshotId,
});

// Generate variants
const variants = await agent.runVariants(
  { messages: [...] },
  2
);

// Pick one
await agent.continueFrom(variants[i].snapshotId);`}</pre>
      </aside>
    </div>
  );
}

function variantText(v: AgentVariant): string {
  if (!v.message) return '(no message)';
  return (
    v.message.content
      ?.map((p: any) => p.text)
      .filter(Boolean)
      .join('') || JSON.stringify(v.message, null, 2)
  );
}
