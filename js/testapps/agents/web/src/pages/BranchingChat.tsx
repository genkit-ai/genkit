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

import type { Part } from 'genkit/beta';

import { remoteAgent, type AgentAPI } from 'genkit/beta/client';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';

// ---------------------------------------------------------------------------
// Branching Chat — "Pick Your Variant" UI
//
// Demonstrates:
//   • Session branching via snapshotId — forking a conversation into
//     two independent timelines from the same checkpoint
//   • Parallel chats started from the same snapshotId (agent.chat({ snapshotId }))
//   • The user picks which variant to continue from, selecting a branch
//   • Abandoned branches remain in the store (immutable snapshots)
//   • URL-based session persistence + restore on reload (agent.loadChat)
// ---------------------------------------------------------------------------

const ENDPOINT = '/api/branchingAgent';

/** A settled chat message (user or chosen model response). */
interface ChatMessage {
  role: 'user' | 'model';
  text: string;
}

/** A pair of variant responses waiting for user selection. */
interface VariantPair {
  a: { text: string; snapshotId: string };
  b: { text: string; snapshotId: string };
}

export default function BranchingChat() {
  const { snapshotId: urlSnapshotId } = useParams<{ snapshotId: string }>();
  const navigate = useNavigate();

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [variants, setVariants] = useState<VariantPair | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [restoring, setRestoring] = useState(!!urlSnapshotId);
  const [hasSession, setHasSession] = useState(!!urlSnapshotId);

  // The agent client. We branch by starting fresh chats from a snapshotId.
  const agent: AgentAPI = useMemo(() => remoteAgent({ url: ENDPOINT }), []);

  // The snapshotId of the current branch point.
  const snapshotIdRef = useRef<string | undefined>(urlSnapshotId);

  // ── Restore session from snapshotId on mount ───────────────────────
  useEffect(() => {
    if (!urlSnapshotId) return;

    let cancelled = false;

    async function restore() {
      try {
        const chat = await agent.loadChat({ snapshotId: urlSnapshotId! });
        if (cancelled) return;

        const restored: ChatMessage[] = [];
        for (const msg of chat.messages) {
          const role = msg.role as ChatMessage['role'];
          if (role !== 'user' && role !== 'model') continue;
          const textParts = (msg.content || [])
            .filter((p: Part) => p.text)
            .map((p: Part) => p.text);
          if (textParts.length > 0) {
            restored.push({ role, text: textParts.join('') });
          }
        }
        setMessages(restored);
        snapshotIdRef.current = chat.snapshotId ?? urlSnapshotId;
        setHasSession(true);
      } catch (err: any) {
        if (!cancelled) {
          setError(`Failed to restore session: ${err.message}`);
        }
      } finally {
        if (!cancelled) setRestoring(false);
      }
    }

    restore();
    return () => {
      cancelled = true;
    };
  }, [agent]); // Only run on mount

  // ── Send a message and generate two variants ─────────────────────────
  const handleSend = useCallback(
    async (text: string) => {
      if (loading || variants) return;

      setMessages((prev) => [...prev, { role: 'user', text }]);
      setInput('');
      setLoading(true);
      setError(null);
      setVariants(null);

      // Both branches start from the same snapshotId (or a fresh session).
      const init = snapshotIdRef.current
        ? { snapshotId: snapshotIdRef.current }
        : undefined;

      try {
        // Fire two independent chats in parallel from the same branch point.
        // Each chat tracks its own resulting snapshotId.
        const [resA, resB] = await Promise.all([
          agent.chat(init).send(text),
          agent.chat(init).send(text),
        ]);

        setVariants({
          a: { text: resA.text, snapshotId: resA.snapshotId ?? '' },
          b: { text: resB.text, snapshotId: resB.snapshotId ?? '' },
        });
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    },
    [loading, variants, agent]
  );

  // ── User picks a variant ─────────────────────────────────────────────
  const handlePick = useCallback(
    (which: 'a' | 'b') => {
      if (!variants) return;

      const chosen = variants[which];
      snapshotIdRef.current = chosen.snapshotId;
      setHasSession(true);

      // Push the chosen snapshotId into the URL for persistence.
      navigate(`/branching/${chosen.snapshotId}`, { replace: true });

      setMessages((prev) => [...prev, { role: 'model', text: chosen.text }]);
      setVariants(null);
    },
    [variants, navigate]
  );

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (input.trim()) handleSend(input.trim());
    }
  };

  if (restoring) {
    return (
      <div className="page-with-sidebar">
        <div className="chat-panel">
          <div className="chat-header">
            <h2>🔀 Branching Chat</h2>
            <span className="chat-desc">Restoring session…</span>
          </div>
          <div className="chat-messages">
            <div className="message">
              <div className="message-role">system</div>
              <div className="message-text loading">
                Restoring session from snapshot {urlSnapshotId}…
              </div>
            </div>
          </div>
        </div>
        <aside className="info-sidebar" />
      </div>
    );
  }

  return (
    <div className="page-with-sidebar">
      <div className="chat-panel">
        <div className="chat-header">
          <div className="chat-header-top">
            <h2>🔀 Branching Chat</h2>
            {hasSession && (
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

        {/* ── Message list ──────────────────────────────────────────── */}
        <div className="chat-messages">
          {messages.length === 0 && !loading && !variants && (
            <div className="chat-empty">
              Send a message to start. Each response will show two variants —
              pick your favorite to choose which branch to follow.
            </div>
          )}

          {messages.map((msg, i) => (
            <div
              key={i}
              className={`message ${msg.role === 'user' ? 'message-user' : ''}`}>
              <div className="message-role">
                {msg.role === 'user' ? 'You' : 'Model'}
              </div>
              <div className="message-text">{msg.text}</div>
            </div>
          ))}

          {/* ── Variant picker ────────────────────────────────────── */}
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
                <button
                  className="variant-card"
                  onClick={() => handlePick('a')}>
                  <div className="variant-card-badge">A</div>
                  <div className="variant-card-text">{variants.a.text}</div>
                  <div className="variant-card-action">Use this ✓</div>
                </button>
                <button
                  className="variant-card"
                  onClick={() => handlePick('b')}>
                  <div className="variant-card-badge">B</div>
                  <div className="variant-card-text">{variants.b.text}</div>
                  <div className="variant-card-action">Use this ✓</div>
                </button>
              </div>
            </div>
          )}

          {error && (
            <div className="message message-system">
              <div className="message-role">system</div>
              <div className="message-text">Error: {error}</div>
            </div>
          )}
        </div>

        {/* ── Input ───────────────────────────────────────────────── */}
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

      {/* ── Info sidebar ──────────────────────────────────────────────── */}
      <aside className="info-sidebar">
        <h3>📋 How It Works</h3>
        <ol>
          <li>
            User sends a message. The client starts{' '}
            <strong>two parallel</strong> chats, both from the same{' '}
            <code>{'{ snapshotId }'}</code>.
          </li>
          <li>
            Each chat creates an <strong>independent branch</strong> from the
            same conversation checkpoint. The LLM's non-determinism produces
            different responses.
          </li>
          <li>Both variants are displayed side-by-side. The user picks one.</li>
          <li>
            The chosen variant's <code>snapshotId</code> becomes the new branch
            point for the next turn and is pushed into the URL for persistence.
          </li>
          <li>
            On reload, the client calls <code>agent.loadChat()</code> with the
            URL's snapshotId to restore the conversation history.
          </li>
        </ol>

        <h4>Key Concept</h4>
        <p>
          A <code>snapshotId</code> is an <strong>immutable checkpoint</strong>.
          You can branch from it as many times as you want — each branch creates
          a new, independent snapshot. This is like Git: the original commit
          doesn't change when you create branches from it.
        </p>

        <h4>Key APIs</h4>
        <pre>{`// Branch: two chats from same snapshot
const [a, b] = await Promise.all([
  agent.chat({ snapshotId })
    .send(text),
  agent.chat({ snapshotId })
    .send(text),
]);

// a.snapshotId !== b.snapshotId
// Both branch from the same point
// Pick one to continue from`}</pre>
      </aside>
    </div>
  );
}
