import type { AgentInit, AgentInput, AgentOutput } from 'genkit/beta';
import { runFlow } from 'genkit/beta/client';
import { useCallback, useEffect, useRef, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';

// ---------------------------------------------------------------------------
// Branching Chat — "Pick Your Variant" UI
//
// Demonstrates:
//   • Session branching via snapshotId — forking a conversation into
//     two independent timelines from the same checkpoint
//   • Parallel `runFlow()` calls from the same snapshotId
//   • The user picks which variant to continue from, selecting a branch
//   • Abandoned branches remain in the store (immutable snapshots)
//   • URL-based session persistence + restore on reload
// ---------------------------------------------------------------------------

const ENDPOINT = '/api/branchingAgent';
const STATE_ENDPOINT = '/api/branchingAgent/state';

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

  // The snapshotId of the current branch point.
  const snapshotIdRef = useRef<string | undefined>(urlSnapshotId);

  // ── Restore session from snapshotId on mount ───────────────────────
  useEffect(() => {
    if (!urlSnapshotId) return;

    let cancelled = false;

    async function restore() {
      try {
        const snapshot = (await runFlow({
          url: STATE_ENDPOINT,
          input: urlSnapshotId,
        })) as any;

        if (cancelled) return;

        if (snapshot?.state?.messages) {
          const restored: ChatMessage[] = [];
          for (const msg of snapshot.state.messages) {
            const role = msg.role as ChatMessage['role'];
            if (role !== 'user' && role !== 'model') continue;
            const textParts = (msg.content || [])
              .filter((p: any) => p.text)
              .map((p: any) => p.text);
            if (textParts.length > 0) {
              restored.push({ role, text: textParts.join('') });
            }
          }
          setMessages(restored);
          snapshotIdRef.current = snapshot.snapshotId ?? urlSnapshotId;
        }
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
  }, []); // Only run on mount

  // ── Send a message and generate two variants ─────────────────────────
  const handleSend = useCallback(
    async (text: string) => {
      if (loading || variants) return;

      setMessages((prev) => [...prev, { role: 'user', text }]);
      setInput('');
      setLoading(true);
      setError(null);
      setVariants(null);

      const msgInput: AgentInput = {
        messages: [{ role: 'user' as const, content: [{ text }] }],
      };

      // Both calls branch from the same snapshotId (or fresh session).
      const init: AgentInit = snapshotIdRef.current
        ? { snapshotId: snapshotIdRef.current }
        : {};

      try {
        // Fire two requests in parallel from the same branch point.
        const [resultA, resultB] = await Promise.all([
          runFlow<AgentOutput, AgentInit>({
            url: ENDPOINT,
            input: msgInput,
            init,
          }),
          runFlow<AgentOutput, AgentInit>({
            url: ENDPOINT,
            input: msgInput,
            init,
          }),
        ]);

        const textA = extractText(resultA);
        const textB = extractText(resultB);

        setVariants({
          a: { text: textA, snapshotId: resultA?.snapshotId ?? '' },
          b: { text: textB, snapshotId: resultB?.snapshotId ?? '' },
        });
      } catch (err: any) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    },
    [loading, variants]
  );

  // ── User picks a variant ─────────────────────────────────────────────
  const handlePick = useCallback(
    (which: 'a' | 'b') => {
      if (!variants) return;

      const chosen = variants[which];
      snapshotIdRef.current = chosen.snapshotId;

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
            {snapshotIdRef.current && (
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
            User sends a message. The client fires <strong>two parallel</strong>{' '}
            <code>runFlow()</code> calls, both with the same{' '}
            <code>{'init: { snapshotId }'}</code>.
          </li>
          <li>
            Each call creates an <strong>independent branch</strong> from the
            same conversation checkpoint. The LLM's non-determinism produces
            different responses.
          </li>
          <li>Both variants are displayed side-by-side. The user picks one.</li>
          <li>
            The chosen variant's <code>snapshotId</code> becomes the new branch
            point for the next turn and is pushed into the URL for persistence.
          </li>
          <li>
            On reload, the client calls <code>/state</code> with the URL's
            snapshotId to restore the conversation history.
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
        <pre>{`// Branch: two calls from same snapshot
const [a, b] = await Promise.all([
  runFlow({
    url: '/api/branchingAgent',
    input: { messages: [...] },
    init: { snapshotId },
  }),
  runFlow({
    url: '/api/branchingAgent',
    input: { messages: [...] },
    init: { snapshotId },
  }),
]);

// a.snapshotId !== b.snapshotId
// Both branch from the same point
// Pick one to continue from`}</pre>
      </aside>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Helper: extract text from a session flow result
// ---------------------------------------------------------------------------
function extractText(result: AgentOutput): string {
  if (!result) return '(no result)';
  const msg = result.message;
  if (!msg) return JSON.stringify(result, null, 2);
  const parts: string[] = [];
  for (const p of msg.content || []) {
    if (p.text) parts.push(p.text);
  }
  return parts.join('') || JSON.stringify(result, null, 2);
}
