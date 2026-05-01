import { useCallback, useEffect, useRef, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { runFlow, streamFlow } from 'genkit/beta/client';
import type { AgentInit, AgentInput, AgentOutput, AgentStreamChunk } from 'genkit/beta';
import { ChatUI, type Message } from '../components/ChatUI';

// ---------------------------------------------------------------------------
// Weather Chat — multi-turn streaming chat with tool-calling + session restore
//
// Demonstrates:
//   • streamFlow() for streaming responses
//   • Multi-turn session via `init: { state }` round-tripping
//   • Rendering streamed tool calls and tool responses in real time
//   • Restoring a session from a snapshotId (URL-based session persistence)
//   • Using the `/state` endpoint to fetch snapshot data on page load
// ---------------------------------------------------------------------------

const ENDPOINT = '/api/weatherAgent';
const STATE_ENDPOINT = '/api/weatherAgent/state';

export default function WeatherChat() {
  const { snapshotId: urlSnapshotId } = useParams<{ snapshotId: string }>();
  const navigate = useNavigate();

  const [messages, setMessages] = useState<Message[]>([]);
  const [streamingText, setStreamingText] = useState('');
  const [loading, setLoading] = useState(false);
  const [restoring, setRestoring] = useState(!!urlSnapshotId);

  // Session tracking
  const stateRef = useRef<any>(undefined);
  const snapshotIdRef = useRef<string | undefined>(urlSnapshotId);

  // ── Restore session from snapshotId on mount ───────────────────────
  useEffect(() => {
    if (!urlSnapshotId) return;

    let cancelled = false;

    async function restore() {
      try {
        // Call the /state endpoint to fetch the snapshot data.
        // The getSnapshotDataAction takes a snapshotId string as input
        // and returns a SessionSnapshot with the full session state.
        const snapshot = (await runFlow({
          url: STATE_ENDPOINT,
          input: urlSnapshotId,
        })) as any;

        if (cancelled) return;

        if (snapshot?.state?.messages) {
          // Reconstruct chat messages from the session history.
          const restored: Message[] = [];
          for (const msg of snapshot.state.messages) {
            const role = msg.role as Message['role'];
            const textParts = (msg.content || [])
              .filter((p: any) => p.text)
              .map((p: any) => p.text);

            if (textParts.length > 0) {
              restored.push({ role, text: textParts.join('') });
            }

            // Also show tool calls/responses from history
            for (const p of msg.content || []) {
              if (p.toolRequest) {
                restored.push({
                  role: 'tool',
                  text: `🔧 ${p.toolRequest.name}(${JSON.stringify(p.toolRequest.input)})`,
                });
              }
              if (p.toolResponse) {
                restored.push({
                  role: 'tool',
                  text: `✅ ${p.toolResponse.name} → ${JSON.stringify(p.toolResponse.output)}`,
                });
              }
            }
          }
          setMessages(restored);

          // Use the snapshot's state for continuing the conversation.
          // Since the weatherAgent uses a server store, we can use snapshotId.
          snapshotIdRef.current = snapshot.snapshotId;
          // Also keep the state for the `state` init path.
          stateRef.current = snapshot.state;
        }
      } catch (err: any) {
        if (!cancelled) {
          setMessages([
            {
              role: 'system',
              text: `Failed to restore session: ${err.message}`,
            },
          ]);
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

  const handleSend = useCallback(
    async (text: string) => {
      if (loading) return;

      setMessages((prev) => [...prev, { role: 'user', text }]);
      setLoading(true);
      setStreamingText('');

      // ── Build the request ──────────────────────────────────────────────
      const input: AgentInput = {
        messages: [{ role: 'user', content: [{ text }] }],
      };

      // Prefer state over snapshotId for multi-turn.
      const init: AgentInit = stateRef.current
        ? { state: stateRef.current }
        : snapshotIdRef.current
          ? { snapshotId: snapshotIdRef.current }
          : {};

      try {
        // ── Stream the response ────────────────────────────────────────
        const response = streamFlow<AgentOutput, AgentStreamChunk, AgentInit>({ url: ENDPOINT, input, init });

        let accumulated = '';
        for await (const chunk of response.stream) {
          const c = chunk;
          const mc = c?.modelChunk;
          if (!mc) continue;

          for (const part of mc.content || []) {
            if (part.text) {
              accumulated += part.text;
              setStreamingText(accumulated);
            } else if (part.toolRequest) {
              const tr = part.toolRequest;
              setMessages((prev) => [
                ...prev,
                {
                  role: 'tool',
                  text: `🔧 Calling ${tr.name}(${JSON.stringify(tr.input)})`,
                },
              ]);
            } else if (part.toolResponse) {
              const tr = part.toolResponse;
              setMessages((prev) => [
                ...prev,
                {
                  role: 'tool',
                  text: `✅ ${tr.name} → ${JSON.stringify(tr.output)}`,
                },
              ]);
            }
          }
        }

        // ── Read the final result ──────────────────────────────────────
        const result = await response.output;
        setStreamingText('');

        // Save session state for the next turn.
        if (result?.state) stateRef.current = result.state;

        // Update the snapshotId and push it into the URL.
        if (result?.snapshotId) {
          snapshotIdRef.current = result.snapshotId;
          // Update URL so the user can bookmark or hard-reload this session.
          navigate(`/weather/${result.snapshotId}`, { replace: true });
        }

        const replyText = extractText(result);
        setMessages((prev) => [
          ...prev,
          { role: 'model', text: replyText || accumulated },
        ]);
      } catch (err: any) {
        setStreamingText('');
        setMessages((prev) => [
          ...prev,
          { role: 'system', text: `Error: ${err.message}` },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [loading, navigate]
  );

  if (restoring) {
    return (
      <div className="page-with-sidebar">
        <div className="chat-panel">
          <div className="chat-header">
            <h2>Weather Agent</h2>
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
      <ChatUI
        title="Weather Agent"
        description="Multi-turn chat with tool-calling. Ask about the weather in any city. Session persists in the URL."
        messages={messages}
        streamingText={streamingText}
        loading={loading}
        onSend={handleSend}
        headerAction={
          snapshotIdRef.current ? (
            <Link to="/weather" className="btn btn-new-session" reloadDocument>
              ✨ New Session
            </Link>
          ) : null
        }
      />

      <aside className="info-sidebar">
        <h3>📋 How It Works</h3>
        <ol>
          <li>
            Client sends user message via <code>streamFlow()</code> — responses
            arrive as they're generated.
          </li>
          <li>
            The model can invoke <strong>tools</strong> (e.g. <code>getWeather</code>).
            Tool calls and responses render inline in the chat.
          </li>
          <li>
            Each response returns a <code>state</code> object and a{' '}
            <code>snapshotId</code>. The state is sent back on the next turn via{' '}
            <code>{'init: { state }'}</code> for multi-turn context.
          </li>
          <li>
            The <code>snapshotId</code> is pushed into the URL, so you can
            bookmark or share the session link.
          </li>
          <li>
            On page load with a <code>:snapshotId</code> in the URL, the client
            calls the <code>/state</code> endpoint to restore the full
            conversation history.
          </li>
        </ol>

        <h4>Key APIs</h4>
        <pre>{`// Streaming multi-turn
const response = streamFlow({
  url: '/api/weatherAgent',
  input: { messages: [...] },
  init: { state },
});

for await (const chunk of response.stream) {
  // chunk.modelChunk.content[]
  // → .text, .toolRequest, .toolResponse
}

const result = await response.output;
// result.state → send back next turn
// result.snapshotId → push to URL

// Restore session
runFlow({
  url: '/api/weatherAgent/state',
  input: snapshotId,
});`}</pre>

        <h4>Session Persistence</h4>
        <p>
          This demo uses a <strong>server-side session store</strong>. The{' '}
          <code>snapshotId</code> is a key into the store — the full message
          history lives on the server. The client also receives{' '}
          <code>state</code> for stateless round-tripping as a fallback.
        </p>
      </aside>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Helper: pull displayable text out of a session flow result
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
