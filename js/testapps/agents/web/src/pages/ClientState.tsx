import type {
  AgentInit,
  AgentInput,
  AgentOutput,
  AgentStreamChunk,
} from 'genkit/beta';
import { streamFlow } from 'genkit/beta/client';
import { useCallback, useRef, useState } from 'react';
import { ChatUI, type Message } from '../components/ChatUI';

// ---------------------------------------------------------------------------
// Client-Managed State — weather chat with NO server store
//
// Demonstrates:
//   • streamFlow() with client-managed state (no server-side store)
//   • The client owns the `state` blob: receive it from the server,
//     store it locally, and send it back on every subsequent turn
//   • Tool calling works identically to the server-stored variant
//   • A "State Inspector" panel shows the raw state JSON so you can
//     see exactly what's being round-tripped (including message history)
//
// Compare with WeatherChat — same UX, but here the server is fully
// stateless. All session state lives in the blob the client round-trips.
// ---------------------------------------------------------------------------

const ENDPOINT = '/api/clientStateAgent';

export default function ClientState() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [streamingText, setStreamingText] = useState('');
  const [loading, setLoading] = useState(false);

  // The client owns this state blob. It's returned by the server on every
  // turn and must be sent back on the next turn via `init: { state }`.
  const [stateDisplay, setStateDisplay] = useState<string>(
    '(no state yet — first turn will create it)'
  );
  const stateRef = useRef<any>(undefined);

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

      // On the first turn, `init` is empty (no prior state).
      // On subsequent turns, we send back the state blob from the last turn.
      // This is the KEY difference from server-stored flows — we always
      // send `state`, never `snapshotId`.
      const init: AgentInit = stateRef.current
        ? { state: stateRef.current }
        : {};

      try {
        // ── Stream the response ────────────────────────────────────────
        const response = streamFlow<AgentOutput, AgentStreamChunk, AgentInit>({
          url: ENDPOINT,
          input,
          init,
        });

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

        // Store the state blob for the next turn.
        if (result?.state) {
          stateRef.current = result.state;
          setStateDisplay(JSON.stringify(result.state, null, 2));
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
    [loading]
  );

  return (
    <div className="client-state-layout">
      <ChatUI
        title="Client-Managed Weather Chat"
        description="Same weather agent, but NO server store. The client round-trips the full session state."
        messages={messages}
        streamingText={streamingText}
        loading={loading}
        onSend={handleSend}
      />
      {/* State Inspector — shows the raw state JSON being round-tripped */}
      <aside className="state-inspector">
        <h3>📦 Session State (client-owned)</h3>
        <p className="state-inspector-hint">
          This is the raw <code>state</code> blob returned by the server. It
          contains the full message history, custom data, and artifacts. The
          client stores it and sends it back on every subsequent turn via{' '}
          <code>init: {'{ state }'}</code>.
        </p>
        <pre className="state-inspector-json">{stateDisplay}</pre>
      </aside>
    </div>
  );
}

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
