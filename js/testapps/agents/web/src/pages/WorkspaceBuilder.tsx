import { useCallback, useRef, useState } from 'react';
import { streamFlow } from 'genkit/beta/client';
import { ChatUI, type Message } from '../components/ChatUI';

// ---------------------------------------------------------------------------
// Workspace Builder — artifacts alongside chat
//
// Demonstrates:
//   • streamFlow() with artifact production
//   • Reading `result.artifacts` from the session flow response
//   • Multi-turn session via `init: { state }` round-tripping
//   • Displaying generated code artifacts in a side panel
// ---------------------------------------------------------------------------

const ENDPOINT = '/api/workspaceAgent';

interface Artifact {
  name: string;
  parts: Array<{ text?: string }>;
}

export default function WorkspaceBuilder() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [streamingText, setStreamingText] = useState('');
  const [loading, setLoading] = useState(false);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);

  // Session state — returned by the server, sent back on the next turn.
  const stateRef = useRef<any>(undefined);

  const handleSend = useCallback(
    async (text: string) => {
      if (loading) return;

      setMessages((prev) => [...prev, { role: 'user', text }]);
      setLoading(true);
      setStreamingText('');

      // ── Build the request ──────────────────────────────────────────────
      const input = {
        messages: [{ role: 'user' as const, content: [{ text }] }],
      };

      const init = stateRef.current ? { state: stateRef.current } : {};

      try {
        // ── Stream the response ────────────────────────────────────────
        const response = streamFlow({ url: ENDPOINT, input, init });

        let accumulated = '';
        for await (const chunk of response.stream) {
          const c = chunk as any;
          if (c?.modelChunk?.content) {
            for (const part of c.modelChunk.content) {
              if (part.text) {
                accumulated += part.text;
                setStreamingText(accumulated);
              }
            }
          }
        }

        // ── Read the final result ──────────────────────────────────────
        const result = (await response.output) as any;
        setStreamingText('');

        // Save session state for the next turn.
        if (result?.state) stateRef.current = result.state;

        // Update artifacts — the workspace agent returns an `artifacts`
        // array in the result whenever the emitArtifact tool was called.
        if (result?.artifacts) {
          setArtifacts(result.artifacts);
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
    <div className="workspace-layout">
      <ChatUI
        title="Workspace Builder"
        description="Generates code artifacts via an emitArtifact tool. Try: 'Write poem.txt with a poem about AI'"
        messages={messages}
        streamingText={streamingText}
        loading={loading}
        onSend={handleSend}
      />

      {/* Artifacts panel — shows generated files */}
      <aside className="artifacts-sidebar">
        <h3>🛠️ Artifacts</h3>
        {artifacts.length === 0 ? (
          <p className="artifacts-empty">
            No artifacts yet. Ask the agent to generate a file.
          </p>
        ) : (
          artifacts.map((a, i) => (
            <div key={i} className="artifact">
              <div className="artifact-name">{a.name}</div>
              <pre className="artifact-content">
                {a.parts
                  ?.filter((p) => p.text)
                  .map((p) => p.text)
                  .join('\n')}
              </pre>
            </div>
          ))
        )}
      </aside>
    </div>
  );
}

function extractText(result: any): string {
  if (!result) return '(no result)';
  const msg = result.message;
  if (!msg) return JSON.stringify(result, null, 2);
  const parts: string[] = [];
  for (const p of msg.content || []) {
    if (p.text) parts.push(p.text);
  }
  return parts.join('') || JSON.stringify(result, null, 2);
}
