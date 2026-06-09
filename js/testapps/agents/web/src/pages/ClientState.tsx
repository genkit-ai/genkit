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

import { AgentError, remoteAgent, type AgentChat } from 'genkit/beta/client';
import { useCallback, useMemo, useRef, useState } from 'react';
import { ChatUI, type ChatMessage } from '../components/ChatUI';

// ---------------------------------------------------------------------------
// Client-Managed State — weather chat with NO server store
//
// Demonstrates:
//   • remoteAgent({ stateManagement: 'client' }) — no server-side store
//   • The chat owns the `state` blob: it's returned by the server, tracked
//     by the chat, and sent back automatically on every subsequent turn
//   • Tool calling works identically to the server-stored variant
//   • A "State Inspector" panel shows the raw state JSON so you can
//     see exactly what's being round-tripped (including message history)
//
// Compare with WeatherChat — same UX, but here the server is fully
// stateless. All session state lives in the blob the chat round-trips.
// ---------------------------------------------------------------------------

const ENDPOINT = '/api/weatherAgentStateless';

export default function ClientState() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streamingText, setStreamingText] = useState('');
  const [loading, setLoading] = useState(false);

  // The chat owns the state blob. It's returned by the server on every turn
  // and sent back automatically on the next turn. We display the raw blob.
  const [stateDisplay, setStateDisplay] = useState<string>(
    '(no state yet — first turn will create it)'
  );

  // A client-managed chat: state is round-tripped, never a snapshotId.
  const agent = useMemo(
    () => remoteAgent({ url: ENDPOINT, stateManagement: 'client' }),
    []
  );
  const chatRef = useRef<AgentChat | null>(null);

  const handleSend = useCallback(
    async (text: string) => {
      if (loading) return;

      setMessages((prev) => [...prev, { role: 'user', text }]);
      setLoading(true);
      setStreamingText('');

      // Lazily create the chat on the first turn.
      if (!chatRef.current) {
        chatRef.current = agent.chat();
      }
      const chat = chatRef.current;

      try {
        // ── Stream the response ────────────────────────────────────────
        const turn = chat.sendStream(text);

        let accumulated = '';
        for await (const chunk of turn.stream) {
          if (chunk.text) {
            accumulated += chunk.text;
            setStreamingText(accumulated);
          }
          for (const tr of chunk.toolRequests) {
            const req = tr.toolRequest;
            setMessages((prev) => [
              ...prev,
              {
                role: 'tool',
                text: `🔧 Calling ${req.name}(${JSON.stringify(req.input)})`,
              },
            ]);
          }
          for (const part of chunk.raw.modelChunk?.content || []) {
            if (part.toolResponse) {
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
        const res = await turn.response;
        setStreamingText('');

        // Show the full state blob the client round-trips. `res.raw.state` is
        // the complete SessionState (messages + custom + artifacts).
        if (res.raw.state) {
          setStateDisplay(JSON.stringify(res.raw.state, null, 2));
        }

        setMessages((prev) => [
          ...prev,
          { role: 'model', text: res.text || accumulated },
        ]);
      } catch (err: any) {
        setStreamingText('');
        // A turn can fail gracefully: it throws an AgentError but the chat
        // keeps the last-good state, so the conversation can continue.
        if (err instanceof AgentError) {
          if (err.state !== undefined || chat.state !== undefined) {
            setStateDisplay(
              JSON.stringify(err.response.raw.state ?? chat.state, null, 2)
            );
          }
          setMessages((prev) => [
            ...prev,
            {
              role: 'system',
              text:
                `⚠️ Turn failed (${err.status}): ${err.message}. ` +
                `Your session state was preserved — you can keep chatting.`,
            },
          ]);
          return;
        }
        setMessages((prev) => [
          ...prev,
          { role: 'system', text: `Connection error: ${err.message}` },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [loading, agent]
  );

  return (
    <div className="client-state-layout">
      <ChatUI
        title="Client-Managed Weather Chat"
        description="Same weather agent, but NO server store. The client round-trips the full session state."
        suggestions={[
          'What is the weather like in London?',
          'Is it sunny in Tokyo right now?',
        ]}
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
          contains the full message history, custom data, and artifacts. The{' '}
          <code>chat</code> stores it and sends it back automatically on every
          subsequent turn.
        </p>
        <pre className="state-inspector-json">{stateDisplay}</pre>
      </aside>
    </div>
  );
}
