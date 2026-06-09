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

import { remoteAgent, type AgentChat } from 'genkit/beta/client';
import { useCallback, useMemo, useRef, useState } from 'react';
import { ChatUI, type ChatMessage } from '../components/ChatUI';

// ---------------------------------------------------------------------------
// Client-Managed State — weather chat with NO server store
//
// Demonstrates:
//   • The `remoteAgent` client with client-managed state (no server store)
//   • The client owns the session state: the `remoteAgent` client tracks it
//     across turns and round-trips it to the stateless server automatically
//   • Tool calling works identically to the server-stored variant
//   • A "State Inspector" panel shows the raw state JSON so you can
//     see exactly what's being round-tripped (including message history)
//
// Compare with WeatherChat — same UX, but here the server is fully
// stateless. All session state lives in the blob the client round-trips.
// ---------------------------------------------------------------------------

const ENDPOINT = '/api/weatherAgentStateless';

export default function ClientState() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streamingText, setStreamingText] = useState('');
  const [loading, setLoading] = useState(false);

  // A human-readable view of the client-owned state blob, refreshed each turn.
  const [stateDisplay, setStateDisplay] = useState<string>(
    '(no state yet — first turn will create it)'
  );

  // Typed HTTP client. With a stateless server, the client owns and
  // round-trips the full session state on every turn.
  const agent = useMemo(() => remoteAgent({ url: ENDPOINT }), []);

  // The conversation. Created on first send and reused for follow-up turns so
  // state (messages + custom + artifacts) is threaded automatically.
  const chatRef = useRef<AgentChat | null>(null);

  const handleSend = useCallback(
    async (text: string) => {
      if (loading) return;

      setMessages((prev) => [...prev, { role: 'user', text }]);
      setLoading(true);
      setStreamingText('');

      // Lazily create the chat. No initial state — the first turn creates it.
      if (!chatRef.current) {
        chatRef.current = agent.chat();
      }
      const chat = chatRef.current;

      try {
        const turn = chat.sendStream(text);

        let accumulated = '';
        for await (const chunk of turn.stream) {
          // ── Tool calls/responses — render inline from the raw chunk ──
          for (const part of chunk.raw.modelChunk?.content ?? []) {
            if (part.toolRequest) {
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

          // ── Model text chunks ──
          if (chunk.text) {
            accumulated = chunk.accumulatedText;
            setStreamingText(accumulated);
          }
        }

        const res = await turn.response;
        setStreamingText('');

        // Surface the full client-owned state blob in the inspector. We show
        // the complete state (messages + custom + artifacts) round-tripped by
        // the client.
        setStateDisplay(JSON.stringify(res.raw.state ?? null, null, 2));

        setMessages((prev) => [
          ...prev,
          { role: 'model', text: res.text || accumulated },
        ]);
      } catch (err: any) {
        // The turn failed. The client preserves the last-good state, so the
        // session stays usable — surface the error but keep chatting.
        setStreamingText('');
        setMessages((prev) => [
          ...prev,
          {
            role: 'system',
            text:
              `⚠️ Turn failed (${err.status ?? 'INTERNAL'}): ${err.message}. ` +
              `Your session state was preserved — you can keep chatting.`,
          },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [agent, loading]
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
          This is the raw <code>state</code> blob the client round-trips. It
          contains the full message history, custom data, and artifacts. The{' '}
          <code>remoteAgent</code> client stores it and sends it back on every
          subsequent turn automatically.
        </p>
        <pre className="state-inspector-json">{stateDisplay}</pre>
      </aside>
    </div>
  );
}
