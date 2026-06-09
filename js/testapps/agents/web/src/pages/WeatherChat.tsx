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
import { remoteAgent, type AgentChat } from 'genkit/beta/client';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { ChatUI, type ChatMessage } from '../components/ChatUI';

// ---------------------------------------------------------------------------
// Weather Chat — multi-turn streaming chat with tool-calling + session restore
//
// Demonstrates:
//   • The `remoteAgent` client for streaming responses (chat.sendStream)
//   • Multi-turn session — the client threads the session across turns
//   • Rendering streamed tool calls and tool responses in real time
//   • Restoring a session from a snapshotId (URL-based session persistence)
//     via `agent.loadChat({ snapshotId })`
// ---------------------------------------------------------------------------

const ENDPOINT = '/api/weatherAgent';

export default function WeatherChat() {
  const { snapshotId: urlSnapshotId } = useParams<{ snapshotId: string }>();
  const navigate = useNavigate();

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streamingText, setStreamingText] = useState('');
  const [loading, setLoading] = useState(false);
  const [restoring, setRestoring] = useState(!!urlSnapshotId);

  // Typed HTTP client for the weather agent. Uses a server-side session store,
  // so each turn resumes the latest snapshot automatically.
  const agent = useMemo(() => remoteAgent({ url: ENDPOINT }), []);

  // The conversation. Created on first send, restored from a snapshot on mount
  // when a :snapshotId is present in the URL.
  const chatRef = useRef<AgentChat | null>(null);

  // ── Restore session from snapshotId on mount ───────────────────────
  useEffect(() => {
    if (!urlSnapshotId) return;

    let cancelled = false;

    async function restore() {
      try {
        // Load the chat from the server snapshot — history is restored for us.
        const chat = await agent.loadChat({ snapshotId: urlSnapshotId! });

        if (cancelled) return;

        // Reconstruct chat messages from the restored session history.
        const restored: ChatMessage[] = [];
        for (const msg of chat.messages) {
          const role = msg.role as ChatMessage['role'];
          const textParts = (msg.content || [])
            .filter((p: Part) => p.text)
            .map((p: Part) => p.text);

          if (textParts.length > 0) {
            restored.push({ role, text: textParts.join('') });
          }

          // Also show tool calls/responses from history.
          for (const p of (msg.content as Part[]) || []) {
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
        chatRef.current = chat;
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

      // Lazily create the chat on the first turn (no snapshot to restore).
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

        // Push the latest snapshotId into the URL so the user can bookmark or
        // hard-reload this session.
        if (chat.snapshotId) {
          navigate(`/weather/${chat.snapshotId}`, { replace: true });
        }

        setMessages((prev) => [
          ...prev,
          { role: 'model', text: res.text || accumulated },
        ]);
      } catch (err: any) {
        // A turn can fail gracefully: the client preserves the last-good
        // snapshot, so the session stays restorable. Surface the error but
        // keep the session usable.
        setStreamingText('');
        if (err.snapshotId) {
          navigate(`/weather/${err.snapshotId}`, { replace: true });
        }
        setMessages((prev) => [
          ...prev,
          {
            role: 'system',
            text:
              `⚠️ Turn failed (${err.status ?? 'INTERNAL'}): ${err.message}. ` +
              `The last-good snapshot was preserved — you can keep chatting.`,
          },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [agent, loading, navigate]
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
        suggestions={[
          'What is the weather like in London?',
          'Is it sunny in Tokyo right now?',
          'Compare the weather in Paris and New York.',
        ]}
        messages={messages}
        streamingText={streamingText}
        loading={loading}
        onSend={handleSend}
        headerAction={
          chatRef.current?.snapshotId ? (
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
            Client sends user message via <code>chat.sendStream()</code> —
            responses arrive as they're generated.
          </li>
          <li>
            The model can invoke <strong>tools</strong> (e.g.{' '}
            <code>getWeather</code>). Tool calls and responses render inline in
            the chat.
          </li>
          <li>
            The <code>remoteAgent</code> client tracks the session across turns
            automatically — no manual state threading.
          </li>
          <li>
            After each turn, the <code>chat.snapshotId</code> is pushed into the
            URL, so you can bookmark or share the session link.
          </li>
          <li>
            On page load with a <code>:snapshotId</code> in the URL, the client
            calls <code>agent.loadChat({'{ snapshotId }'})</code> to restore the
            full conversation history.
          </li>
        </ol>

        <h4>Key APIs</h4>
        <pre>{`// Streaming multi-turn
const agent = remoteAgent({ url: '/api/weatherAgent' });
const chat = agent.chat();

const turn = chat.sendStream('Weather in Tokyo?');
for await (const chunk of turn.stream) {
  // chunk.text → streamed text
  // chunk.raw.modelChunk.content → tool req/resp
}
const res = await turn.response;
// chat.snapshotId → push to URL

// Restore session
const chat = await agent.loadChat({ snapshotId });`}</pre>

        <h4>Session Persistence</h4>
        <p>
          This demo uses a <strong>server-side session store</strong>. The{' '}
          <code>snapshotId</code> is a key into the store — the full message
          history lives on the server and is restored via{' '}
          <code>agent.loadChat()</code>.
        </p>
      </aside>
    </div>
  );
}
