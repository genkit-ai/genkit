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

import type { MessageData, Part } from 'genkit/beta';
import { AgentError, remoteAgent, type AgentChat } from 'genkit/beta/client';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { ChatUI, type ChatMessage } from '../components/ChatUI';

// ---------------------------------------------------------------------------
// Weather Chat — multi-turn streaming chat with tool-calling + session restore
//
// Demonstrates the ergonomic `remoteAgent` client:
//   • agent.chat() — a stateful conversation that tracks snapshotId/state
//   • chat.send(text) — returns a turn with `.stream` and `.response`
//   • agent.loadChat({ snapshotId }) — restore a session from the URL
//   • Graceful failures surface as `AgentError` while the session stays usable
// ---------------------------------------------------------------------------

const ENDPOINT = '/api/weatherAgent';

export default function WeatherChat() {
  const { snapshotId: urlSnapshotId } = useParams<{ snapshotId: string }>();
  const navigate = useNavigate();

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streamingText, setStreamingText] = useState('');
  const [loading, setLoading] = useState(false);
  const [restoring, setRestoring] = useState(!!urlSnapshotId);
  const [hasSession, setHasSession] = useState(!!urlSnapshotId);

  // The agent client and the live chat that tracks snapshotId/state/messages.
  const agent = useMemo(() => remoteAgent({ url: ENDPOINT }), []);
  const chatRef = useRef<AgentChat | null>(null);

  // ── Restore session from snapshotId on mount ───────────────────────
  useEffect(() => {
    if (!urlSnapshotId) return;

    let cancelled = false;

    async function restore() {
      try {
        // loadChat fetches the snapshot and returns a chat with history +
        // state restored, ready to continue the conversation.
        const chat = await agent.loadChat({ snapshotId: urlSnapshotId! });
        if (cancelled) return;

        chatRef.current = chat;
        setMessages(messagesToChat(chat.messages));
        setHasSession(true);
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
  }, [agent]); // Only run on mount

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
          // Tool requests are exposed directly on the chunk.
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
          // Tool responses live on the raw chunk content.
          for (const part of toolResponses(chunk.raw.modelChunk?.content)) {
            const tr = part.toolResponse!;
            setMessages((prev) => [
              ...prev,
              {
                role: 'tool',
                text: `✅ ${tr.name} → ${JSON.stringify(tr.output)}`,
              },
            ]);
          }
        }

        // ── Read the final result ──────────────────────────────────────
        const res = await turn.response;
        setStreamingText('');

        // The chat tracks the latest snapshotId — push it into the URL so
        // the user can bookmark or hard-reload this session.
        if (chat.snapshotId) {
          setHasSession(true);
          navigate(`/weather/${chat.snapshotId}`, { replace: true });
        }

        setMessages((prev) => [
          ...prev,
          { role: 'model', text: res.text || accumulated },
        ]);
      } catch (err: any) {
        setStreamingText('');
        // A turn can fail gracefully: it throws an AgentError but the chat
        // keeps the last-good snapshot/state, so the session stays usable.
        if (err instanceof AgentError) {
          if (chat.snapshotId) {
            setHasSession(true);
            navigate(`/weather/${chat.snapshotId}`, { replace: true });
          }
          setMessages((prev) => [
            ...prev,
            {
              role: 'system',
              text:
                `⚠️ Turn failed (${err.status}): ${err.message}. ` +
                `The last-good snapshot was preserved — you can keep chatting.`,
            },
          ]);
          return;
        }
        // Only transport/connection errors reach here.
        setMessages((prev) => [
          ...prev,
          { role: 'system', text: `Connection error: ${err.message}` },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [loading, navigate, agent]
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
          hasSession ? (
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
            Client sends user message via <code>chat.send()</code> — responses
            arrive as they're generated on <code>turn.stream</code>.
          </li>
          <li>
            The model can invoke <strong>tools</strong> (e.g.{' '}
            <code>getWeather</code>). Tool calls and responses render inline in
            the chat.
          </li>
          <li>
            The <code>chat</code> tracks <code>snapshotId</code> and{' '}
            <code>state</code> automatically across turns — no manual threading.
          </li>
          <li>
            The <code>chat.snapshotId</code> is pushed into the URL, so you can
            bookmark or share the session link.
          </li>
          <li>
            On page load with a <code>:snapshotId</code> in the URL, the client
            calls <code>agent.loadChat()</code> to restore the full conversation
            history.
          </li>
        </ol>

        <h4>Key APIs</h4>
        <pre>{`// Create a client + chat
const agent = remoteAgent({
  url: '/api/weatherAgent',
});
const chat = agent.chat();

// Streaming multi-turn
const turn = chat.sendStream('Weather in Tokyo?');
for await (const chunk of turn.stream) {
  // chunk.text, chunk.toolRequests
}

const res = await turn.response;
// res.text, res.snapshotId, res.state
// chat.snapshotId → push to URL

// Restore session
const chat =
  await agent.loadChat({ snapshotId });`}</pre>

        <h4>Session Persistence</h4>
        <p>
          This demo uses a <strong>server-side session store</strong>. The{' '}
          <code>snapshotId</code> is a key into the store — the full message
          history lives on the server and the <code>chat</code> tracks it for
          you.
        </p>
      </aside>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Filter the toolResponse parts out of a content array. */
function toolResponses(content?: Part[]): Part[] {
  return (content || []).filter((p) => p.toolResponse);
}

/** Rebuild displayable chat messages from restored session history. */
function messagesToChat(history: MessageData[]): ChatMessage[] {
  const restored: ChatMessage[] = [];
  for (const msg of history) {
    const role = msg.role as ChatMessage['role'];
    const textParts = (msg.content || [])
      .filter((p: Part) => p.text)
      .map((p: Part) => p.text);

    if (textParts.length > 0) {
      restored.push({ role, text: textParts.join('') });
    }

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
  return restored;
}
