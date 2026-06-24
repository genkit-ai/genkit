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
// Trip Planner — demonstrates definePromptAgent with a .prompt file
//
// This page is structurally similar to WeatherChat but targets the
// tripPlannerAgent, whose prompt is defined in `prompts/tripPlanner.prompt`
// and wired via `ai.definePromptAgent({ promptName: 'tripPlanner' })`.
//
// Demonstrates:
//   • Agent whose prompt lives in a .prompt file (dotprompt)
//   • definePromptAgent — prompt file + agent wiring separated
//   • Streaming multi-turn chat with tool calls
//   • Session restore via snapshotId in URL
// ---------------------------------------------------------------------------

const ENDPOINT = '/api/tripPlannerAgent';

export default function TripPlanner() {
  const { snapshotId: urlSnapshotId } = useParams<{ snapshotId: string }>();
  const navigate = useNavigate();

  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streamingText, setStreamingText] = useState('');
  const [loading, setLoading] = useState(false);
  const [restoring, setRestoring] = useState(!!urlSnapshotId);

  // Typed HTTP client for the agent (server-managed FileSessionStore).
  const agent = useMemo(() => remoteAgent({ url: ENDPOINT }), []);

  // The live conversation. Reused across turns so history is threaded for us.
  const chatRef = useRef<AgentChat | null>(null);
  const snapshotIdRef = useRef<string | undefined>(urlSnapshotId);

  // ── Restore session from snapshotId on mount ───────────────────────
  useEffect(() => {
    if (!urlSnapshotId) return;

    let cancelled = false;

    async function restore() {
      try {
        // Load the chat from the server snapshot — history is restored for us.
        const chat = await agent.loadChat({ snapshotId: urlSnapshotId! });

        if (cancelled) return;

        const restored: ChatMessage[] = [];
        for (const msg of chat.messages) {
          const role = msg.role as ChatMessage['role'];
          const textParts = (msg.content || [])
            .filter((p: Part) => p.text)
            .map((p: Part) => p.text);

          if (textParts.length > 0) {
            restored.push({ role, text: textParts.join('') });
          }

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
        snapshotIdRef.current = chat.snapshotId;
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

      // Lazily create the chat (fresh session) the first time we send.
      if (!chatRef.current) {
        chatRef.current = agent.chat();
      }
      const chat = chatRef.current;

      try {
        const turn = chat.sendStream(text);

        let accumulated = '';
        for await (const chunk of turn.stream) {
          for (const part of chunk.raw.modelChunk?.content || []) {
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

        const res = await turn.response;
        setStreamingText('');

        if (res.snapshotId) {
          snapshotIdRef.current = res.snapshotId;
          navigate(`/trip-planner/${res.snapshotId}`, { replace: true });
        }

        setMessages((prev) => [
          ...prev,
          { role: 'model', text: res.text || accumulated },
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
    [agent, loading, navigate]
  );

  if (restoring) {
    return (
      <div className="page-with-sidebar">
        <div className="chat-panel">
          <div className="chat-header">
            <h2>Trip Planner</h2>
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
        title="Trip Planner"
        description="Multi-turn travel assistant powered by a .prompt file and definePromptAgent."
        suggestions={[
          'I want to plan a trip to Paris. What should I see there?',
          'Find me flights from New York to Tokyo.',
          'What are the top attractions in London?',
        ]}
        messages={messages}
        streamingText={streamingText}
        loading={loading}
        onSend={handleSend}
        headerAction={
          snapshotIdRef.current ? (
            <Link
              to="/trip-planner"
              className="btn btn-new-session"
              reloadDocument>
              ✨ New Session
            </Link>
          ) : null
        }
      />

      <aside className="info-sidebar">
        <h3>📋 How It Works</h3>
        <p>
          This agent demonstrates <code>definePromptAgent</code> — the prompt
          template lives in a <strong>.prompt file</strong> (
          <code>prompts/tripPlanner.prompt</code>) rather than being defined
          inline in code.
        </p>

        <h4>Prompt File</h4>
        <pre>{`---
model: googleai/gemini-flash-latest
tools:
  - getAttractions
  - getFlightInfo
---

{{role "system"}}
You are a friendly trip planning
assistant...

{{history}}`}</pre>

        <h4>Agent Wiring</h4>
        <pre>{`// Tools are defined in code
const getAttractions = ai.defineTool(...);
const getFlightInfo = ai.defineTool(...);

// Agent is wired from the .prompt file
const tripPlannerAgent =
  ai.definePromptAgent({
    promptName: 'tripPlanner',
    store: new FileSessionStore(...),
  });`}</pre>

        <h4>Why use definePromptAgent?</h4>
        <ul>
          <li>
            <strong>Separation of concerns</strong> — prompt authors can edit{' '}
            <code>.prompt</code> files without touching code
          </li>
          <li>
            <strong>Reuse</strong> — the same prompt can power multiple agents
            with different stores or configurations
          </li>
          <li>
            <strong>Dotprompt features</strong> — use Handlebars templates,{' '}
            <code>{'{{history}}'}</code>, roles, helpers, and partials
          </li>
        </ul>

        <h4>Key APIs</h4>
        <pre>{`// Client-side streaming
const agent = remoteAgent({
  url: '/api/tripPlannerAgent',
});
const chat = agent.chat();

const turn = chat.sendStream(text);
for await (const chunk of turn.stream) {
  if (chunk.text) appendText(chunk.text);
}

const res = await turn.response;`}</pre>
      </aside>
    </div>
  );
}
