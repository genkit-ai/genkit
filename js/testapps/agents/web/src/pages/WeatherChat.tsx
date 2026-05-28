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

import { useCallback, useEffect, useMemo } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { ChatUI, type ChatMessage } from '../components/ChatUI';
import { useGenkitAgent } from '../genkit-react';

// ---------------------------------------------------------------------------
// Weather Chat — MIGRATED TO v2 HOOKS
//
// Before: 345 LOC managing streamFlow, stateRef, snapshotIdRef, restoration,
// chunk parsing, tool rendering, and URL updates by hand.
//
// After: ~130 LOC of UI. All wire-protocol concerns absorbed by
// `useGenkitAgent`:
//   - continuationId opaque-token round-tripping (no state/snapshotId fork)
//   - URL-bookmark restoration via `resumeFromContinuation`
//   - walkAgentEvent dispatch — text, tool calls, tool responses
//   - committed message history + in-flight streaming text
// ---------------------------------------------------------------------------

const ENDPOINT = '/api/weatherAgent';

export default function WeatherChat() {
  const { snapshotId: urlSnapshotId } = useParams<{ snapshotId: string }>();
  const navigate = useNavigate();

  // Translate a snapshotId from the URL into a v2 continuation token.
  const resumeFromContinuation = urlSnapshotId
    ? `v1:${urlSnapshotId}`
    : undefined;

  const agent = useGenkitAgent({
    url: ENDPOINT,
    resumeFromContinuation,
  });

  // Push the snapshot back to the URL when it changes, so the page is
  // bookmarkable.
  useEffect(() => {
    if (!agent.continuationId) return;
    const sid = agent.continuationId.startsWith('v1:')
      ? agent.continuationId.slice(3)
      : null;
    if (!sid) return;
    if (sid === urlSnapshotId) return;
    navigate(`/weather/${sid}`, { replace: true });
  }, [agent.continuationId, urlSnapshotId, navigate]);

  const handleSend = useCallback(
    (text: string) => {
      if (agent.phase === 'streaming') return;
      agent.submit({ messages: [{ role: 'user', content: [{ text }] }] });
    },
    [agent]
  );

  // Project the hook's reactive state into the ChatUI's message format.
  const chatMessages = useMemo<ChatMessage[]>(() => {
    return agent.messages.flatMap((msg) =>
      messageToChatRows(msg as { role: string; content?: any[] })
    );
  }, [agent.messages]);

  const inFlightRows = useMemo<ChatMessage[]>(() => {
    const rows: ChatMessage[] = [];
    for (const tc of agent.toolCalls) {
      if (tc.state === 'call') {
        rows.push({
          role: 'tool',
          text: `🔧 Calling ${tc.name}(${JSON.stringify(tc.input)})`,
        });
      } else if (tc.state === 'result') {
        rows.push({
          role: 'tool',
          text: `✅ ${tc.name} → ${JSON.stringify(tc.output)}`,
        });
      } else {
        rows.push({ role: 'tool', text: `❌ ${tc.name} failed` });
      }
    }
    return rows;
  }, [agent.toolCalls]);

  return (
    <div className="page-with-sidebar">
      <ChatUI
        title="Weather Agent"
        description="Multi-turn chat with tool-calling. Migrated to useGenkitAgent (v2 events + continuationId)."
        suggestions={[
          'What is the weather like in London?',
          'Is it sunny in Tokyo right now?',
          'Compare the weather in Paris and New York.',
        ]}
        messages={[...chatMessages, ...inFlightRows]}
        streamingText={agent.streamingText}
        loading={agent.phase === 'streaming'}
        onSend={handleSend}
        headerAction={
          agent.continuationId ? (
            <Link to="/weather" className="btn btn-new-session" reloadDocument>
              ✨ New Session
            </Link>
          ) : null
        }
      />

      <aside className="info-sidebar">
        <h3>📋 v2 Migration</h3>
        <ol>
          <li>
            <code>useGenkitAgent</code> owns the streamFlow connection, the
            chunk dispatch, and the continuation-token round-trip.
          </li>
          <li>
            On mount, if a snapshotId is in the URL, the hook calls{' '}
            <code>/state</code> and rehydrates messages + custom state +
            artifacts.
          </li>
          <li>
            On every turn, the hook walks the v2 <code>event</code> field on
            each chunk (via <code>walkAgentEvent</code>) and surfaces{' '}
            <code>streamingText</code>, <code>toolCalls</code>,{' '}
            <code>pendingInterrupt</code>, etc. as reactive React state.
          </li>
          <li>
            The single opaque <code>continuationId</code> replaces the
            state-vs-snapshotId fork from v1.
          </li>
        </ol>

        <h4>Page code</h4>
        <pre>{`const agent = useGenkitAgent({
  url: '/api/weatherAgent',
  resumeFromContinuation,
});

agent.submit({ messages: [...] });
agent.streamingText;       // live tokens
agent.toolCalls;           // tool lifecycle
agent.messages;            // committed
agent.continuationId;      // bookmark this
agent.pendingInterrupt;    // user input needed
agent.respondToInterrupt(out);`}</pre>
      </aside>
    </div>
  );
}

function messageToChatRows(msg: { role: string; content?: any[] }): ChatMessage[] {
  const rows: ChatMessage[] = [];
  const textParts: string[] = [];
  for (const p of msg.content ?? []) {
    if (p.text) textParts.push(p.text);
    if (p.toolRequest) {
      rows.push({
        role: 'tool',
        text: `🔧 ${p.toolRequest.name}(${JSON.stringify(p.toolRequest.input)})`,
      });
    }
    if (p.toolResponse) {
      rows.push({
        role: 'tool',
        text: `✅ ${p.toolResponse.name} → ${JSON.stringify(p.toolResponse.output)}`,
      });
    }
  }
  if (textParts.length > 0) {
    rows.unshift({ role: msg.role as ChatMessage['role'], text: textParts.join('') });
  }
  return rows;
}
