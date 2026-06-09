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
// Workspace Builder — artifacts alongside chat
//
// Demonstrates:
//   • chat.send() with artifact production
//   • Reading `res.artifacts` (or `chat.artifacts`) from the agent response
//   • Multi-turn session — the chat tracks state automatically
//   • Displaying generated code artifacts in a side panel
// ---------------------------------------------------------------------------

const ENDPOINT = '/api/workspaceAgent';

interface Artifact {
  name?: string;
  parts: Array<{ text?: string }>;
}

export default function WorkspaceBuilder() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streamingText, setStreamingText] = useState('');
  const [loading, setLoading] = useState(false);
  const [artifacts, setArtifacts] = useState<Artifact[]>([]);

  // The agent client and the live chat that tracks state/artifacts.
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
        }

        // ── Read the final result ──────────────────────────────────────
        const res = await turn.response;
        setStreamingText('');

        // Update artifacts — the workspace agent emits artifacts whenever the
        // emitArtifact tool was called. The chat aggregates them across turns.
        if (res.artifacts?.length) {
          setArtifacts(res.artifacts as Artifact[]);
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
    [loading, agent]
  );

  return (
    <div className="workspace-layout">
      <ChatUI
        title="Workspace Builder"
        description="Generates code artifacts via an emitArtifact tool."
        suggestions={[
          'Write poem.txt with a poem about AI.',
          'Create hello.py with a Python hello world script.',
          'Generate a README.md for a todo app project.',
        ]}
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
