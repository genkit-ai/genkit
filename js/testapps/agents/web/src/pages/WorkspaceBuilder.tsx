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
//   • The `remoteAgent` client with artifact production
//   • Reading streamed `chunk.artifact` and the final `chat.artifacts`
//   • Multi-turn session — the client threads state across turns for us
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

  // Typed HTTP client. Tracks session state across turns for us.
  const agent = useMemo(() => remoteAgent({ url: ENDPOINT }), []);

  // The conversation. Created on first send and reused for follow-up turns.
  const chatRef = useRef<AgentChat | null>(null);

  const handleSend = useCallback(
    async (text: string) => {
      if (loading) return;

      setMessages((prev) => [...prev, { role: 'user', text }]);
      setLoading(true);
      setStreamingText('');

      if (!chatRef.current) {
        chatRef.current = agent.chat();
      }
      const chat = chatRef.current;

      try {
        const turn = chat.sendStream(text);

        let accumulated = '';
        for await (const chunk of turn.stream) {
          // ── Artifacts stream in as they're emitted ──
          if (chunk.artifact) {
            setArtifacts([...chat.artifacts] as Artifact[]);
          }
          // ── Model text chunks ──
          if (chunk.text) {
            accumulated = chunk.accumulatedText;
            setStreamingText(accumulated);
          }
        }

        const res = await turn.response;
        setStreamingText('');

        // Reflect the authoritative final artifact set.
        setArtifacts([...res.artifacts] as Artifact[]);

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
    [agent, loading]
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
