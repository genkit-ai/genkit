/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

import { useMemo } from 'react';
import { ChatUI, type ChatMessage } from '../components/ChatUI';
import { useGenkitAgent } from '../genkit-react';

// Workspace Builder — artifacts demonstration. `agent.artifacts` is a
// reactive array fed both by `artifact-emitted` events during streaming and
// by `result.state.artifacts` at turn end.

const ENDPOINT = '/api/workspaceAgent';

interface ArtifactDisplay {
  name?: string;
  parts: Array<{ text?: string }>;
}

export default function WorkspaceBuilder() {
  const agent = useGenkitAgent({ url: ENDPOINT });

  const handleSend = (text: string) => {
    if (agent.phase === 'streaming') return;
    agent.submit({ messages: [{ role: 'user', content: [{ text }] }] });
  };

  const chatMessages = useMemo<ChatMessage[]>(
    () => agent.messages.flatMap(messageToChatRows),
    [agent.messages]
  );

  const artifacts = agent.artifacts as ArtifactDisplay[];

  return (
    <div className="workspace-layout">
      <ChatUI
        title="Workspace Builder — v2"
        description="Generates code artifacts. The v2 hook surfaces them as a reactive `agent.artifacts` array."
        suggestions={[
          'Write poem.txt with a poem about AI.',
          'Create hello.py with a Python hello world script.',
          'Generate a README.md for a todo app project.',
        ]}
        messages={chatMessages}
        streamingText={agent.streamingText}
        loading={agent.phase === 'streaming'}
        onSend={handleSend}
      />

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
                {a.parts?.filter((p) => p.text).map((p) => p.text).join('\n')}
              </pre>
            </div>
          ))
        )}
      </aside>
    </div>
  );
}

function messageToChatRows(msg: any): ChatMessage[] {
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
    rows.unshift({ role: msg.role, text: textParts.join('') });
  }
  return rows;
}
