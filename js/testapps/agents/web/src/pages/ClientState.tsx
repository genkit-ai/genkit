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

// Client-managed state — same as WeatherChat but the server has no store.
// The structured `continuation` field carries the full state inline as
// `{ kind: 'state', state: ... }` on every turn. The client round-trips
// the same object back — no encoding, no decoding, no `kind` to guess.

const ENDPOINT = '/api/weatherAgentStateless';

export default function ClientState() {
  const agent = useGenkitAgent({ url: ENDPOINT });

  const handleSend = (text: string) => {
    if (agent.phase === 'streaming') return;
    agent.submit({ messages: [{ role: 'user', content: [{ text }] }] });
  };

  const chatMessages = useMemo<ChatMessage[]>(
    () => agent.messages.flatMap(messageToChatRows),
    [agent.messages]
  );

  // Show the structured continuation as-is. No decoding step.
  const stateDisplay = agent.continuation
    ? agent.continuation.kind === 'state'
      ? JSON.stringify(agent.continuation.state, null, 2)
      : '(server-stored agent — no state blob)'
    : '(no state yet — first turn will create it)';

  return (
    <div className="client-state-layout">
      <ChatUI
        title="Client-Managed Weather Chat — v2"
        description="Stateless agent. The structured continuation carries the full state inline as `{ kind: 'state', state: ... }`. Round-trip is one object."
        suggestions={[
          'What is the weather like in London?',
          'Is it sunny in Tokyo right now?',
        ]}
        messages={chatMessages}
        streamingText={agent.streamingText}
        loading={agent.phase === 'streaming'}
        onSend={handleSend}
      />
      <aside className="state-inspector">
        <h3>📦 continuation (client-owned)</h3>
        <p className="state-inspector-hint">
          One structured object round-trips on every turn via{' '}
          <code>init.continuation</code>. For this agent (no server store),
          it's <code>{`{ kind: 'state', state: {...} }`}</code> with the full
          session inline. For server-stored agents, it's{' '}
          <code>{`{ kind: 'snapshot', snapshotId }`}</code>. Clients don't have
          to know which.
        </p>
        <pre className="state-inspector-json">{stateDisplay}</pre>
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

