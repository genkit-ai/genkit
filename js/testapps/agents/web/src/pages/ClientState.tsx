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
// With v2 continuationId, the client just round-trips one opaque token;
// it doesn't have to know whether the server is stored or stateless.

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

  // For the state inspector: decode the continuation token to show the
  // raw client-side state blob being round-tripped.
  const stateDisplay = agent.continuationId
    ? decodeStateBlob(agent.continuationId) ?? '(server-stored agent — no state blob)'
    : '(no state yet — first turn will create it)';

  return (
    <div className="client-state-layout">
      <ChatUI
        title="Client-Managed Weather Chat — v2"
        description="Stateless agent. The v2 continuationId encodes the state blob client-side. Round-trip is one opaque string."
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
        <h3>📦 continuationId (client-owned)</h3>
        <p className="state-inspector-hint">
          One opaque token round-trips on every turn via{' '}
          <code>init.continuationId</code>. For this agent (no server store),
          it's a base64-encoded state blob (<code>v1s:...</code> prefix). For
          server-stored agents, the same field holds a snapshotId (
          <code>v1:...</code>). Clients don't have to know which.
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

function decodeStateBlob(continuationId: string): string | null {
  if (!continuationId.startsWith('v1s:')) return null;
  try {
    const b64 = continuationId.slice(4);
    const json = atob(b64);
    return JSON.stringify(JSON.parse(json), null, 2);
  } catch {
    return null;
  }
}
