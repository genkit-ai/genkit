/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

import { useEffect, useMemo } from 'react';
import { Link, useNavigate, useParams } from 'react-router-dom';
import { ChatUI, type ChatMessage } from '../components/ChatUI';
import { useGenkitAgent } from '../genkit-react';

// Trip Planner — definePromptAgent backed by a .prompt file. Migration is
// identical to WeatherChat; the only difference is the URL.

const ENDPOINT = '/api/tripPlannerAgent';

export default function TripPlanner() {
  const { snapshotId: urlSnapshotId } = useParams<{ snapshotId: string }>();
  const navigate = useNavigate();

  const resumeFromContinuation = urlSnapshotId ? `v1:${urlSnapshotId}` : undefined;
  const agent = useGenkitAgent({ url: ENDPOINT, resumeFromContinuation });

  useEffect(() => {
    if (!agent.continuationId) return;
    const sid = agent.continuationId.startsWith('v1:')
      ? agent.continuationId.slice(3)
      : null;
    if (!sid || sid === urlSnapshotId) return;
    navigate(`/trip-planner/${sid}`, { replace: true });
  }, [agent.continuationId, urlSnapshotId, navigate]);

  const handleSend = (text: string) => {
    if (agent.phase === 'streaming') return;
    agent.submit({ messages: [{ role: 'user', content: [{ text }] }] });
  };

  const chatMessages = useMemo<ChatMessage[]>(
    () => agent.messages.flatMap(messageToChatRows),
    [agent.messages]
  );

  return (
    <div className="page-with-sidebar">
      <ChatUI
        title="Trip Planner — v2"
        description="Multi-turn travel assistant powered by a .prompt file and definePromptAgent."
        suggestions={[
          'I want to plan a trip to Paris. What should I see there?',
          'Find me flights from New York to Tokyo.',
          'What are the top attractions in London?',
        ]}
        messages={chatMessages}
        streamingText={agent.streamingText}
        loading={agent.phase === 'streaming'}
        onSend={handleSend}
        headerAction={
          agent.continuationId ? (
            <Link to="/trip-planner" className="btn btn-new-session" reloadDocument>
              ✨ New Session
            </Link>
          ) : null
        }
      />

      <aside className="info-sidebar">
        <h3>📋 definePromptAgent + v2 hooks</h3>
        <p>
          The agent's prompt lives in <code>prompts/tripPlanner.prompt</code>.
          The client is unchanged from the inline-prompt case — same{' '}
          <code>useGenkitAgent</code> hook, same continuationId round-trip.
        </p>
        <h4>Page code</h4>
        <pre>{`const agent = useGenkitAgent({
  url: '/api/tripPlannerAgent',
  resumeFromContinuation,
});

agent.submit({ messages: [...] });`}</pre>
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
