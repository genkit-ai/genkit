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

// Sub-agent orchestration. The `call_agent` tool calls render with special
// labels so the user can see delegation in flight.

const ENDPOINT = '/api/orchestratorAgent';

export default function SubAgentChat() {
  const agent = useGenkitAgent({ url: ENDPOINT });

  const handleSend = (text: string) => {
    if (agent.phase === 'streaming') return;
    agent.submit({ messages: [{ role: 'user', content: [{ text }] }] });
  };

  const chatMessages = useMemo<ChatMessage[]>(
    () => agent.messages.flatMap(messageToChatRows),
    [agent.messages]
  );

  // In-flight tool calls also get rendered with delegation labels.
  const inFlight = useMemo<ChatMessage[]>(() => {
    return agent.toolCalls.map((tc) => {
      const inp = tc.input as { agent?: string; task?: string } | undefined;
      const out = tc.output as
        | { response?: string; agentName?: string }
        | undefined;
      if (tc.name === 'call_agent') {
        if (tc.state === 'call' && inp?.agent) {
          return {
            role: 'tool' as const,
            text: `🤝 Delegating to "${inp.agent}": ${inp.task ?? ''}`,
          };
        }
        if (tc.state === 'result' && out?.agentName) {
          return {
            role: 'tool' as const,
            text: `✅ "${out.agentName}" responded: ${out.response ?? ''}`,
          };
        }
      }
      return tc.state === 'result'
        ? {
            role: 'tool' as const,
            text: `✅ ${tc.name} → ${JSON.stringify(tc.output)}`,
          }
        : {
            role: 'tool' as const,
            text: `🔧 ${tc.name}(${JSON.stringify(tc.input)})`,
          };
    });
  }, [agent.toolCalls]);

  return (
    <div className="page-with-sidebar">
      <ChatUI
        title="Sub-Agent Orchestrator — v2"
        description="Orchestrator delegating to researcher and coder sub-agents."
        suggestions={[
          'Research the history of the Internet.',
          'Write a Python function to calculate Fibonacci numbers.',
          'Explain quantum computing in simple terms.',
        ]}
        messages={[...chatMessages, ...inFlight]}
        streamingText={agent.streamingText}
        loading={agent.phase === 'streaming'}
        onSend={handleSend}
        renderMarkdown
      />

      <aside className="info-sidebar">
        <h3>🤝 v2 Sub-Agent Delegation</h3>
        <p>
          The orchestrator's <code>call_agent</code> tool surfaces as a regular
          tool request through the v2 event stream. The hook exposes it via{' '}
          <code>agent.toolCalls</code>; this page formats it with delegation
          labels for the UI.
        </p>
        <p>
          A future iteration would emit a distinct <code>sub-agent-start</code>{' '}
          / <code>sub-agent-event</code> / <code>sub-agent-end</code> event type
          so sub-agent activity could be rendered with full nesting (status
          updates, intermediate artifacts).
        </p>
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
      const inp = p.toolRequest.input as
        | { agent?: string; task?: string }
        | undefined;
      const label =
        p.toolRequest.name === 'call_agent' && inp?.agent
          ? `🤝 Delegated to "${inp.agent}": ${inp.task ?? ''}`
          : `🔧 ${p.toolRequest.name}(${JSON.stringify(p.toolRequest.input)})`;
      rows.push({ role: 'tool', text: label });
    }
    if (p.toolResponse) {
      const out = p.toolResponse.output as
        | { response?: string; agentName?: string }
        | undefined;
      const label =
        p.toolResponse.name === 'call_agent' && out?.agentName
          ? `✅ "${out.agentName}" responded: ${out.response ?? ''}`
          : `✅ ${p.toolResponse.name} → ${JSON.stringify(p.toolResponse.output)}`;
      rows.push({ role: 'tool', text: label });
    }
  }
  if (textParts.length > 0) {
    rows.unshift({ role: msg.role, text: textParts.join('') });
  }
  return rows;
}
