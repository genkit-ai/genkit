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

// Research Agent — defineCustomAgent multi-step orchestration. v2 typed
// status/progress events drive the status bar; `agent.customState` carries
// the structured research state for the sidebar.

const ENDPOINT = '/api/researchAgent';

interface SubAnswer {
  question: string;
  answer: string;
}
interface ResearchState {
  subQuestions: string[];
  subAnswers: SubAnswer[];
}

export default function ResearchAgent() {
  const agent = useGenkitAgent<ResearchState>({ url: ENDPOINT });

  const handleSend = (text: string) => {
    if (agent.phase === 'streaming') return;
    agent.submit({ messages: [{ role: 'user', content: [{ text }] }] });
  };

  const chatMessages = useMemo<ChatMessage[]>(
    () => agent.messages.flatMap(messageToChatRows),
    [agent.messages]
  );

  const research = agent.customState;
  const statusLine = agent.progress
    ? `${agent.progress.label ?? 'Working'} (${agent.progress.current}/${agent.progress.total})`
    : agent.statusLabel;

  return (
    <div className="research-layout">
      <ChatUI
        title="Research Agent — v2"
        description="Multi-step research via defineCustomAgent. Status/progress events drive the indicator below."
        suggestions={[
          'What are the impacts of AI on education?',
          'Compare solar and wind energy.',
          'Explain the pros and cons of remote work.',
        ]}
        messages={chatMessages}
        streamingText={agent.streamingText}
        loading={agent.phase === 'streaming'}
        onSend={handleSend}
        renderMarkdown>
        {statusLine && (
          <div className="research-status-bar">
            <span className="research-status-dot" />
            {statusLine}
          </div>
        )}
      </ChatUI>

      <aside className="research-sidebar">
        <h3>🔬 Research Process</h3>
        <p className="research-sidebar-hint">
          Custom-agent flow: decompose → research → synthesize. Typed status
          events (<code>{`{ type: 'progress', current, total, label }`}</code>)
          drive the indicator above.
        </p>

        {!research ? (
          <div className="research-empty">
            Ask a question to see the research process unfold.
          </div>
        ) : (
          <>
            <div className="research-section">
              <h4>📋 Sub-Questions</h4>
              <ol className="research-questions">
                {research.subQuestions.map((q, i) => (
                  <li key={i} className="research-question">
                    {q}
                  </li>
                ))}
              </ol>
            </div>
            {research.subAnswers.length > 0 && (
              <div className="research-section">
                <h4>📝 Research Findings</h4>
                {research.subAnswers.map((sa, i) => (
                  <details key={i} className="research-answer" open={i === 0}>
                    <summary className="research-answer-q">
                      {i + 1}. {sa.question}
                    </summary>
                    <div className="research-answer-text">{sa.answer}</div>
                  </details>
                ))}
              </div>
            )}
          </>
        )}

        <hr className="research-divider" />
        <h4>Server emission</h4>
        <pre className="research-code">{`// Typed status events — first-class contract
sendChunk({ type: 'status', label: 'Decomposing…' });
sendChunk({
  type: 'progress',
  label: 'Researching',
  current: i + 1, total: n,
});
sendChunk({ type: 'model-chunk', chunk });`}</pre>
      </aside>
    </div>
  );
}

function messageToChatRows(msg: any): ChatMessage[] {
  const rows: ChatMessage[] = [];
  const textParts: string[] = [];
  for (const p of msg.content ?? []) {
    if (p.text) textParts.push(p.text);
  }
  if (textParts.length > 0) {
    rows.push({ role: msg.role, text: textParts.join('') });
  }
  return rows;
}
