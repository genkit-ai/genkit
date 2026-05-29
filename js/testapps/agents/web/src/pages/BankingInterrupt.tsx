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

import { useMemo, useState } from 'react';
import { ChatUI, type ChatMessage } from '../components/ChatUI';
import { useGenkitAgent } from '../genkit-react';

// ---------------------------------------------------------------------------
// Banking Interrupt — MIGRATED TO v2 HOOKS
//
// Before: 358 LOC with stateRef + snapshotIdRef, manual `findInterrupt`
// scanning `result.message.content`, hand-rolled `resume.respond` payload
// construction, and a separate handler path for resumption.
//
// After: the v2 `interrupt` event surfaces through `useGenkitAgent` as
// `pendingInterrupt`. Resuming is one call: `agent.respondToInterrupt(out)`.
// No snapshotId routing, no content scanning, no separate code path.
// ---------------------------------------------------------------------------

const ENDPOINT = '/api/bankingAgent';

interface ApprovalInput {
  action?: string;
  details?: string;
}

export default function BankingInterrupt() {
  const agent = useGenkitAgent({ url: ENDPOINT });
  const [feedback, setFeedback] = useState('');

  const handleSend = (text: string) => {
    if (agent.phase === 'streaming') return;
    agent.submit({ messages: [{ role: 'user', content: [{ text }] }] });
  };

  const handleApprove = () => {
    agent.respondToInterrupt({
      approved: true,
      feedback: feedback || undefined,
    });
    setFeedback('');
  };

  const handleDeny = () => {
    agent.respondToInterrupt({
      approved: false,
      feedback: feedback || undefined,
    });
    setFeedback('');
  };

  const chatMessages = useMemo<ChatMessage[]>(() => {
    return agent.messages.flatMap((m) =>
      messageToChatRows(m as { role: string; content?: any[] })
    );
  }, [agent.messages]);

  const interrupt = agent.pendingInterrupt;
  const interruptInput = (interrupt?.input ?? {}) as ApprovalInput;

  return (
    <div className="page-with-sidebar">
      <ChatUI
        title="Banking Agent (Interrupt) — v2"
        description="Banking assistant that requests user approval before transfers. Now driven by useGenkitAgent + in-stream interrupt events."
        suggestions={[
          'Transfer $500 to my savings account.',
          'Send $200 to account ACME-1234.',
          'What is my account balance?',
        ]}
        messages={chatMessages}
        streamingText={agent.streamingText}
        loading={agent.phase === 'streaming'}
        onSend={handleSend}
        inputDisabled={!!interrupt}>
        {interrupt && interrupt.toolName === 'userApproval' && (
          <div className="interrupt-dialog">
            <h3>⚠️ Approval Required</h3>
            <p>
              <strong>Action:</strong> {interruptInput.action ?? 'Unknown'}
            </p>
            <p>
              <strong>Details:</strong> {interruptInput.details ?? '(none)'}
            </p>
            <textarea
              className="interrupt-feedback"
              placeholder="Optional feedback…"
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
              rows={2}
            />
            <div className="interrupt-buttons">
              <button className="btn btn-approve" onClick={handleApprove}>
                Approve
              </button>
              <button className="btn btn-deny" onClick={handleDeny}>
                Deny
              </button>
            </div>
          </div>
        )}
        {agent.error && (
          <div className="message">
            <div className="message-role">system</div>
            <div className="message-text">Error: {agent.error.message}</div>
          </div>
        )}
      </ChatUI>

      <aside className="info-sidebar">
        <h3>📋 v2 Interrupt Pattern</h3>
        <ol>
          <li>
            User asks: "Transfer $500 to savings". The agent emits an{' '}
            <code>interrupt</code> event on the stream when the{' '}
            <code>userApproval</code> tool fires.
          </li>
          <li>
            <code>useGenkitAgent</code> surfaces it as{' '}
            <code>pendingInterrupt</code> with addressable{' '}
            <code>toolCallId</code>, <code>toolName</code>, and{' '}
            <code>input</code>.
          </li>
          <li>
            UI shows the approval dialog. User clicks Approve →{' '}
            <code>agent.respondToInterrupt({'{ approved: true }'})</code>.
          </li>
          <li>
            The hook constructs the <code>resume.respond</code> payload with the
            right ref, sends it back to the same agent URL, and continues
            streaming the result.
          </li>
        </ol>

        <h4>Page code</h4>
        <pre>{`const agent = useGenkitAgent({ url });

// Detection
agent.pendingInterrupt
  // { toolCallId, toolName, input, kind }

// Resumption
agent.respondToInterrupt(output);  // for defineInterrupt tools
agent.restartInterrupt(metadata);  // for approval-gated tools

// No snapshotId routing.
// No findInterrupt() scanning.
// No separate resume code path.`}</pre>
      </aside>
    </div>
  );
}

function messageToChatRows(msg: {
  role: string;
  content?: any[];
}): ChatMessage[] {
  const rows: ChatMessage[] = [];
  const textParts: string[] = [];
  for (const p of msg.content ?? []) {
    if (p.text) textParts.push(p.text);
    if (p.toolRequest && p.toolRequest.name !== 'userApproval') {
      rows.push({
        role: 'tool',
        text: `🔧 ${p.toolRequest.name}(${JSON.stringify(p.toolRequest.input)})`,
      });
    }
    if (p.toolResponse && p.toolResponse.name !== 'userApproval') {
      rows.push({
        role: 'tool',
        text: `✅ ${p.toolResponse.name} → ${JSON.stringify(p.toolResponse.output)}`,
      });
    }
  }
  if (textParts.length > 0) {
    rows.unshift({
      role: msg.role as ChatMessage['role'],
      text: textParts.join(''),
    });
  }
  return rows;
}
