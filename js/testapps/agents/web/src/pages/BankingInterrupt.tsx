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

import {
  AgentError,
  remoteAgent,
  type AgentChat,
  type AgentInterrupt,
  type AgentTurn,
} from 'genkit/beta/client';
import { useCallback, useMemo, useRef, useState } from 'react';
import { ChatUI, type ChatMessage } from '../components/ChatUI';

// ---------------------------------------------------------------------------
// Banking Interrupt — interrupt/approval workflow
//
// Demonstrates the ergonomic `remoteAgent` interrupt API:
//   • chat.send() pauses with `res.interrupts` instead of a final answer
//   • Each interrupt exposes `.name`, `.input`, and a `.respond()` builder
//   • chat.resume({ respond: [...] }) continues the paused flow
//   • Inline approval dialog
// ---------------------------------------------------------------------------

const ENDPOINT = '/api/bankingAgent';

interface PendingInterrupt {
  action: string;
  details: string;
  interrupt: AgentInterrupt<{ action?: string; details?: string }>;
}

export default function BankingInterrupt() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streamingText, setStreamingText] = useState('');
  const [loading, setLoading] = useState(false);
  const [interrupt, setInterrupt] = useState<PendingInterrupt | null>(null);
  const [feedback, setFeedback] = useState('');

  // The agent client and the live chat. The chat tracks the snapshot the flow
  // paused on, so resuming works without manual snapshotId threading.
  const agent = useMemo(() => remoteAgent({ url: ENDPOINT }), []);
  const chatRef = useRef<AgentChat | null>(null);

  // ── Send a regular user message ──────────────────────────────────────
  const handleSend = useCallback(
    async (text: string) => {
      if (loading || interrupt) return;

      setMessages((prev) => [...prev, { role: 'user', text }]);
      setLoading(true);
      setStreamingText('');

      if (!chatRef.current) {
        chatRef.current = agent.chat();
      }

      try {
        await runTurn(chatRef.current.sendStream(text));
      } catch (err: any) {
        handleTurnError(err);
      } finally {
        setLoading(false);
      }
    },
    [loading, interrupt, agent]
  );

  // ── Respond to an interrupt (approve or deny) ────────────────────────
  const handleInterruptResponse = useCallback(
    async (approved: boolean) => {
      if (!interrupt || !chatRef.current) return;
      const current = interrupt;
      setInterrupt(null);
      setLoading(true);
      setStreamingText('');

      setMessages((prev) => [
        ...prev,
        {
          role: 'system',
          text: `User ${approved ? 'approved ✅' : 'denied ❌'}: ${feedback || '(no feedback)'}`,
        },
      ]);

      try {
        // Resume the paused flow with the interrupt's `respond` builder. The
        // chat resumes from the snapshot it's tracking.
        await runTurn(
          chatRef.current.resumeStream({
            respond: [
              current.interrupt.respond({
                approved,
                feedback: feedback || undefined,
              }),
            ],
          })
        );
      } catch (err: any) {
        handleTurnError(err, 'resuming');
      } finally {
        setLoading(false);
        setFeedback('');
      }
    },
    [interrupt, feedback]
  );

  // ── Shared: stream a turn, render text, and detect interrupts ────────
  async function runTurn(turn: AgentTurn): Promise<void> {
    let accumulated = '';
    for await (const chunk of turn.stream) {
      if (chunk.text) {
        accumulated += chunk.text;
        setStreamingText(accumulated);
      }
    }

    const res = await turn.response;
    setStreamingText('');

    // The turn paused for approval — surface the dialog.
    const approval = res.interrupts.find((i) => i.name === 'userApproval') as
      | AgentInterrupt<{ action?: string; details?: string }>
      | undefined;
    if (approval) {
      setInterrupt({
        action: approval.input?.action || 'Unknown',
        details: approval.input?.details || '',
        interrupt: approval,
      });
      return;
    }

    // Otherwise show the model's reply.
    const replyText = res.text || accumulated;
    if (replyText) {
      setMessages((prev) => [...prev, { role: 'model', text: replyText }]);
    }
  }

  // ── Shared: surface a turn-level or transport error ──────────────────
  function handleTurnError(err: any, verb = '') {
    setStreamingText('');
    if (err instanceof AgentError) {
      // A turn (including a resume) can fail gracefully — the chat keeps the
      // last-good snapshot so the user can retry the approval.
      setMessages((prev) => [
        ...prev,
        {
          role: 'system',
          text: `⚠️ Turn failed (${err.status}): ${err.message}.`,
        },
      ]);
      return;
    }
    setMessages((prev) => [
      ...prev,
      {
        role: 'system',
        text: `Error${verb ? ` ${verb}` : ''}: ${err.message}`,
      },
    ]);
  }

  return (
    <div className="page-with-sidebar">
      <ChatUI
        title="Banking Agent (Interrupt)"
        description="Banking assistant that requests user approval before transfers."
        suggestions={[
          'Transfer $500 to my savings account.',
          'Send $200 to account ACME-1234.',
          'What is my account balance?',
        ]}
        messages={messages}
        streamingText={streamingText}
        loading={loading}
        onSend={handleSend}
        inputDisabled={!!interrupt}>
        {/* Inline approval dialog — shown when the agent pauses for approval */}
        {interrupt && (
          <div className="interrupt-dialog">
            <h3>⚠️ Approval Required</h3>
            <p>
              <strong>Action:</strong> {interrupt.action}
            </p>
            <p>
              <strong>Details:</strong> {interrupt.details}
            </p>
            <textarea
              className="interrupt-feedback"
              placeholder="Optional feedback…"
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
              rows={2}
            />
            <div className="interrupt-buttons">
              <button
                className="btn btn-approve"
                onClick={() => handleInterruptResponse(true)}>
                Approve
              </button>
              <button
                className="btn btn-deny"
                onClick={() => handleInterruptResponse(false)}>
                Deny
              </button>
            </div>
          </div>
        )}
      </ChatUI>

      <aside className="info-sidebar">
        <h3>📋 How It Works</h3>
        <ol>
          <li>
            User sends a request like <em>"Transfer $500 to savings"</em> via{' '}
            <code>chat.send()</code>.
          </li>
          <li>
            The model decides to call the <code>userApproval</code> tool.
            Instead of a final answer, <code>res.interrupts</code> contains the
            paused tool request with the action details.
          </li>
          <li>
            The client detects the interrupt and shows an inline approval dialog
            — the flow is <strong>paused</strong>.
          </li>
          <li>
            When the user approves or denies, the client calls{' '}
            <code>chat.resume()</code> with the interrupt's{' '}
            <code>respond()</code> output to <strong>resume</strong> from the
            exact point where the flow paused.
          </li>
          <li>
            The model processes the approval result and returns a final
            confirmation or denial message.
          </li>
        </ol>

        <h4>Key APIs</h4>
        <pre>{`// Detect interrupt in the response
const res = await chat.send(text);
const approval = res.interrupts.find(
  (i) => i.name === 'userApproval'
);

// Resume after approval
await chat.resume({
  respond: [
    approval.respond({ approved: true }),
  ],
});`}</pre>

        <h4>Interrupt Pattern</h4>
        <p>
          The interrupt pattern uses <strong>tool calls as control flow</strong>
          . The <code>userApproval</code> tool never executes server-side — it
          exists solely to pause the flow and hand control back to the client.
          The client's <code>resume</code> payload resumes execution.
        </p>
      </aside>
    </div>
  );
}
