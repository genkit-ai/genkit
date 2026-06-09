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
  remoteAgent,
  type AgentChat,
  type AgentInterrupt,
  type AgentResponse,
} from 'genkit/beta/client';
import { useCallback, useMemo, useRef, useState } from 'react';
import { ChatUI, type ChatMessage } from '../components/ChatUI';

// ---------------------------------------------------------------------------
// Banking Interrupt — interrupt/approval workflow
//
// Demonstrates:
//   • The `remoteAgent` client with a server-side session store
//   • Detecting an interrupt: `response.interrupts` is non-empty when the
//     agent pauses for approval (the `userApproval` tool request)
//   • Resuming after interrupt: `chat.resumeStream(interrupt.respond(output))`
//     continues the flow from the exact point it paused
//   • Inline approval dialog
// ---------------------------------------------------------------------------

const ENDPOINT = '/api/bankingAgent';

interface PendingInterrupt {
  interrupt: AgentInterrupt;
  action: string;
  details: string;
}

export default function BankingInterrupt() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streamingText, setStreamingText] = useState('');
  const [loading, setLoading] = useState(false);
  const [interrupt, setInterrupt] = useState<PendingInterrupt | null>(null);
  const [feedback, setFeedback] = useState('');

  // Typed HTTP client. This agent uses a server-side store; the client tracks
  // the snapshot across turns and resumes interrupts for us.
  const agent = useMemo(() => remoteAgent({ url: ENDPOINT }), []);

  // The conversation. Created on first send and reused for follow-up turns
  // (and to resume interrupts).
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
      const chat = chatRef.current;

      try {
        const res = await streamAndCollect(chat.sendStream(text));
        processResult(res);
      } catch (err: any) {
        setStreamingText('');
        setMessages((prev) => [
          ...prev,
          {
            role: 'system',
            text: `⚠️ Turn failed (${err.status ?? 'INTERNAL'}): ${err.message}.`,
          },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [agent, loading, interrupt]
  );

  // ── Respond to an interrupt (approve or deny) ────────────────────────
  const handleInterruptResponse = useCallback(
    async (approved: boolean) => {
      if (!interrupt || !chatRef.current) return;
      const chat = chatRef.current;
      const currentInterrupt = interrupt;
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

      // Build the resume payload from the interrupt itself: `respond` returns a
      // toolResponse part matching the interrupt's ref/name and our output.
      const respond = currentInterrupt.interrupt.respond({
        approved,
        feedback: feedback || undefined,
      });

      try {
        const res = await streamAndCollect(
          chat.resumeStream({ respond: [respond] })
        );
        processResult(res);
      } catch (err: any) {
        setStreamingText('');
        setMessages((prev) => [
          ...prev,
          {
            role: 'system',
            text: `Error resuming (${err.status ?? 'INTERNAL'}): ${err.message}`,
          },
        ]);
      } finally {
        setLoading(false);
        setFeedback('');
      }
    },
    [interrupt, feedback]
  );

  // ── Shared: stream a turn and collect chunks ─────────────────────────
  async function streamAndCollect(
    turn: ReturnType<AgentChat['sendStream']>
  ): Promise<AgentResponse> {
    let accumulated = '';
    for await (const chunk of turn.stream) {
      if (chunk.text) {
        accumulated = chunk.accumulatedText;
        setStreamingText(accumulated);
      }
    }

    const res = await turn.response;
    setStreamingText('');

    // Don't render a model bubble for an interrupted turn — the approval dialog
    // shows the tool-request details instead.
    if (res.interrupts.length === 0) {
      const replyText = res.text;
      if (accumulated || replyText) {
        setMessages((prev) => [
          ...prev,
          { role: 'model', text: replyText || accumulated },
        ]);
      }
    }

    return res;
  }

  // ── Process a result: detect interrupts ──────────────────────────────
  function processResult(res: AgentResponse) {
    // The turn paused for approval — surface the first interrupt's details in
    // the dialog.
    const pending = res.interrupts.find((i) => i.name === 'userApproval');
    if (pending) {
      const input = pending.input as
        | { action?: string; details?: string }
        | undefined;
      setInterrupt({
        interrupt: pending,
        action: input?.action || 'Unknown',
        details: input?.details || '',
      });
    }
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
            <code>chat.sendStream()</code>.
          </li>
          <li>
            The model decides to call the <code>userApproval</code> tool.
            Instead of a final answer, the response carries an entry in{' '}
            <code>response.interrupts</code> with the action details.
          </li>
          <li>
            The client detects the interrupt and shows an inline approval dialog
            — the flow is <strong>paused</strong>.
          </li>
          <li>
            When the user approves or denies, the client calls{' '}
            <code>chat.resumeStream(interrupt.respond(output))</code> to{' '}
            <strong>resume</strong> from the exact point where the flow paused.
          </li>
          <li>
            The model processes the approval result and returns a final
            confirmation or denial message.
          </li>
        </ol>

        <h4>Key APIs</h4>
        <pre>{`// Detect interrupt in the response
const res = await turn.response;
const pending = res.interrupts.find(
  (i) => i.name === 'userApproval'
);
// pending.input → action details

// Resume after approval
const respond = pending.respond({ approved: true });
const turn = chat.resumeStream({ respond: [respond] });`}</pre>

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
