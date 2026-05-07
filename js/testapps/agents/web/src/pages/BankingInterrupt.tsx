import type {
  AgentInit,
  AgentInput,
  AgentOutput,
  AgentStreamChunk,
  ToolRequest,
} from 'genkit/beta';
import { streamFlow } from 'genkit/beta/client';
import { useCallback, useRef, useState } from 'react';
import { ChatUI, type Message } from '../components/ChatUI';

// ---------------------------------------------------------------------------
// Banking Interrupt — interrupt/approval workflow
//
// Demonstrates:
//   • streamFlow() with server-side session store (uses snapshotId)
//   • Detecting an interrupt: the result contains a toolRequest for
//     'userApproval' instead of a final answer
//   • Resuming after interrupt: send a toolResponse message with
//     `init: { snapshotId }` to continue the flow
//   • Inline approval dialog
// ---------------------------------------------------------------------------

const ENDPOINT = '/api/bankingAgent';

interface PendingInterrupt {
  ref?: string;
  action: string;
  details: string;
  snapshotId: string;
}

export default function BankingInterrupt() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [streamingText, setStreamingText] = useState('');
  const [loading, setLoading] = useState(false);
  const [interrupt, setInterrupt] = useState<PendingInterrupt | null>(null);
  const [feedback, setFeedback] = useState('');

  // This agent uses a server-side store, so we track state for stateless
  // fallback AND snapshotId for interrupt resumption.
  const stateRef = useRef<any>(undefined);
  const snapshotIdRef = useRef<string | undefined>(undefined);

  // ── Send a regular user message ──────────────────────────────────────
  const handleSend = useCallback(
    async (text: string) => {
      if (loading || interrupt) return;

      setMessages((prev) => [...prev, { role: 'user', text }]);
      setLoading(true);
      setStreamingText('');

      const input: AgentInput = {
        messages: [{ role: 'user', content: [{ text }] }],
      };

      // Prefer state over snapshotId for multi-turn.
      const init: AgentInit = stateRef.current
        ? { state: stateRef.current }
        : snapshotIdRef.current
          ? { snapshotId: snapshotIdRef.current }
          : {};

      try {
        const result = await streamAndCollect(input, init);
        processResult(result);
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
    [loading, interrupt]
  );

  // ── Respond to an interrupt (approve or deny) ────────────────────────
  const handleInterruptResponse = useCallback(
    async (approved: boolean) => {
      if (!interrupt) return;
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

      // ── Build the tool response message ──────────────────────────────
      // To resume an interrupted flow, send a message with role 'tool'
      // containing a toolResponse that matches the interrupt's ref.
      const input: AgentInput = {
        messages: [
          {
            role: 'tool',
            content: [
              {
                toolResponse: {
                  name: 'userApproval',
                  ref: currentInterrupt.ref,
                  output: {
                    approved,
                    feedback: feedback || undefined,
                  },
                },
              },
            ],
          },
        ],
      };

      // Resume from the snapshot where the flow was interrupted.
      // The banking agent uses a server-side store, so snapshotId works.
      const init = { snapshotId: currentInterrupt.snapshotId };

      try {
        const result = await streamAndCollect(input, init);
        processResult(result);
      } catch (err: any) {
        setStreamingText('');
        setMessages((prev) => [
          ...prev,
          { role: 'system', text: `Error resuming: ${err.message}` },
        ]);
      } finally {
        setLoading(false);
        setFeedback('');
      }
    },
    [interrupt, feedback]
  );

  // ── Shared: stream a request and collect chunks ──────────────────────
  async function streamAndCollect(
    input: AgentInput,
    init: AgentInit
  ): Promise<AgentOutput> {
    const response = streamFlow<AgentOutput, AgentStreamChunk, AgentInit>({
      url: ENDPOINT,
      input,
      init,
    });

    let accumulated = '';
    for await (const chunk of response.stream) {
      if (chunk?.modelChunk?.content) {
        for (const part of chunk.modelChunk.content) {
          if (part.text) {
            accumulated += part.text;
            setStreamingText(accumulated);
          }
        }
      }
    }

    const result = await response.output;
    setStreamingText('');

    // Show the model's reply (or the accumulated stream text).
    const replyText = extractText(result);
    if (accumulated || replyText !== '(no result)') {
      setMessages((prev) => [
        ...prev,
        { role: 'model', text: replyText || accumulated },
      ]);
    }

    return result;
  }

  // ── Process a result: update session tracking & detect interrupts ────
  function processResult(result: AgentOutput) {
    if (result?.state) stateRef.current = result.state;
    if (result?.snapshotId) snapshotIdRef.current = result.snapshotId;

    // Check if the result contains an interrupt (userApproval tool request).
    const irpt = findInterrupt(result);
    if (irpt && result.snapshotId) {
      setInterrupt({ ...irpt, snapshotId: result.snapshotId });
    }
  }

  return (
    <div className="page-with-sidebar">
      <ChatUI
        title="Banking Agent (Interrupt)"
        description="Banking assistant that requests user approval before transfers. Try: 'Transfer $500 to savings'"
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
            <code>streamFlow()</code>.
          </li>
          <li>
            The model decides to call the <code>userApproval</code> tool.
            Instead of a final answer, the result contains a{' '}
            <code>toolRequest</code> with the action details.
          </li>
          <li>
            The client detects the <code>toolRequest</code> and shows an inline
            approval dialog — the flow is <strong>paused</strong>.
          </li>
          <li>
            When the user approves or denies, the client sends a{' '}
            <code>toolResponse</code> message with{' '}
            <code>{'init: { snapshotId }'}</code> to <strong>resume</strong>{' '}
            from the exact point where the flow paused.
          </li>
          <li>
            The model processes the approval result and returns a final
            confirmation or denial message.
          </li>
        </ol>

        <h4>Key APIs</h4>
        <pre>{`// Detect interrupt in result
const msg = result.message;
for (const p of msg.content) {
  if (p.toolRequest?.name === 'userApproval') {
    // Show approval dialog
    // Save result.snapshotId
  }
}

// Resume after approval
streamFlow({
  url: '/api/bankingAgent',
  input: {
    messages: [{
      role: 'tool',
      content: [{
        toolResponse: {
          name: 'userApproval',
          ref: interrupt.ref,
          output: { approved: true },
        },
      }],
    }],
  },
  init: { snapshotId },
});`}</pre>

        <h4>Interrupt Pattern</h4>
        <p>
          The interrupt pattern uses <strong>tool calls as control flow</strong>
          . The <code>userApproval</code> tool never executes server-side — it
          exists solely to pause the flow and hand control back to the client.
          The client's <code>toolResponse</code> resumes execution.
        </p>
      </aside>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function extractText(result: AgentOutput): string {
  if (!result) return '(no result)';
  const msg = result.message;
  if (!msg) return JSON.stringify(result, null, 2);
  const parts: string[] = [];
  for (const p of msg.content || []) {
    if (p.text) parts.push(p.text);
    if (p.toolRequest) {
      parts.push(
        `[Tool Request: ${p.toolRequest.name}]\n${JSON.stringify(p.toolRequest.input, null, 2)}`
      );
    }
  }
  return parts.join('') || JSON.stringify(result, null, 2);
}

/** Check if the result contains an interrupt (userApproval tool request). */
function findInterrupt(
  result: AgentOutput
): { ref?: string; action: string; details: string } | null {
  const msg = result?.message;
  if (!msg) return null;
  for (const p of msg.content || []) {
    if (p.toolRequest?.name === 'userApproval') {
      const tr = p.toolRequest as ToolRequest;
      return {
        ref: tr.ref,
        action: (tr.input as any)?.action || 'Unknown',
        details: (tr.input as any)?.details || '',
      };
    }
  }
  return null;
}
