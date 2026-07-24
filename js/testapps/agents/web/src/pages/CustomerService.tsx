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

import { remoteAgent, type AgentChat } from 'genkit/beta/client';
import { useCallback, useMemo, useRef, useState } from 'react';
import { ChatUI, type ChatMessage } from '../components/ChatUI';

// ---------------------------------------------------------------------------
// Customer Service Handoff — a single agent that hosts multiple personas
//
// Demonstrates:
//   • The `handoff` middleware for agent transfer / handoff
//   • The `remoteAgent` client for streaming responses
//   • Inline rendering of `transfer_to_<persona>` tool calls so you can watch
//     control move between the triage, refund, and billing personas
//   • Multi-turn session — the active persona persists across turns
// ---------------------------------------------------------------------------

const ENDPOINT = '/api/customerServiceAgent';

// Maps a transfer tool name (e.g. `transfer_to_refund`) to its persona.
function personaFromToolName(name: string): string | undefined {
  const m = /^transfer_to_(.+)$/.exec(name);
  return m?.[1];
}

export default function CustomerService() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streamingText, setStreamingText] = useState('');
  const [loading, setLoading] = useState(false);
  const [activePersona, setActivePersona] = useState('triage');

  // Typed HTTP client. Tracks session state across turns for us.
  const agent = useMemo(() => remoteAgent({ url: ENDPOINT }), []);

  // The conversation. Created on first send and reused for follow-up turns.
  const chatRef = useRef<AgentChat | null>(null);

  const handleSend = useCallback(
    async (text: string) => {
      if (loading) return;

      setMessages((prev) => [...prev, { role: 'user', text }]);
      setLoading(true);
      setStreamingText('');

      if (!chatRef.current) {
        chatRef.current = agent.chat();
      }
      const chat = chatRef.current;

      try {
        const turn = chat.sendStream(text);

        let accumulated = '';
        for await (const chunk of turn.stream) {
          for (const part of chunk.raw.modelChunk?.content ?? []) {
            if (part.toolRequest) {
              const tr = part.toolRequest;
              const persona = personaFromToolName(tr.name);
              if (persona) {
                const inp = tr.input as { reason?: string } | undefined;
                const label = inp?.reason
                  ? `🔀 Transferring to "${persona}": ${inp.reason}`
                  : `🔀 Transferring to "${persona}"`;
                setMessages((prev) => [...prev, { role: 'tool', text: label }]);
              } else {
                setMessages((prev) => [
                  ...prev,
                  {
                    role: 'tool',
                    text: `🔧 ${tr.name}(${JSON.stringify(tr.input)})`,
                  },
                ]);
              }
            } else if (part.toolResponse) {
              const tr = part.toolResponse;
              const persona = personaFromToolName(tr.name);
              if (persona) {
                const out = tr.output as
                  | { transferred?: boolean; message?: string }
                  | undefined;
                if (out?.transferred) {
                  setActivePersona(persona);
                }
                setMessages((prev) => [
                  ...prev,
                  {
                    role: 'tool',
                    text: out?.transferred
                      ? `✅ Now talking to the "${persona}" specialist.`
                      : `⚠️ ${out?.message ?? 'Transfer declined.'}`,
                  },
                ]);
              } else {
                setMessages((prev) => [
                  ...prev,
                  {
                    role: 'tool',
                    text: `✅ ${tr.name} → ${JSON.stringify(tr.output)}`,
                  },
                ]);
              }
            }
          }

          if (chunk.text) {
            accumulated = chunk.accumulatedText;
            setStreamingText(accumulated);
          }
        }

        const res = await turn.response;
        setStreamingText('');

        setMessages((prev) => [
          ...prev,
          { role: 'model', text: res.text || accumulated },
        ]);
      } catch (err: unknown) {
        setStreamingText('');
        const errMsg = err instanceof Error ? err.message : String(err);
        setMessages((prev) => [
          ...prev,
          { role: 'system', text: `Error: ${errMsg}` },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [agent, loading]
  );

  return (
    <div className="page-with-sidebar">
      <ChatUI
        title="Customer Service"
        description={`Acme Corp support — currently handled by the "${activePersona}" persona. Control transfers between personas as your needs change.`}
        suggestions={[
          'I was charged twice for order #1234 and I would like a refund.',
          'Why is my latest invoice so high?',
          'Hi, I have a question about my account.',
        ]}
        messages={messages}
        streamingText={streamingText}
        loading={loading}
        onSend={handleSend}
        renderMarkdown
      />

      <aside className="info-sidebar">
        <h3>🔀 How It Works</h3>
        <ol>
          <li>
            A single <strong>customerServiceAgent</strong> hosts three{' '}
            <strong>personas</strong> via the <code>handoff</code> middleware:{' '}
            <strong>triage</strong>, <strong>refund</strong>, and{' '}
            <strong>billing</strong>.
          </li>
          <li>
            Each persona has its own <em>system prompt</em> and <em>toolset</em>.
            Only the active persona's tools are visible to the model.
          </li>
          <li>
            The middleware injects a <code>transfer_to_&lt;persona&gt;</code>{' '}
            tool. When the model calls it, the active system prompt and visible
            tools are swapped for all subsequent turns.
          </li>
          <li>
            Unlike delegation (<code>agents</code> middleware), control{' '}
            <strong>transfers</strong> — the user keeps talking directly to the
            specialist, who can transfer back or to another persona.
          </li>
        </ol>

        <h4>Personas</h4>
        <ul>
          <li>
            <strong>triage</strong> — greets, diagnoses, and routes (no tools).
          </li>
          <li>
            <strong>refund</strong> — <code>lookupOrder</code>,{' '}
            <code>issueRefund</code>.
          </li>
          <li>
            <strong>billing</strong> — <code>getInvoice</code>.
          </li>
        </ul>

        <h4>Try It</h4>
        <ul>
          <li>
            <em>"I want a refund for order #1234"</em> → triage transfers to
            refund
          </li>
          <li>
            <em>"Why is my invoice so high?"</em> → triage transfers to billing
          </li>
          <li>
            Ask a billing question mid-refund → refund transfers back to triage,
            which routes to billing
          </li>
        </ul>

        <h4>Key APIs</h4>
        <pre>{`const agent = ai.defineAgent({
  name: 'customerServiceAgent',
  model: '...',
  system: 'Shared brand voice…',
  use: [
    handoff({
      personas: [
        { name: 'triage', system: '…' },
        {
          name: 'refund',
          system: '…',
          tools: ['lookupOrder', 'issueRefund'],
        },
        {
          name: 'billing',
          system: '…',
          tools: ['getInvoice'],
        },
      ],
      defaultPersona: 'triage',
      maxTransfers: 5,
    }),
  ],
});`}</pre>

        <h4>Architecture</h4>
        <p>
          The <code>handoff</code> middleware from{' '}
          <code>@genkit-ai/middleware</code> runs on every turn of the tool
          loop. It tracks the active persona by scanning the message history for
          the most recent successful transfer, then layers that persona's system
          prompt onto the agent's shared prompt and gates the visible tools by
          name.
        </p>
      </aside>
    </div>
  );
}
