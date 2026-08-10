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
// Sub-Agent Delegation — orchestrator delegates to researcher & coder
//
// Demonstrates:
//   • The `agents` middleware for sub-agent delegation
//   • The `remoteAgent` client for streaming orchestrator responses
//   • Inline rendering of `call_agent` tool calls showing delegation
//   • Multi-turn session — the client threads state across turns for us
//   • Markdown rendering for code-heavy responses
// ---------------------------------------------------------------------------

const ENDPOINT = '/api/orchestratorAgent';

export default function SubAgentChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streamingText, setStreamingText] = useState('');
  const [loading, setLoading] = useState(false);

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
              // Format call_agent delegations nicely
              const inp = tr.input as
                | { agent?: string; task?: string }
                | undefined;
              const label =
                tr.name === 'call_agent' && inp?.agent
                  ? `🤝 Delegating to "${inp.agent}": ${inp.task ?? ''}`
                  : `🔧 ${tr.name}(${JSON.stringify(tr.input)})`;
              setMessages((prev) => [...prev, { role: 'tool', text: label }]);
            } else if (part.toolResponse) {
              const tr = part.toolResponse;
              // Format call_agent responses nicely
              const out = tr.output as
                | { response?: string; agentName?: string }
                | undefined;
              const label =
                tr.name === 'call_agent' && out?.agentName
                  ? `✅ "${out.agentName}" responded: ${out.response ?? ''}`
                  : `✅ ${tr.name} → ${JSON.stringify(tr.output)}`;
              setMessages((prev) => [...prev, { role: 'tool', text: label }]);
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
        title="Sub-Agent Orchestrator"
        description="An orchestrator that delegates research tasks to a researcher agent and coding tasks to a coder agent."
        suggestions={[
          'Research the history of the Internet.',
          'Write a Python function to calculate Fibonacci numbers.',
          'Explain quantum computing in simple terms.',
        ]}
        messages={messages}
        streamingText={streamingText}
        loading={loading}
        onSend={handleSend}
        renderMarkdown
      />

      <aside className="info-sidebar">
        <h3>🤝 How It Works</h3>
        <ol>
          <li>
            The <strong>orchestrator</strong> agent has two sub-agents wired via
            the <code>agents</code> middleware: <strong>researcher</strong> and{' '}
            <strong>coder</strong>.
          </li>
          <li>
            The middleware injects a <code>call_agent</code> tool that the
            orchestrator model can invoke to delegate tasks.
          </li>
          <li>
            When the model calls <code>call_agent</code>, the middleware
            intercepts it, runs the sub-agent via its <code>.run()</code>{' '}
            method, and returns the sub-agent's response as the tool result.
          </li>
          <li>
            The orchestrator synthesizes sub-agent responses into a final answer
            for the user.
          </li>
        </ol>

        <h4>Try It</h4>
        <ul>
          <li>
            <em>"Research the best sorting algorithms"</em> → delegates to
            researcher
          </li>
          <li>
            <em>"Write a quicksort in TypeScript"</em> → delegates to coder
          </li>
          <li>
            <em>
              "Research sorting algorithms then write a TypeScript quicksort"
            </em>{' '}
            → delegates to both sequentially
          </li>
        </ul>

        <h4>Key APIs</h4>
        <pre>{`// Define sub-agents
const researcher = ai.defineAgent({
  name: 'researcher',
  model: '...',
  system: '...',
  tools: [getWebResults],
});

const coder = ai.defineAgent({
  name: 'coder',
  model: '...',
  system: '...',
});

// Wire into orchestrator via middleware
const orchestrator = ai.defineAgent({
  name: 'orchestrator',
  model: '...',
  system: '...',
  use: [
    agents({
      agents: ['researcher', 'coder'],
    }),
  ],
});`}</pre>

        <h4>Architecture</h4>
        <p>
          The <code>agents</code> middleware from{' '}
          <code>@genkit-ai/middleware</code> resolves sub-agents from the
          registry (<code>/agent/name</code>), calls their <code>.run()</code>{' '}
          method with the delegated task, and extracts the text response.
          Interrupts from sub-agents propagate up automatically.
        </p>
      </aside>
    </div>
  );
}
