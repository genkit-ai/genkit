import type {
  AgentInit,
  AgentInput,
  AgentOutput,
  AgentStreamChunk,
} from 'genkit/beta';
import { streamFlow } from 'genkit/beta/client';
import { useCallback, useRef, useState } from 'react';
import { ChatUI, type Message } from '../components/ChatUI';

// ---------------------------------------------------------------------------
// Sub-Agent Delegation — orchestrator delegates to researcher & coder
//
// Demonstrates:
//   • The `agents` middleware for sub-agent delegation
//   • streamFlow() for streaming orchestrator responses
//   • Inline rendering of `call_agent` tool calls showing delegation
//   • Multi-turn session via state round-tripping
//   • Markdown rendering for code-heavy responses
// ---------------------------------------------------------------------------

const ENDPOINT = '/api/orchestratorAgent';

export default function SubAgentChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [streamingText, setStreamingText] = useState('');
  const [loading, setLoading] = useState(false);

  // Session tracking
  const stateRef = useRef<AgentOutput['state']>(undefined);

  const handleSend = useCallback(
    async (text: string) => {
      if (loading) return;

      setMessages((prev) => [...prev, { role: 'user', text }]);
      setLoading(true);
      setStreamingText('');

      const input: AgentInput = {
        messages: [{ role: 'user', content: [{ text }] }],
      };

      const init: AgentInit = stateRef.current
        ? { state: stateRef.current }
        : {};

      try {
        const response = streamFlow<AgentOutput, AgentStreamChunk, AgentInit>({
          url: ENDPOINT,
          input,
          init,
        });

        let accumulated = '';
        for await (const chunk of response.stream) {
          const mc = chunk?.modelChunk;
          if (!mc) continue;

          for (const part of mc.content || []) {
            if (part.text) {
              accumulated += part.text;
              setStreamingText(accumulated);
            } else if (part.toolRequest) {
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
        }

        const result = await response.output;
        setStreamingText('');

        if (result?.state) stateRef.current = result.state;

        const replyText = extractText(result);
        setMessages((prev) => [
          ...prev,
          { role: 'model', text: replyText || accumulated },
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
    [loading]
  );

  return (
    <div className="page-with-sidebar">
      <ChatUI
        title="Sub-Agent Orchestrator"
        description="An orchestrator that delegates research tasks to a researcher agent and coding tasks to a coder agent."
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

// ---------------------------------------------------------------------------
// Helper: pull displayable text out of a session flow result
// ---------------------------------------------------------------------------
function extractText(result: AgentOutput): string {
  if (!result) return '(no result)';
  const msg = result.message;
  if (!msg) return JSON.stringify(result, null, 2);
  const parts: string[] = [];
  for (const p of msg.content || []) {
    if (p.text) parts.push(p.text);
  }
  return parts.join('') || JSON.stringify(result, null, 2);
}
