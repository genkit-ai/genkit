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
// Research Agent — Multi-Step Orchestration (defineCustomAgent)
//
// Demonstrates capabilities that REQUIRE defineCustomAgent:
//   • Multi-step orchestration — multiple sequential model calls
//   • Custom status streaming — sendChunk({ status }) for progress updates
//   • Multiple models — fast model for decomposition, main model for research
//   • Direct session control — manually managing messages and custom state
//
// The backend:
//   1. Decomposes the question into 2–3 sub-questions (fast model)
//   2. Researches each sub-question (main model, with status updates)
//   3. Synthesizes all sub-answers into a final response (streamed)
// ---------------------------------------------------------------------------

const ENDPOINT = '/api/customAgent';

interface SubAnswer {
  question: string;
  answer: string;
}

interface ResearchState {
  subQuestions: string[];
  subAnswers: SubAnswer[];
}

export default function ResearchAgent() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [streamingText, setStreamingText] = useState('');
  const [loading, setLoading] = useState(false);
  const [statusText, setStatusText] = useState<string | null>(null);

  // Research state — extracted from result.state.custom each turn
  const [researchState, setResearchState] = useState<ResearchState | null>(
    null
  );

  // Session state — round-tripped to the server each turn
  const stateRef = useRef<any>(undefined);

  const handleSend = useCallback(
    async (text: string) => {
      if (loading) return;

      setMessages((prev) => [...prev, { role: 'user', text }]);
      setLoading(true);
      setStreamingText('');
      setStatusText(null);
      setResearchState(null);

      // ── Build the request ──────────────────────────────────────────────
      const input: AgentInput = {
        messages: [{ role: 'user', content: [{ text }] }],
      };

      const init: AgentInit = stateRef.current
        ? { state: stateRef.current }
        : {
            state: {
              custom: { subQuestions: [], subAnswers: [] } as ResearchState,
              messages: [],
              artifacts: [],
            },
          };

      try {
        // ── Stream the response ────────────────────────────────────────
        const response = streamFlow<AgentOutput, AgentStreamChunk, AgentInit>({
          url: ENDPOINT,
          input,
          init,
        });

        let accumulated = '';
        for await (const chunk of response.stream) {
          // ── Status chunks — progress indicators from the orchestrator ──
          if (chunk?.status) {
            setStatusText(chunk.status as string);
          }

          // ── Model text chunks — the final synthesis ────────────────────
          const mc = chunk?.modelChunk;
          if (!mc) continue;

          for (const part of mc.content || []) {
            if (part.text) {
              accumulated += part.text;
              setStreamingText(accumulated);
            }
          }
        }

        // ── Read the final result ──────────────────────────────────────
        const result = await response.output;
        setStreamingText('');
        setStatusText(null);

        // Save session state for the next turn.
        if (result?.state) {
          stateRef.current = result.state;

          // Extract the research state for the sidebar.
          const custom = result.state.custom as ResearchState | undefined;
          if (custom) {
            setResearchState(custom);
          }
        }

        const replyText = extractText(result);
        setMessages((prev) => [
          ...prev,
          { role: 'model', text: replyText || accumulated },
        ]);
      } catch (err: any) {
        setStreamingText('');
        setStatusText(null);
        setMessages((prev) => [
          ...prev,
          { role: 'system', text: `Error: ${err.message}` },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [loading]
  );

  return (
    <div className="research-layout">
      <ChatUI
        title="Research Agent"
        description='Multi-step research via defineCustomAgent. Try: "What are the impacts of AI on education?" or "Compare solar and wind energy"'
        messages={messages}
        streamingText={streamingText}
        loading={loading}
        onSend={handleSend}
        renderMarkdown>
        {/* Status indicator — shows orchestration progress */}
        {statusText && (
          <div className="research-status-bar">
            <span className="research-status-dot" />
            {statusText}
          </div>
        )}
      </ChatUI>

      {/* Research Process Panel — shows the decomposition and sub-answers */}
      <aside className="research-sidebar">
        <h3>🔬 Research Process</h3>
        <p className="research-sidebar-hint">
          Shows the multi-step orchestration: decomposition → research →
          synthesis. This is only possible with <code>defineCustomAgent</code>.
        </p>

        {!researchState ? (
          <div className="research-empty">
            Ask a question to see the research process unfold.
          </div>
        ) : (
          <>
            {/* Sub-questions */}
            <div className="research-section">
              <h4>📋 Sub-Questions</h4>
              <p className="research-section-hint">
                Generated by a fast model (<code>gemini-flash-lite</code>)
              </p>
              <ol className="research-questions">
                {researchState.subQuestions.map((q, i) => (
                  <li key={i} className="research-question">
                    {q}
                  </li>
                ))}
              </ol>
            </div>

            {/* Sub-answers */}
            {researchState.subAnswers.length > 0 && (
              <div className="research-section">
                <h4>📝 Research Findings</h4>
                <p className="research-section-hint">
                  Each answered by the main model (<code>gemini-flash</code>)
                </p>
                {researchState.subAnswers.map((sa, i) => (
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

        <h4>📋 How It Works</h4>
        <ol className="research-howto">
          <li>
            Uses <code>defineCustomAgent</code> for full control of the handler
            — orchestrating multiple model calls.
          </li>
          <li>
            <strong>Step 1:</strong> Fast model decomposes the question →{' '}
            <code>sendChunk({'{ status }'})</code>
          </li>
          <li>
            <strong>Step 2:</strong> Main model researches each sub-question →{' '}
            status updates per sub-question
          </li>
          <li>
            <strong>Step 3:</strong> Main model synthesizes a final response →{' '}
            <code>sendChunk({'{ modelChunk }'})</code>
          </li>
          <li>
            Research state stored in <code>session.custom</code> via{' '}
            <code>session.updateCustom()</code>.
          </li>
        </ol>

        <h4>Why defineCustomAgent?</h4>
        <pre className="research-code">{`// Can't do this with defineAgent:
// 1. Multiple sequential model calls
const decompose = await ai.generate({ model: 'fast', ... });
const research = await ai.generate({ model: 'main', ... });
const synthesis = ai.generateStream({ model: 'main', ... });

// 2. Custom status streaming between steps
sendChunk({ status: 'Researching...' });

// 3. Direct session & message management
sess.session.addMessages([response.message]);`}</pre>
      </aside>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Helper: extract displayable text from a session flow result
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
