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
// Research Agent — Multi-Step Orchestration (defineCustomAgent)
//
// Demonstrates capabilities that REQUIRE defineCustomAgent:
//   • Multi-step orchestration — multiple sequential model calls
//   • Custom-state streaming — the backend mutates a typed `status` field on
//     the session's custom state; the runtime auto-emits a custom-state update.
//     The `remoteAgent` client applies it for us and surfaces the full,
//     post-patch state as `chunk.custom`, so we never touch JSON Patch directly.
//   • Multiple models — fast model for decomposition, main model for research
//   • Direct session control — manually managing messages and custom state
//
// The backend:
//   1. Decomposes the question into 2–3 sub-questions (fast model)
//   2. Researches each sub-question (main model, with status updates)
//   3. Synthesizes all sub-answers into a final response (streamed)
// ---------------------------------------------------------------------------

const ENDPOINT = '/api/researchAgent';

interface SubAnswer {
  question: string;
  answer: string;
}

interface ResearchState {
  /** Human-readable progress indicator, driven by streamed custom updates. */
  status?: string;
  subQuestions: string[];
  subAnswers: SubAnswer[];
}

export default function ResearchAgent() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streamingText, setStreamingText] = useState('');
  const [loading, setLoading] = useState(false);
  const [statusText, setStatusText] = useState<string | null>(null);

  // Research state — kept live from each chunk's `custom` payload.
  const [researchState, setResearchState] = useState<ResearchState | null>(
    null
  );

  // Typed HTTP client for the agent. The client tracks session state across
  // turns and applies streamed custom-state updates for us.
  const agent = useMemo(
    () => remoteAgent<ResearchState>({ url: ENDPOINT }),
    []
  );

  // The conversation. Created on first send (seeding the initial custom state)
  // and reused for follow-up turns so history is threaded automatically.
  const chatRef = useRef<AgentChat<ResearchState> | null>(null);

  const handleSend = useCallback(
    async (text: string) => {
      if (loading) return;

      setMessages((prev) => [...prev, { role: 'user', text }]);
      setLoading(true);
      setStreamingText('');
      setStatusText(null);
      setResearchState(null);

      // Lazily create the chat, seeding the initial custom state.
      if (!chatRef.current) {
        chatRef.current = agent.chat({
          state: {
            custom: { subQuestions: [], subAnswers: [] },
            messages: [],
            artifacts: [],
          },
        });
      }
      const chat = chatRef.current;

      try {
        const turn = chat.sendStream(text);

        let accumulated = '';
        for await (const chunk of turn.stream) {
          // ── Custom-state updates — fresh, post-patch state each time ──
          if (chunk.custom) {
            setResearchState(chunk.custom);
            setStatusText(chunk.custom.status ?? null);
          } else if (chunk.text) {
            // ── Model text chunks — the final synthesis ──
            accumulated = chunk.accumulatedText;
            setStreamingText(accumulated);
          }
        }

        const res = await turn.response;
        setStreamingText('');
        setStatusText(null);

        // Reflect the authoritative final custom state in the sidebar.
        if (chat.state) {
          setResearchState(chat.state);
        }

        setMessages((prev) => [
          ...prev,
          { role: 'model', text: res.text || accumulated },
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
    [agent, loading]
  );

  return (
    <div className="research-layout">
      <ChatUI
        title="Research Agent"
        description="Multi-step research via defineCustomAgent."
        suggestions={[
          'What are the impacts of AI on education?',
          'Compare solar and wind energy.',
          'Explain the pros and cons of remote work.',
        ]}
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
            <code>session.updateCustom({'{ status }'})</code>
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

        <h4>Client: read chunk.custom</h4>
        <pre className="research-code">{`const agent = remoteAgent<ResearchState>({
  url: '/api/researchAgent',
});
const chat = agent.chat();

const turn = chat.sendStream(text);
for await (const chunk of turn.stream) {
  if (chunk.custom) {
    // full, post-patch custom state (fresh ref)
    setResearchState(chunk.custom);
  } else if (chunk.text) {
    appendStreamingText(chunk.text);
  }
}`}</pre>

        <h4>Why defineCustomAgent?</h4>
        <pre className="research-code">{`// Can't do this with defineAgent:
// 1. Multiple sequential model calls
const decompose = await ai.generate({ model: 'fast', ... });
const research = await ai.generate({ model: 'main', ... });
const synthesis = ai.generateStream({ model: 'main', ... });

// 2. Custom-state streaming between steps — mutate custom
//    state; the runtime auto-emits a custom-state update.
session.updateCustom((s) => ({ ...s, status: 'Researching...' }));

// 3. Direct session & message management
sess.addMessages([response.message]);`}</pre>
      </aside>
    </div>
  );
}
