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

/**
 * Multi-Step Research Agent — demonstrates defineCustomAgent
 *
 * This sample showcases capabilities that REQUIRE defineCustomAgent and
 * cannot be achieved with the simpler defineAgent:
 *
 *   • Multi-step orchestration — multiple sequential model calls with
 *     custom logic between them
 *   • Custom status streaming — sendChunk({ status }) to stream typed
 *     progress indicators to the client between model calls
 *   • Direct session control — manually managing messages and custom state
 *   • Multiple models — using a fast model for decomposition and a capable
 *     model for research and synthesis
 *
 * Flow:
 *   1. Decompose: Fast model breaks the user's question into 2–3 sub-questions
 *   2. Research: Main model answers each sub-question (with status updates)
 *   3. Synthesize: Main model combines sub-answers into a final response
 *      (streamed to the client via modelChunk)
 */

import { z } from 'genkit';
import { ai } from './genkit.js';

// ---------------------------------------------------------------------------
// Typed custom state — tracks research steps
// ---------------------------------------------------------------------------

interface ResearchState {
  subQuestions: string[];
  subAnswers: Array<{ question: string; answer: string }>;
}

// ---------------------------------------------------------------------------
// Custom Agent — multi-step orchestration
// ---------------------------------------------------------------------------

export const customAgent = ai.defineCustomAgent(
  { name: 'customAgent' },
  async (sess, { sendChunk }) => {
    let lastMessage: any;

    await sess.run(async (input) => {
      const userText =
        input.messages?.[input.messages.length - 1]?.content[0]?.text || '';

      // Build conversation history context (available for ALL steps).
      // sess.run() adds the current input.messages to the session before
      // calling this handler, so getMessages() includes everything.
      const priorMessages = sess.session.getMessages();
      const historyContext =
        priorMessages.length > 1
          ? '\nConversation history:\n' +
            priorMessages
              .slice(0, -1) // Exclude the current user message (already in `userText`)
              .map((m) => `${m.role}: ${m.content.map((c) => c.text || '').join('')}`)
              .join('\n') +
            '\n'
          : '';

      // ── Step 1: Decompose the question ────────────────────────────────
      sendChunk({ status: 'Decomposing question into sub-topics…' });

      const decompose = await ai.generate({
        model: 'googleai/gemini-flash-lite-latest',
        prompt: `You are a research planner. Given a user question, break it into exactly 2-3 focused sub-questions that together would provide a comprehensive answer. Return ONLY the sub-questions as a JSON array of strings, no other text.
${historyContext}
User question: "${userText}"`,
        output: { format: 'json', schema: z.array(z.string()).min(2).max(3) },
      });

      const subQuestions: string[] = decompose.output ?? [userText];

      // Store decomposition in custom state
      const session = ai.currentSession<ResearchState>();
      session.updateCustom(() => ({
        subQuestions,
        subAnswers: [],
      }));

      // ── Step 2: Research each sub-question ────────────────────────────
      const subAnswers: Array<{ question: string; answer: string }> = [];

      for (let i = 0; i < subQuestions.length; i++) {
        const q = subQuestions[i];
        sendChunk({ status: `Researching (${i + 1}/${subQuestions.length}): ${q}` });

        const research = await ai.generate({
          model: 'googleai/gemini-flash-latest',
          prompt: `Answer this question concisely but thoroughly in 2-3 paragraphs. Be specific and factual.

Question: ${q}`,
        });

        subAnswers.push({ question: q, answer: research.text });
      }

      // Update custom state with all sub-answers
      session.updateCustom((state) => ({
        ...state!,
        subAnswers,
      }));

      // ── Step 3: Synthesize final response ─────────────────────────────
      sendChunk({ status: 'Synthesizing final response…' });

      const researchContext = subAnswers
        .map((sa, i) => `### Sub-question ${i + 1}: ${sa.question}\n${sa.answer}`)
        .join('\n\n');

      const synthesisStream = ai.generateStream({
        model: 'googleai/gemini-flash-latest',
        prompt: `You are a research synthesizer. Based on the research below, write a comprehensive, well-structured answer to the original question. Use markdown formatting.
${historyContext}
Current question: "${userText}"

Research findings:
${researchContext}

Write a clear, cohesive response that integrates all the research findings. Don't just list the sub-answers — synthesize them into a unified narrative. If there is conversation history, take it into account for context.`,
      });

      // Stream the final synthesis to the client
      for await (const chunk of synthesisStream.stream) {
        sendChunk({ modelChunk: chunk });
      }

      const synthesisResponse = await synthesisStream.response;
      lastMessage = synthesisResponse.message;

      // Add the final synthesized response to the session messages
      if (lastMessage) {
        sess.session.addMessages([lastMessage]);
      }

      sendChunk({ status: 'Done' });
    });

    return {
      message: lastMessage || {
        role: 'model' as const,
        content: [{ text: 'Research complete.' }],
      },
    };
  }
);

// ---------------------------------------------------------------------------
// Test flow
// ---------------------------------------------------------------------------

export const testCustomAgent = ai.defineFlow(
  {
    name: 'testCustomAgent',
    inputSchema: z
      .string()
      .default('What are the environmental and economic impacts of electric vehicles?'),
    outputSchema: z.any(),
  },
  async (text, { sendChunk }) => {
    const res = await customAgent.run(
      {
        messages: [{ role: 'user', content: [{ text }] }],
      },
      {
        init: {
          state: {
            custom: { subQuestions: [], subAnswers: [] } as ResearchState,
            messages: [],
            artifacts: [],
          },
        },
        onChunk: sendChunk,
      }
    );
    return res.result;
  }
);
