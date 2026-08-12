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

'use client';

import { useChat } from '@ai-sdk/react';
import {
  GenkitChatTransport,
  restartInterrupt,
} from '@genkit-ai/vercel-ai/client';
import { lastAssistantMessageIsCompleteWithToolCalls } from 'ai';
import { useCallback, useMemo, useState } from 'react';

import {
  Conversation,
  ConversationContent,
  ConversationEmptyState,
  ConversationScrollButton,
} from '@/components/ai-elements/conversation';
import {
  Message,
  MessageContent,
  MessageResponse,
} from '@/components/ai-elements/message';
import {
  PromptInput,
  PromptInputFooter,
  PromptInputSubmit,
  PromptInputTextarea,
  type PromptInputMessage,
} from '@/components/ai-elements/prompt-input';
import {
  Tool,
  ToolContent,
  ToolHeader,
  ToolInput,
  ToolOutput,
} from '@/components/ai-elements/tool';

import { CloudSunIcon, LandmarkIcon } from 'lucide-react';

/**
 * Lightweight shape for tool invocation parts in the AI SDK v6 per-tool
 * format (`type: "tool-<name>"`). TypeScript can't narrow
 * `UIMessagePart` via `startsWith`, so we use this instead of `any`.
 */
interface ToolPartLike {
  type: `tool-${string}`;
  toolCallId?: string;
  toolName?: string;
  state?:
    | 'input-streaming'
    | 'input-available'
    | 'approval-requested'
    | 'approval-responded'
    | 'output-available'
    | 'output-error'
    | 'output-denied';
  input?: Record<string, unknown>;
  output?: unknown;
  errorText?: string;
}

// ---------------------------------------------------------------------------
// Agent configs
// ---------------------------------------------------------------------------

type AgentKey = 'weather' | 'banking';

const agents: Record<
  AgentKey,
  {
    label: string;
    endpoint: string;
    description: string;
    icon: React.ReactNode;
  }
> = {
  weather: {
    label: 'Weather Agent',
    endpoint: '/api/chat/weather',
    description: 'Ask about the weather anywhere in the world',
    icon: <CloudSunIcon className="size-5" />,
  },
  banking: {
    label: 'Banking Agent',
    endpoint: '/api/chat/banking',
    description:
      'Ask about banking — try requesting a money transfer to see the interrupt flow',
    icon: <LandmarkIcon className="size-5" />,
  },
};

// ---------------------------------------------------------------------------
// Chat panel for a single agent
// ---------------------------------------------------------------------------

function ChatPanel({ agentKey }: { agentKey: AgentKey }) {
  const agent = agents[agentKey];

  // Conversation state is fully server-managed: the transport sends the
  // `useChat` `id` to the agent as its Genkit `sessionId`, and the agent
  // persists per-session state server-side. The `id` must be a bare UUID.
  const transport = useMemo(
    () => new GenkitChatTransport({ url: agent.endpoint }),
    [agent.endpoint]
  );

  // A stable UUID for this chat session (regenerated when the agent changes,
  // since each panel is keyed by `activeAgent` and remounts).
  const chatId = useMemo(() => crypto.randomUUID(), []);

  // `addToolResult` records the user's resolution of an interrupted tool;
  // `sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithToolCalls`
  // tells useChat to automatically re-submit (resume) once every pending
  // tool call has a result. These are AI SDK v6 native HITL primitives — no
  // manual setMessages/flushSync/phantom-message dance required.
  const { messages, status, sendMessage, addToolResult } = useChat({
    id: chatId,
    transport,
    sendAutomaticallyWhen: lastAssistantMessageIsCompleteWithToolCalls,
  });

  const handleSubmit = useCallback(
    (msg: PromptInputMessage) => {
      sendMessage({ text: msg.text });
    },
    [sendMessage]
  );

  /**
   * Resolve a `userApproval` interrupt by supplying the tool's output. The
   * Genkit agent resumes with the user's decision as the tool response.
   */
  const handleApproval = useCallback(
    (toolName: string, toolCallId: string, approved: boolean) => {
      addToolResult({
        tool: toolName,
        toolCallId,
        output: {
          approved,
          feedback: approved ? 'User approved' : 'User rejected',
        },
      });
    },
    [addToolResult]
  );

  /**
   * Resolve a restartable interrupt by asking the agent to *re-run* the tool
   * (rather than supplying a final output). `restartInterrupt()` produces a
   * marker the Genkit transport turns into a `resume.restart`; any metadata
   * passed here arrives as the tool's `resumed` argument server-side.
   */
  const handleRestart = useCallback(
    (toolName: string, toolCallId: string) => {
      addToolResult({
        tool: toolName,
        toolCallId,
        output: restartInterrupt({ confirmedAt: new Date().toISOString() }),
      });
    },
    [addToolResult]
  );

  return (
    <div className="flex h-full flex-col">
      <Conversation>
        <ConversationContent>
          {messages.length === 0 ? (
            <ConversationEmptyState
              icon={agent.icon}
              title={agent.label}
              description={agent.description}
            />
          ) : (
            messages.map((message) => (
              <Message key={message.id} from={message.role}>
                <MessageContent>
                  {message.parts.map((part, i) => {
                    // Text parts
                    if (part.type === 'text' && part.text) {
                      return (
                        <MessageResponse key={i}>{part.text}</MessageResponse>
                      );
                    }

                    // Tool parts (v6 format: type === "tool-<name>")
                    if (part.type.startsWith('tool-')) {
                      const toolPart = part as unknown as ToolPartLike;
                      // Derive tool name from type (e.g. "tool-userApproval" → "userApproval")
                      const derivedToolName =
                        toolPart.toolName ||
                        part.type.split('-').slice(1).join('-');
                      // An interrupt is a tool call that's paused awaiting
                      // user input (state `input-available`, no output yet).
                      const isInterrupt = toolPart.state === 'input-available';
                      // `userApproval` is resolved by supplying a response;
                      // `getExchangeRate` is resolved by *restarting* the tool.
                      const isApproval = derivedToolName === 'userApproval';
                      const isRestartable =
                        derivedToolName === 'getExchangeRate';
                      const toolCallId = toolPart.toolCallId ?? '';

                      return (
                        <Tool key={i}>
                          <ToolHeader
                            type={toolPart.type}
                            state={toolPart.state ?? 'input-available'}
                          />
                          <ToolContent>
                            <ToolInput input={toolPart.input} />

                            {/* Approval interrupt: approve / reject buttons */}
                            {isInterrupt && isApproval && (
                              <div className="space-y-3">
                                <div className="rounded-md border border-yellow-200 bg-yellow-50 p-3 text-sm dark:border-yellow-800 dark:bg-yellow-950">
                                  <p className="font-medium text-yellow-800 dark:text-yellow-200">
                                    Approval Required
                                  </p>
                                  <p className="mt-1 text-yellow-700 dark:text-yellow-300">
                                    {String(toolPart.input?.action ?? '')}:{' '}
                                    {String(toolPart.input?.details ?? '')}
                                  </p>
                                </div>
                                <div className="flex items-center justify-end gap-2">
                                  <button
                                    type="button"
                                    className="rounded-md border px-3 py-1.5 text-sm font-medium text-muted-foreground hover:bg-muted"
                                    onClick={() =>
                                      handleApproval(
                                        derivedToolName,
                                        toolCallId,
                                        false
                                      )
                                    }>
                                    Reject
                                  </button>
                                  <button
                                    type="button"
                                    className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90"
                                    onClick={() =>
                                      handleApproval(
                                        derivedToolName,
                                        toolCallId,
                                        true
                                      )
                                    }>
                                    Approve
                                  </button>
                                </div>
                              </div>
                            )}

                            {/* Restartable interrupt: re-run the tool */}
                            {isInterrupt && isRestartable && (
                              <div className="space-y-3">
                                <div className="rounded-md border border-blue-200 bg-blue-50 p-3 text-sm dark:border-blue-800 dark:bg-blue-950">
                                  <p className="font-medium text-blue-800 dark:text-blue-200">
                                    Confirm to continue
                                  </p>
                                  <p className="mt-1 text-blue-700 dark:text-blue-300">
                                    Look up the live exchange rate for{' '}
                                    {String(toolPart.input?.fromCurrency ?? '')}{' '}
                                    → {String(toolPart.input?.toCurrency ?? '')}
                                    ?
                                  </p>
                                </div>
                                <div className="flex items-center justify-end gap-2">
                                  <button
                                    type="button"
                                    className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90"
                                    onClick={() =>
                                      handleRestart(derivedToolName, toolCallId)
                                    }>
                                    Look up rate
                                  </button>
                                </div>
                              </div>
                            )}

                            <ToolOutput
                              output={toolPart.output}
                              errorText={toolPart.errorText}
                            />
                          </ToolContent>
                        </Tool>
                      );
                    }

                    return null;
                  })}
                </MessageContent>
              </Message>
            ))
          )}
        </ConversationContent>
        <ConversationScrollButton />
      </Conversation>

      <div className="border-t p-4">
        <PromptInput onSubmit={handleSubmit}>
          <PromptInputTextarea placeholder={`Message ${agent.label}...`} />
          <PromptInputFooter>
            <div />
            <PromptInputSubmit status={status} />
          </PromptInputFooter>
        </PromptInput>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page — agent selector + chat panel
// ---------------------------------------------------------------------------

export default function Home() {
  const [activeAgent, setActiveAgent] = useState<AgentKey>('weather');

  return (
    <div className="flex h-screen flex-col">
      {/* Header with agent selector */}
      <header className="flex items-center gap-4 border-b px-6 py-3">
        <h1 className="text-lg font-semibold">Genkit + AI Elements</h1>
        <div className="flex gap-1 rounded-lg bg-muted p-1">
          {(Object.keys(agents) as AgentKey[]).map((key) => (
            <button
              key={key}
              type="button"
              onClick={() => setActiveAgent(key)}
              className={`flex items-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium transition-colors ${
                activeAgent === key
                  ? 'bg-background text-foreground shadow-sm'
                  : 'text-muted-foreground hover:text-foreground'
              }`}>
              {agents[key].icon}
              {agents[key].label}
            </button>
          ))}
        </div>
      </header>

      {/* Chat area */}
      <main className="flex-1 overflow-hidden">
        <ChatPanel key={activeAgent} agentKey={activeAgent} />
      </main>
    </div>
  );
}
