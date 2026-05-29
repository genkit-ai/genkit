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
import { GenkitChatTransport } from '@genkit-ai/vercel-ai/client';
import type { UIMessage } from 'ai';
import { useCallback, useMemo, useState } from 'react';
import { flushSync } from 'react-dom';

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

  const transport = useMemo(
    () => new GenkitChatTransport({ url: agent.endpoint }),
    [agent.endpoint]
  );

  const { messages, status, sendMessage, setMessages } = useChat({
    transport,
  });

  const handleSubmit = useCallback(
    (msg: PromptInputMessage) => {
      sendMessage({ text: msg.text });
    },
    [sendMessage]
  );

  /**
   * Handle interrupt resolution — when the user approves or rejects an
   * interrupt, we update the tool part's output and state, then re-send
   * the messages so the backend can resume.
   *
   * We use `flushSync` to ensure React commits the `setMessages` update
   * synchronously before `sendMessage` reads the messages array. This
   * avoids a race condition that would occur with `setTimeout`.
   *
   * Note: `sendMessage({ text: '' })` appends a phantom empty-text user
   * message to the conversation. The transport ignores this for resume
   * payloads — it detects the resolved tool results in the assistant
   * message and sends a resume request instead.
   */
  const handleInterruptResolve = useCallback(
    (
      messageId: string,
      toolCallId: string,
      toolName: string,
      approved: boolean
    ) => {
      flushSync(() => {
        setMessages((prev: UIMessage[]) =>
          prev.map((msg) => {
            if (msg.id !== messageId) return msg;
            return {
              ...msg,
              parts: msg.parts.map((part) => {
                const tp = part as unknown as ToolPartLike;
                if (
                  part.type === `tool-${toolName}` &&
                  tp.toolCallId === toolCallId
                ) {
                  return {
                    ...part,
                    state: 'output-available' as const,
                    output: {
                      approved,
                      feedback: approved ? 'User approved' : 'User rejected',
                    },
                  } as typeof part;
                }
                return part;
              }),
            };
          })
        );
      });

      // State is now flushed — safe to trigger resume immediately.
      sendMessage({ text: '' });
    },
    [setMessages, sendMessage]
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
                      const isInterrupt =
                        derivedToolName === 'userApproval' &&
                        toolPart.state === 'input-available';

                      return (
                        <Tool key={i}>
                          <ToolHeader
                            type={toolPart.type}
                            state={toolPart.state ?? 'input-available'}
                          />
                          <ToolContent>
                            <ToolInput input={toolPart.input} />

                            {/* Interrupt: show approve/reject buttons */}
                            {isInterrupt && (
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
                                      handleInterruptResolve(
                                        message.id,
                                        toolPart.toolCallId ?? '',
                                        derivedToolName,
                                        false
                                      )
                                    }>
                                    Reject
                                  </button>
                                  <button
                                    type="button"
                                    className="rounded-md bg-primary px-3 py-1.5 text-sm font-medium text-primary-foreground hover:bg-primary/90"
                                    onClick={() =>
                                      handleInterruptResolve(
                                        message.id,
                                        toolPart.toolCallId ?? '',
                                        derivedToolName,
                                        true
                                      )
                                    }>
                                    Approve
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
