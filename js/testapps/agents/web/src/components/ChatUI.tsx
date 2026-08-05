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

import { useEffect, useRef } from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// ---------------------------------------------------------------------------
// Shared chat chrome — renders messages, input box, and send button.
// Contains NO genkit logic. Each page owns its own session/streaming code.
// ---------------------------------------------------------------------------

export interface ChatMessage {
  role: 'user' | 'model' | 'system' | 'tool';
  text: string;
  /** Optional reasoning/thinking content — rendered as a collapsible block. */
  reasoning?: string;
  /** Optional detail content — rendered as a terminal-style box below the text. */
  detail?: string;
}

interface Props {
  /** Display title shown in the chat header. */
  title: string;
  /** Short description below the title. */
  description?: string;
  /** Conversation messages to render. */
  messages: ChatMessage[];
  /** Partial text being streamed in (shown with a cursor). */
  streamingText?: string;
  /** Whether the agent is currently processing. */
  loading?: boolean;
  /** Called when the user submits a message. */
  onSend: (text: string) => void;
  /** Disable input (e.g. during interrupt). */
  inputDisabled?: boolean;
  /** Optional extra content rendered between messages and input (e.g. interrupt dialog, artifacts). */
  children?: React.ReactNode;
  /** Optional action element rendered in the header (e.g. "New Session" button). */
  headerAction?: React.ReactNode;
  /** Render model messages as markdown instead of plain text. */
  renderMarkdown?: boolean;
  /** Suggested prompts shown in the empty state for easy copy-paste or click-to-send. */
  suggestions?: string[];
  /** Partial reasoning text being streamed (shown as animated "thinking..." block). */
  streamingReasoning?: string;
}

export function ChatUI({
  title,
  description,
  messages,
  streamingText,
  loading,
  onSend,
  inputDisabled,
  children,
  headerAction,
  renderMarkdown,
  suggestions,
  streamingReasoning,
}: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    // Use double-rAF to ensure DOM has fully updated before scrolling
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        if (scrollRef.current) {
          scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
      });
    });
  }, [messages, streamingText, streamingReasoning]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      const text = inputRef.current?.value.trim();
      if (text) {
        onSend(text);
        if (inputRef.current) inputRef.current.value = '';
      }
    }
  };

  const handleSend = () => {
    const text = inputRef.current?.value.trim();
    if (text) {
      onSend(text);
      if (inputRef.current) inputRef.current.value = '';
    }
  };

  const disabled = loading || inputDisabled;

  return (
    <div className="chat-panel">
      <div className="chat-header">
        <div className="chat-header-top">
          <h2>{title}</h2>
          {headerAction && (
            <div className="chat-header-action">{headerAction}</div>
          )}
        </div>
        {description && <span className="chat-desc">{description}</span>}
      </div>

      <div className="chat-messages" ref={scrollRef}>
        {messages.length === 0 && !streamingText && !loading && (
          <div className="empty-state">
            <p>
              Send a message to start a conversation with{' '}
              <strong>{title}</strong>
            </p>
            {suggestions && suggestions.length > 0 && (
              <div className="suggestions">
                <span className="suggestions-label">Try one of these:</span>
                <div className="suggestions-list">
                  {suggestions.map((s, i) => (
                    <button
                      key={i}
                      className="suggestion-chip"
                      onClick={() => onSend(s)}
                      disabled={disabled}>
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
        {messages.map((m, i) => {
          const isUser = m.role === 'user';
          const isSystem = m.role === 'system';
          const isTool = m.role === 'tool';
          return (
            <div key={i}>
              {/* Collapsible reasoning block (if message has reasoning) */}
              {m.reasoning && (
                <details className="thinking-block">
                  <summary className="thinking-summary">
                    <span className="thinking-icon">🧠</span> Thinking…
                  </summary>
                  <div className="thinking-content">
                    <Markdown remarkPlugins={[remarkGfm]}>
                      {m.reasoning}
                    </Markdown>
                  </div>
                </details>
              )}
              {/* Only show the message bubble if there's actual text content */}
              {m.text && (
                <div
                  className={`message ${isUser ? 'message-user' : ''} ${isSystem ? 'message-system' : ''} ${isTool ? 'message-tool' : ''}`}>
                  <div className="message-role">{m.role}</div>
                  <div
                    className={`message-text ${isTool ? 'message-text-mono' : ''}`}>
                    {renderMarkdown && m.role === 'model' ? (
                      <div className="markdown-body">
                        <Markdown remarkPlugins={[remarkGfm]}>
                          {m.text}
                        </Markdown>
                      </div>
                    ) : (
                      m.text.split('\n').map((line, j) => (
                        <span key={j}>
                          {line}
                          {j < m.text.split('\n').length - 1 && <br />}
                        </span>
                      ))
                    )}
                  </div>
                  {/* Terminal-style detail box (e.g. shell output, file content) */}
                  {m.detail && <pre className="message-detail">{m.detail}</pre>}
                </div>
              )}
            </div>
          );
        })}
        {/* Live streaming reasoning indicator */}
        {streamingReasoning && !streamingText && (
          <details className="thinking-block thinking-streaming" open>
            <summary className="thinking-summary">
              <span className="thinking-icon thinking-pulse">🧠</span> Thinking…
            </summary>
            <div className="thinking-content">
              <Markdown remarkPlugins={[remarkGfm]}>
                {streamingReasoning}
              </Markdown>
            </div>
          </details>
        )}
        {streamingText && (
          <div className="message">
            <div className="message-role">model</div>
            <div className="message-text streaming">
              {renderMarkdown ? (
                <div className="markdown-body">
                  <Markdown remarkPlugins={[remarkGfm]}>
                    {streamingText}
                  </Markdown>
                  <span>▊</span>
                </div>
              ) : (
                <>{streamingText}▊</>
              )}
            </div>
          </div>
        )}
        {loading && !streamingText && !streamingReasoning && (
          <div className="message">
            <div className="message-role">model</div>
            <div className="message-text loading">Thinking…</div>
          </div>
        )}
      </div>

      {children}

      <div className="chat-input-area">
        <textarea
          ref={inputRef}
          className="chat-input"
          placeholder={`Message ${title}…`}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          rows={2}
        />
        <button
          className="btn btn-send"
          onClick={handleSend}
          disabled={disabled}>
          Send
        </button>
      </div>
    </div>
  );
}
