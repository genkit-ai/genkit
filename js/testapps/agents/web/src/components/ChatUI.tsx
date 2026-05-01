import { useEffect, useRef } from 'react';
import Markdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

// ---------------------------------------------------------------------------
// Shared chat chrome — renders messages, input box, and send button.
// Contains NO genkit logic. Each page owns its own session/streaming code.
// ---------------------------------------------------------------------------

export interface Message {
  role: 'user' | 'model' | 'system' | 'tool';
  text: string;
}

interface Props {
  /** Display title shown in the chat header. */
  title: string;
  /** Short description below the title. */
  description?: string;
  /** Conversation messages to render. */
  messages: Message[];
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
}: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo(0, scrollRef.current.scrollHeight);
  }, [messages, streamingText]);

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
            Send a message to start a conversation with <strong>{title}</strong>
          </div>
        )}
        {messages.map((m, i) => {
          const isUser = m.role === 'user';
          const isSystem = m.role === 'system';
          const isTool = m.role === 'tool';
          return (
            <div
              key={i}
              className={`message ${isUser ? 'message-user' : ''} ${isSystem ? 'message-system' : ''} ${isTool ? 'message-tool' : ''}`}>
              <div className="message-role">{m.role}</div>
              <div
                className={`message-text ${isTool ? 'message-text-mono' : ''}`}>
                {renderMarkdown && m.role === 'model' ? (
                  <div className="markdown-body">
                    <Markdown remarkPlugins={[remarkGfm]}>{m.text}</Markdown>
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
            </div>
          );
        })}
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
        {loading && !streamingText && (
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
