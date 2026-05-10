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

import type {
  AgentInit,
  AgentInput,
  AgentOutput,
  AgentStreamChunk,
  ToolRequest,
} from 'genkit/beta';
import { streamFlow } from 'genkit/beta/client';
import { useCallback, useEffect, useRef, useState } from 'react';
import { ChatUI, type Message } from '../components/ChatUI';

// ---------------------------------------------------------------------------
// Coding Agent — AI coding assistant with filesystem access
//
// Demonstrates:
//   • filesystem middleware (list_files, read_file, write_file, search_and_replace)
//   • skills middleware (use_skill for coding conventions)
//   • toolApproval middleware (interrupt for write operations)
//   • run_shell tool with AI-powered safety gate (interrupt for risky commands)
//   • Server-side session store for interrupt resumption
//   • Real-time file explorer showing workspace contents
//   • Inline tool approval dialog with file content preview
//   • Markdown rendering for code-heavy responses
//   • Uses `toolRestarts` (not toolResponse) to resume interrupted tools
// ---------------------------------------------------------------------------

const API_BASE = 'http://localhost:8080';
const ENDPOINT = `${API_BASE}/api/codingAgent`;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface WorkspaceFile {
  name: string;
  path: string;
  type: 'file' | 'directory';
  children?: WorkspaceFile[];
}

interface PendingApproval {
  toolName: string;
  ref?: string;
  input: any;
  snapshotId: string;
}

interface PendingQuestion {
  question: string;
  options: string[];
  ref?: string;
  snapshotId: string;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function CodingAgent() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [streamingText, setStreamingText] = useState('');
  const [streamingReasoning, setStreamingReasoning] = useState('');
  const [loading, setLoading] = useState(false);
  const [approval, setApproval] = useState<PendingApproval | null>(null);
  const [question, setQuestion] = useState<PendingQuestion | null>(null);
  const [customAnswer, setCustomAnswer] = useState('');

  // File explorer state
  const [files, setFiles] = useState<WorkspaceFile[]>([]);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);
  const [fileContent, setFileContent] = useState<string>('');
  const [fileLoading, setFileLoading] = useState(false);

  // Session tracking
  const stateRef = useRef<any>(undefined);
  const snapshotIdRef = useRef<string | undefined>(undefined);

  // ── Fetch workspace file tree ──────────────────────────────────────────
  const refreshFiles = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/api/workspace/files`);
      const data = await res.json();
      setFiles(data.files || []);
    } catch {
      // ignore
    }
  }, []);

  // Load files on mount and after each agent response
  useEffect(() => {
    refreshFiles();
  }, [refreshFiles]);

  // ── Fetch a single file's content ──────────────────────────────────────
  const viewFile = useCallback(async (filePath: string) => {
    setSelectedFile(filePath);
    setFileLoading(true);
    try {
      const res = await fetch(
        `${API_BASE}/api/workspace/file?path=${encodeURIComponent(filePath)}`
      );
      const data = await res.json();
      setFileContent(data.content || '');
    } catch {
      setFileContent('(failed to load file)');
    } finally {
      setFileLoading(false);
    }
  }, []);

  // ── Send a regular user message ──────────────────────────────────────
  const handleSend = useCallback(
    async (text: string) => {
      if (loading || approval || question) return;

      setMessages((prev) => [...prev, { role: 'user', text }]);
      setLoading(true);
      setStreamingText('');
      setStreamingReasoning('');

      const input: AgentInput = {
        messages: [{ role: 'user', content: [{ text }] }],
      };

      const init: AgentInit = stateRef.current
        ? { state: stateRef.current }
        : snapshotIdRef.current
          ? { snapshotId: snapshotIdRef.current }
          : {};

      try {
        const result = await streamAndCollect(input, init);
        processResult(result);
      } catch (err: unknown) {
        setStreamingText('');
        const errMsg = err instanceof Error ? err.message : String(err);
        setMessages((prev) => [
          ...prev,
          { role: 'system', text: `Error: ${errMsg}` },
        ]);
      } finally {
        setLoading(false);
        refreshFiles();
      }
    },
    [loading, approval, question, refreshFiles]
  );

  // ── Respond to a tool approval interrupt ──────────────────────────────
  const handleApprovalResponse = useCallback(
    async (approved: boolean) => {
      if (!approval) return;
      const currentApproval = approval;
      setApproval(null);
      setLoading(true);
      setStreamingText('');
      setStreamingReasoning('');

      setMessages((prev) => [
        ...prev,
        {
          role: 'system',
          text: approved
            ? `✅ Approved: ${currentApproval.toolName}`
            : `❌ Denied: ${currentApproval.toolName}`,
        },
      ]);

      const init: AgentInit = { snapshotId: currentApproval.snapshotId };
      let input: AgentInput;

      if (approved) {
        // Use toolRestarts to resume the interrupted tool.
        // This maps to generate()'s resume: { restart: [...] } in definePromptAgent.
        // The metadata.resumed.toolApproved flag is checked by:
        //   • toolApproval middleware (for write_file, search_and_replace)
        //   • run_shell tool handler (for risky shell commands)
        input = {
          toolRestarts: [
            {
              toolRequest: {
                name: currentApproval.toolName,
                ref: currentApproval.ref,
                input: currentApproval.input,
              },
              metadata: { resumed: { toolApproved: true } },
            },
          ],
        };
      } else {
        // For denial, send a user message so the model knows the tool was rejected.
        // We don't use toolRestarts for denial because that would re-trigger the
        // interrupt in a loop.
        input = {
          messages: [
            {
              role: 'user',
              content: [
                {
                  text:
                    `I denied the "${currentApproval.toolName}" tool call` +
                    (currentApproval.toolName === 'run_shell'
                      ? ` for command: "${currentApproval.input?.command}".`
                      : ` for file: "${currentApproval.input?.filePath}".`) +
                    ` Please continue without executing it, or suggest an alternative.`,
                },
              ],
            },
          ],
        };
      }

      try {
        const result = await streamAndCollect(input, init);
        processResult(result);
      } catch (err: unknown) {
        setStreamingText('');
        const errMsg = err instanceof Error ? err.message : String(err);
        setMessages((prev) => [
          ...prev,
          { role: 'system', text: `Error resuming: ${errMsg}` },
        ]);
      } finally {
        setLoading(false);
        refreshFiles();
      }
    },
    [approval, refreshFiles]
  );

  // ── Shared: stream a request and collect chunks ──────────────────────
  async function streamAndCollect(
    input: AgentInput,
    init: AgentInit
  ): Promise<AgentOutput> {
    const response = streamFlow<AgentOutput, AgentStreamChunk, AgentInit>({
      url: ENDPOINT,
      input,
      init,
    });

    let accumulated = '';
    let accumulatedReasoning = '';
    for await (const chunk of response.stream) {
      const mc = chunk?.modelChunk;
      if (!mc) continue;

      for (const part of mc.content || []) {
        if ((part as any).reasoning) {
          // Accumulate reasoning/thinking content
          accumulatedReasoning += (part as any).reasoning;
          setStreamingReasoning(accumulatedReasoning);
        } else if (part.text) {
          // Filesystem middleware injects file contents as text chunks —
          // show as a tool message but don't dump raw content into streaming text
          const fsMeta = (part as any).metadata;
          if (fsMeta?.filesystemMiddlewareTool) {
            const toolName = fsMeta.filesystemMiddlewareTool;
            const filePath = fsMeta.filePath || '';
            if (toolName === 'read_file' && filePath) {
              setMessages((prev) => [
                ...prev,
                { role: 'tool', text: `📖 Reading ${filePath}` },
              ]);
            }
            continue;
          }
          accumulated += part.text;
          setStreamingText(accumulated);
        } else if (part.toolRequest) {
          const tr = part.toolRequest;
          // Skip interrupt tools — they're shown in approval/question dialogs
          // and would appear twice (once in original stream, once on resume)
          if (
            tr.name === 'ask_user' ||
            tr.name === 'write_file' ||
            tr.name === 'search_and_replace' ||
            tr.name === 'read_file'
          )
            continue;
          const msg = formatToolRequest(tr.name, tr.input);
          setMessages((prev) => [...prev, { role: 'tool', ...msg }]);
        } else if (part.toolResponse) {
          const tr = part.toolResponse;
          // Skip read_file response — the "📖 Reading" message already covers it
          if (tr.name === 'read_file') continue;
          const msg = formatToolResponse(tr.name, tr.output);
          setMessages((prev) => [...prev, { role: 'tool', ...msg }]);
        }
      }
    }

    const result = await response.output;
    setStreamingText('');
    setStreamingReasoning('');

    const replyText = extractText(result);
    if (accumulated || replyText || accumulatedReasoning) {
      setMessages((prev) => [
        ...prev,
        {
          role: 'model',
          text: replyText || accumulated || '',
          reasoning: accumulatedReasoning || undefined,
        },
      ]);
    }

    return result;
  }

  // ── Respond to an ask_user interrupt ───────────────────────────────────
  const handleQuestionResponse = useCallback(
    async (answer: string) => {
      if (!question) return;
      const currentQuestion = question;
      setQuestion(null);
      setCustomAnswer('');
      setLoading(true);
      setStreamingText('');
      setStreamingReasoning('');

      setMessages((prev) => [
        ...prev,
        { role: 'system', text: `💬 Answer: ${answer}` },
      ]);

      const init: AgentInit = { snapshotId: currentQuestion.snapshotId };

      // Use the respond pattern — send the tool response as a role='tool'
      // message in input.messages. The tool never executes; we provide
      // the output directly as a ToolResponsePart in the message content.
      const input: AgentInput = {
        messages: [
          {
            role: 'tool' as const,
            content: [
              {
                toolResponse: {
                  name: 'ask_user',
                  ref: currentQuestion.ref,
                  output: { answer },
                },
                metadata: { interruptResponse: true },
              },
            ],
          },
        ],
      };

      try {
        const result = await streamAndCollect(input, init);
        processResult(result);
      } catch (err: unknown) {
        setStreamingText('');
        const errMsg = err instanceof Error ? err.message : String(err);
        setMessages((prev) => [
          ...prev,
          { role: 'system', text: `Error resuming: ${errMsg}` },
        ]);
      } finally {
        setLoading(false);
        refreshFiles();
      }
    },
    [question, refreshFiles]
  );

  // ── Process a result: update session tracking & detect interrupts ────
  function processResult(result: AgentOutput) {
    if (result?.state) stateRef.current = result.state;
    if (result?.snapshotId) snapshotIdRef.current = result.snapshotId;

    // Check for interrupts
    const interrupt = findToolInterrupt(result);
    if (interrupt && result.snapshotId) {
      if (interrupt.name === 'ask_user') {
        // ask_user interrupt — show question dialog
        setQuestion({
          question: interrupt.input?.question || 'What would you like to do?',
          options: interrupt.input?.options || [],
          ref: interrupt.ref,
          snapshotId: result.snapshotId,
        });
      } else {
        // Tool approval interrupt (write_file, search_and_replace, run_shell)
        setApproval({
          toolName: interrupt.name,
          ref: interrupt.ref,
          input: interrupt.input,
          snapshotId: result.snapshotId,
        });
      }
    }
  }

  // ── New session ────────────────────────────────────────────────────────
  const handleNewSession = useCallback(() => {
    setMessages([]);
    setStreamingText('');
    setStreamingReasoning('');
    setApproval(null);
    setQuestion(null);
    setCustomAnswer('');
    stateRef.current = undefined;
    snapshotIdRef.current = undefined;
    refreshFiles();
  }, [refreshFiles]);

  return (
    <div className="coding-agent-layout">
      {/* Main chat panel */}
      <ChatUI
        title="Coding Agent"
        description="AI coding assistant with filesystem access, skills, tool approval, and shell execution."
        suggestions={[
          'I want to build something fun. Ask me to pick 3 quick/simple project ideas.',
          'Create a TypeScript Express hello world app',
          'List the files in the workspace',
          'Create a Python script that generates fibonacci numbers',
        ]}
        messages={messages}
        streamingText={streamingText}
        streamingReasoning={streamingReasoning}
        loading={loading}
        onSend={handleSend}
        inputDisabled={!!approval || !!question}
        renderMarkdown
        headerAction={
          <button className="btn-new-session" onClick={handleNewSession}>
            New Session
          </button>
        }>
        {/* Tool approval dialog */}
        {approval && (
          <div className="approval-dialog">
            <h3>⚠️ Tool Approval Required</h3>
            <div className="approval-tool-name">
              <span className="approval-label">Tool:</span>{' '}
              <code>{approval.toolName}</code>
            </div>

            {approval.toolName === 'write_file' && (
              <div className="approval-details">
                <div className="approval-file-path">
                  <span className="approval-label">File:</span>{' '}
                  <code>{approval.input?.filePath}</code>
                </div>
                <div className="approval-content-preview">
                  <span className="approval-label">Content:</span>
                  <pre className="approval-code">
                    {approval.input?.content || '(empty)'}
                  </pre>
                </div>
              </div>
            )}

            {approval.toolName === 'search_and_replace' && (
              <div className="approval-details">
                <div className="approval-file-path">
                  <span className="approval-label">File:</span>{' '}
                  <code>{approval.input?.filePath}</code>
                </div>
                <div className="approval-content-preview">
                  <span className="approval-label">Edits:</span>
                  {(approval.input?.edits || []).map(
                    (edit: string, i: number) => (
                      <pre key={i} className="approval-code approval-diff">
                        {edit}
                      </pre>
                    )
                  )}
                </div>
              </div>
            )}

            {approval.toolName === 'run_shell' && (
              <div className="approval-details">
                <div className="approval-file-path">
                  <span className="approval-label">Command:</span>{' '}
                  <code className="approval-command">
                    {approval.input?.command}
                  </code>
                </div>
                <div className="approval-content-preview">
                  <div className="approval-warning">
                    🛡️ This shell command was flagged as potentially dangerous
                    by the AI safety gate. Review the command carefully before
                    approving.
                  </div>
                </div>
              </div>
            )}

            <div className="approval-buttons">
              <button
                className="btn btn-approve"
                onClick={() => handleApprovalResponse(true)}>
                ✅ Approve
              </button>
              <button
                className="btn btn-deny"
                onClick={() => handleApprovalResponse(false)}>
                ❌ Deny
              </button>
            </div>
          </div>
        )}

        {/* Ask user question dialog */}
        {question && (
          <div className="ask-user-dialog">
            <h3>❓ Question from Agent</h3>
            <p className="ask-user-question">{question.question}</p>

            <div className="ask-user-options">
              {question.options.map((opt, i) => (
                <button
                  key={i}
                  className="ask-user-option"
                  onClick={() => handleQuestionResponse(opt)}>
                  {opt}
                </button>
              ))}
            </div>

            <div className="ask-user-custom-section">
              <span className="ask-user-custom-label">Or write your own:</span>
              <div className="ask-user-custom-row">
                <input
                  type="text"
                  className="ask-user-custom"
                  placeholder="Type your answer…"
                  value={customAnswer}
                  onChange={(e) => setCustomAnswer(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && customAnswer.trim()) {
                      handleQuestionResponse(customAnswer.trim());
                    }
                  }}
                />
                <button
                  className="btn btn-send"
                  disabled={!customAnswer.trim()}
                  onClick={() => {
                    if (customAnswer.trim()) {
                      handleQuestionResponse(customAnswer.trim());
                    }
                  }}>
                  Send
                </button>
              </div>
            </div>
          </div>
        )}
      </ChatUI>

      {/* File explorer sidebar */}
      <aside className="file-explorer">
        <div className="file-explorer-header">
          <h3>📁 Workspace</h3>
          <button
            className="btn-refresh-files"
            onClick={refreshFiles}
            title="Refresh file list">
            🔄
          </button>
        </div>

        {files.length === 0 ? (
          <p className="file-explorer-empty">Workspace is empty.</p>
        ) : (
          <div className="file-tree">
            <FileTree
              files={files}
              selectedFile={selectedFile}
              onSelect={viewFile}
            />
          </div>
        )}

        {/* File content viewer */}
        {selectedFile && (
          <div className="file-viewer">
            <div className="file-viewer-header">
              <span className="file-viewer-path">{selectedFile}</span>
              <button
                className="file-viewer-close"
                onClick={() => {
                  setSelectedFile(null);
                  setFileContent('');
                }}>
                ✕
              </button>
            </div>
            <pre className="file-viewer-content">
              {fileLoading ? 'Loading…' : fileContent}
            </pre>
          </div>
        )}
      </aside>
    </div>
  );
}

// ---------------------------------------------------------------------------
// File Tree component
// ---------------------------------------------------------------------------

function FileTree({
  files,
  selectedFile,
  onSelect,
  depth = 0,
}: {
  files: WorkspaceFile[];
  selectedFile: string | null;
  onSelect: (path: string) => void;
  depth?: number;
}) {
  return (
    <>
      {files.map((f) => (
        <div key={f.path}>
          {f.type === 'directory' ? (
            <DirectoryNode
              file={f}
              selectedFile={selectedFile}
              onSelect={onSelect}
              depth={depth}
            />
          ) : (
            <button
              className={`file-tree-item ${selectedFile === f.path ? 'selected' : ''}`}
              style={{ paddingLeft: `${12 + depth * 16}px` }}
              onClick={() => onSelect(f.path)}>
              <span className="file-icon">📄</span>
              <span className="file-name">{f.name}</span>
            </button>
          )}
        </div>
      ))}
    </>
  );
}

function DirectoryNode({
  file,
  selectedFile,
  onSelect,
  depth,
}: {
  file: WorkspaceFile;
  selectedFile: string | null;
  onSelect: (path: string) => void;
  depth: number;
}) {
  const [expanded, setExpanded] = useState(false);

  return (
    <>
      <button
        className="file-tree-item file-tree-dir"
        style={{ paddingLeft: `${12 + depth * 16}px` }}
        onClick={() => setExpanded(!expanded)}>
        <span className="file-icon">{expanded ? '📂' : '📁'}</span>
        <span className="file-name">{file.name}</span>
      </button>
      {expanded && file.children && (
        <FileTree
          files={file.children}
          selectedFile={selectedFile}
          onSelect={onSelect}
          depth={depth + 1}
        />
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// Tool display formatters
// ---------------------------------------------------------------------------

/** Truncate a string, adding ellipsis if it exceeds maxLen. */
function truncate(s: string, maxLen = 200): string {
  return s.length > maxLen ? s.substring(0, maxLen) + '…' : s;
}

/** Show first N lines of content with a "(+X more lines)" note. */
function previewLines(content: string, maxLines = 20): string {
  const lines = content.split('\n');
  if (lines.length <= maxLines) return content;
  return (
    lines.slice(0, maxLines).join('\n') +
    `\n… (+${lines.length - maxLines} more lines)`
  );
}

interface ToolMsg {
  text: string;
  detail?: string;
}

/** Pretty-format a tool request for inline display. */
function formatToolRequest(name: string, input: any): ToolMsg {
  switch (name) {
    case 'write_file':
      return {
        text: `📝 Writing ${input?.filePath || 'file'}`,
        detail: previewLines(input?.content || '', 20),
      };

    case 'read_file':
      return { text: `📖 Reading ${input?.filePath || 'file'}` };

    case 'list_files':
      return { text: `📁 Listing files in ${input?.directory || '/'}` };

    case 'search_and_replace': {
      const file = input?.filePath || 'file';
      const edits: any[] = input?.edits || [];
      if (edits.length === 0) return { text: `✏️ Editing ${file}` };
      const diffPreview = edits
        .slice(0, 3)
        .map((e: any) => {
          if (typeof e === 'string') return truncate(e, 150);
          const search = truncate(String(e.search || ''), 80);
          const replace = truncate(String(e.replace || ''), 80);
          return `"${search}" → "${replace}"`;
        })
        .join('\n');
      const moreNote =
        edits.length > 3 ? `\n… (+${edits.length - 3} more edits)` : '';
      return { text: `✏️ Editing ${file}`, detail: diffPreview + moreNote };
    }

    case 'run_shell':
      return { text: `🖥️ $ ${input?.command || '(unknown command)'}` };

    case 'use_skill':
      return { text: `📚 Loading skill: ${input?.skillName || '(unknown)'}` };

    default: {
      const inputStr =
        typeof input === 'object'
          ? truncate(JSON.stringify(input), 300)
          : truncate(String(input ?? ''), 300);
      return { text: `🔧 ${name}`, detail: inputStr };
    }
  }
}

/** Pretty-format a tool response for inline display. */
function formatToolResponse(name: string, output: any): ToolMsg {
  const outputStr =
    typeof output === 'string' ? output : JSON.stringify(output);

  switch (name) {
    case 'write_file':
      return { text: '✅ File written' };

    case 'read_file':
      return { text: '✅ File content:', detail: previewLines(outputStr, 15) };

    case 'list_files': {
      // Parse JSON array and show a nice file list
      let fileList = outputStr;
      try {
        const files = typeof output === 'string' ? JSON.parse(output) : output;
        if (Array.isArray(files)) {
          fileList = files
            .map((f: any) => `${f.isDirectory ? '📁' : '📄'} ${f.path}`)
            .join('\n');
        }
      } catch {
        /* use raw string */
      }
      return { text: '✅ Files:', detail: fileList || '(empty)' };
    }

    case 'search_and_replace':
      return { text: '✅ Edits applied' };

    case 'run_shell': {
      // Extract stdout/stderr from structured output
      let shellText = outputStr;
      if (typeof output === 'object' && output !== null) {
        const parts: string[] = [];
        if (output.stdout) parts.push(output.stdout);
        if (output.stderr) parts.push(`(stderr) ${output.stderr}`);
        if (output.exitCode !== undefined && output.exitCode !== 0)
          parts.push(`Exit code: ${output.exitCode}`);
        shellText = parts.join('\n') || '(no output)';
      }
      return { text: '✅ Shell output:', detail: truncate(shellText, 500) };
    }

    case 'use_skill':
      return { text: '✅ Skill loaded' };

    default:
      return { text: `✅ ${name}`, detail: truncate(outputStr, 400) };
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function extractText(result: AgentOutput): string {
  if (!result) return '';
  const msg = result.message;
  if (!msg) return '';
  const parts: string[] = [];
  for (const p of msg.content || []) {
    if (p.text) parts.push(p.text);
    // Skip interrupt tool requests — they're handled by the question/approval dialogs
    if (
      p.toolRequest &&
      p.toolRequest.name !== 'ask_user' &&
      p.toolRequest.name !== 'write_file' &&
      p.toolRequest.name !== 'search_and_replace' &&
      p.toolRequest.name !== 'run_shell'
    ) {
      parts.push(
        `[Tool Request: ${p.toolRequest.name}]\n${JSON.stringify(p.toolRequest.input, null, 2)}`
      );
    }
  }
  return parts.join('');
}

function findToolInterrupt(
  result: AgentOutput
): { name: string; ref?: string; input: any } | null {
  const msg = result?.message;
  if (!msg) return null;
  for (const p of msg.content || []) {
    if (p.toolRequest) {
      const tr = p.toolRequest as ToolRequest;
      // These tools can trigger interrupts:
      //   • write_file, search_and_replace — from toolApproval middleware
      //   • run_shell — from the AI safety gate in the tool handler
      //   • ask_user — always interrupts (defineInterrupt)
      if (
        tr.name === 'write_file' ||
        tr.name === 'search_and_replace' ||
        tr.name === 'run_shell' ||
        tr.name === 'ask_user'
      ) {
        return { name: tr.name, ref: tr.ref, input: tr.input };
      }
    }
  }
  return null;
}
