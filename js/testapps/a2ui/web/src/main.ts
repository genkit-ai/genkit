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
 * A2UI testapp frontend.
 *
 * Talks to the Genkit A2UI agent with `remoteAgent` from `genkit/beta/client`,
 * renders prose in a simple chat log, and renders each A2UI surface with the
 * `@a2ui/lit` renderer driven by `@a2ui/web_core`'s `MessageProcessor`.
 *
 * A2UI travels as `data` parts on the agent stream; we pull them off each chunk
 * with `a2uiEnvelopes` (from `@genkit-ai/a2ui/client`) and feed the whole
 * envelopes to the renderer. Surface actions (e.g. button presses) are sent back
 * to the agent as the next turn.
 */

import { basicCatalog, Context } from '@a2ui/lit/v0_9';
// Importing the v0_9 entry registers <a2ui-surface> and all basic components.
import '@a2ui/lit/v0_9';
import { renderMarkdown } from '@a2ui/markdown-it';
import { MessageProcessor } from '@a2ui/web_core/v0_9';
import { injectBasicCatalogStyles } from '@a2ui/web_core/v0_9/basic_catalog';
import {
  a2uiEnvelopes,
  actionToMessage,
  type A2uiClientAction,
} from '@genkit-ai/a2ui/client';
import { ContextProvider } from '@lit/context';
import { remoteAgent, type AgentChat } from 'genkit/beta/client';
import './style.css';

// Inject the basic catalog's default styles into the document.
injectBasicCatalogStyles();

// The basic catalog's `Text` component turns a heading `variant` into Markdown
// (h2 -> "## ...") and renders it via a MarkdownRenderer pulled from Lit
// context. Provide one (backed by `@a2ui/markdown-it`) on <body> so every
// `<a2ui-surface>` inherits it; without it, headings render as literal "##".
//
// `markdown-it` always appends a trailing newline to its HTML output. Because
// the `Text` host is `display: inline-block`, that whitespace renders as a
// stray selectable blank line *and* inflates the Text box, which throws off the
// `align-items: center` vertical alignment of icons next to titles/labels in
// Rows and Buttons. Trim the output so every Text is exactly its visible size.
const renderMarkdownTrimmed = ((value: string, options: unknown) =>
  renderMarkdown(value, options as any).then((html) =>
    html.trim()
  )) as typeof renderMarkdown;

new ContextProvider(document.body as any, {
  context: Context.markdown,
  initialValue: renderMarkdownTrimmed,
});

// A typed client for the agent. `remoteAgent` manages the session id for us, so
// a single `chat` keeps the whole conversation server-side.
const agent = remoteAgent({ url: '/api/uiAgent' });
const chat: AgentChat = agent.chat();

const app = document.getElementById('app')!;
app.innerHTML = `
  <header>
    <h1>Genkit + A2UI</h1>
    <p class="subtitle">
      An agent that streams generative UI surfaces. Try:
      <button class="chip" data-prompt="What's the weather in Tokyo?">weather in Tokyo</button>
      <button class="chip" data-prompt="Compare the weather in London, Paris and Rome.">compare 3 cities</button>
      <button class="chip" data-prompt="Give me a short signup form (name and email) with a submit button.">a signup form</button>
    </p>
  </header>
  <main id="log" class="log"></main>
  <form id="composer" class="composer">
    <input id="input" type="text" autocomplete="off"
      placeholder="Ask for something visual…" />
    <button id="send" type="submit">Send</button>
  </form>
`;

const log = document.getElementById('log') as HTMLDivElement;
const form = document.getElementById('composer') as HTMLFormElement;
const input = document.getElementById('input') as HTMLInputElement;
const sendBtn = document.getElementById('send') as HTMLButtonElement;

/** Shared message processor + a place to route surface actions back to the agent. */
const processor = new MessageProcessor([basicCatalog], (action) => {
  onSurfaceAction(action as unknown as A2uiClientAction);
});

// When a surface is (re)created, drop its renderer into the newest agent bubble.
let pendingSurfaceSlot: HTMLDivElement | null = null;
processor.onSurfaceCreated((surface: any) => {
  const slot = pendingSurfaceSlot ?? newAgentBubble();
  pendingSurfaceSlot = null;
  slot.querySelector('a2ui-surface')?.remove();
  const el = document.createElement('a2ui-surface') as any;
  el.surface = surface;
  slot.appendChild(el);
  scrollToBottom();
});

function addBubble(role: 'user' | 'agent'): HTMLDivElement {
  const bubble = document.createElement('div');
  bubble.className = `bubble ${role}`;
  log.appendChild(bubble);
  scrollToBottom();
  return bubble;
}

function newAgentBubble(): HTMLDivElement {
  return addBubble('agent');
}

function scrollToBottom() {
  log.scrollTop = log.scrollHeight;
}

async function send(text: string) {
  if (!text.trim()) return;
  input.value = '';
  setBusy(true);

  const userBubble = addBubble('user');
  userBubble.textContent = text;

  await runTurn(text);
  setBusy(false);
}

/** Runs a single agent turn, streaming prose + surfaces into the log. */
async function runTurn(message: string | Record<string, unknown>) {
  const agentBubble = newAgentBubble();
  const prose = document.createElement('div');
  prose.className = 'prose';
  agentBubble.appendChild(prose);
  // Any surface created during this turn renders into this bubble.
  pendingSurfaceSlot = agentBubble;

  try {
    const turn = chat.sendStream(message as any);
    for await (const chunk of turn.stream) {
      if (chunk.text) {
        prose.textContent += chunk.text;
        scrollToBottom();
      }
      // A2UI rides as data parts on the raw chunk; extract whole envelopes.
      const envelopes = a2uiEnvelopes(chunk.raw);
      if (envelopes.length > 0) {
        processor.processMessages(envelopes as any);
      }
    }
    await turn.response;
  } catch (err) {
    prose.classList.add('error');
    prose.textContent = `Error: ${(err as Error).message}`;
  }
  if (!prose.textContent) prose.remove();
}

/** A surface fired an action (e.g. button press): send it back to the agent. */
async function onSurfaceAction(action: A2uiClientAction) {
  const label = addBubble('user');
  label.classList.add('action');
  label.textContent = `▶ ${action.name}`;
  setBusy(true);
  // Wrap the action's message as an AgentInput (`{ message }`).
  await runTurn({ message: actionToMessage(action) } as Record<
    string,
    unknown
  >);
  setBusy(false);
}

function setBusy(busy: boolean) {
  input.disabled = busy;
  sendBtn.disabled = busy;
  if (!busy) input.focus();
}

form.addEventListener('submit', (e) => {
  e.preventDefault();
  void send(input.value);
});

document.querySelectorAll('.chip').forEach((chip) => {
  chip.addEventListener('click', () => {
    void send((chip as HTMLElement).dataset.prompt || '');
  });
});

input.focus();
