/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

import { Component, computed, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { injectGenkitAgent } from '../genkit-angular';

interface DisplayRow {
  role: 'user' | 'model' | 'tool';
  text: string;
}

@Component({
  selector: 'app-weather-chat',
  imports: [FormsModule],
  template: `
    <h2>Weather agent</h2>
    <p class="desc">
      Multi-turn chat with tool calling. Same backend
      (<code>/api/weatherAgent</code>) as the React testapp.
    </p>

    @for (m of displayed(); track $index) {
      <div class="msg msg-{{ m.role }}">
        <span class="role">{{ m.role }}</span>
        <span class="text">{{ m.text }}</span>
      </div>
    }

    @if (state().streamingText) {
      <div class="msg msg-model streaming">
        <span class="role">model</span>
        <span class="text">{{ state().streamingText }}<span class="cursor">▊</span></span>
      </div>
    }

    @if (state().phase === 'streaming' && !state().streamingText) {
      <div class="msg msg-system">
        <span class="text">Thinking…</span>
      </div>
    }

    <form class="composer" (submit)="onSubmit($event)">
      <input
        type="text"
        [(ngModel)]="draft"
        name="draft"
        [disabled]="state().phase === 'streaming'"
        placeholder="Ask about the weather…" />
      <button type="submit" [disabled]="!draft() || state().phase === 'streaming'">
        Send
      </button>
    </form>

    <div class="suggestions">
      @for (s of suggestions; track s) {
        <button class="chip" (click)="ask(s)">{{ s }}</button>
      }
    </div>
  `,
  styles: [`
    h2 { margin: 0 0 4px; }
    .desc { color: #667085; font-size: 13px; margin: 0 0 20px; }
    .msg { padding: 10px 14px; border-radius: 8px; margin: 8px 0; max-width: 720px; background: #fff; border: 1px solid #e4e7ec; }
    .role { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; color: #98a2b3; margin-right: 8px; }
    .text { white-space: pre-wrap; }
    .msg-user { background: #eaf0ff; border-color: #cdddff; }
    .msg-tool { background: #f1f5f9; font-family: ui-monospace, monospace; font-size: 13px; }
    .msg-system { color: #667085; font-style: italic; }
    .streaming .cursor { animation: blink 1s steps(2) infinite; }
    @keyframes blink { 50% { opacity: 0; } }
    .composer { display: flex; gap: 8px; margin: 20px 0; }
    .composer input { flex: 1; padding: 10px 12px; border-radius: 8px; border: 1px solid #cdd5e0; font-size: 14px; }
    .composer button { padding: 0 16px; border-radius: 8px; border: 0; background: #3158d6; color: #fff; font-weight: 600; cursor: pointer; }
    .composer button:disabled { background: #98a2b3; cursor: not-allowed; }
    .suggestions { display: flex; gap: 8px; flex-wrap: wrap; }
    .chip { padding: 6px 12px; border-radius: 999px; border: 1px solid #d0d5dd; background: #fff; cursor: pointer; font-size: 13px; }
    .chip:hover { background: #f1f5f9; }
  `],
})
export class WeatherChatComponent {
  readonly draft = signal('');
  readonly suggestions = [
    'What is the weather like in London?',
    'Is it sunny in Tokyo right now?',
    'Compare the weather in Paris and New York.',
  ];

  private readonly agent = injectGenkitAgent({ url: '/api/weatherAgent' });
  readonly state = this.agent.state;

  // Project agent.messages + in-flight toolCalls into the display rows.
  readonly displayed = computed<DisplayRow[]>(() => {
    const out: DisplayRow[] = [];
    for (const m of this.state().messages) {
      const role = m.role as DisplayRow['role'];
      const textParts = (m.content || [])
        .filter((p) => p['text'])
        .map((p) => String(p['text']))
        .join('');
      if (textParts) out.push({ role, text: textParts });
      for (const p of m.content || []) {
        if (p.toolRequest) {
          out.push({
            role: 'tool',
            text: `🔧 ${p.toolRequest.name}(${JSON.stringify(p.toolRequest.input)})`,
          });
        }
        if (p.toolResponse) {
          out.push({
            role: 'tool',
            text: `✅ ${p.toolResponse.name} → ${JSON.stringify(p.toolResponse.output)}`,
          });
        }
      }
    }
    // Live tool calls during streaming.
    if (this.state().phase === 'streaming') {
      for (const tc of this.state().toolCalls) {
        if (tc.state === 'call') {
          out.push({ role: 'tool', text: `🔧 Calling ${tc.name}…` });
        } else if (tc.state === 'error') {
          out.push({
            role: 'tool',
            text: `❌ ${tc.name} failed: ${tc.errorText ?? ''}`,
          });
        }
      }
    }
    return out;
  });

  ask(text: string) {
    this.agent.submit({
      messages: [{ role: 'user', content: [{ text }] }],
    });
  }

  onSubmit(e: Event) {
    e.preventDefault();
    const text = this.draft().trim();
    if (!text) return;
    this.draft.set('');
    this.ask(text);
  }
}
