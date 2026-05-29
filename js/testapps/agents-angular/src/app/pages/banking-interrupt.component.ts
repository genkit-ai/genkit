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

@Component({
  selector: 'app-banking-interrupt',
  imports: [FormsModule],
  template: `
    <h2>Banking agent (in-stream interrupt)</h2>
    <p class="desc">
      Sensitive operations trigger an in-stream
      <code>interrupt</code> event. Same backend
      (<code>/api/bankingAgent</code>) as the React testapp.
    </p>

    @for (m of displayed(); track $index) {
      <div class="msg msg-{{ m.role }}">
        <span class="role">{{ m.role }}</span>
        <span class="text">{{ m.text }}</span>
      </div>
    }

    @if (state().streamingText) {
      <div class="msg msg-model">
        <span class="text">{{ state().streamingText }}<span class="cursor">▊</span></span>
      </div>
    }

    @if (state().pendingInterrupt; as pi) {
      <div class="interrupt">
        <h3>⚠️ Approval required</h3>
        <div><strong>Tool:</strong> <code>{{ pi.toolName }}</code></div>
        <pre>{{ inputJson(pi.input) }}</pre>
        <div class="actions">
          <button class="approve" (click)="agent.respondToInterrupt({ approved: true })">
            ✅ Approve
          </button>
          <button class="deny" (click)="agent.respondToInterrupt({ approved: false })">
            ❌ Deny
          </button>
        </div>
      </div>
    }

    @if (!state().pendingInterrupt) {
      <form class="composer" (submit)="onSubmit($event)">
        <input
          type="text"
          [(ngModel)]="draft"
          name="draft"
          [disabled]="state().phase === 'streaming'"
          placeholder="Try: Transfer $5000 from checking to savings" />
        <button type="submit" [disabled]="!draft() || state().phase === 'streaming'">
          Send
        </button>
      </form>
    }

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
    .msg-system { color: #667085; font-style: italic; }
    .cursor { animation: blink 1s steps(2) infinite; }
    @keyframes blink { 50% { opacity: 0; } }
    .interrupt { border: 2px solid #f59e0b; background: #fffbeb; padding: 16px; border-radius: 10px; margin: 16px 0; }
    .interrupt h3 { margin: 0 0 8px; }
    .interrupt pre { background: #fff; padding: 10px; border-radius: 6px; font-size: 13px; overflow-x: auto; }
    .actions { display: flex; gap: 8px; margin-top: 12px; }
    .approve, .deny { padding: 8px 16px; border: 0; border-radius: 6px; font-weight: 600; cursor: pointer; color: #fff; }
    .approve { background: #16a34a; }
    .deny { background: #dc2626; }
    .composer { display: flex; gap: 8px; margin: 20px 0; }
    .composer input { flex: 1; padding: 10px 12px; border-radius: 8px; border: 1px solid #cdd5e0; font-size: 14px; }
    .composer button { padding: 0 16px; border-radius: 8px; border: 0; background: #3158d6; color: #fff; font-weight: 600; cursor: pointer; }
    .composer button:disabled { background: #98a2b3; cursor: not-allowed; }
    .suggestions { display: flex; gap: 8px; flex-wrap: wrap; }
    .chip { padding: 6px 12px; border-radius: 999px; border: 1px solid #d0d5dd; background: #fff; cursor: pointer; font-size: 13px; }
    .chip:hover { background: #f1f5f9; }
  `],
})
export class BankingInterruptComponent {
  readonly draft = signal('');
  readonly suggestions = [
    'Transfer $500 to my savings account',
    'Transfer $5000 from checking to savings',
    'What is my account balance?',
  ];

  readonly agent = injectGenkitAgent({ url: '/api/bankingAgent' });
  readonly state = this.agent.state;

  readonly displayed = computed(() => {
    const out: Array<{ role: string; text: string }> = [];
    for (const m of this.state().messages) {
      const textParts = (m.content || [])
        .filter((p) => p['text'])
        .map((p) => String(p['text']))
        .join('');
      if (textParts) out.push({ role: m.role, text: textParts });
    }
    return out;
  });

  inputJson(input: unknown): string {
    return JSON.stringify(input, null, 2);
  }

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
