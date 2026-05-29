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
  selector: 'app-background-agent',
  imports: [FormsModule],
  template: `
    <h2>Background research agent</h2>
    <p class="desc">
      Submit a topic with <code>detach: true</code>. The session
      auto-transitions to <code>phase: 'background'</code> on the
      in-stream <code>detached</code> event and polls the snapshot
      endpoint internally. Same backend
      (<code>/api/backgroundAgent</code>) as the React testapp.
    </p>

    @if (state().phase === 'idle' || state().phase === 'streaming') {
      <form (submit)="onSubmit($event)">
        <textarea
          [(ngModel)]="topic"
          name="topic"
          rows="3"
          [disabled]="state().phase === 'streaming'"
          placeholder="e.g. The impact of quantum computing on cybersecurity"
        ></textarea>
        <button type="submit"
          [disabled]="!topic().trim() || state().phase === 'streaming'">
          {{ state().phase === 'streaming' ? 'Submitting…' : '🚀 Generate report (background)' }}
        </button>
      </form>
    }

    @if (state().phase === 'background') {
      <div class="status">
        <h3>⏳ Processing in background…</h3>
        <p class="meta">
          The session is polling the snapshot endpoint internally;
          messages update reactively as the snapshot evolves.
        </p>
        <p><code>snapshotId: {{ state().snapshotId }}</code></p>
        @if (state().status?.label) {
          <p>Status: {{ state().status?.label }}</p>
        }
        <button class="abort" (click)="agent.abort()">✋ Abort</button>
      </div>
    }

    @if (state().phase === 'done') {
      <div class="result">
        <div class="result-header">
          <span class="badge done">✅ Complete</span>
          <code>{{ state().snapshotId }}</code>
          <button class="reset" (click)="agent.reset(); topic.set('')">
            New report
          </button>
        </div>
        <div class="report">{{ report() }}</div>
      </div>
    }

    @if (state().phase === 'error' && state().error) {
      <div class="error">
        <span class="badge failed">❌ Failed</span>
        <p>{{ state().error?.message }}</p>
      </div>
    }
  `,
  styles: [`
    h2 { margin: 0 0 4px; }
    .desc { color: #667085; font-size: 13px; margin: 0 0 20px; }
    form { display: flex; flex-direction: column; gap: 10px; max-width: 720px; }
    textarea { padding: 12px; border-radius: 8px; border: 1px solid #cdd5e0; font-size: 14px; font-family: inherit; }
    button { padding: 10px 16px; border-radius: 8px; border: 0; background: #3158d6; color: #fff; font-weight: 600; cursor: pointer; align-self: flex-start; }
    button:disabled { background: #98a2b3; cursor: not-allowed; }
    .status { background: #fffbeb; border: 1px solid #f59e0b; border-radius: 10px; padding: 20px; max-width: 720px; }
    .status h3 { margin: 0 0 8px; }
    .status .meta { color: #667085; font-size: 13px; }
    .status code { background: #fff; padding: 2px 6px; border-radius: 4px; font-size: 12px; }
    .abort { background: #dc2626; margin-top: 12px; }
    .result { max-width: 720px; }
    .result-header { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
    .badge { padding: 4px 10px; border-radius: 999px; font-size: 12px; font-weight: 600; }
    .badge.done { background: #d1fae5; color: #065f46; }
    .badge.failed { background: #fee2e2; color: #991b1b; }
    .reset { background: #6b7280; margin-left: auto; }
    .report { background: #fff; padding: 20px; border-radius: 10px; border: 1px solid #e4e7ec; white-space: pre-wrap; line-height: 1.6; }
    .error { background: #fee2e2; border: 1px solid #fecaca; padding: 16px; border-radius: 10px; max-width: 720px; }
  `],
})
export class BackgroundAgentComponent {
  readonly topic = signal('');

  readonly agent = injectGenkitAgent<unknown, { label: string }>({
    url: '/api/backgroundAgent',
  });
  readonly state = this.agent.state;

  // Pull the final report text out of the agent's messages.
  readonly report = computed<string>(() => {
    const models = this.state().messages.filter((m) => m.role === 'model');
    const last = models[models.length - 1];
    if (!last) return '';
    return (last.content ?? [])
      .filter((p) => p['text'])
      .map((p) => String(p['text']))
      .join('');
  });

  onSubmit(e: Event) {
    e.preventDefault();
    const t = this.topic().trim();
    if (!t) return;
    this.agent.submit({
      messages: [{ role: 'user', content: [{ text: t }] }],
      detach: true,
    });
  }
}
