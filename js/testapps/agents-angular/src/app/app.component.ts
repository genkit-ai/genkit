/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

import { Component } from '@angular/core';
import { RouterLink, RouterLinkActive, RouterOutlet } from '@angular/router';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, RouterLink, RouterLinkActive],
  template: `
    <nav class="sidebar">
      <h1>Genkit Agents (Angular)</h1>
      <p class="subtitle">
        Same backend, same <code>AgentSession</code>, Angular front-end.
      </p>
      <ul>
        <li><a routerLink="/weather" routerLinkActive="active">Weather</a></li>
        <li><a routerLink="/banking" routerLinkActive="active">Banking interrupt</a></li>
        <li><a routerLink="/background" routerLinkActive="active">Background</a></li>
      </ul>
    </nav>
    <main>
      <router-outlet />
    </main>
  `,
  styles: [`
    :host { display: flex; height: 100vh; font-family: system-ui, sans-serif; }
    .sidebar { width: 240px; background: #1c1f26; color: #e3e6ec; padding: 24px 20px; box-sizing: border-box; }
    .sidebar h1 { font-size: 16px; margin: 0 0 6px; }
    .sidebar .subtitle { font-size: 12px; color: #98a2b3; margin: 0 0 24px; }
    .sidebar ul { list-style: none; padding: 0; margin: 0; }
    .sidebar li { margin: 4px 0; }
    .sidebar a { color: #cdd5e0; text-decoration: none; display: block; padding: 8px 10px; border-radius: 6px; font-size: 14px; }
    .sidebar a:hover { background: #262a33; }
    .sidebar a.active { background: #3158d6; color: #fff; }
    main { flex: 1; overflow: auto; padding: 32px; background: #f5f7fa; }
  `],
})
export class AppComponent {}
