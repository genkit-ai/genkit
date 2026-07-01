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

import { NavLink, Outlet } from 'react-router-dom';

// ---------------------------------------------------------------------------
// App shell — sidebar nav + content outlet for each demo page.
// ---------------------------------------------------------------------------

const NAV_ITEMS = [
  { to: '/coding-agent', icon: '💻', label: 'Coding Agent' },
  { to: '/weather', icon: '🌤️', label: 'Weather Chat' },
  { to: '/client-state', icon: '🌤️', label: 'Weather Chat (Stateless)' },
  { to: '/banking', icon: '🏦', label: 'Banking (Interrupt)' },
  { to: '/workspace', icon: '🛠️', label: 'Workspace Builder' },
  { to: '/background', icon: '⏳', label: 'Background (Detach)' },
  { to: '/branching', icon: '🔀', label: 'Branching (Variants)' },
  { to: '/tasks', icon: '✅', label: 'Task Tracker (Custom State)' },
  { to: '/research', icon: '🔬', label: 'Research (Custom Agent)' },
  { to: '/subagents', icon: '🤝', label: 'Sub-Agent Delegation' },
  { to: '/trip-planner', icon: '✈️', label: 'Trip Planner (Prompt File)' },
];

export default function App() {
  return (
    <div className="app">
      <aside className="sidebar">
        <h1 className="sidebar-title">🔥 Genkit Agents</h1>
        <p className="sidebar-subtitle">Genkit Agent Demos</p>

        <nav className="nav-list">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }: { isActive: boolean }) =>
                `nav-item ${isActive ? 'active' : ''}`
              }>
              <span className="nav-icon">{item.icon}</span>
              <span className="nav-label">{item.label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <span className="sidebar-hint">
            Each page is a self-contained sample showing how to use the{' '}
            <code>genkit/beta/client</code> library.
          </span>
        </div>
      </aside>

      <main className="main">
        <Outlet />
      </main>
    </div>
  );
}
