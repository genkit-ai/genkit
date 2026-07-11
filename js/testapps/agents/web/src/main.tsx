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

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import App from './App';
import './App.css';

// Lazy-load each page so the imports stay self-contained.
import BackgroundAgent from './pages/BackgroundAgent';
import BankingInterrupt from './pages/BankingInterrupt';
import BranchingChat from './pages/BranchingChat';
import ClientState from './pages/ClientState';
import CodingAgent from './pages/CodingAgent';
import ResearchAgent from './pages/ResearchAgent';
import SubAgentChat from './pages/SubAgentChat';
import TaskTracker from './pages/TaskTracker';
import TripPlanner from './pages/TripPlanner';
import WeatherChat from './pages/WeatherChat';
import WorkspaceBuilder from './pages/WorkspaceBuilder';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<App />}>
          <Route index element={<Navigate to="/weather" replace />} />
          <Route path="weather" element={<WeatherChat />} />
          <Route path="weather/:snapshotId" element={<WeatherChat />} />
          <Route path="client-state" element={<ClientState />} />
          <Route path="banking" element={<BankingInterrupt />} />
          <Route path="workspace" element={<WorkspaceBuilder />} />
          <Route path="background" element={<BackgroundAgent />} />
          <Route path="branching" element={<BranchingChat />} />
          <Route path="branching/:snapshotId" element={<BranchingChat />} />
          <Route path="tasks" element={<TaskTracker />} />
          <Route path="research" element={<ResearchAgent />} />
          <Route path="subagents" element={<SubAgentChat />} />
          <Route path="trip-planner" element={<TripPlanner />} />
          <Route path="trip-planner/:snapshotId" element={<TripPlanner />} />
          <Route path="coding-agent" element={<CodingAgent />} />
          <Route path="coding-agent/:snapshotId" element={<CodingAgent />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>
);
