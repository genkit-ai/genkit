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
import ResearchAgent from './pages/ResearchAgent';
import SubAgentChat from './pages/SubAgentChat';
import TaskTracker from './pages/TaskTracker';
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
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>
);
