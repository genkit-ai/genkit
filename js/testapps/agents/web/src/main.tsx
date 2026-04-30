import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import App from './App';
import './App.css';

// Lazy-load each page so the imports stay self-contained.
import WeatherChat from './pages/WeatherChat';
import Translator from './pages/Translator';
import ClientState from './pages/ClientState';
import BankingInterrupt from './pages/BankingInterrupt';
import WorkspaceBuilder from './pages/WorkspaceBuilder';
import BackgroundAgent from './pages/BackgroundAgent';
import BranchingChat from './pages/BranchingChat';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route element={<App />}>
          <Route index element={<Navigate to="/weather" replace />} />
          <Route path="weather" element={<WeatherChat />} />
          <Route path="weather/:snapshotId" element={<WeatherChat />} />
          <Route path="translator" element={<Translator />} />
          <Route path="client-state" element={<ClientState />} />
          <Route path="banking" element={<BankingInterrupt />} />
          <Route path="workspace" element={<WorkspaceBuilder />} />
          <Route path="background" element={<BackgroundAgent />} />
          <Route path="branching" element={<BranchingChat />} />
          <Route path="branching/:snapshotId" element={<BranchingChat />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>
);
