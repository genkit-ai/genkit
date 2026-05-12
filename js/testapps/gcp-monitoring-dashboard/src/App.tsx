import { Route, Routes } from 'react-router-dom';
import { Layout } from './components/Layout';
import { ErrorBoundary } from './components/ErrorBoundary';
import { AuthGate } from './components/AuthGate';
import { DashboardProvider } from './contexts/DashboardContext';
import { FeaturePage } from './pages/FeaturePage';
import { OverviewPage } from './pages/OverviewPage';
import { TraceViewerPage } from './pages/TraceViewerPage';

export default function App() {
  return (
    <ErrorBoundary>
      <AuthGate>
        <DashboardProvider>
          <Routes>
            <Route element={<Layout />}>
              <Route
                path="/"
                element={
                  <ErrorBoundary>
                    <OverviewPage />
                  </ErrorBoundary>
                }
              />
              <Route
                path="/feature/:featureName"
                element={
                  <ErrorBoundary>
                    <FeaturePage />
                  </ErrorBoundary>
                }
              />
              <Route
                path="/trace/:traceId"
                element={
                  <ErrorBoundary>
                    <TraceViewerPage />
                  </ErrorBoundary>
                }
              />
            </Route>
          </Routes>
        </DashboardProvider>
      </AuthGate>
    </ErrorBoundary>
  );
}
