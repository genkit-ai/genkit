import { Route, Routes } from 'react-router-dom';
import { Layout } from './components/Layout';
import { FeaturePage } from './pages/FeaturePage';
import { OverviewPage } from './pages/OverviewPage';
import { TraceViewerPage } from './pages/TraceViewerPage';

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<OverviewPage />} />
        <Route path="/feature/:featureName" element={<FeaturePage />} />
        <Route path="/trace/:traceId" element={<TraceViewerPage />} />
      </Route>
    </Routes>
  );
}
