import { RequestsChart } from '../components/charts/RequestsChart';
import { SuccessRateChart } from '../components/charts/SuccessRateChart';
import { LatencyChart } from '../components/charts/LatencyChart';
import { FeatureTable } from '../components/FeatureTable';
import { useDashboard } from '../contexts/DashboardContext';
import { AlertTriangle } from 'lucide-react';

export function OverviewPage() {
  const { projectId } = useDashboard();

  if (!projectId) {
    return (
      <div className="flex flex-col items-center justify-center py-24 text-center">
        <AlertTriangle className="mb-4 h-10 w-10 text-accent-orange" />
        <h2 className="mb-2 text-lg font-semibold text-text-primary">
          No project selected
        </h2>
        <p className="text-sm text-text-secondary">
          Select a GCP project using the dropdown in the header to view
          monitoring data.
        </p>
      </div>
    );
  }

  return (
    <div>
      {/* Charts row */}
      <div className="mb-8 grid grid-cols-1 gap-4 md:grid-cols-3">
        <RequestsChart />
        <SuccessRateChart />
        <LatencyChart />
      </div>

      {/* Feature table */}
      <div>
        <h3 className="section-title mb-3">
          Features
        </h3>
        <FeatureTable />
      </div>
    </div>
  );
}
