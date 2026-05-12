import { ArrowLeft, Loader2 } from 'lucide-react';
import { Link, useParams } from 'react-router-dom';
import { useMetricsOverview } from '../hooks/useMetrics';
import { StabilityMetrics } from '../components/StabilityMetrics';
import { RequestsChart } from '../components/charts/RequestsChart';
import { SuccessRateChart } from '../components/charts/SuccessRateChart';
import { LatencyChart } from '../components/charts/LatencyChart';
import { TokenChart } from '../components/charts/TokenChart';
import { TracesList } from '../components/TracesList';

export function FeaturePage() {
  const { featureName } = useParams<{ featureName: string }>();
  const { data: overview, isLoading } = useMetricsOverview();

  // Find this feature's overview data from the aggregated metrics
  const feature = overview?.features?.find(
    (f) => f.name === featureName
  );

  return (
    <div>
      {/* Breadcrumb */}
      <div className="mb-6">
        <div className="mb-1 flex items-center gap-2 text-sm text-text-secondary">
          <Link to="/" className="hover:text-accent-blue">
            Genkit
          </Link>
          <span>›</span>
          <span className="text-text-primary">{featureName}</span>
        </div>
        <div className="flex items-center gap-3">
          <Link to="/" className="text-text-secondary hover:text-text-primary">
            <ArrowLeft className="h-5 w-5" />
          </Link>
          <h2 className="text-xl font-semibold">{featureName}</h2>
        </div>
      </div>

      {/* Stability Metrics Bar */}
      {isLoading ? (
        <div className="card mb-6 flex items-center justify-center p-8">
          <Loader2 className="h-5 w-5 animate-spin text-text-tertiary" />
        </div>
      ) : feature ? (
        <StabilityMetrics feature={feature} />
      ) : (
        <div className="card mb-6 p-4 text-center text-sm text-text-tertiary">
          No aggregated metrics found for this feature in the selected time
          range.
        </div>
      )}

      {/* 2x2 Chart Grid */}
      <div className="mb-8 grid grid-cols-1 gap-4 md:grid-cols-2">
        <RequestsChart featureName={featureName} />
        <TokenChart featureName={featureName} />
        <SuccessRateChart featureName={featureName} />
        <LatencyChart featureName={featureName} />
      </div>

      {/* Traces List */}
      {featureName && <TracesList featureName={featureName} />}
    </div>
  );
}
