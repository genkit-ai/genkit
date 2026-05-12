import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { useDashboard } from '../contexts/DashboardContext';

/**
 * Fetch the feature overview (aggregated metrics per feature).
 */
export function useMetricsOverview() {
  const { projectId, startTime, endTime } = useDashboard();

  return useQuery({
    queryKey: ['metrics', 'overview', projectId, startTime, endTime],
    queryFn: () =>
      api.metricsOverview({
        projectId: projectId || undefined,
        startTime,
        endTime,
      }),
    enabled: !!projectId,
    staleTime: 60_000,
  });
}

/**
 * Fetch time series data for a specific metric type.
 */
export function useTimeSeries(
  metricType: string,
  options?: {
    groupBy?: string;
    reducer?: string;
    aligner?: string;
    filter?: string;
    enabled?: boolean;
  }
) {
  const { projectId, startTime, endTime } = useDashboard();

  return useQuery({
    queryKey: [
      'metrics',
      'timeseries',
      metricType,
      projectId,
      startTime,
      endTime,
      options?.groupBy,
      options?.filter,
    ],
    queryFn: () =>
      api.timeseries({
        metricType,
        projectId: projectId || undefined,
        startTime,
        endTime,
        groupBy: options?.groupBy,
        reducer: options?.reducer,
        aligner: options?.aligner,
        filter: options?.filter,
      }),
    enabled: (options?.enabled !== false) && !!projectId,
    staleTime: 60_000,
  });
}
