import { useQuery } from '@tanstack/react-query';
import { api } from '../api/client';
import { useDashboard } from '../contexts/DashboardContext';

/**
 * Fetch a list of traces, optionally filtered by feature name.
 */
export function useTraceList(params?: {
  featureName?: string;
  status?: string;
  pageSize?: number;
  pageToken?: string;
}) {
  const { projectId, startTime, endTime } = useDashboard();

  return useQuery({
    queryKey: [
      'traces',
      'list',
      projectId,
      startTime,
      endTime,
      params?.featureName,
      params?.status,
      params?.pageSize,
      params?.pageToken,
    ],
    queryFn: () =>
      api.traces({
        projectId: projectId || undefined,
        startTime,
        endTime,
        featureName: params?.featureName,
        status: params?.status,
        pageSize: params?.pageSize || 20,
        pageToken: params?.pageToken,
      }),
    enabled: !!projectId,
    staleTime: 30_000,
  });
}

/**
 * Fetch full trace detail with span tree.
 */
export function useTraceDetail(traceId: string | undefined) {
  const { projectId } = useDashboard();

  return useQuery({
    queryKey: ['traces', 'detail', projectId, traceId],
    queryFn: () => api.traceDetail(traceId!, { projectId: projectId || undefined }),
    enabled: !!projectId && !!traceId,
    staleTime: 300_000, // 5 min (traces are immutable)
  });
}
