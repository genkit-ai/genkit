import {
  createContext,
  useCallback,
  useContext,
  useMemo,
  useState,
  type ReactNode,
} from 'react';
import { useSearchParams } from 'react-router-dom';
import { presetToTimeRange, type TimeRangePreset } from '../api/client';

interface DashboardState {
  projectId: string | null;
  setProjectId: (id: string) => void;
  timeRangePreset: TimeRangePreset;
  setTimeRangePreset: (preset: TimeRangePreset) => void;
  startTime: string;
  endTime: string;
  /** Refresh time range (recompute endTime = now) */
  refresh: () => void;
}

const DashboardContext = createContext<DashboardState | null>(null);

export function DashboardProvider({ children }: { children: ReactNode }) {
  const [searchParams, setSearchParams] = useSearchParams();

  // Project ID from URL or null (will be filled from auth/status)
  const [projectId, setProjectIdState] = useState<string | null>(
    searchParams.get('project') || null
  );

  // Time range preset from URL or default 24h
  const [preset, setPresetState] = useState<TimeRangePreset>(
    (searchParams.get('range') as TimeRangePreset) || '24h'
  );

  // Refresh counter to force time range recomputation
  const [refreshKey, setRefreshKey] = useState(0);

  const setProjectId = useCallback(
    (id: string) => {
      setProjectIdState(id);
      setSearchParams((prev) => {
        const next = new URLSearchParams(prev);
        next.set('project', id);
        return next;
      });
    },
    [setSearchParams]
  );

  const setTimeRangePreset = useCallback(
    (p: TimeRangePreset) => {
      setPresetState(p);
      setRefreshKey((k) => k + 1);
      setSearchParams((prev) => {
        const next = new URLSearchParams(prev);
        next.set('range', p);
        return next;
      });
    },
    [setSearchParams]
  );

  const refresh = useCallback(() => {
    setRefreshKey((k) => k + 1);
  }, []);

  // Compute time range from preset (memoized on preset + refreshKey)
  const { startTime, endTime } = useMemo(() => {
    void refreshKey; // depend on refreshKey to recompute
    return presetToTimeRange(preset);
  }, [preset, refreshKey]);

  const value = useMemo<DashboardState>(
    () => ({
      projectId,
      setProjectId,
      timeRangePreset: preset,
      setTimeRangePreset,
      startTime,
      endTime,
      refresh,
    }),
    [projectId, setProjectId, preset, setTimeRangePreset, startTime, endTime, refresh]
  );

  return (
    <DashboardContext.Provider value={value}>
      {children}
    </DashboardContext.Provider>
  );
}

export function useDashboard(): DashboardState {
  const ctx = useContext(DashboardContext);
  if (!ctx) {
    throw new Error('useDashboard must be used within a DashboardProvider');
  }
  return ctx;
}
