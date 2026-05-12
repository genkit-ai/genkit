import { Router, Request, Response } from 'express';
import { cache, CacheTTL } from '../cache.js';
import {
  queryFeatureOverview,
  queryTimeSeries,
  normalizeTimeSeries,
  computeAlignmentPeriod,
} from '../gcp/monitoring.js';
import { getProjectId } from '../auth.js';

const router = Router();

/**
 * Helper to resolve projectId from query param or ADC default.
 */
async function resolveProjectId(req: Request): Promise<string> {
  const projectId =
    (req.query.projectId as string) || (await getProjectId());
  if (!projectId) {
    throw new Error(
      'No projectId provided and no default project found. ' +
        'Pass ?projectId=... or run: gcloud config set project <PROJECT>'
    );
  }
  return projectId;
}

/**
 * Helper to parse time range from query params with defaults.
 */
function parseTimeRange(req: Request): { startTime: string; endTime: string } {
  const endTime =
    (req.query.endTime as string) || new Date().toISOString();
  const startTime =
    (req.query.startTime as string) ||
    new Date(Date.now() - 24 * 60 * 60 * 1000).toISOString(); // default: last 24h
  return { startTime, endTime };
}

/**
 * GET /api/metrics/overview
 *
 * Returns aggregated feature-level metrics.
 *
 * Query params:
 *   projectId  - GCP project ID (optional, defaults to ADC project)
 *   startTime  - ISO 8601 start time (optional, defaults to 24h ago)
 *   endTime    - ISO 8601 end time (optional, defaults to now)
 */
router.get('/overview', async (req: Request, res: Response) => {
  try {
    const projectId = await resolveProjectId(req);
    const { startTime, endTime } = parseTimeRange(req);

    const cacheKey = `metrics:overview:${projectId}:${startTime}:${endTime}`;

    const features = await cache.getOrFetch(
      cacheKey,
      CacheTTL.METRICS,
      () => queryFeatureOverview(projectId, startTime, endTime)
    );

    res.json({ features });
  } catch (err) {
    const status = (err as Error).message?.includes('401') ? 401 : 500;
    res.status(status).json({
      error: err instanceof Error ? err.message : 'Failed to fetch metrics overview',
    });
  }
});

/**
 * GET /api/metrics/timeseries
 *
 * Returns raw time series data for charting.
 *
 * Query params:
 *   projectId       - GCP project ID (optional)
 *   metricType      - Full metric type string (required)
 *   startTime       - ISO 8601 start time (optional, defaults to 24h ago)
 *   endTime         - ISO 8601 end time (optional, defaults to now)
 *   alignmentPeriod - e.g. "60s", "3600s" (optional, auto-computed)
 *   aligner         - e.g. "ALIGN_SUM", "ALIGN_RATE" (optional, default ALIGN_SUM)
 *   reducer         - Cross-series reducer (optional)
 *   groupBy         - Comma-separated group-by fields (optional)
 *   filter          - Additional filter beyond metric.type (optional)
 */
router.get('/timeseries', async (req: Request, res: Response) => {
  try {
    const projectId = await resolveProjectId(req);
    const { startTime, endTime } = parseTimeRange(req);

    const metricType = req.query.metricType as string;
    if (!metricType) {
      res.status(400).json({
        error: 'metricType query parameter is required',
        hint: 'e.g. ?metricType=workload.googleapis.com/genkit/feature/requests',
      });
      return;
    }

    const alignmentPeriod =
      (req.query.alignmentPeriod as string) ||
      computeAlignmentPeriod(startTime, endTime);
    const aligner = (req.query.aligner as string) || 'ALIGN_DELTA';
    const reducer = req.query.reducer as string | undefined;
    const groupByRaw = req.query.groupBy as string | undefined;
    const groupByFields = groupByRaw
      ? groupByRaw.split(',').map((f) => f.trim())
      : undefined;
    const additionalFilter = req.query.filter as string | undefined;

    const cacheKey = `metrics:ts:${projectId}:${metricType}:${startTime}:${endTime}:${alignmentPeriod}:${aligner}:${reducer || ''}:${groupByRaw || ''}:${additionalFilter || ''}`;

    const timeSeries = await cache.getOrFetch(
      cacheKey,
      CacheTTL.METRICS,
      async () => {
        const raw = await queryTimeSeries({
          projectId,
          metricType,
          startTime,
          endTime,
          alignmentPeriod,
          perSeriesAligner: aligner,
          crossSeriesReducer: reducer,
          groupByFields,
          filter: additionalFilter,
        });
        return normalizeTimeSeries(raw);
      }
    );

    res.json({ timeSeries });
  } catch (err) {
    const status = (err as Error).message?.includes('401') ? 401 : 500;
    res.status(status).json({
      error: err instanceof Error ? err.message : 'Failed to fetch time series',
    });
  }
});

export default router;
