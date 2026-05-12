import { Router, Request, Response } from 'express';
import { cache, CacheTTL } from '../cache.js';
import {
  listTraces,
  getTrace,
  buildSpanTree,
  normalizeTraceForList,
  flattenSpanTree,
} from '../gcp/tracing.js';
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
 * GET /api/traces
 *
 * List traces with optional filtering and pagination.
 *
 * Query params:
 *   projectId  - GCP project ID (optional, defaults to ADC project)
 *   startTime  - ISO 8601 start time (optional, defaults to 24h ago)
 *   endTime    - ISO 8601 end time (optional, defaults to now)
 *   filter     - Cloud Trace filter string (optional)
 *               e.g. "genkit/isRoot:true" or "genkit/feature:myFlow"
 *   featureName - Shorthand: filter by feature name (optional)
 *   status     - Shorthand: filter by status "success" or "error" (optional)
 *   pageSize   - Number of traces per page (optional, default 20)
 *   pageToken  - Pagination token from previous response (optional)
 *   orderBy    - Sort order (optional, default newest first)
 */
router.get('/', async (req: Request, res: Response) => {
  try {
    const projectId = await resolveProjectId(req);
    const { startTime, endTime } = parseTimeRange(req);

    // Build filter from explicit filter param or shorthand params
    let filter = req.query.filter as string | undefined;
    const featureName = req.query.featureName as string | undefined;
    const status = req.query.status as string | undefined;

    if (!filter) {
      const filterParts: string[] = [];
      if (featureName) {
        filterParts.push(`genkit/feature:${featureName}`);
      }
      if (status) {
        filterParts.push(`genkit/state:${status}`);
      }
      // Default: only show root traces (genkit traces with isRoot label)
      if (!featureName && !status) {
        filterParts.push('genkit/isRoot:true');
      }
      if (filterParts.length > 0) {
        filter = filterParts.join(' ');
      }
    }

    const pageSize = Math.min(
      parseInt(req.query.pageSize as string, 10) || 20,
      100
    );
    const pageToken = req.query.pageToken as string | undefined;
    const orderBy = req.query.orderBy as string | undefined;

    const cacheKey = `traces:list:${projectId}:${startTime}:${endTime}:${filter || ''}:${pageSize}:${pageToken || ''}:${orderBy || ''}`;

    const result = await cache.getOrFetch(
      cacheKey,
      CacheTTL.TRACE_LIST,
      async () => {
        const { traces: rawTraces, nextPageToken } = await listTraces({
          projectId,
          startTime,
          endTime,
          filter,
          pageSize,
          pageToken,
          orderBy,
        });

        const traces = rawTraces.map(normalizeTraceForList);

        return {
          traces,
          nextPageToken,
        };
      }
    );

    res.json(result);
  } catch (err) {
    const status = (err as Error).message?.includes('401') ? 401 : 500;
    res.status(status).json({
      error: err instanceof Error ? err.message : 'Failed to list traces',
    });
  }
});

/**
 * GET /api/traces/:traceId
 *
 * Get full trace detail with all spans organized as a tree.
 *
 * Query params:
 *   projectId - GCP project ID (optional, defaults to ADC project)
 */
router.get('/:traceId', async (req: Request, res: Response) => {
  try {
    const projectId = await resolveProjectId(req);
    const { traceId } = req.params;

    if (!traceId) {
      res.status(400).json({ error: 'traceId is required' });
      return;
    }

    const cacheKey = `traces:detail:${projectId}:${traceId}`;

    const result = await cache.getOrFetch(
      cacheKey,
      CacheTTL.TRACE_DETAIL,
      async () => {
        const rawTrace = await getTrace(projectId, traceId);
        const rootSpan = buildSpanTree(rawTrace.spans || []);
        const spans = rootSpan ? flattenSpanTree(rootSpan) : [];

        return {
          traceId,
          rootSpan,
          spans,
        };
      }
    );

    res.json(result);
  } catch (err) {
    const errMsg = err instanceof Error ? err.message : 'Failed to fetch trace';
    const status = errMsg.includes('404')
      ? 404
      : errMsg.includes('401')
        ? 401
        : 500;
    res.status(status).json({ error: errMsg });
  }
});

export default router;
