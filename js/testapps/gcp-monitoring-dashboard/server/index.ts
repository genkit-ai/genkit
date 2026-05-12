import cors from 'cors';
import express from 'express';
import { getAuthClient, getProjectId } from './auth.js';
import { cache } from './cache.js';

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

// Health check
app.get('/api/health', async (_req, res) => {
  try {
    const projectId = await getProjectId();
    res.json({
      status: 'ok',
      timestamp: new Date().toISOString(),
      defaultProject: projectId || null,
      cacheSize: cache.size(),
    });
  } catch (err) {
    res.json({
      status: 'ok',
      timestamp: new Date().toISOString(),
      defaultProject: null,
      authError: err instanceof Error ? err.message : 'Unknown error',
      cacheSize: cache.size(),
    });
  }
});

// Auth status check
app.get('/api/auth/status', async (_req, res) => {
  try {
    const client = await getAuthClient();
    const projectId = await getProjectId();

    // Try to get credentials to verify they work
    const credentials = await client.getAccessToken();

    res.json({
      authenticated: !!credentials.token,
      projectId: projectId || null,
    });
  } catch (err) {
    res.status(401).json({
      authenticated: false,
      error: err instanceof Error ? err.message : 'Unknown error',
      hint: 'Run: gcloud auth application-default login',
    });
  }
});

// Cache management
app.post('/api/cache/clear', (_req, res) => {
  const cleared = cache.size();
  cache.clear();
  res.json({ cleared, message: `Cleared ${cleared} cached entries` });
});

app.get('/api/cache/stats', (_req, res) => {
  res.json({
    size: cache.size(),
    maxSize: 1000,
  });
});

// Placeholder routes for metrics and traces (will be implemented in Stage 2)
app.get('/api/projects', async (_req, res) => {
  try {
    const projectId = await getProjectId();
    // For now, just return the default project from ADC
    const projects = projectId
      ? [{ projectId, name: projectId }]
      : [];
    res.json({ projects });
  } catch (err) {
    res.status(500).json({
      error: err instanceof Error ? err.message : 'Failed to list projects',
    });
  }
});

app.get('/api/metrics/overview', (_req, res) => {
  res.json({
    features: [],
    _note: 'Not yet implemented — coming in Stage 2',
  });
});

app.get('/api/metrics/timeseries', (_req, res) => {
  res.json({
    timeSeries: [],
    _note: 'Not yet implemented — coming in Stage 2',
  });
});

app.get('/api/traces', (_req, res) => {
  res.json({
    traces: [],
    _note: 'Not yet implemented — coming in Stage 2',
  });
});

app.get('/api/traces/:traceId', (req, res) => {
  res.json({
    traceId: req.params.traceId,
    spans: [],
    _note: 'Not yet implemented — coming in Stage 2',
  });
});

// Error handling middleware
app.use(
  (
    err: Error,
    _req: express.Request,
    res: express.Response,
    _next: express.NextFunction
  ) => {
    console.error('Server error:', err);
    res.status(500).json({
      error: err.message || 'Internal server error',
    });
  }
);

app.listen(PORT, () => {
  console.log(`\n🔍 Genkit Monitoring API server running at http://localhost:${PORT}`);
  console.log(`   Health check: http://localhost:${PORT}/api/health\n`);
});
