import cors from 'cors';
import express from 'express';
import { getAuthClient, getProjectId } from './auth.js';
import { cache } from './cache.js';
import metricsRouter from './routes/metrics.js';
import tracesRouter from './routes/traces.js';

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

// Projects
app.get('/api/projects', async (_req, res) => {
  try {
    const projectId = await getProjectId();
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

// Mount route modules
app.use('/api/metrics', metricsRouter);
app.use('/api/traces', tracesRouter);

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
  console.log(`   Health check: http://localhost:${PORT}/api/health`);
  console.log(`   Metrics:      http://localhost:${PORT}/api/metrics/overview`);
  console.log(`   Traces:       http://localhost:${PORT}/api/traces\n`);
});
