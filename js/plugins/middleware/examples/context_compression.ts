/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { vertexAI } from '@genkit-ai/google-genai';
import { genkit, z } from 'genkit/beta';
import { contextCompression } from '../src/index.js';

const ai = genkit({
  plugins: [vertexAI({ location: 'global' })],
});

// Tool 1: Searches database for matching report IDs
const searchReports = ai.defineTool(
  {
    name: 'searchReports',
    description:
      'Searches for document and incident report IDs related to a topic.',
    inputSchema: z.object({
      topic: z.string().describe('The topic or system to search for'),
    }),
    outputSchema: z.array(z.string()),
  },
  async () => {
    return ['report-101', 'report-102', 'report-103'];
  }
);

// Tool 2: Fetches detailed content for a report ID
const fetchReport = ai.defineTool(
  {
    name: 'fetchReport',
    description:
      'Retrieves the detailed text and metrics for a specific report ID.',
    inputSchema: z.object({
      reportId: z.string().describe('The report ID to retrieve'),
    }),
    outputSchema: z.object({
      id: z.string(),
      title: z.string(),
      details: z.string(),
      metrics: z.record(z.any()),
    }),
  },
  async ({ reportId }) => {
    const database: Record<
      string,
      { title: string; details: string; metrics: any }
    > = {
      'report-101': {
        title: 'Project Alpha Performance Audit Q1',
        details:
          'Audit conducted across cluster us-central. Average latency was 240ms under 50k QPS load. ' +
          'Database query optimization resolved 4 out of 5 slow queries identified in cache warmup. ' +
          'CPU utilization remained within 65% across all pod replicas.',
        metrics: { p99LatencyMs: 420, p50LatencyMs: 180, cacheHitRate: 0.94 },
      },
      'report-102': {
        title: 'Project Alpha Security & Access Log',
        details:
          'Quarterly access control review. 14 service accounts were audited; 2 stale credentials were revoked. ' +
          'All API endpoints now enforce mTLS and token rotation intervals were reduced to 1 hour. ' +
          'No unauthorized intrusion attempts detected.',
        metrics: {
          auditedAccounts: 14,
          revokedTokens: 2,
          rotationIntervalHours: 1,
        },
      },
      'report-103': {
        title: 'Project Alpha Deployment Incident Post-Mortem',
        details:
          'Root cause analysis for service interruption on Feb 12. Automated canary failed to halt rollout ' +
          'due to an unhandled rejection in the telemetry health-check probe. Rollback took 14 minutes. ' +
          'Fix deployed: probe timeout reduced from 30s to 5s with strict fail-closed assertion.',
        metrics: { downtimeMinutes: 14, affectedUsersRatio: 0.04 },
      },
    };

    const report = database[reportId] || {
      title: 'Unknown report',
      details: 'Report details are not available for this identifier.',
      metrics: {},
    };

    return {
      id: reportId,
      ...report,
    };
  }
);

/**
 * Define research agent equipped with context compression.
 * When asked to investigate Project Alpha, the agent will make multiple
 * tool calls sequentially in a single generate loop (Turn 0 -> Turn 1 -> Turn 2...),
 * triggering context compression directly within the intra-call tool loop.
 */
export const researchAgent = ai.defineAgent({
  name: 'researchAgent',
  model: vertexAI.model('gemini-flash-latest'),
  system:
    'You are an investigative research assistant. When asked about a project, always search for ' +
    'reports first, then fetch every report individually to synthesize a comprehensive summary.',
  tools: [searchReports, fetchReport],
  maxTurns: 10,
  use: [
    contextCompression({
      maxInputTokens: 200, // Triggers compression as tool responses accumulate in the generate loop
      toolResponses: {
        maxChars: 120, // Truncate verbose tool responses beyond 120 chars
        preserveRecent: 1, // Keep most recent tool response intact
      },
      maxMessages: 10,
    }),
  ],
});
