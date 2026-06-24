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

import {
  generateMiddleware,
  ToolAction,
  z,
  type GenerateMiddleware,
} from 'genkit';
import { tool, type Artifact } from 'genkit/beta';

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

export const ArtifactsOptionsSchema = z.object({
  readonly: z
    .boolean()
    .optional()
    .describe(
      'When true, only the read_artifact tool is provided — the model ' +
        'cannot create or update artifacts. Defaults to false.'
    ),
});

export type ArtifactsOptions = z.infer<typeof ArtifactsOptionsSchema>;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Extracts the text content from an artifact's parts, joining multiple
 * text parts with newlines.
 */
function extractArtifactText(artifact: Artifact): string {
  return (artifact.parts ?? [])
    .map((p) => p.text ?? '')
    .filter((t) => t.length > 0)
    .join('\n');
}

/**
 * Builds the `<artifacts>` system prompt block listing available artifacts.
 */
function buildArtifactListing(artifacts: Artifact[]): string {
  if (artifacts.length === 0) {
    return (
      `<artifacts>\n` +
      `No artifacts are currently available in the session.\n` +
      `</artifacts>`
    );
  }

  const listing = artifacts
    .map((a) => {
      const text = extractArtifactText(a);
      const sizeHint = text.length > 0 ? ` (${text.length} chars)` : '';
      const source = a.metadata?.source ? ` [from: ${a.metadata.source}]` : '';
      return `  - ${a.name || '(unnamed)'}${sizeHint}${source}`;
    })
    .join('\n');

  return (
    `<artifacts>\n` +
    `The following artifacts are available in the session. Use the ` +
    `read_artifact tool to view their content.\n` +
    `${listing}\n` +
    `</artifacts>`
  );
}

// ---------------------------------------------------------------------------
// Middleware
// ---------------------------------------------------------------------------

/**
 * Creates a middleware that gives the model tools to interact with session
 * artifacts, and injects an artifact listing into the system prompt.
 *
 * **Tools provided:**
 *
 * - `read_artifact` — reads an artifact by name from the session and returns
 *   its text content.
 * - `write_artifact` (unless `readonly: true`) — creates or updates an
 *   artifact in the session. Artifacts are deduplicated by name: writing to
 *   an existing name replaces the artifact.
 *
 * **System prompt injection:**
 *
 * An `<artifacts>` block is injected into (or appended to) the system message
 * on each generate turn, listing the names and sizes of all artifacts
 * currently in the session. This lets the model know what's available without
 * consuming context on the full content.
 *
 * This middleware is useful standalone (e.g. for a workspace-builder agent
 * that creates files as artifacts) or combined with the `agents` middleware
 * using `artifactStrategy: 'session'`, where sub-agent artifacts are merged
 * into the parent session and the model accesses them via these tools.
 *
 * @example
 * ```typescript
 * // Standalone: agent that creates and reads artifacts
 * const builder = ai.defineAgent({
 *   name: 'builder',
 *   system: 'You are a code generator. Use write_artifact to create files.',
 *   use: [artifacts()],
 * });
 *
 * // Combined with agents middleware (session strategy)
 * const orchestrator = ai.defineAgent({
 *   name: 'orchestrator',
 *   system: 'You coordinate sub-agents and review their work.',
 *   use: [
 *     agents({
 *       agents: ['researcher', 'coder'],
 *       artifactStrategy: 'session',
 *     }),
 *     artifacts({ readonly: true }), // can read sub-agent artifacts
 *   ],
 * });
 * ```
 */
export const artifacts: GenerateMiddleware<typeof ArtifactsOptionsSchema> =
  generateMiddleware(
    {
      name: 'artifacts',
      description:
        'Provides read_artifact and write_artifact tools for interacting ' +
        'with session artifacts, and injects an artifact listing into the ' +
        'system prompt.',
      configSchema: ArtifactsOptionsSchema,
    },
    ({ config, ai }) => {
      const readonly = config?.readonly ?? false;

      // ── read_artifact tool ──────────────────────────────────────────

      const readArtifactTool = tool(
        {
          name: 'read_artifact',
          description:
            'Reads the content of a named artifact from the session. ' +
            'Use this to inspect artifacts produced by sub-agents or ' +
            'previously created artifacts.',
          inputSchema: z.object({
            name: z.string().describe('The name of the artifact to read.'),
          }),
          outputSchema: z.object({
            name: z.string().describe('The artifact name.'),
            content: z.string().describe('The text content of the artifact.'),
            found: z
              .boolean()
              .describe('Whether the artifact was found in the session.'),
          }),
        },
        async (input) => {
          // `ai.currentSession()` throws when there is no active session.
          let session: ReturnType<typeof ai.currentSession>;
          try {
            session = ai.currentSession();
          } catch {
            return {
              name: input.name,
              content: 'Error: no active session.',
              found: false,
            };
          }

          const artifacts = session.getArtifacts();
          const artifact = artifacts.find((a) => a.name === input.name);

          if (!artifact) {
            return {
              name: input.name,
              content: `Artifact "${input.name}" not found.`,
              found: false,
            };
          }

          return {
            name: input.name,
            content: extractArtifactText(artifact),
            found: true,
          };
        }
      );

      // ── write_artifact tool ─────────────────────────────────────────

      const writeArtifactTool = tool(
        {
          name: 'write_artifact',
          description:
            'Creates or updates a named artifact in the session. ' +
            'If an artifact with the same name already exists, it will be ' +
            'replaced. Use this to produce files, reports, code, or other ' +
            'deliverables.',
          inputSchema: z.object({
            name: z
              .string()
              .describe(
                'A unique name for the artifact (e.g. a filename like "report.md").'
              ),
            content: z
              .string()
              .describe('The full text content of the artifact.'),
          }),
          outputSchema: z.object({
            status: z
              .string()
              .describe(
                'Confirmation that the artifact was created or updated.'
              ),
          }),
        },
        async (input) => {
          // `ai.currentSession()` throws when there is no active session.
          let session: ReturnType<typeof ai.currentSession>;
          try {
            session = ai.currentSession();
          } catch {
            return { status: 'Error: no active session.' };
          }

          session.addArtifacts([
            {
              name: input.name,
              parts: [{ text: input.content }],
            },
          ]);

          return { status: `Artifact "${input.name}" saved successfully.` };
        }
      );

      // ── Assemble tools ──────────────────────────────────────────────

      const artifactTools: ToolAction<any, any>[] = [readArtifactTool];
      if (!readonly) {
        artifactTools.push(writeArtifactTool);
      }

      return {
        tools: artifactTools,

        generate: async (envelope, ctx, next) => {
          const { request } = envelope;

          // ── Build artifact listing for the system prompt ──────────
          // `ai.currentSession()` throws when there is no active session.
          let currentArtifacts: Artifact[] = [];
          try {
            currentArtifacts = ai.currentSession().getArtifacts();
          } catch {
            // No active session — nothing to list.
          }

          const artifactListing = buildArtifactListing(currentArtifacts);

          // ── Inject / update the listing in the system message ─────
          const messages = [...request.messages];
          const MARKER_KEY = 'artifacts-middleware-listing';

          // Remove any previously injected listing (refresh each turn).
          for (let i = 0; i < messages.length; i++) {
            const msg = messages[i];
            const filteredContent = msg.content.filter(
              (part) => !part.metadata?.[MARKER_KEY]
            );
            if (filteredContent.length !== msg.content.length) {
              messages[i] = { ...msg, content: filteredContent };
            }
          }

          // Append to existing system message or create one.
          const systemIdx = messages.findIndex((m) => m.role === 'system');
          if (systemIdx !== -1) {
            messages[systemIdx] = {
              ...messages[systemIdx],
              content: [
                ...messages[systemIdx].content,
                {
                  text: artifactListing,
                  metadata: { [MARKER_KEY]: true },
                },
              ],
            };
          } else {
            messages.unshift({
              role: 'system',
              content: [
                {
                  text: artifactListing,
                  metadata: { [MARKER_KEY]: true },
                },
              ],
            });
          }

          return next({ ...envelope, request: { ...request, messages } }, ctx);
        },
      };
    }
  );
