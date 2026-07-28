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

/**
 * Generate middleware exposing an Open Knowledge Format (OKF) bundle to the
 * model: a `<knowledge>` index is injected into the system prompt (the root
 * `index.md` when present, otherwise synthesized from concept frontmatter) and
 * a `lookup_knowledge` tool serves concept bodies by bundle-relative path.
 *
 * OKF (https://github.com/GoogleCloudPlatform/knowledge-catalog) is markdown +
 * YAML frontmatter; `type` is the only required field. Per the consumer
 * conformance rules, concepts are never rejected for missing optional fields
 * and unknown types are tolerated. Lifecycle handling: `status: deprecated`
 * concepts are omitted from the synthesized index (still loadable by path);
 * concepts past `stale_after` are flagged as stale.
 *
 * @module @genkit-ai/agent-dirs/okf
 */

import { generateMiddleware, z, type GenerateMiddleware } from 'genkit';
import { tool } from 'genkit/beta';
import * as fs from 'node:fs';
import * as path from 'node:path';

export const OkfKnowledgeOptionsSchema = z.object({
  /** Paths to directories containing OKF bundles. */
  knowledgePaths: z
    .array(z.string())
    .describe('Paths to directories containing OKF markdown bundles.'),
});

export type OkfKnowledgeOptions = z.infer<typeof OkfKnowledgeOptionsSchema>;

interface OkfConcept {
  /** Bundle-relative path, starting with '/'. */
  bundlePath: string;
  filePath: string;
  type?: string;
  title?: string;
  description?: string;
  status?: string;
  staleAfter?: string;
}

const INJECTED_MARKER = 'okf-knowledge';

function parseFrontmatter(content: string): Record<string, string> {
  const match = /^---\s*\r?\n([^]*?)\r?\n---/.exec(content);
  if (!match) return {};
  const fields: Record<string, string> = {};
  for (const key of ['type', 'title', 'description', 'status', 'stale_after']) {
    const m = new RegExp(`^${key}:\\s*(.+)$`, 'm').exec(match[1]);
    if (m) fields[key] = m[1].trim().replace(/^['"]|['"]$/g, '');
  }
  return fields;
}

function isStale(concept: OkfConcept): boolean {
  if (!concept.staleAfter) return false;
  const stale = Date.parse(concept.staleAfter);
  return !Number.isNaN(stale) && stale < Date.now();
}

/**
 * Middleware that exposes OKF bundles under `knowledgePaths` via a system
 * prompt index and a `lookup_knowledge` tool.
 */
export const okfKnowledge: GenerateMiddleware<
  typeof OkfKnowledgeOptionsSchema
> = generateMiddleware(
  {
    name: 'okfKnowledge',
    description:
      'Injects an index of Open Knowledge Format documents and a ' +
      'lookup_knowledge tool for loading them on demand.',
    configSchema: OkfKnowledgeOptionsSchema,
  },
  ({ config }) => {
    const roots = (config?.knowledgePaths ?? ['knowledge']).map((p) =>
      path.resolve(p)
    );
    const concepts = new Map<string, OkfConcept>();
    let rootIndexBody: string | undefined;
    let scanPromise: Promise<void> | null = null;

    async function scanDir(root: string, dir: string): Promise<void> {
      let entries: fs.Dirent[];
      try {
        entries = await fs.promises.readdir(dir, { withFileTypes: true });
      } catch {
        return;
      }
      for (const entry of entries) {
        const filePath = path.join(dir, entry.name);
        if (entry.isDirectory() && !entry.name.startsWith('.')) {
          await scanDir(root, filePath);
          continue;
        }
        if (!entry.isFile() || path.extname(entry.name) !== '.md') continue;
        if (entry.name === 'log.md') continue;
        const bundlePath =
          '/' + path.relative(root, filePath).split(path.sep).join('/');
        let content: string;
        try {
          content = await fs.promises.readFile(filePath, 'utf8');
        } catch {
          continue;
        }
        if (entry.name === 'index.md') {
          if (bundlePath === '/index.md') {
            rootIndexBody = content.replace(/^---[^]*?---\s*/, '').trim();
          }
          continue;
        }
        const fm = parseFrontmatter(content);
        concepts.set(bundlePath, {
          bundlePath,
          filePath,
          type: fm.type,
          title: fm.title,
          description: fm.description,
          status: fm.status,
          staleAfter: fm.stale_after,
        });
      }
    }

    function ensureScanned(): Promise<void> {
      if (!scanPromise) {
        scanPromise = (async () => {
          for (const root of roots) {
            await scanDir(root, root);
          }
        })();
      }
      return scanPromise;
    }

    function synthesizeIndex(): string {
      return Array.from(concepts.values())
        .filter((c) => c.status !== 'deprecated')
        .map((c) => {
          const label = c.title ?? c.bundlePath;
          const meta = [
            c.type,
            isStale(c) ? 'STALE - verify before relying on this' : undefined,
          ]
            .filter(Boolean)
            .join(', ');
          const desc = c.description ? ` - ${c.description}` : '';
          return ` - ${c.bundlePath}: ${label}${meta ? ` (${meta})` : ''}${desc}`;
        })
        .sort()
        .join('\n');
    }

    const lookupKnowledgeTool = tool(
      {
        name: 'lookup_knowledge',
        description:
          'Loads a knowledge document by its bundle-relative path ' +
          "(as listed in the <knowledge> index, e.g. '/carriers.md').",
        inputSchema: z.object({
          path: z.string().describe('Bundle-relative path of the document.'),
        }),
        outputSchema: z.string(),
      },
      async (input) => {
        await ensureScanned();
        const normalized = input.path.startsWith('/')
          ? input.path
          : `/${input.path}`;
        const concept = concepts.get(normalized);
        if (!concept) {
          throw new Error(
            `Knowledge document '${input.path}' not found. Available: ` +
              Array.from(concepts.keys()).join(', ')
          );
        }
        // Concepts are only served from the scanned map, so a crafted path
        // can never escape the bundle roots.
        return fs.promises.readFile(concept.filePath, 'utf8');
      }
    );

    return {
      tools: [lookupKnowledgeTool],
      generate: async (envelope, ctx, next) => {
        await ensureScanned();
        if (concepts.size === 0 && !rootIndexBody) {
          return next(envelope, ctx);
        }

        const indexText =
          `<knowledge>\n` +
          `You have a knowledge base of documents. Load a document with the ` +
          `lookup_knowledge tool before answering questions its entry ` +
          `covers; do not answer from memory what the knowledge base can ` +
          `answer.\n` +
          `${rootIndexBody ?? synthesizeIndex()}\n` +
          `</knowledge>`;

        const messages = [...envelope.request.messages];
        let replaced = false;
        for (let i = 0; i < messages.length && !replaced; i++) {
          const content = messages[i].content;
          for (let j = 0; j < content.length; j++) {
            const part = content[j];
            if (part.text && part.metadata?.[INJECTED_MARKER] === true) {
              if (part.text !== indexText) {
                const newContent = [...content];
                newContent[j] = {
                  text: indexText,
                  metadata: { [INJECTED_MARKER]: true },
                };
                messages[i] = { ...messages[i], content: newContent };
              }
              replaced = true;
              break;
            }
          }
        }
        if (!replaced) {
          const systemIndex = messages.findIndex((m) => m.role === 'system');
          const indexPart = {
            text: `\n\n${indexText}`,
            metadata: { [INJECTED_MARKER]: true },
          };
          if (systemIndex !== -1) {
            messages[systemIndex] = {
              ...messages[systemIndex],
              content: [...messages[systemIndex].content, indexPart],
            };
          } else {
            messages.unshift({ role: 'system', content: [indexPart] });
          }
        }

        return next(
          { ...envelope, request: { ...envelope.request, messages } },
          ctx
        );
      },
    };
  }
);
