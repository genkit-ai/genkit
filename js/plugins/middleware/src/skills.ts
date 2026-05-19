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

import * as fs from 'fs';
import {
  X_GENKIT_ALLOW_CUSTOM,
  X_GENKIT_DATA_SOURCE,
  annotateSchema,
  generateMiddleware,
  z,
  type GenerateMiddleware,
} from 'genkit';
import { tool } from 'genkit/beta';
import * as path from 'path';

/**
 * Action that crawls the project directory for any directory containing a SKILL.md file.
 * This is used as a data source for the Dev UI.
 */
export const listSkillsAction = {
  name: 'folders/list',
  actionType: 'custom' as const,
  inputSchema: z.void(),
  outputSchema: z.array(
    z.object({
      label: z.string(),
      value: z.string(),
    })
  ),
  handler: async () => {
    const findSkillDirs = async (dir: string, results: string[] = []) => {
      try {
        const files = await fs.promises.readdir(dir);
        for (const file of files) {
          if (file === 'node_modules' || file === '.git' || file === 'lib')
            continue;
          const fullPath = path.join(dir, file);
          const stat = await fs.promises.stat(fullPath).catch(() => null);
          if (stat?.isDirectory()) {
            const hasSkillMd = await fs.promises
              .access(path.join(fullPath, 'SKILL.md'))
              .then(() => true)
              .catch(() => false);
            if (hasSkillMd) {
              results.push(path.relative(process.cwd(), fullPath));
            }
            await findSkillDirs(fullPath, results);
          }
        }
      } catch (e) {
        // Ignore errors for unreadable directories
      }
      return results;
    };
    const dirs = await findSkillDirs('.');
    return dirs.map((d) => ({ label: d, value: d }));
  },
};

export const SkillsOptionsSchema = z.object({
  /**
   * Paths to directories containing skills.
   * @default ['skills']
   */
  skillPaths: annotateSchema(
    z
      .array(z.string())
      .optional()
      .describe('Paths to directories containing skills.'),
    {
      [X_GENKIT_DATA_SOURCE]: '/custom/middleware:skills/folders/list',
      [X_GENKIT_ALLOW_CUSTOM]: true,
    }
  ),
});

export type SkillsOptions = z.infer<typeof SkillsOptionsSchema>;

/**
 * Creates a middleware that scans for skills in specified paths.
 * Injects a system prompt listing available skills and provides a `use_skill` tool.
 */
export const skills: GenerateMiddleware<typeof SkillsOptionsSchema> =
  generateMiddleware(
    {
      name: 'skills',
      description: 'Injects system instructions and tools for using skills.',
      configSchema: SkillsOptionsSchema,
    },
    ({ config }) => {
      const skillPaths = config?.skillPaths ?? ['skills'];
      const skillCache = new Map<
        string,
        { path: string; description: string }
      >();

      function parseFrontmatter(content: string) {
        const match = /^---\s*\r?\n([^]*?)\r?\n---/.exec(content);
        if (!match) return null;

        const yaml = match[1];
        const nameMatch = /^name:\s*(.+)/m.exec(yaml);
        const descriptionMatch = /^description:\s*(.+)/m.exec(yaml);

        return {
          name: nameMatch ? nameMatch[1].trim() : undefined,
          description: descriptionMatch
            ? descriptionMatch[1].trim()
            : undefined,
        };
      }

      let scanPromise: Promise<void> | null = null;

      function ensureSkillsScanned(): Promise<void> {
        if (!scanPromise) {
          scanPromise = (async () => {
            skillCache.clear();

            for (const p of skillPaths) {
              const dirPath = path.resolve(p);
              try {
                const files = await fs.promises.readdir(dirPath, {
                  withFileTypes: true,
                });
                for (const file of files) {
                  if (file.isDirectory() && !file.name.startsWith('.')) {
                    const skillDir = path.join(dirPath, file.name);
                    const skillMdPath = path.join(skillDir, 'SKILL.md');
                    try {
                      const content = await fs.promises.readFile(
                        skillMdPath,
                        'utf-8'
                      );
                      let description = 'No description provided.';
                      const fm = parseFrontmatter(content);
                      if (fm?.description) {
                        description = fm.description;
                      }
                      skillCache.set(file.name, {
                        path: skillMdPath,
                        description,
                      });
                    } catch (e) {
                      // ignore file read errors
                    }
                  }
                }
              } catch (e) {
                // ignore directory read errors
              }
            }
          })();
        }
        return scanPromise;
      }

      const useSkillTool = tool(
        {
          name: 'use_skill',
          description: 'Use a skill by its name.',
          inputSchema: z.object({
            skillName: z.string().describe('The name of the skill to use.'),
          }),
          outputSchema: z.string(),
        },
        async (input) => {
          await ensureSkillsScanned();
          const info = skillCache.get(input.skillName);
          if (!info) {
            throw new Error(`Skill '${input.skillName}' not found.`);
          }

          try {
            return await fs.promises.readFile(info.path, 'utf-8');
          } catch (e) {
            throw new Error(`Failed to read skill "${input.skillName}": ${e}`);
          }
        }
      );

      return {
        tools: [useSkillTool],
        generate: async (envelope, ctx, next) => {
          const { request } = envelope;
          await ensureSkillsScanned();
          if (skillCache.size === 0) return next(envelope, ctx);

          const skillsList = Array.from(skillCache.entries())
            .map(([name, info]) => {
              if (info.description !== 'No description provided.') {
                return ` - ${name} - ${info.description}`;
              }
              return ` - ${name}`;
            })
            .join('\n');

          const systemPromptText =
            `<skills>\n` +
            `You have access to a library of skills that serve as specialized instructions/personas.\n` +
            `Strongly prefer to use them when working on anything related to them.\n` +
            `Only use them once to load the context.\n` +
            `Here are the available skills:\n` +
            `${skillsList}\n` +
            `</skills>`;

          const messages = [...request.messages];
          let injectedPart: any | undefined;
          let injectedMsgIndex = -1;
          let injectedPartIndex = -1;

          for (let i = 0; i < messages.length; i++) {
            const msg = messages[i];
            for (let j = 0; j < msg.content.length; j++) {
              const p = msg.content[j];
              if (p.text && p.metadata?.['skills-instructions'] === true) {
                injectedPart = p;
                injectedMsgIndex = i;
                injectedPartIndex = j;
                break;
              }
            }
            if (injectedPart) break;
          }

          if (injectedPart) {
            if (injectedPart.text !== systemPromptText) {
              const newContent = [...messages[injectedMsgIndex].content];
              newContent[injectedPartIndex] = {
                text: systemPromptText,
                metadata: { 'skills-instructions': true },
              };
              messages[injectedMsgIndex] = {
                ...messages[injectedMsgIndex],
                content: newContent as any,
              };
            }
          } else {
            const systemMsgIndex = messages.findIndex(
              (m) => m.role === 'system'
            );
            if (systemMsgIndex !== -1) {
              messages[systemMsgIndex] = {
                ...messages[systemMsgIndex],
                content: [
                  ...messages[systemMsgIndex].content,
                  {
                    text: systemPromptText,
                    metadata: { 'skills-instructions': true },
                  },
                ],
              };
            } else {
              messages.unshift({
                role: 'system',
                content: [
                  {
                    text: systemPromptText,
                    metadata: { 'skills-instructions': true },
                  },
                ],
              });
            }
          }

          return next({ ...envelope, request: { ...request, messages } }, ctx);
        },
      };
    }
  );

// Add the discovery action to the plugin's init hook
const originalPlugin = skills.plugin;
skills.plugin = (options) => {
  const p = originalPlugin(options);
  const originalInit = p.init;
  p.init = async () => {
    const { action } = require('genkit');
    const supplementalActions = [
      action(listSkillsAction as any, listSkillsAction.handler),
    ];
    if (originalInit) {
      const originalActions = await originalInit();
      return [...(originalActions || []), ...supplementalActions];
    }
    return supplementalActions;
  };
  return p;
};
