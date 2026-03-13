/**
 * Copyright 2025 Google LLC
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

import { RuntimeManager } from '@genkit-ai/tools-common/manager';
import { z } from 'zod';
import { startDevProcessManager, startManager } from '../utils/manager-utils';

export interface McpToolOptions {
  projectRoot: string;
  explicitProjectRoot: boolean;
  timeout?: number;
  manager: McpRuntimeManager;
}

export function getCommonSchema(
  explicitProjectRoot: boolean,
  shape: z.ZodRawShape = {}
): z.ZodRawShape {
  return !explicitProjectRoot
    ? shape
    : {
        projectRoot: z
          .string()
          .describe(
            'The path to the current project root (a.k.a workspace directory or project directory); type: string'
          ),
        ...shape,
      };
}

export function resolveProjectRoot(
  explicitProjectRoot: boolean,
  opts: {
    [x: string]: any;
  },
  fallback: string
): string | { content: any[]; isError: boolean } {
  if (explicitProjectRoot && !opts?.projectRoot) {
    return {
      content: [
        { type: 'text', text: 'Project root is required for this tool.' },
      ],
      isError: true,
    };
  }
  return opts?.projectRoot ?? fallback;
}

/** Genkit Runtime manager specifically for the MCP server. Allows lazy
 * initialization and dev process manangement. */
export class McpRuntimeManager {
  private manager: RuntimeManager | undefined;
  private currentProjectRoot: string | undefined;

  async getManager(projectRoot: string) {
    if (this.manager && this.currentProjectRoot === projectRoot) {
      return this.manager;
    }
    if (this.manager) {
      await this.manager.stop();
    }
    this.manager = await startManager({
      projectRoot,
      manageHealth: true,
    });
    this.currentProjectRoot = projectRoot;
    return this.manager;
  }

  async getManagerWithDevProcess(params: {
    projectRoot: string;
    command: string;
    args: string[];
    explicitProjectRoot: boolean;
    timeout?: number;
  }): Promise<RuntimeManager> {
    const { projectRoot, command, args, timeout, explicitProjectRoot } = params;
    if (this.manager) {
      await this.manager.stop();
    }
    const devManager = await startDevProcessManager(
      projectRoot,
      command,
      args,
      {
        nonInteractive: true,
        healthCheck: timeout !== 0,
        timeout,
        cwd: explicitProjectRoot ? projectRoot : undefined,
      }
    );
    this.manager = devManager.manager;
    this.currentProjectRoot = projectRoot;
    return this.manager;
  }

  async kill() {
    if (this.manager) {
      await this.manager.stop();
    }
  }
}
