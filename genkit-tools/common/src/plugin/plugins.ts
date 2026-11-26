/**
 * Copyright 2024 Google LLC
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

import { execSync } from 'child_process';
import * as clc from 'colorette';
import { z } from 'zod';

const SupportedFlagValuesSchema = z.union([
  z.undefined(),
  z.string(),
  z.boolean(),
  z.array(z.string()),
]);

export const BaseToolPluginActionSchema = z.object({
  args: z.optional(
    z.array(
      z.object({
        description: z.string(),
        flag: z.string(), // Flag uses Commander syntax; i.e. '-q, --quiet'
        defaultValue: SupportedFlagValuesSchema,
      })
    )
  ),
  hook: z
    .function()
    .args(z.optional(z.record(z.string(), SupportedFlagValuesSchema)))
    .returns(z.union([z.void(), z.promise(z.void())])),
});

export const ToolPluginActionSchema = BaseToolPluginActionSchema.extend({
  action: z.string(),
  helpText: z.string(),
});

export const ToolPluginSubCommandsSchema = z.object({
  login: z.optional(BaseToolPluginActionSchema),
  deploy: z.optional(BaseToolPluginActionSchema),
});

export const ToolPluginSchema = z.object({
  name: z.string(),
  keyword: z.string(),
  actions: z.array(ToolPluginActionSchema),
  subCommands: z.optional(ToolPluginSubCommandsSchema),
});

export type SupportedFlagValues = z.infer<typeof SupportedFlagValuesSchema>;
export type BaseToolPluginAction = z.infer<typeof BaseToolPluginActionSchema>;
export type ToolPluginAction = z.infer<typeof ToolPluginActionSchema>;
export type ToolPlugin = z.infer<typeof ToolPluginSchema>;
export type SpecialAction = keyof z.infer<typeof ToolPluginSubCommandsSchema>;

const SEPARATOR = '===========================';

/**
 * Executes the command given, returning the contents of STDOUT.
 *
 * The contents of STDOUT are hidden during execution. This function will
 * also print out the commands that will be run for transparency to the user.
 */
export function cliCommand(command: string, options?: string): void {
  const commandString = command + (options ? ` ${options}` : '');
  process.stderr.write(`Running ${clc.bold(commandString)}...\n${SEPARATOR}\n`);

  try {
    execSync(commandString, { stdio: 'inherit', encoding: 'utf8' });
  } catch (e) {
    process.stderr.write(`${SEPARATOR}\n\n`);
    throw e;
  }

  process.stderr.write(`${SEPARATOR}\n\n`);
}
