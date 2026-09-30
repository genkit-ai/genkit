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

import {
  findToolsConfig,
  type BaseToolPluginAction,
  type SpecialAction,
  type SupportedFlagValues,
  type ToolPlugin,
} from '@genkit-ai/tools-common/plugin';
import { logger } from '@genkit-ai/tools-common/utils';
import * as clc from 'colorette';
import { Command } from 'commander';

/** Gets plugin commands based on the tools config file, if present. */
export async function getPluginCommands(): Promise<Command[]> {
  const config = await findToolsConfig();
  return (config?.cliPlugins || []).map(pluginToCommander);
}

/** Gets special-case commands for plugins in the config file. */
export async function getPluginSubCommand(
  commandString: SpecialAction
): Promise<Command | undefined> {
  const config = await findToolsConfig();
  const actions = (config?.cliPlugins || [])
    .filter((p) => !!p.subCommands?.[commandString])
    .map((p) => ({
      keyword: p.keyword,
      name: p.name,
      ...p.subCommands![commandString]!,
    }));
  const command = new Command(commandString).description(
    `${humanReadableCommand(commandString)} using tools plugins`
  );

  if (!actions.length) {
    return undefined;
  }

  for (const a of actions) {
    const subcmd = command
      .command(a.keyword)
      .description(a.name + ' ' + clc.italic('(plugin)'));
    attachPluginActionToCommand(subcmd, a);
  }

  return command;
}

export function attachPluginActionToCommand(
  cmd: Command,
  action: BaseToolPluginAction
): void {
  for (const o of action.args || []) {
    cmd.option(o.flag, o.description, o.defaultValue);
  }
  cmd.action(async (options: Record<string, SupportedFlagValues>) => {
    await action.hook(options);
  });
}

function pluginToCommander(p: ToolPlugin): Command {
  const cmd = new Command(p.keyword).description(
    p.name + ' ' + clc.italic('(plugin)')
  );
  for (const a of p.actions) {
    const subcmd = cmd.command(a.action).description(a.helpText);
    attachPluginActionToCommand(subcmd, a);
  }

  // Default action to catch unknown commands.
  cmd.action((_, { args }: { args: string[] }) => {
    logger.error(`"${clc.bold(args[0])}" is not a known ${p.name} command.`);
    process.exitCode = 1;
  });
  return cmd;
}

function humanReadableCommand(c: string): string {
  return c[0].toUpperCase() + c.slice(1);
}
