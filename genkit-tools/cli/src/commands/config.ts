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
  ANALYTICS_OPT_OUT_CONFIG_TAG,
  ConfigEvent,
  getUserSettings,
  logger,
  record,
  setUserSettings,
} from '@genkit-ai/tools-common/utils';
import * as clc from 'colorette';
import { Command } from 'commander';

export const UPDATE_NOTIFICATIONS_OPT_OUT_CONFIG_TAG =
  'updateNotificationsOptOut';

const CONFIG_TAGS: Record<
  string,
  (value: string) => string | boolean | number
> = {
  [ANALYTICS_OPT_OUT_CONFIG_TAG]: (value) => {
    let o: boolean | undefined;
    try {
      o = JSON.parse(value);
    } finally {
      if (typeof o !== 'boolean') throw new Error('Expected boolean');
      return o;
    }
  },
  [UPDATE_NOTIFICATIONS_OPT_OUT_CONFIG_TAG]: (value) => {
    let o: boolean | undefined;
    try {
      o = JSON.parse(value);
    } finally {
      if (typeof o !== 'boolean') throw new Error('Expected boolean');
      return o;
    }
  },
};

export const config = new Command('config');

config
  .description('set development environment configuration')
  .command('get')
  .argument('<tag>', `The config tag to get. One of [${readableTagsHint()}]`)
  .action((tag) => {
    if (!CONFIG_TAGS[tag]) {
      logger.error(
        `Unknown config tag "${clc.bold(tag)}.\nValid options: ${readableTagsHint()}`
      );
      return;
    }

    const userSettings = getUserSettings();
    if (userSettings[tag] !== undefined) {
      logger.info(userSettings[tag]);
    } else {
      logger.info(clc.italic('(unset)'));
    }
  });

config
  .command('set')
  .argument('<tag>', `The config tag to get. One of [${readableTagsHint()}]`)
  .argument('<value>', 'The value to set tag to')
  .action(async (tag, value) => {
    if (!CONFIG_TAGS[tag]) {
      logger.error(
        `Unknown config tag "${clc.bold(tag)}.\nValid options: ${readableTagsHint()}`
      );
      return;
    }

    let parsedValue: string | boolean | number;
    try {
      parsedValue = CONFIG_TAGS[tag](value);
    } catch (e: any) {
      logger.error(`Invalid type for "${clc.bold(tag)}.\n${e.message}`);
      return;
    }

    await record(new ConfigEvent(tag));

    const userSettings = getUserSettings();
    setUserSettings({
      ...userSettings,
      [tag]: parsedValue,
    });

    logger.info(`Set "${clc.bold(tag)}" to "${clc.bold(value)}".`);
  });

function readableTagsHint() {
  return Object.keys(CONFIG_TAGS).map(clc.bold).join(', ');
}
