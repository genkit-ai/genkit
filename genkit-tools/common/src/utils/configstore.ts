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

import Configstore from 'configstore';
import * as fs from 'fs';
import * as path from 'path';
import { toolsPackage } from './package';
import { findProjectRoot } from './utils';

const USER_SETTINGS_TAG = 'userSettings';
const PROJECT_SETTINGS_TAG = 'projectSettings';

export const configstore = new Configstore(toolsPackage.name);

export function getUserSettings(): Record<string, string | boolean | number> {
  return configstore.get(USER_SETTINGS_TAG) || {};
}

export function setUserSettings(s: Record<string, string | boolean | number>) {
  configstore.set(USER_SETTINGS_TAG, s);
}

export async function getProjectConfigStore(): Promise<Configstore> {
  const projectRoot = await findProjectRoot();
  const dotGenkitDir = path.join(projectRoot, '.genkit');
  const configFilePath = path.join(dotGenkitDir, 'genkit.json');
  return new Configstore('genkit-config', {}, { configPath: configFilePath });
}

export async function getProjectSettings(): Promise<
  Record<string, string | boolean | number>
> {
  const projectRoot = await findProjectRoot();
  const dotGenkitDir = path.join(projectRoot, '.genkit');
  if (!fs.existsSync(dotGenkitDir)) {
    return {};
  }
  const store = await getProjectConfigStore();
  return store.get(PROJECT_SETTINGS_TAG) || {};
}

export async function setProjectSettings(
  s: Record<string, string | boolean | number>
) {
  const projectRoot = await findProjectRoot();
  const dotGenkitDir = path.join(projectRoot, '.genkit');
  if (!fs.existsSync(dotGenkitDir)) {
    fs.mkdirSync(dotGenkitDir, { recursive: true });
  }
  const store = await getProjectConfigStore();
  store.set(PROJECT_SETTINGS_TAG, s);
}
