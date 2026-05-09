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

import { z } from 'genkit';
import { ModelInfo } from 'genkit/model';
import { compatOaiImageModelRef } from '../image';

/** XAI image generation ModelRef helper, same as the OpenAI-compatible model specification. */
export function xaiImageModelRef<
  CustomOptions extends z.ZodTypeAny = z.ZodTypeAny,
>(params: {
  name: string;
  info?: ModelInfo;
  configSchema?: CustomOptions;
  config?: any;
}) {
  return compatOaiImageModelRef({ ...params, namespace: 'xai' });
}

export const SUPPORTED_IMAGE_MODELS = {
  'grok-2-image-1212': xaiImageModelRef({
    name: 'grok-2-image-1212',
  }),
};
