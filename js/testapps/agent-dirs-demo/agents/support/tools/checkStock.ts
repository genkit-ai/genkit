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

// Factory-form tool: the native-API escape hatch. The factory owns the name.

import { z } from 'genkit';
import type { GenkitBeta } from 'genkit/beta';

export default (ai: GenkitBeta) =>
  ai.defineTool(
    {
      name: 'checkStock',
      description: 'Checks whether a product is in stock.',
      inputSchema: z.object({ sku: z.string() }),
      outputSchema: z.object({ inStock: z.boolean() }),
    },
    async ({ sku }) => ({ inStock: !sku.endsWith('0') })
  );
