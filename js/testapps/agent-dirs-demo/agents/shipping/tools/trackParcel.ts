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

import { defineDirTool } from '@genkit-ai/agent-dirs';
import { z } from 'genkit';

export default defineDirTool(
  {
    description: 'Tracks a parcel by tracking number.',
    inputSchema: z.object({ trackingNumber: z.string() }),
    outputSchema: z.object({ status: z.string(), location: z.string() }),
  },
  async ({ trackingNumber }) => ({
    status: `Parcel ${trackingNumber} is out for delivery`,
    location: 'Springfield depot',
  })
);
