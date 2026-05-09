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

import { ai } from '../config/genkit';

export const getStoreInfoTool = ai.defineTool(
  {
    name: 'getStoreInfo',
    description:
      'Get store information including hours, location, and contact details',
  },
  async () => {
    return {
      success: true,
      store: {
        name: 'TechStore Computer Shop',
        address: '123 Tech Street, Silicon Valley, CA 94000',
        phone: '(555) 123-4567',
        email: 'info@techstore.com',
        hours: {
          monday: '9:00 AM - 7:00 PM',
          tuesday: '9:00 AM - 7:00 PM',
          wednesday: '9:00 AM - 7:00 PM',
          thursday: '9:00 AM - 7:00 PM',
          friday: '9:00 AM - 8:00 PM',
          saturday: '10:00 AM - 6:00 PM',
          sunday: 'Closed',
        },
      },
    };
  }
);
