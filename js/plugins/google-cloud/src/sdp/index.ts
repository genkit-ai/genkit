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

import { generateMiddleware } from 'genkit';

// Option 1: Configure redaction options inline.
export interface SdpInlineConfig {
  /**
   * 1. Select which infoTypes to inspect and replace.
   * Default: ['CREDIT_CARD_NUMBER', 'EMAIL_ADDRESS', 'PHONE_NUMBER']
   * All available infoTypes: https://cloud.google.com/sensitive-data-protection/docs/infotypes-reference#descriptions
   */
  infoTypes?: string[];

  /**
   * 2. Select the transformation method.
   * - 'INFOTYPE' (default): Replaces with the type name (e.g., [EMAIL_ADDRESS])
   * - 'MASK': Replaces characters with a symbol (e.g., *****)
   * - 'CUSTOM_STRING': Replaces with a fixed string (e.g., [REDACTED])
   */
  transformation?: 'INFOTYPE' | 'MASK' | 'CUSTOM_STRING';

  /** Masking configuration when transformation is 'MASK'. Defaults to '*' } */
  maskConfig?: { maskingCharacter: string };

  /** Custom string configuration when transformation is 'CUSTOM_STRING'. Defaults to '[REDACTED]' */
  customConfig?: string;
}

// Option 2: Create a custom config template in the Google Cloud Console.
// Create in Google Cloud Console > Security > Sensitive Data Protection > Configuration.
// Instructions: https://docs.cloud.google.com/sensitive-data-protection/docs/create-inspection-template
export interface SdpTemplateConfig {
  inspectTemplateName: string;
  deidentifyTemplateName: string;
}

export interface SdpOptions {
  templates?: SdpTemplateConfig;
  inline?: SdpInlineConfig;

  projectId?: string; // (Optional) Explicitly set the Google Cloud Project ID
}

export const sensitiveDataProtection = generateMiddleware<SdpOptions>(
  { name: 'sensitiveDataProtection' },
  ({ config, pluginConfig }) => {
    return {
      generate: async (envelope, ctx, next) => {
        // Intercept input
        const res = await next(envelope, ctx);
        // Intercept output
        return res;
      },
    };
  }
);
