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

/**
 * Metadata keys used for UI-specific schema annotations.
 */
export const GENKIT_UI_METADATA = {
  /**
   * Provides data to populate both standard and customized UI widgets.
   *
   * e.g.
   *
   * {
   *   [GENKIT_UI_METADATA.DATA_SOURCE]: {
   *     action: '/custom/foo',
   *     allowCustomValues: true
   *   }
   * }
   */
  DATA_SOURCE: 'x-genkit-ui-data-source',

  /**
   * Provides a display name or label for the UI. By default, the Dev UI will
   * transform schema fields to "Sentence case". There are times where this is
   * not desirable and needs to be overwritten, such as abbreviations or
   * acronyms (e.g. "Top P", "Top K") and proper nouns (e.g. "Google Search
   * retrieval").
   *
   * e.g. { [GENKIT_UI_METADATA.DISPLAY_NAME]: 'Top P' }
   */
  DISPLAY_NAME: 'x-genkit-ui-display-name',

  /**
   * Specifies the UI component (widget) to use for a schema field. Commonly
   * used for model and middleware configuration. Useful to resolve ambiguity
   * when multiple inputs could apply, or to provide a tailored user experience
   * for complex inputs.
   *
   * e.g. { [GENKIT_UI_METADATA.WIDGET]: GENKIT_UI_WIDGETS.MODEL_LIST }
   */
  WIDGET: 'x-genkit-ui-widget',
} as const;

/**
 * Standard UI widget names used with GENKIT_UI_METADATA.WIDGET.
 */
export const GENKIT_UI_WIDGETS = {
  /** A widget for configuring a list of models. */
  MODEL_LIST: 'model-list-config',
  /** A widget for configuring LLM model safety settings. */
  SAFETY_SETTINGS: 'safety-settings',
} as const;
