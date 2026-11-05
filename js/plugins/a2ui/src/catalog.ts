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
 * A2UI catalog description used by the {@link a2ui} middleware.
 *
 * A catalog pins the set of components a surface may render. The middleware uses
 * it for two things: (1) telling the model what it may render (prompt
 * injection), and (2) validating emitted envelopes only reference known
 * components. The renderer on the client registers a matching `@a2ui/*` catalog
 * under the same `id`.
 *
 * @module
 */

import { BASIC_CATALOG_ID } from './types.js';

/**
 * The registry value type under which A2UI catalogs are stored, so the
 * {@link a2ui} middleware can look them up by id (and, in the future, tooling
 * can list them). Register with `ai.registry.registerValue(...)` via
 * {@link loadCatalog}.
 */
export const A2UI_CATALOG_VALUE_TYPE = 'a2ui-catalog';

/**
 * The default catalog id used by the {@link a2ui} middleware when none is
 * given. Resolves to the bundled {@link basicCatalog}.
 */
export const DEFAULT_CATALOG_ID = 'basic';

/** A component the model may use, plus a short description of its props. */
export interface A2uiCatalogComponent {
  /** The component type name, e.g. `Text`. */
  name: string;
  /** One-line description of what the component renders. */
  description: string;
  /**
   * A short, model-facing description of the component's props. Kept as plain
   * text (rather than a JSON Schema) to keep the injected prompt compact.
   */
  props: string;
}

/** A parsed catalog: an id plus the components it exposes. */
export interface A2uiCatalog {
  /** Globally-unique catalog id (also used as `catalogId` on `createSurface`). */
  id: string;
  /** The components available in this catalog. */
  components: A2uiCatalogComponent[];
}

/**
 * The set of icon names the basic catalog's `Icon` component supports. Names
 * outside this list render as literal text (the renderer degrades gracefully),
 * so the prompt lists them to steer the model toward valid names. Note the
 * middleware validates component *types* against the catalog but does not
 * validate individual `Icon` name values.
 */
export const BASIC_ICON_NAMES = [
  'accountCircle',
  'add',
  'arrowBack',
  'arrowForward',
  'attachFile',
  'calendarToday',
  'call',
  'camera',
  'check',
  'close',
  'delete',
  'download',
  'edit',
  'event',
  'error',
  'fastForward',
  'favorite',
  'favoriteOff',
  'folder',
  'help',
  'home',
  'info',
  'locationOn',
  'lock',
  'lockOpen',
  'mail',
  'menu',
  'moreVert',
  'moreHoriz',
  'notificationsOff',
  'notifications',
  'pause',
  'payment',
  'person',
  'phone',
  'photo',
  'play',
  'print',
  'refresh',
  'rewind',
  'search',
  'send',
  'settings',
  'share',
  'shoppingCart',
  'skipNext',
  'skipPrevious',
  'star',
  'starHalf',
  'starOff',
  'stop',
  'upload',
  'visibility',
  'visibilityOff',
  'volumeDown',
  'volumeMute',
  'volumeOff',
  'volumeUp',
  'warning',
] as const;

/**
 * The A2UI "Basic Catalog" (v0.9), mirroring the components published by
 * `@a2ui/web_core`'s basic catalog. Use this to render standard UI without
 * defining your own design system.
 */
export const basicCatalog: A2uiCatalog = {
  id: BASIC_CATALOG_ID,
  components: [
    {
      name: 'Text',
      description:
        'Displays a run of text. For headings/titles set the `variant` prop ' +
        '(h1..h5) rather than embedding Markdown; the text itself may use ' +
        'inline Markdown.',
      props:
        'text: string (required); variant?: one of h1|h2|h3|h4|h5|caption|body.',
    },
    {
      name: 'Image',
      description: 'Displays an image from a URL.',
      props:
        'url: string (required); description?: string; fit?: contain|cover|fill|none|scaleDown; variant?: icon|avatar|smallFeature|mediumFeature|largeFeature|header.',
    },
    {
      name: 'Icon',
      description:
        'Displays a named material icon. `name` MUST be one of the exact names ' +
        'listed below — do NOT invent names (e.g. there is no "cloud", "air", ' +
        'or "thermostat"). If none fits, omit the Icon rather than guessing.',
      props: `name: one of ${BASIC_ICON_NAMES.join(', ')} (required, exact).`,
    },
    {
      name: 'Row',
      description: 'Lays out children horizontally.',
      props:
        'children: string[] of component ids (required); justify?: start|center|end|spaceAround|spaceBetween|spaceEvenly|stretch; align?: start|center|end|stretch.',
    },
    {
      name: 'Column',
      description: 'Lays out children vertically.',
      props:
        'children: string[] of component ids (required); justify?: start|center|end|spaceBetween|spaceAround|spaceEvenly|stretch; align?: start|center|end|stretch.',
    },
    {
      name: 'List',
      description: 'A list of children.',
      props:
        'children: string[] of component ids (required); direction?: vertical|horizontal; listStyle?: ordered|unordered|none.',
    },
    {
      name: 'Card',
      description: 'A visually-contained card wrapping a single child.',
      props:
        'child: string id of a single child component (required; wrap multiple in a Column/Row).',
    },
    {
      name: 'Divider',
      description: 'A horizontal or vertical separator line.',
      props: 'axis?: horizontal|vertical.',
    },
    {
      name: 'Button',
      description: 'A clickable button that fires an action back to the agent.',
      props:
        'child: string id of a child (usually a Text) (required); variant?: default|primary|borderless; action: { event: { name: string, context?: object } } (required — the event name is sent back to the agent when clicked).',
    },
    {
      name: 'TextField',
      description: 'A single- or multi-line text input.',
      props:
        'label: string (required); value?: string or { path } binding; variant?: shortText|longText|number|obscured.',
    },
    {
      name: 'CheckBox',
      description: 'A labeled checkbox.',
      props:
        'label: string (required); value: boolean or { path } binding (required).',
    },
    {
      name: 'Slider',
      description: 'A numeric slider.',
      props:
        'max: number (required); value: number or { path } binding (required); label?: string; min?: number; step?: number.',
    },
  ],
};

/**
 * Builds the "make it look good" styling tips, scoped to the components the
 * catalog actually provides so a custom catalog is never told to emit
 * components it lacks (which would then fail `validate: 'strict'`).
 */
function renderStyleTips(has: Set<string>): string {
  const tips: string[] = [];
  const containers = ['Card', 'Column', 'Row'].filter((c) => has.has(c));
  if (containers.length) {
    tips.push(
      `- Group related content with layout components (${containers.join(
        '/'
      )}) and give it a clear hierarchy.`
    );
  }
  if (has.has('Text')) {
    tips.push(
      '- Give titles a heading `variant` (e.g. h2/h3) and secondary text the ' +
        '`caption` variant instead of embedding "#"/"##" heading markers in ' +
        'the text.'
    );
  }
  const accents = ['Icon', 'Divider', 'Image'].filter((c) => has.has(c));
  if (accents.length) {
    tips.push(
      `- Use ${accents.join(
        '/'
      )} to add visual meaning and separate sections where it helps.`
    );
  }
  if (has.has('Button')) {
    tips.push('- Give primary buttons `variant: "primary"`.');
  }
  return tips.length
    ? `\n\nMake it look good, not bland:\n${tips.join('\n')}`
    : '';
}

/**
 * Builds a worked example. Uses a rich Card/Column/Text layout when the catalog
 * supports it (the common case, e.g. the basic catalog); otherwise falls back
 * to a minimal example built only from components the catalog provides, so the
 * example never references unknown components.
 */
function renderExample(catalog: A2uiCatalog, has: Set<string>): string {
  if (has.has('Card') && has.has('Column') && has.has('Text')) {
    return `

Example (a small weather card):
\`\`\`a2ui
[
  { "createSurface": { "surfaceId": "SURFACE_ID", "catalogId": "${catalog.id}" } },
  { "updateComponents": { "surfaceId": "SURFACE_ID", "components": [
    { "id": "root", "component": "Card", "child": "body" },
    { "id": "body", "component": "Column", "children": ["title", "temp"] },
    { "id": "title", "component": "Text", "text": "Weather in Tokyo", "variant": "h3" },
    { "id": "temp", "component": "Text", "text": { "path": "/temp" } }
  ] } },
  { "updateDataModel": { "surfaceId": "SURFACE_ID", "path": "/temp", "value": "18°C" } }
]
\`\`\``;
  }
  // Minimal fallback: root uses whatever the catalog's first component is.
  const rootComponent = catalog.components[0]?.name ?? 'Text';
  return `

Example (a minimal surface):
\`\`\`a2ui
[
  { "createSurface": { "surfaceId": "SURFACE_ID", "catalogId": "${catalog.id}" } },
  { "updateComponents": { "surfaceId": "SURFACE_ID", "components": [
    { "id": "root", "component": "${rootComponent}" }
  ] } }
]
\`\`\``;
}

/**
 * Renders a catalog into model-facing instructions describing the A2UI protocol
 * and the available components. Injected into the system prompt by the
 * middleware when `instructions !== 'none'`.
 */
export function renderCatalogInstructions(catalog: A2uiCatalog): string {
  const componentDocs = catalog.components
    .map((c) => `- ${c.name}: ${c.description} Props: ${c.props}`)
    .join('\n');

  const has = new Set(catalog.components.map((c) => c.name));
  const styleSection = renderStyleTips(has);
  const exampleSection = renderExample(catalog, has);

  // Forms guidance only applies if the catalog has input components.
  const inputs = ['TextField', 'CheckBox', 'Slider'].filter((c) => has.has(c));
  const inputList = inputs.join(', ');
  const formsSection = inputs.length
    ? `
- Forms: input components (${inputList}) do NOT send their values automatically.
  To capture what the user entered you MUST do BOTH of these:
  1. Bind each input's \`value\` to a data-model path, e.g.
     \`{ "component": "TextField", "label": "Email", "value": { "path": "/email" } }\`.
     Typing updates the data model at that path.
  2. On the submit \`Button\`, echo those same paths in
     \`action.event.context\` so their current values are sent back to you, e.g.
     \`"context": { "email": { "path": "/email" }, "name": { "path": "/name" } }\`.
  Without the \`{ path }\` bindings and the button \`context\`, the action arrives
  with an empty \`context\` and the entered values are lost.`
    : '';

  return `# Rendering UI with A2UI

You can render rich, interactive UI (not just text) by emitting an A2UI surface.
When a result is better *shown* than *told* (weather, lists, forms, comparisons,
confirmations, anything visual or interactive), render a UI surface.

To render UI, output a single fenced code block tagged \`a2ui\` containing a JSON
array of A2UI envelope messages. You may still write normal prose before it.

Rules:
- The UI is an ADJACENCY LIST: a flat array of components. Build the tree using
  string \`id\` references, NOT nested objects. Exactly one component MUST have
  \`id: "root"\`.
- Every component has a \`component\` (type name) and an \`id\`. Container
  components reference their children by id via a \`children\` array; single-child
  wrappers reference one \`child\` id.
- Values can be literals, or a data-model binding \`{ "path": "/somePath" }\`.
- Use \`createSurface\` first (with \`catalogId\`), then \`updateComponents\` to add
  the component list, then optionally \`updateDataModel\` to set data. You may
  combine them in one array, in order.
- Interactive components fire an \`action\` with an event \`name\`; that name is
  sent back to you when the user interacts, so choose meaningful names.${formsSection}
- When a user interacts with a surface (e.g. presses a button) and you respond
  with updated UI, RE-RENDER THE WHOLE SURFACE: start again with
  \`createSurface\` followed by \`updateComponents\`. Do not emit a bare
  \`updateDataModel\`/\`updateComponents\` expecting a previous surface to still
  exist.${styleSection}

The catalogId to use is:
"${catalog.id}"

Available components:
${componentDocs}${exampleSection}

Do not explain the JSON; just render the block. Use "SURFACE_ID" literally as a
placeholder for the surface id — the system replaces it with a real id.`;
}

/** The literal placeholder the model is told to use for surface ids. */
export const SURFACE_ID_PLACEHOLDER = 'SURFACE_ID';
