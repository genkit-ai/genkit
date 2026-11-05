# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Bundled basic catalog and the system-prompt instructions it produces."""

from __future__ import annotations

from dataclasses import dataclass
from string import Template
from typing import Any, cast

from ._types import BASIC_CATALOG_ID

BASIC_ICON_NAMES = (
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
)


@dataclass(frozen=True)
class A2uiCatalogComponent:
    name: str
    description: str
    props: str


@dataclass(frozen=True)
class A2uiCatalog:
    id: str
    components: tuple[A2uiCatalogComponent, ...]

    def as_value(self) -> dict[str, object]:
        return {
            'id': self.id,
            'components': [{'name': c.name, 'description': c.description, 'props': c.props} for c in self.components],
        }

    @staticmethod
    def from_value(value: object) -> A2uiCatalog | None:
        if isinstance(value, A2uiCatalog):
            return value
        if not isinstance(value, dict):
            return None
        payload = cast(dict[str, Any], value)
        catalog_id = payload.get('id')
        raw = payload.get('components')
        if not isinstance(catalog_id, str) or not catalog_id or not isinstance(raw, list):
            return None
        components: list[A2uiCatalogComponent] = []
        for item in raw:
            row = cast(dict[str, Any], item) if isinstance(item, dict) else None
            if row is None or not isinstance(row.get('name'), str):
                return None
            components.append(
                A2uiCatalogComponent(
                    name=row['name'],
                    description=row['description'] if isinstance(row.get('description'), str) else '',
                    props=row['props'] if isinstance(row.get('props'), str) else '',
                )
            )
        return A2uiCatalog(id=catalog_id, components=tuple(components))


BASIC_CATALOG = A2uiCatalog(
    id=BASIC_CATALOG_ID,
    components=(
        A2uiCatalogComponent(
            name='Text',
            description=(
                'Displays a run of text. For headings/titles set the `variant` prop '
                '(h1..h5) rather than embedding Markdown; the text itself may use '
                'inline Markdown.'
            ),
            props='text: string (required); variant?: one of h1|h2|h3|h4|h5|caption|body.',
        ),
        A2uiCatalogComponent(
            name='Image',
            description='Displays an image from a URL.',
            props=(
                'url: string (required); description?: string; '
                'fit?: contain|cover|fill|none|scaleDown; '
                'variant?: icon|avatar|smallFeature|mediumFeature|largeFeature|header.'
            ),
        ),
        A2uiCatalogComponent(
            name='Icon',
            description=(
                'Displays a named material icon. `name` MUST be one of the exact names '
                'listed below — do NOT invent names (e.g. there is no "cloud", "air", '
                'or "thermostat"). If none fits, omit the Icon rather than guessing.'
            ),
            props=f'name: one of {", ".join(BASIC_ICON_NAMES)} (required, exact).',
        ),
        A2uiCatalogComponent(
            name='Row',
            description='Lays out children horizontally.',
            props=(
                'children: string[] of component ids (required); '
                'justify?: start|center|end|spaceAround|spaceBetween|spaceEvenly|stretch; '
                'align?: start|center|end|stretch.'
            ),
        ),
        A2uiCatalogComponent(
            name='Column',
            description='Lays out children vertically.',
            props=(
                'children: string[] of component ids (required); '
                'justify?: start|center|end|spaceBetween|spaceAround|spaceEvenly|stretch; '
                'align?: start|center|end|stretch.'
            ),
        ),
        A2uiCatalogComponent(
            name='List',
            description='A list of children.',
            props=(
                'children: string[] of component ids (required); '
                'direction?: vertical|horizontal; listStyle?: ordered|unordered|none.'
            ),
        ),
        A2uiCatalogComponent(
            name='Card',
            description='A visually-contained card wrapping a single child.',
            props='child: string id of a single child component (required; wrap multiple in a Column/Row).',
        ),
        A2uiCatalogComponent(
            name='Divider',
            description='A horizontal or vertical separator line.',
            props='axis?: horizontal|vertical.',
        ),
        A2uiCatalogComponent(
            name='Button',
            description='A clickable button that fires an action back to the agent.',
            props=(
                'child: string id of a child (usually a Text) (required); '
                'variant?: default|primary|borderless; '
                'action: { event: { name: string, context?: object } } '
                '(required — the event name is sent back to the agent when clicked).'
            ),
        ),
        A2uiCatalogComponent(
            name='TextField',
            description='A single- or multi-line text input.',
            props=(
                'label: string (required); value?: string or { path } binding; '
                'variant?: shortText|longText|number|obscured.'
            ),
        ),
        A2uiCatalogComponent(
            name='CheckBox',
            description='A labeled checkbox.',
            props='label: string (required); value: boolean or { path } binding (required).',
        ),
        A2uiCatalogComponent(
            name='Slider',
            description='A numeric slider.',
            props=(
                'max: number (required); value: number or { path } binding (required); '
                'label?: string; min?: number; step?: number.'
            ),
        ),
    ),
)


FORMS_SECTION = Template("""
- Forms: input components ($input_list) do NOT send their values automatically.
  To capture what the user entered you MUST do BOTH of these:
  1. Bind each input's `value` to a data-model path, e.g.
     `{ "component": "TextField", "label": "Email", "value": { "path": "/email" } }`.
     Typing updates the data model at that path.
  2. On the submit `Button`, echo those same paths in
     `action.event.context` so their current values are sent back to you, e.g.
     `"context": { "email": { "path": "/email" }, "name": { "path": "/name" } }`.
  Without the `{ path }` bindings and the button `context`, the action arrives
  with an empty `context` and the entered values are lost.""")

WEATHER_EXAMPLE = Template("""

Example (a small weather card):
```a2ui
[
  { "createSurface": { "surfaceId": "SURFACE_ID", "catalogId": "$catalog_id" } },
  { "updateComponents": { "surfaceId": "SURFACE_ID", "components": [
    { "id": "root", "component": "Card", "child": "body" },
    { "id": "body", "component": "Column", "children": ["title", "temp"] },
    { "id": "title", "component": "Text", "text": "Weather in Tokyo", "variant": "h3" },
    { "id": "temp", "component": "Text", "text": { "path": "/temp" } }
  ] } },
  { "updateDataModel": { "surfaceId": "SURFACE_ID", "path": "/temp", "value": "18°C" } }
]
```""")

MINIMAL_EXAMPLE = Template("""

Example (a minimal surface):
```a2ui
[
  { "createSurface": { "surfaceId": "SURFACE_ID", "catalogId": "$catalog_id" } },
  { "updateComponents": { "surfaceId": "SURFACE_ID", "components": [
    { "id": "root", "component": "$root" }
  ] } }
]
```""")

INSTRUCTIONS = Template("""# Rendering UI with A2UI

You can render rich, interactive UI (not just text) by emitting an A2UI surface.
When a result is better *shown* than *told* (weather, lists, forms, comparisons,
confirmations, anything visual or interactive), render a UI surface.

To render UI, output a single fenced code block tagged `a2ui` containing a JSON
array of A2UI envelope messages. You may still write normal prose before it.

Rules:
- The UI is an ADJACENCY LIST: a flat array of components. Build the tree using
  string `id` references, NOT nested objects. Exactly one component MUST have
  `id: "root"`.
- Every component has a `component` (type name) and an `id`. Container
  components reference their children by id via a `children` array; single-child
  wrappers reference one `child` id.
- Values can be literals, or a data-model binding `{ "path": "/somePath" }`.
- Use `createSurface` first (with `catalogId`), then `updateComponents` to add
  the component list, then optionally `updateDataModel` to set data. You may
  combine them in one array, in order.
- Interactive components fire an `action` with an event `name`; that name is
  sent back to you when the user interacts, so choose meaningful names.$forms_section
- When a user interacts with a surface (e.g. presses a button) and you respond
  with updated UI, RE-RENDER THE WHOLE SURFACE: start again with
  `createSurface` followed by `updateComponents`. Do not emit a bare
  `updateDataModel`/`updateComponents` expecting a previous surface to still
  exist.$style_section

The catalogId to use is:
"$catalog_id"

Available components:
$component_docs$example_section

Do not explain the JSON; just render the block. Use "SURFACE_ID" literally as a
placeholder for the surface id — the system replaces it with a real id.""")


def render_catalog_instructions(catalog: A2uiCatalog) -> str:
    has = {c.name for c in catalog.components}
    inputs = [name for name in ('TextField', 'CheckBox', 'Slider') if name in has]
    forms_section = FORMS_SECTION.substitute(input_list=', '.join(inputs)) if inputs else ''
    return INSTRUCTIONS.substitute(
        forms_section=forms_section,
        style_section=render_style_tips(has=has),
        catalog_id=catalog.id,
        component_docs='\n'.join(f'- {c.name}: {c.description} Props: {c.props}' for c in catalog.components),
        example_section=render_example(catalog=catalog, has=has),
    )


def render_style_tips(*, has: set[str]) -> str:
    tips: list[str] = []
    containers = [c for c in ('Card', 'Column', 'Row') if c in has]
    if containers:
        tips.append(
            f'- Group related content with layout components ({"/".join(containers)}) and give it a clear hierarchy.'
        )
    if 'Text' in has:
        tips.append(
            '- Give titles a heading `variant` (e.g. h2/h3) and secondary text the '
            '`caption` variant instead of embedding "#"/"##" heading markers in '
            'the text.'
        )
    accents = [c for c in ('Icon', 'Divider', 'Image') if c in has]
    if accents:
        tips.append(f'- Use {"/".join(accents)} to add visual meaning and separate sections where it helps.')
    if 'Button' in has:
        tips.append('- Give primary buttons `variant: "primary"`.')
    if not tips:
        return ''
    return '\n\nMake it look good, not bland:\n' + '\n'.join(tips)


def render_example(*, catalog: A2uiCatalog, has: set[str]) -> str:
    if {'Card', 'Column', 'Text'} <= has:
        return WEATHER_EXAMPLE.substitute(catalog_id=catalog.id)
    root = catalog.components[0].name if catalog.components else 'Text'
    return MINIMAL_EXAMPLE.substitute(catalog_id=catalog.id, root=root)
