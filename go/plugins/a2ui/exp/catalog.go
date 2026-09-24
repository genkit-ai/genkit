// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// SPDX-License-Identifier: Apache-2.0

package exp

import (
	"fmt"
	"strings"
)

// CatalogComponent is a component the model may use, plus a short description
// of its props.
type CatalogComponent struct {
	// Name is the component type name, e.g. "Text". It must match the renderer
	// type.
	Name string `json:"name"`
	// Description is a one-line summary of what the component renders and when
	// to use it.
	Description string `json:"description"`
	// Props is a short, model-facing description of the component's props. Kept
	// as plain text (rather than a JSON Schema) to keep the injected prompt
	// compact.
	Props string `json:"props"`
}

// Catalog pins the set of components a surface may render. The [Surfaces]
// middleware uses it to tell the model what it may render (prompt injection)
// and to validate that emitted envelopes only reference known components. The
// renderer on the client registers a matching @a2ui/* catalog under the same
// ID.
type Catalog struct {
	// ID is a globally-unique catalog id (also used as catalogId on
	// createSurface).
	ID string `json:"id"`
	// Components are the components available in this catalog.
	Components []CatalogComponent `json:"components"`
}

// basicIconNames is the set of icon names the basic catalog's Icon component
// supports. Names outside this list render as literal text (the renderer
// degrades gracefully), so the prompt lists them to steer the model toward
// valid names. The middleware validates component types against the catalog but
// does not validate individual Icon name values.
var basicIconNames = []string{
	"accountCircle", "add", "arrowBack", "arrowForward", "attachFile",
	"calendarToday", "call", "camera", "check", "close", "delete", "download",
	"edit", "event", "error", "fastForward", "favorite", "favoriteOff",
	"folder", "help", "home", "info", "locationOn", "lock", "lockOpen", "mail",
	"menu", "moreVert", "moreHoriz", "notificationsOff", "notifications",
	"pause", "payment", "person", "phone", "photo", "play", "print", "refresh",
	"rewind", "search", "send", "settings", "share", "shoppingCart", "skipNext",
	"skipPrevious", "star", "starHalf", "starOff", "stop", "upload",
	"visibility", "visibilityOff", "volumeDown", "volumeMute", "volumeOff",
	"volumeUp", "warning",
}

// BasicCatalog returns the A2UI "Basic Catalog" (v0.9), mirroring the
// components published by @a2ui/web_core's basic catalog. Use it to render
// standard UI without defining your own design system.
func BasicCatalog() *Catalog {
	return &Catalog{
		ID: BasicCatalogID,
		Components: []CatalogComponent{
			{
				Name: "Text",
				Description: "Displays a run of text. For headings/titles set the `variant` prop " +
					"(h1..h5) rather than embedding Markdown; the text itself may use inline Markdown.",
				Props: "text: string (required); variant?: one of h1|h2|h3|h4|h5|caption|body.",
			},
			{
				Name:        "Image",
				Description: "Displays an image from a URL.",
				Props:       "url: string (required); description?: string; fit?: contain|cover|fill|none|scaleDown; variant?: icon|avatar|smallFeature|mediumFeature|largeFeature|header.",
			},
			{
				Name: "Icon",
				Description: "Displays a named material icon. `name` MUST be one of the exact names " +
					"listed below — do NOT invent names (e.g. there is no \"cloud\", \"air\", or " +
					"\"thermostat\"). If none fits, omit the Icon rather than guessing.",
				Props: fmt.Sprintf("name: one of %s (required, exact).", strings.Join(basicIconNames, ", ")),
			},
			{
				Name:        "Row",
				Description: "Lays out children horizontally.",
				Props:       "children: string[] of component ids (required); justify?: start|center|end|spaceAround|spaceBetween|spaceEvenly|stretch; align?: start|center|end|stretch.",
			},
			{
				Name:        "Column",
				Description: "Lays out children vertically.",
				Props:       "children: string[] of component ids (required); justify?: start|center|end|spaceBetween|spaceAround|spaceEvenly|stretch; align?: start|center|end|stretch.",
			},
			{
				Name:        "List",
				Description: "A list of children.",
				Props:       "children: string[] of component ids (required); direction?: vertical|horizontal; listStyle?: ordered|unordered|none.",
			},
			{
				Name:        "Card",
				Description: "A visually-contained card wrapping a single child.",
				Props:       "child: string id of a single child component (required; wrap multiple in a Column/Row).",
			},
			{
				Name:        "Divider",
				Description: "A horizontal or vertical separator line.",
				Props:       "axis?: horizontal|vertical.",
			},
			{
				Name:        "Button",
				Description: "A clickable button that fires an action back to the agent.",
				Props:       "child: string id of a child (usually a Text) (required); variant?: default|primary|borderless; action: { event: { name: string, context?: object } } (required — the event name is sent back to the agent when clicked).",
			},
			{
				Name:        "TextField",
				Description: "A single- or multi-line text input.",
				Props:       "label: string (required); value?: string or { path } binding; variant?: shortText|longText|number|obscured.",
			},
			{
				Name:        "CheckBox",
				Description: "A labeled checkbox.",
				Props:       "label: string (required); value: boolean or { path } binding (required).",
			},
			{
				Name:        "Slider",
				Description: "A numeric slider.",
				Props:       "max: number (required); value: number or { path } binding (required); label?: string; min?: number; step?: number.",
			},
		},
	}
}

// componentNames returns the set of component names in the catalog.
func (c *Catalog) componentNames() map[string]bool {
	set := make(map[string]bool, len(c.Components))
	for _, comp := range c.Components {
		set[comp.Name] = true
	}
	return set
}

// renderStyleTips builds the "make it look good" styling tips, scoped to the
// components the catalog actually provides so a custom catalog is never told to
// emit components it lacks (which would then fail strict validation).
func renderStyleTips(has map[string]bool) string {
	var tips []string
	var containers []string
	for _, c := range []string{"Card", "Column", "Row"} {
		if has[c] {
			containers = append(containers, c)
		}
	}
	if len(containers) > 0 {
		tips = append(tips, fmt.Sprintf(
			"- Group related content with layout components (%s) and give it a clear hierarchy.",
			strings.Join(containers, "/")))
	}
	if has["Text"] {
		tips = append(tips, "- Give titles a heading `variant` (e.g. h2/h3) and secondary text the "+
			"`caption` variant instead of embedding \"#\"/\"##\" heading markers in the text.")
	}
	var accents []string
	for _, c := range []string{"Icon", "Divider", "Image"} {
		if has[c] {
			accents = append(accents, c)
		}
	}
	if len(accents) > 0 {
		tips = append(tips, fmt.Sprintf(
			"- Use %s to add visual meaning and separate sections where it helps.",
			strings.Join(accents, "/")))
	}
	if has["Button"] {
		tips = append(tips, "- Give primary buttons `variant: \"primary\"`.")
	}
	if len(tips) == 0 {
		return ""
	}
	return "\n\nMake it look good, not bland:\n" + strings.Join(tips, "\n")
}

// renderExample builds a worked example. It uses a rich Card/Column/Text layout
// when the catalog supports it (the common case, e.g. the basic catalog);
// otherwise it falls back to a minimal example built only from components the
// catalog provides, so the example never references unknown components.
func renderExample(catalog *Catalog, has map[string]bool) string {
	if has["Card"] && has["Column"] && has["Text"] {
		return fmt.Sprintf(`

Example (a small weather card):
`+"```"+`a2ui
[
  { "createSurface": { "surfaceId": "SURFACE_ID", "catalogId": "%s" } },
  { "updateComponents": { "surfaceId": "SURFACE_ID", "components": [
    { "id": "root", "component": "Card", "child": "body" },
    { "id": "body", "component": "Column", "children": ["title", "temp"] },
    { "id": "title", "component": "Text", "text": "Weather in Tokyo", "variant": "h3" },
    { "id": "temp", "component": "Text", "text": { "path": "/temp" } }
  ] } },
  { "updateDataModel": { "surfaceId": "SURFACE_ID", "path": "/temp", "value": "18°C" } }
]
`+"```", catalog.ID)
	}
	rootComponent := "Text"
	if len(catalog.Components) > 0 {
		rootComponent = catalog.Components[0].Name
	}
	return fmt.Sprintf(`

Example (a minimal surface):
`+"```"+`a2ui
[
  { "createSurface": { "surfaceId": "SURFACE_ID", "catalogId": "%s" } },
  { "updateComponents": { "surfaceId": "SURFACE_ID", "components": [
    { "id": "root", "component": "%s" }
  ] } }
]
`+"```", catalog.ID, rootComponent)
}

// RenderCatalogInstructions renders a catalog into model-facing instructions
// describing the A2UI protocol and the available components. It is injected
// into the system prompt by the middleware when Instructions is
// InstructionsSystem.
func RenderCatalogInstructions(catalog *Catalog) string {
	if catalog == nil {
		return ""
	}
	var componentDocs strings.Builder
	for i, c := range catalog.Components {
		if i > 0 {
			componentDocs.WriteString("\n")
		}
		fmt.Fprintf(&componentDocs, "- %s: %s Props: %s", c.Name, c.Description, c.Props)
	}

	has := catalog.componentNames()
	styleSection := renderStyleTips(has)
	exampleSection := renderExample(catalog, has)

	// Forms guidance only applies if the catalog has input components.
	var inputs []string
	for _, c := range []string{"TextField", "CheckBox", "Slider"} {
		if has[c] {
			inputs = append(inputs, c)
		}
	}
	formsSection := ""
	if len(inputs) > 0 {
		formsSection = fmt.Sprintf(`
- Forms: input components (%s) do NOT send their values automatically.
  To capture what the user entered you MUST do BOTH of these:
  1. Bind each input's `+"`value`"+` to a data-model path, e.g.
     `+"`"+`{ "component": "TextField", "label": "Email", "value": { "path": "/email" } }`+"`"+`.
     Typing updates the data model at that path.
  2. On the submit `+"`Button`"+`, echo those same paths in
     `+"`action.event.context`"+` so their current values are sent back to you, e.g.
     `+"`"+`"context": { "email": { "path": "/email" }, "name": { "path": "/name" } }`+"`"+`.
  Without the `+"`{ path }`"+` bindings and the button `+"`context`"+`, the action arrives
  with an empty `+"`context`"+` and the entered values are lost.`, strings.Join(inputs, ", "))
	}

	return fmt.Sprintf(`# Rendering UI with A2UI

You can render rich, interactive UI (not just text) by emitting an A2UI surface.
When a result is better *shown* than *told* (weather, lists, forms, comparisons,
confirmations, anything visual or interactive), render a UI surface.

To render UI, output a single fenced code block tagged `+"`a2ui`"+` containing a JSON
array of A2UI envelope messages. You may still write normal prose before it.

Rules:
- The UI is an ADJACENCY LIST: a flat array of components. Build the tree using
  string `+"`id`"+` references, NOT nested objects. Exactly one component MUST have
  `+"`id: \"root\"`"+`.
- Every component has a `+"`component`"+` (type name) and an `+"`id`"+`. Container
  components reference their children by id via a `+"`children`"+` array; single-child
  wrappers reference one `+"`child`"+` id.
- Values can be literals, or a data-model binding `+"`{ \"path\": \"/somePath\" }`"+`.
- Use `+"`createSurface`"+` first (with `+"`catalogId`"+`), then `+"`updateComponents`"+` to add
  the component list, then optionally `+"`updateDataModel`"+` to set data. You may
  combine them in one array, in order.
- Interactive components fire an `+"`action`"+` with an event `+"`name`"+`; that name is
  sent back to you when the user interacts, so choose meaningful names.%s
- When a user interacts with a surface (e.g. presses a button) and you respond
  with updated UI, RE-RENDER THE WHOLE SURFACE: start again with
  `+"`createSurface`"+` followed by `+"`updateComponents`"+`. Do not emit a bare
  `+"`updateDataModel`"+`/`+"`updateComponents`"+` expecting a previous surface to still
  exist.%s

The catalogId to use is:
"%s"

Available components:
%s%s

Do not explain the JSON; just render the block. Use "SURFACE_ID" literally as a
placeholder for the surface id — the system replaces it with a real id.`,
		formsSection, styleSection, catalog.ID, componentDocs.String(), exampleSection)
}
