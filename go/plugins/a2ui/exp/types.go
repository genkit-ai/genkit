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

// Package exp provides in-preview A2UI ("Agent to UI") support for Genkit
// agents.
//
// A2UI is a transport-agnostic, JSON-based streaming UI protocol
// (https://a2ui.org/). An A2UI-enabled agent can stream not just prose, but
// rich, interactive UI "surfaces" that a client renders incrementally.
//
// The whole server-side integration is the [ai.Middleware], added to a
// [github.com/firebase/genkit/go/ai.Generate] call via
// [github.com/firebase/genkit/go/ai.WithUse]. It injects the catalog's
// capabilities into the system prompt, then intercepts model output (streamed
// chunks and the final message), extracts a2ui fenced blocks, validates them
// against the catalog, and rewrites them into a2ui data parts.
//
// Examples here import this package as a2uix
// ("github.com/firebase/genkit/go/plugins/a2ui/exp"), the alias used across the
// Genkit docs and samples.
//
// APIs in this package are under active development and may change in any
// minor version release.
package exp

// A2UIMimeType identifies an A2UI payload. It is stamped onto the
// metadata.mimeType of the Genkit data part that carries A2UI envelopes,
// matching the A2A binding of the A2UI spec exactly.
const A2UIMimeType = "application/a2ui+json"

// DefaultVersion is the default A2UI protocol version stamped on emitted
// envelopes.
const DefaultVersion = "v0.9"

// SupportedVersions is the set of A2UI protocol versions the plugin can stamp on
// emitted envelopes. [Surfaces.Version] is validated against it so a typo cannot
// stamp a version the renderer will reject at runtime. Matches the JS plugin's
// SUPPORTED_VERSIONS.
var SupportedVersions = []string{"v0.9", "v0.9.1"}

// supportedVersions is SupportedVersions as a set for O(1) validation.
var supportedVersions = func() map[string]bool {
	m := make(map[string]bool, len(SupportedVersions))
	for _, v := range SupportedVersions {
		m[v] = true
	}
	return m
}()

// supportedVersionList returns the supported versions quoted, for error
// messages.
func supportedVersionList() []string {
	out := make([]string, len(SupportedVersions))
	for i, v := range SupportedVersions {
		out[i] = `"` + v + `"`
	}
	return out
}

// validValidateModes is the set of accepted [ValidateMode] values, used to
// reject a typo like "strick" that would otherwise silently downgrade strict
// validation to the warn default.
var validValidateModes = map[ValidateMode]bool{
	ValidateStrict: true,
	ValidateWarn:   true,
	ValidateOff:    true,
}

// BasicCatalogID is the catalog id of the A2UI "Basic Catalog" (v0.9).
// Surfaces created with the basic catalog reference this id, and the client
// renderer registers a catalog under the same id.
const BasicCatalogID = "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json"

// CatalogValueType is the registry value type under which A2UI catalogs are
// stored (key `/a2ui-catalog/<id>`), so the middleware can look them up by id
// and tooling (e.g. the Dev UI's GET /api/values?type=a2ui-catalog) can list
// them. Matches the JS plugin's value type.
const CatalogValueType = "a2ui-catalog"

// DefaultCatalogID is the id used when [Surfaces] specifies neither Catalog nor
// CatalogID. It resolves to the bundled basic catalog.
const DefaultCatalogID = "basic"

// catalogRegistryKey builds the registry key an A2UI catalog is stored under.
func catalogRegistryKey(id string) string {
	return "/" + CatalogValueType + "/" + id
}

// surfaceIDPlaceholder is the literal placeholder the model is told to use for
// surface ids; the middleware replaces it with a real id.
const surfaceIDPlaceholder = "SURFACE_ID"

// Envelope is a single A2UI envelope message (e.g. createSurface,
// updateComponents, updateDataModel, deleteSurface). It is represented as a
// generic JSON object because the protocol is open-ended and versioned; the
// middleware only inspects a few well-known keys.
//
// A component within an updateComponents envelope is a single entry in an A2UI
// adjacency list: UI is a flat list of components, and the tree is reconstructed
// via id references, with exactly one component having id "root". Beyond
// component/id, every component carries catalog-specific props, so components
// are handled as generic map[string]any objects rather than a dedicated type.
type Envelope = map[string]any
