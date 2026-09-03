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
	"log/slog"
	"reflect"

	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/internal/base"
)

// LoadCatalog registers an A2UI catalog in the Genkit registry under the key
// `/a2ui-catalog/<id>` (using the catalog's own ID), so the [ai.Middleware] can
// resolve it by id via [Surfaces.CatalogID], and tooling such as the Dev UI can
// enumerate catalogs (GET /api/values?type=a2ui-catalog). This mirrors the JS
// and Dart plugins, keeping the catalog representation identical across
// runtimes.
//
// Re-registering the same id is idempotent (the existing registration is kept),
// so calling it more than once, or registering the basic catalog twice, is
// safe. Registering a different catalog under an existing id keeps the original
// and logs a warning, so an edited catalog re-loaded under the same id is not
// silently ignored. It uses a register-if-absent primitive, so concurrent
// callers racing on the same id cannot panic.
//
// Example:
//
//	a2uix.LoadCatalog(g, myCatalog)
//	// ... then reference it by id:
//	ai.WithUse(&a2uix.Surfaces{CatalogID: myCatalog.ID})
func LoadCatalog(g *genkit.Genkit, catalog *Catalog) error {
	if catalog == nil {
		return fmt.Errorf("a2ui: LoadCatalog: catalog is nil")
	}
	if catalog.ID == "" {
		return fmt.Errorf("a2ui: LoadCatalog: catalog has no ID")
	}
	if catalog.Components == nil {
		return fmt.Errorf("a2ui: LoadCatalog: catalog %q has no components", catalog.ID)
	}
	key := catalogRegistryKey(catalog.ID)
	// Register-if-absent closes the check-then-register race against
	// RegisterValue (which panics on a duplicate key): concurrent LoadCatalog
	// calls for the same id all resolve to the same stored catalog, and the
	// losers simply observe that an entry already existed.
	if !genkit.DefineValueIfAbsent(g, key, catalog) {
		// Compare by content, not pointer identity: BasicCatalog() and a
		// re-read LoadCatalogFile each allocate a fresh *Catalog, so a pointer
		// check would warn on genuinely idempotent re-registration and stay
		// silent for none of the cases the warning exists for. DeepEqual warns
		// only when the incoming catalog actually differs from the stored one.
		if existing, ok := genkit.LookupValue(g, key).(*Catalog); ok && !reflect.DeepEqual(existing, catalog) {
			slog.Warn("a2ui: LoadCatalog: a different catalog is already registered under this id; keeping the existing one",
				"id", catalog.ID)
		}
	}
	return nil
}

// LoadCatalogFile reads an A2UI catalog from a JSON file and registers it with
// [LoadCatalog]. The file must contain an object with an "id" string and a
// "components" array (see [Catalog]).
func LoadCatalogFile(g *genkit.Genkit, path string) (*Catalog, error) {
	catalog, err := readCatalogFile(path)
	if err != nil {
		return nil, err
	}
	if err := LoadCatalog(g, catalog); err != nil {
		return nil, err
	}
	return catalog, nil
}

// RegisterBasicCatalog registers the bundled [BasicCatalog] in the registry so
// it appears alongside custom catalogs in tooling. The middleware falls back to
// the basic catalog even without this call; register it explicitly to surface
// it in the Dev UI. Idempotent.
func RegisterBasicCatalog(g *genkit.Genkit) error {
	return LoadCatalog(g, BasicCatalog())
}

// readCatalogFile reads and parses a catalog from a JSON file (without
// registering it). base.ReadJSONFile handles the read-plus-decode; only the
// a2ui-specific error wrapping and the components check live here.
func readCatalogFile(path string) (*Catalog, error) {
	var catalog Catalog
	if err := base.ReadJSONFile(path, &catalog); err != nil {
		return nil, fmt.Errorf("a2ui: failed to read catalog file %q: %w", path, err)
	}
	if catalog.Components == nil {
		return nil, fmt.Errorf("a2ui: catalog file %q must have a \"components\" array", path)
	}
	return &catalog, nil
}

// resolveCatalog resolves the catalog for a turn, given the middleware config.
// Resolution order: an explicit *Catalog on the config, then a CatalogID looked
// up from the registry, then the bundled basic catalog for the default id.
// g may be nil (e.g. a bare-registry test), in which case only the explicit
// catalog or the basic fallback are available.
func resolveCatalog(g *genkit.Genkit, catalog *Catalog, catalogID string) (*Catalog, error) {
	if catalog != nil {
		return catalog, nil
	}
	id := catalogID
	if id == "" {
		id = DefaultCatalogID
	}
	if g != nil {
		if v := genkit.LookupValue(g, catalogRegistryKey(id)); v != nil {
			if c, ok := v.(*Catalog); ok {
				return c, nil
			}
			return nil, fmt.Errorf("a2ui: registry value %q is not an A2UI catalog", catalogRegistryKey(id))
		}
	}
	// Fall back to the bundled basic catalog for the default id (and also when
	// a caller referenced the basic catalog by its full id).
	if id == DefaultCatalogID || id == BasicCatalogID {
		return BasicCatalog(), nil
	}
	return nil, fmt.Errorf(
		"a2ui: no catalog registered under id %q; register one with LoadCatalog(g, catalog) or use the default %q catalog",
		id, DefaultCatalogID)
}
