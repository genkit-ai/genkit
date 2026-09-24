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
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/genkit"
)

func TestRenderCatalogInstructionsBasic(t *testing.T) {
	text := RenderCatalogInstructions(BasicCatalog())

	for _, want := range []string{
		"Rendering UI with A2UI",
		BasicCatalogID,
		"- Text:",
		"- Button:",
		"Forms:",            // basic catalog has input components
		"Make it look good", // basic catalog has containers
		"Example (a small weather card):",
		"SURFACE_ID",
	} {
		if !strings.Contains(text, want) {
			t.Errorf("instructions missing %q", want)
		}
	}
}

func TestRenderCatalogInstructionsCustomNoInputs(t *testing.T) {
	catalog := &Catalog{
		ID: "https://example.com/catalog.json",
		Components: []CatalogComponent{
			{Name: "Banner", Description: "A banner.", Props: "title: string."},
		},
	}
	text := RenderCatalogInstructions(catalog)

	if strings.Contains(text, "Forms:") {
		t.Error("no-input catalog should not include forms guidance")
	}
	if !strings.Contains(text, "- Banner:") {
		t.Error("instructions missing custom component")
	}
	// The minimal example should use the only component as root, not reference
	// components the catalog lacks.
	if strings.Contains(text, `"component": "Card"`) {
		t.Error("example should not reference components absent from the catalog")
	}
	if !strings.Contains(text, "Example (a minimal surface):") {
		t.Error("expected minimal example fallback")
	}
}

func TestLoadCatalogFile(t *testing.T) {
	g := genkit.Init(context.Background())
	dir := t.TempDir()
	path := filepath.Join(dir, "catalog.json")
	content := `{"id":"https://x/c.json","components":[{"name":"Text","description":"d","props":"p"}]}`
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}

	catalog, err := LoadCatalogFile(g, path)
	if err != nil {
		t.Fatal(err)
	}
	if catalog.ID != "https://x/c.json" {
		t.Errorf("id = %q, want https://x/c.json", catalog.ID)
	}
	if len(catalog.Components) != 1 || catalog.Components[0].Name != "Text" {
		t.Errorf("components = %v, want one Text component", catalog.Components)
	}

	// It must be registered under the a2ui-catalog value type so tooling can
	// enumerate it (GET /api/values?type=a2ui-catalog).
	v := genkit.LookupValue(g, catalogRegistryKey("https://x/c.json"))
	if v == nil {
		t.Fatal("catalog was not registered in the registry")
	}
	if _, ok := v.(*Catalog); !ok {
		t.Errorf("registered value type = %T, want %T", v, (*Catalog)(nil))
	}
}

func TestLoadCatalogFileErrors(t *testing.T) {
	g := genkit.Init(context.Background())
	if _, err := LoadCatalogFile(g, filepath.Join(t.TempDir(), "missing.json")); err == nil {
		t.Error("expected error for missing file")
	}

	dir := t.TempDir()
	bad := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(bad, []byte("{ not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadCatalogFile(g, bad); err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestLoadCatalogIdempotent(t *testing.T) {
	g := genkit.Init(context.Background())
	c := &Catalog{ID: "https://x/dup.json", Components: []CatalogComponent{{Name: "Text", Description: "d", Props: "p"}}}
	if err := LoadCatalog(g, c); err != nil {
		t.Fatal(err)
	}
	// A second registration of the same id must not panic or error.
	if err := LoadCatalog(g, c); err != nil {
		t.Fatalf("re-registering the same catalog id should be idempotent, got %v", err)
	}
}

func TestListValuesSurfacesCatalog(t *testing.T) {
	g := genkit.Init(context.Background())
	if err := RegisterBasicCatalog(g); err != nil {
		t.Fatal(err)
	}
	custom := &Catalog{ID: "https://x/custom.json", Components: []CatalogComponent{{Name: "Banner", Description: "b", Props: "p"}}}
	if err := LoadCatalog(g, custom); err != nil {
		t.Fatal(err)
	}

	// Emulate the Dev UI's GET /api/values?type=a2ui-catalog: filter ListValues
	// by the a2ui-catalog prefix.
	prefix := "/" + CatalogValueType + "/"
	found := map[string]bool{}
	for key := range genkit.ListValues(g) {
		if strings.HasPrefix(key, prefix) {
			found[strings.TrimPrefix(key, prefix)] = true
		}
	}
	if !found[BasicCatalogID] {
		t.Errorf("basic catalog not listed under %s; found=%v", CatalogValueType, found)
	}
	if !found["https://x/custom.json"] {
		t.Errorf("custom catalog not listed under %s; found=%v", CatalogValueType, found)
	}
}

func TestResolveCatalog(t *testing.T) {
	g := genkit.Init(context.Background())
	custom := &Catalog{ID: "https://x/res.json", Components: []CatalogComponent{{Name: "Banner", Description: "b", Props: "p"}}}
	if err := LoadCatalog(g, custom); err != nil {
		t.Fatal(err)
	}

	// Explicit catalog wins.
	inline := &Catalog{ID: "inline"}
	if got, _ := resolveCatalog(g, inline, "https://x/res.json"); got != inline {
		t.Errorf("explicit catalog should take precedence")
	}

	// CatalogID resolves from the registry.
	got, err := resolveCatalog(g, nil, "https://x/res.json")
	if err != nil {
		t.Fatal(err)
	}
	if got.ID != "https://x/res.json" {
		t.Errorf("resolved id = %q, want https://x/res.json", got.ID)
	}

	// Empty id falls back to the bundled basic catalog.
	got, err = resolveCatalog(g, nil, "")
	if err != nil {
		t.Fatal(err)
	}
	if got.ID != BasicCatalogID {
		t.Errorf("default resolved id = %q, want basic catalog id", got.ID)
	}

	// Unknown id errors.
	if _, err := resolveCatalog(g, nil, "https://x/nope.json"); err == nil {
		t.Error("expected error resolving an unregistered catalog id")
	}
}
