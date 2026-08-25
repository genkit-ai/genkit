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
	"testing"

	"github.com/firebase/genkit/go/internal/schematest"
)

// TestConfigSchemasDocumented asserts that every field the Dev UI renders for
// a middleware config carries a description. Schema inference does not read Go
// doc comments -- only a `jsonschema_description:"..."` struct tag reaches the
// schema -- so a field documented in Go alone shows up as a bare input box.
//
// Walking the descriptors rather than a fixed list of types means a new
// middleware, or a new field on an existing one, cannot ship undocumented.
func TestConfigSchemasDocumented(t *testing.T) {
	descs, err := (&Middleware{}).Middlewares(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(descs) == 0 {
		t.Fatal("plugin returned no middleware descriptors")
	}
	for _, d := range descs {
		if d.Description == "" {
			t.Errorf("middleware %q has no description", d.Name)
		}
		schematest.AssertDescribed(t, d.Name, d.ConfigSchema)
	}
}

// TestDescriptionsUseTheDedicatedTag guards the whole package against the
// silent truncation described on [schematest.AssertNoInlineDescriptions]. It
// covers the tool input schemas too, which no middleware descriptor exposes.
func TestDescriptionsUseTheDedicatedTag(t *testing.T) {
	schematest.AssertNoInlineDescriptions(t, ".")
}
