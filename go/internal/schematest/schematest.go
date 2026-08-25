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

// Package schematest provides assertions for the JSON schemas a plugin
// publishes, so that the fields a user configures in the Dev UI stay
// documented.
package schematest

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// AssertDescribed reports every property in schema, at any depth, that lacks a
// description or whose description does not end in a period.
//
// Schema inference does not read Go doc comments: [github.com/invopop/jsonschema]
// runs without a comment map, so a field is described only by a struct tag. The
// Dev UI renders that description as the tooltip on the help icon beside the
// field, so a field without one offers the user no explanation at all.
//
// name identifies the schema in failure messages, e.g. the middleware name.
func AssertDescribed(t *testing.T, name string, schema map[string]any) {
	t.Helper()
	assertDescribed(t, name, schema)
}

func assertDescribed(t *testing.T, path string, schema map[string]any) {
	t.Helper()
	props, _ := schema["properties"].(map[string]any)
	names := make([]string, 0, len(props))
	for n := range props {
		names = append(names, n)
	}
	sort.Strings(names)
	for _, n := range names {
		prop, ok := props[n].(map[string]any)
		if !ok {
			continue
		}
		field := path + "." + n
		desc, _ := prop["description"].(string)
		switch {
		case strings.TrimSpace(desc) == "":
			t.Errorf("%s: no description; add a `jsonschema_description:\"...\"` tag (a Go doc comment alone does not reach the schema)", field)
		case !strings.HasSuffix(desc, "."):
			t.Errorf("%s: description does not end in a period: %q", field, desc)
		}
		assertDescribed(t, field, prop)
	}
	if items, ok := schema["items"].(map[string]any); ok {
		assertDescribed(t, path+"[]", items)
	}
}

// AssertNoInlineDescriptions reports every `jsonschema:"description=..."` tag
// in the .go files directly under dir, which must instead be written as a
// `jsonschema_description` tag.
//
// [github.com/invopop/jsonschema] splits the `jsonschema` tag on commas to
// separate its keywords, so an inline description silently loses everything
// from its first comma onward: `description=If true, descend into
// subdirectories.` reaches the schema as "If true". The dedicated tag is read
// whole, so punctuation needs no thought.
//
//	`json:"recursive,omitempty" jsonschema_description:"If true, descend into subdirectories."`
//
// Both forms produce the same schema, and a field may carry the dedicated tag
// alongside a `jsonschema` tag holding other keywords such as an enum.
//
// Unlike [AssertDescribed], this reads the source rather than a published
// schema, so it also covers schemas a plugin never exposes as a descriptor,
// such as tool input types.
func AssertNoInlineDescriptions(t *testing.T, dir string) {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("read %s: %v", dir, err)
	}
	fset := token.NewFileSet()
	checked := 0
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".go") {
			continue
		}
		path := filepath.Join(dir, e.Name())
		file, err := parser.ParseFile(fset, path, nil, 0)
		if err != nil {
			t.Fatalf("parse %s: %v", path, err)
		}
		checked++
		ast.Inspect(file, func(n ast.Node) bool {
			t.Helper()
			field, ok := n.(*ast.Field)
			if !ok || field.Tag == nil {
				return true
			}
			raw, err := strconv.Unquote(field.Tag.Value)
			if err != nil {
				return true
			}
			tag := reflect.StructTag(raw).Get("jsonschema")
			for _, seg := range strings.Split(tag, ",") {
				// Segments are trimmed before the check because the library
				// does not trim them itself: " description=..." matches no
				// keyword and is dropped without a word, so the field ends up
				// with no description at all. Reporting it here is the only
				// thing that catches that on a type no descriptor publishes.
				if !strings.HasPrefix(strings.TrimSpace(seg), "description=") {
					continue
				}
				t.Errorf("%s: a description inside a jsonschema tag does not survive the library's comma splitting; move it to a jsonschema_description tag\n\tfull tag: %s",
					fmt.Sprintf("%s:%d", path, fset.Position(field.Tag.Pos()).Line), tag)
			}
			return true
		})
	}
	if checked == 0 {
		t.Fatalf("no .go files found in %s", dir)
	}
}
