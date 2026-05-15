// Copyright 2025 Google LLC
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

// A simple, self-contained code generator for JSON Schema.
// It converts a JSON Schema to equivalent Go api.
package main

import (
	"bytes"
	"cmp"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"go/format"
	"log"
	"os"
	"path"
	"path/filepath"
	"slices"
	"strings"
	"unicode"

	gocmp "github.com/google/go-cmp/cmp"
	"golang.org/x/exp/maps"
)

var (
	outputDir  = flag.String("outdir", "", "directory to write to, or '-' for stdout")
	noFormat   = flag.Bool("nofmt", false, "do not format output")
	configFile = flag.String("config", "", "config filename")

	// fieldOmitEmptyTag maps schemas (e.g., "ModelResponseChunk") to fields (e.g., "index")
	// that should not receive the `omitempty` JSON tag.
	fieldOmitEmptyTag = map[string]map[string]struct{}{
		"ModelResponseChunk": {
			"index": {}, // fields should be as defined in core/schemas.config
		},
		"Operation": {
			"action": {},
			"done":   {},
			"id":     {},
		},
	}
)

func main() {
	log.SetPrefix("jsonschemagen: ")
	log.SetFlags(0)
	flag.Usage = func() {
		out := flag.CommandLine.Output()
		fmt.Fprintf(out, "usage: jsonschemagen JSON_SCHEMA_FILE DEFAULT_PKG\n")
		flag.PrintDefaults()
	}
	flag.Parse()
	if flag.NArg() != 2 {
		flag.Usage()
		os.Exit(1)
	}
	if err := run(flag.Arg(0), flag.Arg(1), *configFile, *outputDir); err != nil {
		log.Fatal(err)
	}
}

func run(infile, defaultPkgPath, configFile, outRoot string) error {
	// Unmarshal the file, which is a JSON object with a "$defs" key that contains
	// all the type definitions.
	data, err := os.ReadFile(infile)
	if err != nil {
		return err
	}

	var schema *Schema
	if err := json.Unmarshal(data, &schema); err != nil {
		return err
	}

	// Verify that we've captured all the information in the file.
	// This checks that our Schema struct represents enough of JSONSchema to
	// capture everything in the input file.
	// (Even if we put everything in JSONSchema into our struct, there's still no
	// guarantee that JSONSchema won't add more, or that the input file contains
	// some extension that we don't know about.)
	if err := checkSchemaIsComplete(schema, data); err != nil {
		return err
	}

	// Read the config file, if any.
	var cfg config
	if configFile != "" {
		cfg, err = parseConfigFile(configFile)
		if err != nil {
			return err
		}
	}

	// The defs field of the top-level schema is the only interesting part.
	// It is a map from type name to JSON schema for that api.
	schemas := schema.Defs

	// Many of the types are anonymous, used directly as the type of fields.
	// We would end up generating something like
	//     someField struct { x int; y bool}
	// While that is legal Go, it is hard to construct values of those api.
	// Hoist all anonymous types to top level and name them.
	// We do this as a transformation on the map of schemas.
	nameAnonymousTypes(schemas)

	// Split schemas by package.
	schemasByPackage := map[string]map[string]*Schema{}
	for name, s := range schemas {
		pkgPath := defaultPkgPath
		if ic := cfg.configFor(name); ic != nil && ic.pkgPath != "" {
			pkgPath = ic.pkgPath
		}
		m := schemasByPackage[pkgPath]
		if m == nil {
			m = map[string]*Schema{}
			schemasByPackage[pkgPath] = m
		}
		m[name] = s
	}

	// Generate code by package.
	for pkgPath, schemaMap := range schemasByPackage {
		// Generate code for each type in the package.
		gen := &generator{
			pkgName: path.Base(pkgPath),
			schemas: schemaMap,
			cfg:     cfg,
		}
		src, err := gen.generate()
		if err != nil {
			return err
		}

		// Format and write the source.
		if !*noFormat {
			src, err = format.Source(src)
			if err != nil {
				return fmt.Errorf("file for package %s: %w", pkgPath, err)
			}
		}

		if outRoot == "" {
			outRoot = "."
		}
		outDir := filepath.Join(outRoot, filepath.FromSlash(pkgPath))
		if err := os.MkdirAll(outDir, 0o755); err != nil {
			return err
		}
		outFile := filepath.Join(outDir, "gen.go")
		if err := os.WriteFile(outFile, src, 0o644); err != nil {
			return err
		}
		log.Printf("wrote %s\n", outFile)
	}

	// Report unused config.
	for name := range cfg.itemConfigs {
		if !cfg.used[name] {
			log.Printf("config %s unused", name)
		}
	}
	return nil
}

// checkSchemaIsComplete compares the given schema to the original JSON it was unmarshaled
// from to see if the schema is missing anything.
func checkSchemaIsComplete(s *Schema, orig []byte) error {
	var want, got any
	if err := json.Unmarshal(orig, &want); err != nil {
		return err
	}
	data, err := json.Marshal(s)
	if err != nil {
		return err
	}
	if err := json.Unmarshal(data, &got); err != nil {
		return err
	}
	// Normalize both sides before comparison
	adjustAdditionalProperties(want)
	adjustAdditionalProperties(got)
	diff := gocmp.Diff(want, got)
	if diff != "" {
		return fmt.Errorf("mismatch (-want, -got):\n%s", diff)
	}
	return nil
}

// adjustAdditionalProperties changes additionalProperties keys with the value {not{}} to false.
// It is needed because [Schema.UnmarshalJSON] does the opposite, and we want to compare with
// the input schema, which always uses false.
func adjustAdditionalProperties(x any) {
	if m, ok := x.(map[string]any); ok {
		for k, v := range m {
			if k == "additionalProperties" {
				// TODO: For some reason, m[k] = true fails but m[k] = false works
				// To fix this we convert true=>{} so that schema generation works

				// Convert true to {}
				if b, ok := v.(bool); ok && b {
					m[k] = map[string]any{}
				}
				// Convert {not:{}} to false
				if vm, ok := v.(map[string]any); ok && len(vm) == 1 {
					if nm, ok := vm["not"].(map[string]any); ok && len(nm) == 0 {
						m[k] = false
					}
				}
			}
			if k == "properties" {
				if pm, ok := v.(map[string]any); ok && len(pm) == 0 {
					delete(m, k)
				}
			}
			// TODO: Fix this - causing schemagen issues
			if k == "uniqueItems" {
				delete(m, k)
			}
			adjustAdditionalProperties(v)
		}
	} else if a, ok := x.([]any); ok {
		for _, e := range a {
			adjustAdditionalProperties(e)
		}
	}
}

// refPrefix is the common prefix for all "ref" schema elements.
// All references in this file are to other definitions in the same file.
const refPrefix = "#/$defs/"

// nameAnonymousTypes replaces anonymous types in the schemas with a reference to a named
// type, which it constructs and adds to the map.
func nameAnonymousTypes(schemas map[string]*Schema) {
	var nameFields func(prefix string, props map[string]*Schema)
	nameFields = func(prefix string, props map[string]*Schema) {
		for fieldName, fs := range props {
			if fs.Enum != nil || (fs.Type.Any() == "object" && fs.Properties != nil) {
				fname := adjustIdentifier(fieldName)
				newName := prefix + fname
				schemas[newName] = fs
				props[fieldName] = &Schema{Ref: refPrefix + newName}
				nameFields(prefix+fname, fs.Properties)
			}
		}
	}
	for typeName, ts := range schemas {
		nameFields(typeName, ts.Properties)
	}
}

const license = `
// Copyright 2025 Google LLC
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

`

type generator struct {
	pkgName string
	schemas map[string]*Schema
	cfg     config
	pr      func(string, ...any)
}

// generate produces Go source for the types in schemas.
func (g *generator) generate() ([]byte, error) {
	var buf bytes.Buffer

	g.pr = func(format string, args ...any) { fmt.Fprintf(&buf, format, args...) }

	g.pr("%s\n\n", license)
	g.pr("// This file was generated by jsonschemagen. DO NOT EDIT.\n\n")
	g.pr("package %s\n\n", g.pkgName)

	if pc := g.cfg.configFor(g.pkgName); pc != nil {
		g.pr("import %q\n", pc.pkgPath)
	}

	// Sort the names so the output is deterministic.
	for _, name := range sortedKeys(g.schemas) {
		if ic := g.cfg.configFor(name); ic != nil && ic.omit {
			continue
		}
		if err := g.generateType(name); err != nil {
			return nil, err
		}
	}
	return buf.Bytes(), nil
}

func (g *generator) generateType(name string) (err error) {
	defer func() {
		if err != nil {
			err = fmt.Errorf("%s: %w", name, err)
		}
	}()

	s := g.schemas[name]
	tcfg := g.cfg.configFor(name)
	if tcfg == nil {
		tcfg = &itemConfig{}
	}
	if s.Type.Any() == nil {
		if s.AllOf != nil {
			log.Printf("WARNING: %s: cannot handle allOf", name)
			return nil
		}
		if s.AnyOf != nil {
			log.Printf("WARNING: %s: cannot handle anyOf", name)
			return nil
		}
		return errors.New("no type")
	}
	typ, ok := s.Type.Any().(string)
	if !ok {
		return fmt.Errorf("cannot handle multiple types: %v", s.Type)
	}
	if s.Enum != nil {
		if typ != "string" {
			return fmt.Errorf("don't understand enum with type %q", typ)
		}
		return g.generateStringEnum(name, s, tcfg)
	}

	// Check if we have a typeExpr in the config before checking the type
	if tcfg != nil && tcfg.typeExpr != "" {
		log.Printf("Using typeExpr from config for %s: %q", name, tcfg.typeExpr)
		// Use the type from config directly
		g.generateDoc(s, tcfg)
		goName := tcfg.name
		if goName == "" {
			goName = adjustIdentifier(name)
		}
		g.pr("type %s %s\n\n", goName, tcfg.typeExpr)
		return nil
	}

	switch typ {
	case "object": // a JSONSchema object corresponds to a Go struct
		if s.Properties == nil && s.AdditionalProperties != nil {
			typ, err := g.typeExpr(s)
			if err != nil {
				return err
			}
			g.generateDoc(s, tcfg)
			goName := tcfg.name
			if goName == "" {
				goName = adjustIdentifier(name)
			}
			g.pr("type %s %s\n\n", goName, typ)
			return nil
		}
		if err := g.generateStruct(name, s, tcfg); err != nil {
			return err
		}
	case "array": // a slice
		if s.Items == nil {
			return fmt.Errorf("array type without items")
		}
		// Generate a type alias for the array
		itemType, err := g.typeExpr(s.Items)
		if err != nil {
			return fmt.Errorf("items: %w", err)
		}
		g.generateDoc(s, tcfg)
		goName := tcfg.name
		if goName == "" {
			goName = adjustIdentifier(name)
		}
		g.pr("type %s []%s\n\n", goName, itemType)
	default:
		return fmt.Errorf("don't understand type %q", typ)
	}
	return nil
}

func (g *generator) generateStruct(name string, s *Schema, tcfg *itemConfig) error {
	g.generateDoc(s, tcfg)
	goName := tcfg.name
	if goName == "" {
		goName = adjustIdentifier(name)
	}
	g.pr("type %s struct {\n", goName)
	for _, field := range sortedKeys(s.Properties) {
		fcfg := g.cfg.configFor(name + "." + field)
		if fcfg == nil {
			fcfg = &itemConfig{}
		}
		if fcfg.omit {
			continue
		}
		fs := s.Properties[field]
		// Ignore properties with a non-empty "not" constraint.
		// They are probably the result of inheriting from a base zod type with a "never" constraint.
		// E.g. see EmptyPartSchema and its subtypes in js/ai/src/model.ts.
		if fs.Not != nil {
			continue
		}
		if fcfg.omit {
			continue
		}
		typeExpr := fcfg.typeExpr
		if typeExpr == "" {
			var err error
			typeExpr, err = g.typeExpr(fs)
			if err != nil {
				return fmt.Errorf("%s: %w", field, err)
			}
		}
		g.generateDoc(fs, fcfg)

		jsonTag := fmt.Sprintf(`json:"%s,omitempty"`, field)
		if skipOmitEmpty(goName, field) {
			jsonTag = fmt.Sprintf(`json:"%s"`, field)
		}
		fieldName := fcfg.name
		if fieldName == "" {
			fieldName = adjustIdentifier(field)
		}
		g.pr("  %s %s `%s`\n", fieldName, typeExpr, jsonTag)
	}
	for _, f := range tcfg.fields {
		g.pr("  %s %s\n", f.name, f.typeExpr)
	}
	g.pr("}\n\n")
	return nil
}

// skipOmitEmpty determines whether a schema field should include the
// `omitempty` JSON tag
func skipOmitEmpty(schema, field string) bool {
	fields, ok := fieldOmitEmptyTag[schema]
	if !ok {
		return false
	}
	_, ok = fields[field]
	return ok
}

func (g *generator) generateStringEnum(name string, s *Schema, tcfg *itemConfig) error {
	g.generateDoc(s, tcfg)
	goName := tcfg.name
	if goName == "" {
		goName = adjustIdentifier(name)
	}
	g.pr("type %s string\n", goName)
	g.pr("const (\n")
	for _, v := range s.Enum {
		goVName := goName + adjustIdentifier(v)
		if ic := g.cfg.configFor(goVName); ic != nil {
			if ic.name != "" {
				goVName = ic.name
			}
			g.generateDoc(s, ic)
		}
		g.pr(`  %s %s = "%s"`, goVName, goName, v)
		g.pr("\n")
	}
	g.pr(")\n\n")
	return nil
}

func (g *generator) generateDoc(s *Schema, ic *itemConfig) {
	var lines []string
	if len(ic.docLines) > 0 {
		lines = ic.docLines
	} else if s.Description != "" {
		lines = []string{s.Description}
	}
	if len(lines) > 0 {
		for _, line := range lines {
			g.pr("// %s\n", line)
		}
	}
}

// typeExpr returns a Go type expression denoting the type represented by the schema.
func (g *generator) typeExpr(s *Schema) (string, error) {
	// A reference to another type refers to that type by name. Use the name.
	if s == nil {
		return "any", nil
	}
	if s.Ref != "" {
		s2, name, err := g.resolveRef(s.Ref)
		if err != nil {
			return "", err
		}
		// Nested refs (e.g. "#/$defs/Foo/properties/bar") don't correspond to a
		// generated Go type; inline the resolved sub-schema's type instead.
		if strings.Count(s.Ref, "/") > 2 {
			return g.typeExpr(s2)
		}
		ic := g.cfg.configFor(name)
		if s2 == nil {
			// If there is no schema, perhaps there is a config value.
			if ic != nil && ic.name != "" {
				return ic.name, nil
			}
			return "", fmt.Errorf("unknown type in reference: %q", name)
		}
		// Apply a config that changes the name.
		if ic != nil && ic.name != "" {
			name = ic.name
		}
		if s2.Enum != nil {
			return name, nil
		}
		// If it's not an enum, it's a struct. Use a pointer to it.
		return "*" + name, nil
	}
	// If there is no specified type, assume the schema represents any type.
	if s.Type.Any() == nil {
		return "any", nil
	}
	typ, ok := s.Type.Any().(string)
	if !ok {
		return "", fmt.Errorf("%+v: type not a string", s)
	}
	switch typ {
	case "object": // a struct or map
		if s.Properties == nil {
			// An object with no properties is a map.
			// The key type is always string.
			// The value type is in the additionalProperties schema.
			if s.AdditionalProperties == nil {
				return "", errors.New("empty additionalProperties")
			}
			vte, err := g.typeExpr(s.AdditionalProperties)
			if err != nil {
				return "", err
			}
			return "map[string]" + vte, nil
		}
		// This is an inline struct, which is not going to go well.
		log.Printf("WARNING: ignoring inline struct %v", s.Properties)
		return "any", nil
	case "array": // a slice
		el, err := g.typeExpr(s.Items)
		if err != nil {
			return "", err
		}
		return fmt.Sprintf("[]%s", el), nil
	case "string":
		if s.Enum != nil {
			log.Printf("WARNING: ignoring enum %v", s.Enum)
			return "string", nil
		}
		return "string", nil
	case "boolean":
		return "bool", nil
	case "number":
		return "float64", nil
	case "":
		// Assume the empty schema, which means any api.
		return "any", nil
	default:
		return "", fmt.Errorf("typeExpr can't handle type %q", typ)
	}
}

// resolveRef resolves a JSON schema reference.
// It handles simple references like "#/$defs/Action" and nested ones like
// "#/$defs/ActionMetadata/properties/inputJsonSchema".
func (g *generator) resolveRef(ref string) (*Schema, string, error) {
	name, ok := strings.CutPrefix(ref, refPrefix)
	if !ok {
		return nil, "", fmt.Errorf("ref %q does not begin with prefix %q", ref, refPrefix)
	}
	parts := strings.Split(name, "/")
	s, ok := g.schemas[parts[0]]
	if !ok {
		return nil, "", fmt.Errorf("unknown type in reference: %q", parts[0])
	}
	for i := 1; i < len(parts); i++ {
		switch parts[i] {
		case "properties":
			if i+1 >= len(parts) {
				return nil, "", fmt.Errorf("invalid ref (ends in properties): %q", ref)
			}
			s = s.Properties[parts[i+1]]
			i++
		case "additionalProperties":
			s = s.AdditionalProperties
		case "items":
			s = s.Items
		default:
			return nil, "", fmt.Errorf("cannot handle ref segment %q in %q", parts[i], ref)
		}
		if s == nil {
			return nil, "", fmt.Errorf("ref path not found: %q", ref)
		}
	}
	// The caller mostly cares about the direct name in $defs (parts[0]).
	return s, parts[0], nil
}

// adjustIdentifier returns name with the first letter capitalized
// so it is exported, and makes other idiomatic Go adjustments.
func adjustIdentifier(name string) string {
	// Replace hyphens with camel case
	if strings.Contains(name, "-") {
		parts := strings.Split(name, "-")
		for i := 1; i < len(parts); i++ {
			if len(parts[i]) > 0 {
				parts[i] = strings.ToUpper(parts[i][:1]) + parts[i][1:]
			}
		}
		name = strings.Join(parts, "")
	}

	// "Id" is common; change to "ID".
	if pre, ok := strings.CutSuffix(name, "Id"); ok {
		name = pre + "ID"
	} else if pre, ok := strings.CutSuffix(name, "Ids"); ok {
		name = pre + "IDs"
	}
	return fmt.Sprintf("%c%s", unicode.ToUpper(rune(name[0])), name[1:])
}

func sortedKeys[K cmp.Ordered, V any](m map[K]V) []K {
	keys := maps.Keys(m)
	slices.Sort(keys)
	return keys
}

// config is the configuration for a schema file.
// It describes modifications to the defaults of the code generator.
type config struct {
	itemConfigs map[string]*itemConfig
	used        map[string]bool // which itemConfigs were used
}

func (c config) configFor(name string) *itemConfig {
	if ic := c.itemConfigs[name]; ic != nil {
		c.used[name] = true
		return ic
	}
	return nil
}

// itemConfig is configuration for one item: a type, a field or a package.
// Not all itemConfig fields apply to both, but using one type simplifies the parser.
type itemConfig struct {
	omit     bool
	name     string
	pkgPath  string
	typeExpr string
	docLines []string
	fields   []extraField
}

// extraField represents an additional unexported field to add to a struct.
type extraField struct {
	name     string
	typeExpr string
}

// parseConfigFile parses the config file.
// The config file is line-oriented. Empty lines and lines beginning
// with '#' are ignored.
// Other lines start with a word which names either a package, a type or the
// field of a type (as api.FIELD).
// Except for packages, the names are always the original JSONSchema names, not Go names.
// The rest of the line is a directive; one of
//
//	omit
//	    don't generate code for this item
//	name NAME
//	    use NAME instead of the default name
//	type EXPR
//	    use EXPR for the type expression (for fields only)
//	doc
//	    doc is following lines until the line "."
//	pkg
//	    package path, relative to outdir (last component is package name)
//	import
//	    path of package to import (for packages only)
//	field NAME TYPE
//	    add an unexported field to the struct (for types only)
func parseConfigFile(filename string) (config, error) {
	c := config{
		itemConfigs: map[string]*itemConfig{},
		used:        map[string]bool{},
	}
	filedata, err := os.ReadFile(filename)
	if err != nil {
		return config{}, err
	}

	var n int // line number

	errf := func(format string, args ...any) (config, error) {
		return config{}, fmt.Errorf("%s:%d: %s", filename, n, fmt.Sprintf(format, args...))
	}

	var docItem *itemConfig // if non-empty, collect doc lines here
	for i, ln := range bytes.Split(filedata, []byte("\n")) {
		n = i + 1
		line := strings.TrimSpace(string(ln))
		if docItem != nil {
			if line == "." {
				docItem = nil
			} else {
				docItem.docLines = append(docItem.docLines, line)
			}
			continue
		}
		if len(line) == 0 || line[0] == '#' {
			continue
		}
		words := strings.Fields(line)
		if len(words) < 2 {
			return errf("need NAME DIRECTIVE ...")
		}
		ic := c.itemConfigs[words[0]]
		if ic == nil {
			ic = &itemConfig{}
			c.itemConfigs[words[0]] = ic
		}
		switch words[1] {
		case "omit":
			ic.omit = true
		case "name":
			if len(words) < 3 {
				return errf("need NAME name NEWNAME")
			}
			ic.name = words[2]
		case "type":
			if len(words) < 3 {
				return errf("need NAME type EXPR")
			}
			ic.typeExpr = words[2]
		case "doc":
			docItem = ic
		case "pkg":
			if len(words) < 3 {
				return errf("need NAME pkg PATH")
			}
			ic.pkgPath = words[2]
		case "import":
			if len(words) < 3 {
				return errf("need NAME import PATH")
			}
			ic.pkgPath = words[2]
		case "field":
			if len(words) < 4 {
				return errf("need NAME field FIELDNAME TYPE")
			}
			ic.fields = append(ic.fields, extraField{name: words[2], typeExpr: words[3]})
		default:
			return errf("unknown directive %q", words[1])
		}
	}
	return c, nil
}
