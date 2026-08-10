// Copyright 2026 Google LLC
//
// SPDX-License-Identifier: Apache-2.0

package status_test

import (
	"go/ast"
	"go/types"
	"strconv"
	"strings"
	"testing"

	"golang.org/x/tools/go/packages"
)

// TestErrorArgsUseTheWrapVerb pins that every error handed to [status.Errorf]
// or [status.PublicErrorf] is formatted with %w rather than %v.
//
// The two read identically: the cause's text lands in the message either way,
// so nothing about the error a user sees distinguishes them. What %v drops is
// the chain. status.Errorf records its cause from the wrapping the format
// string performs, so with %v the returned error has no cause at all and
// errors.Is, errors.As, and errors.Unwrap stop reaching past it. Nothing fails
// at the call site, and the loss only shows up wherever someone later tries to
// match the underlying error.
//
// That makes it invisible to a test of any single call site, which is why this
// one reads them all. A call that genuinely wants the text without the chain
// should format the message itself and pass no error.
func TestErrorArgsUseTheWrapVerb(t *testing.T) {
	cfg := &packages.Config{
		Mode: packages.NeedName | packages.NeedFiles | packages.NeedCompiledGoFiles |
			packages.NeedImports | packages.NeedDeps | packages.NeedTypes |
			packages.NeedSyntax | packages.NeedTypesInfo,
	}
	pkgs, err := packages.Load(cfg, "github.com/firebase/genkit/go/...")
	if err != nil {
		t.Fatalf("loading packages: %v", err)
	}
	if len(pkgs) == 0 {
		t.Fatal("no packages loaded")
	}

	errorIface := types.Universe.Lookup("error").Type().Underlying().(*types.Interface)
	var checked int

	for _, pkg := range pkgs {
		if pkg.TypesInfo == nil {
			continue
		}
		for _, file := range pkg.Syntax {
			ast.Inspect(file, func(n ast.Node) bool {
				call, ok := n.(*ast.CallExpr)
				if !ok {
					return true
				}
				fn, ok := calleeName(pkg.TypesInfo, call)
				if !ok || (fn != "Errorf" && fn != "PublicErrorf") {
					return true
				}
				// Errorf(sentinel, format, args...): the format is second and
				// the variadic arguments follow it.
				if len(call.Args) < 3 {
					return true
				}
				format, ok := stringLit(call.Args[1])
				if !ok {
					// A computed format cannot be read here. Rare, and vet's
					// printf analyzer covers the shape.
					return true
				}
				verbs := formatVerbs(format)
				for i, arg := range call.Args[2:] {
					if i >= len(verbs) {
						break
					}
					argType := pkg.TypesInfo.TypeOf(arg)
					if argType == nil || !types.Implements(argType, errorIface) {
						continue
					}
					checked++
					if verbs[i] != 'w' {
						pos := pkg.Fset.Position(call.Pos())
						t.Errorf("%s: %s formats an error (%s) with %%%c, want %%w",
							pos, fn, types.TypeString(argType, nil), verbs[i])
					}
				}
				return true
			})
		}
	}

	// A scan that matched nothing would pass silently forever.
	if checked == 0 {
		t.Fatal("no error arguments found; the scan is not reaching the call sites")
	}
	t.Logf("checked %d error arguments across %d packages", checked, len(pkgs))
}

// calleeName returns the name of the function a call selects, when the call is
// a selector on the status package.
func calleeName(info *types.Info, call *ast.CallExpr) (string, bool) {
	sel, ok := call.Fun.(*ast.SelectorExpr)
	if !ok {
		return "", false
	}
	ident, ok := sel.X.(*ast.Ident)
	if !ok {
		return "", false
	}
	pkgName, ok := info.Uses[ident].(*types.PkgName)
	if !ok || pkgName.Imported().Path() != "github.com/firebase/genkit/go/core/status" {
		return "", false
	}
	return sel.Sel.Name, true
}

// stringLit returns the value of an untyped string literal.
func stringLit(e ast.Expr) (string, bool) {
	lit, ok := e.(*ast.BasicLit)
	if !ok || lit.Kind.String() != "STRING" {
		return "", false
	}
	s, err := strconv.Unquote(lit.Value)
	if err != nil {
		return "", false
	}
	return s, true
}

// formatVerbs returns the verb of each directive in a printf format string, in
// the order the arguments supply them. A %% escape consumes no argument, and a
// * width or precision consumes one, so both are accounted for.
func formatVerbs(format string) []rune {
	var verbs []rune
	for i := 0; i < len(format); i++ {
		if format[i] != '%' {
			continue
		}
		i++
		if i >= len(format) {
			break
		}
		if format[i] == '%' {
			continue
		}
		// Flags, width, and precision sit between the % and the verb. A * in
		// either slot takes its value from an argument of its own.
		for i < len(format) && strings.ContainsRune("+-# 0123456789.", rune(format[i])) {
			i++
		}
		for i < len(format) && format[i] == '*' {
			verbs = append(verbs, '*')
			i++
			for i < len(format) && strings.ContainsRune("+-# 0123456789.", rune(format[i])) {
				i++
			}
		}
		if i < len(format) {
			verbs = append(verbs, rune(format[i]))
		}
	}
	return verbs
}
