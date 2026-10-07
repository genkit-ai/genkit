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

// Package livetest is the shared live checklist every model plugin runs
// against its provider's real API.
//
// The checklist cascades. This package is the first tier: the generate and
// agent journeys every model plugin must pass, plus [RunEmbedder] for plugins
// that serve embedders. A plugin family (the OpenAI-compatible plugins, the
// Anthropic plugins, the Google GenAI backends) wraps [Suite] in a suite of its
// own whose Run calls [Run] and then adds the family's checks. A plugin's live
// test calls its family's runner, or [Run] directly, and keeps only the checks
// that are its alone as plain subtests.
//
// # Running
//
// Live tests spend money, so they run only when GENKIT_LIVE is set:
//
//	GENKIT_LIVE=1 go test -run Live ./plugins/compat_oai/openai/
//
// GENKIT_LIVE=all also runs the expensive cases (see [Expensive]). Without
// the variable or the provider's key, a live test skips, unless the -run
// pattern names live tests: then it fails and names what is missing, since a
// run that asked for a test and quietly skipped it would read as a pass. So
// GENKIT_LIVE=1 go test ./plugins/... runs every suite whose key is set.
//
// A live test's name contains "Live", as in TestPluginLive, and a -run
// pattern names live tests when it contains "Live" too. A unit-test pattern
// that happens to match a live test, such as -run TestPlugin, skips it.
//
// # Capabilities
//
// A case that needs a capability runs when the model's registered
// [ai.ModelSupports] claims it. A claim is a promise, so a claimed capability
// whose case fails is a defect in the plugin or in its catalog. A known
// provider gap is recorded in [Suite.Skip] with its reason; a key there that
// names no case fails the run, so a typo cannot hide a case.
package livetest

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"slices"
	"sort"
	"strings"
	"testing"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core/api"
	"github.com/firebase/genkit/go/genkit"
)

// gateVar is the environment variable that opts into live tests.
const gateVar = "GENKIT_LIVE"

// Env gates a live test and returns the value of the first of names that is
// set, typically the provider's API key under its accepted spellings. Call it
// first in every live test.
//
// Without GENKIT_LIVE, or without any of names, the test skips, or fails when
// the -run pattern names live tests, since a run that asked for a test and
// quietly skipped it would read as a pass. With no names, Env only gates (for
// a local server that needs no key) and returns "". It fails a test whose
// name does not contain "Live", since -run could not name it.
func Env(t *testing.T, names ...string) string {
	t.Helper()
	if top, _, _ := strings.Cut(t.Name(), "/"); !strings.Contains(top, "Live") {
		t.Fatalf("livetest: name the test %s with Live in it, so -run Live selects it", top)
	}
	if os.Getenv(gateVar) == "" {
		msg := fmt.Sprintf("live test: set %s=1 to run it (%s=all adds the expensive cases)", gateVar, gateVar)
		if len(names) > 0 {
			msg += "; it also needs " + strings.Join(names, " or ")
		}
		if runSelected() {
			t.Fatal(msg)
		}
		t.Skip(msg)
	}
	if len(names) == 0 {
		return ""
	}
	for _, name := range names {
		if v := os.Getenv(name); v != "" {
			return v
		}
	}
	msg := fmt.Sprintf("live test: %s is set but %s is not", gateVar, strings.Join(names, " or "))
	if runSelected() {
		t.Fatal(msg)
	}
	t.Skip(msg)
	return ""
}

// Expensive skips t unless GENKIT_LIVE=all. Use it for a case that costs far
// more than a short generation: video, image or speech output, context
// caching, or a sweep over a whole catalog.
func Expensive(t *testing.T) {
	t.Helper()
	if os.Getenv(gateVar) != "all" {
		t.Skipf("expensive live case: set %s=all to run it", gateVar)
	}
}

// runSelected reports whether the -run pattern names live tests. Any other
// pattern can still match one by accident: -run TestPlugin matches
// TestPluginLive.
func runSelected() bool {
	f := flag.Lookup("test.run")
	return f != nil && strings.Contains(f.Value.String(), "Live")
}

// Init returns a Genkit instance with plugins and the experimental surface the
// agent cases need. Plugin live tests build their instance with it.
func Init(t *testing.T, plugins ...api.Plugin) *genkit.Genkit {
	t.Helper()
	return genkit.Init(context.Background(),
		genkit.WithPlugins(plugins...),
		genkit.WithExperimental(),
	)
}

// Suite describes how to drive one provider through the shared checklist.
type Suite struct {
	// Model answers the cheap prompts. Every case that needs no special
	// model runs against it. Required.
	Model ai.ModelArg
	// ReasoningModel emits thinking, usually an [ai.ModelRef] carrying the
	// plugin's thinking config. Nil skips the reasoning cases.
	ReasoningModel ai.ModelArg
	// ReasoningContent reports whether the provider returns the thinking
	// content itself. Endpoints that accept the knob but keep the content
	// server-side leave this false and still get the no-error checks.
	ReasoningContent bool
	// StreamOnlyReasoning skips the non-streaming reasoning cases, for
	// providers that only think on streaming calls.
	StreamOnlyReasoning bool
	// VisionModel accepts inline images. Nil falls back to Model when Model
	// claims media support, and skips the vision cases otherwise.
	VisionModel ai.ModelArg
	// LimitConfig is a request config for Model that caps output at a few
	// tokens, to check the length finish reason. Config types differ per
	// plugin, so the suite cannot build one. Nil skips the case.
	LimitConfig any
	// ToolResponseMedia reports whether the provider reads images returned
	// inside a tool response. Providers that cannot carry them drop them with
	// a warning, so the case only runs where they can.
	ToolResponseMedia bool
	// BadKeyPlugin is the plugin configured with an API key the provider
	// rejects, to check that the refusal classifies as UNAUTHENTICATED. Nil
	// skips the case, for plugins that authenticate ambiently.
	BadKeyPlugin api.Plugin
	// Skip maps a case name, as in "generate/tool choice none", to the reason
	// the provider cannot pass it.
	Skip map[string]string
}

// liveCase is one entry of the checklist.
type liveCase struct {
	name string
	// needs returns why the case cannot run against this suite, or "" when
	// it can.
	needs func(*runner) string
	run   func(*testing.T, *runner)
}

// runner carries what the cases share within one [Run].
type runner struct {
	g     *genkit.Genkit
	s     Suite
	ctx   context.Context
	caps  ai.ModelSupports
	tools *fixtures
	// vision is the model for the image cases, nil when none applies.
	vision ai.ModelArg
	// agents counts the agents defined so far, to keep their names unique.
	agents int
}

// Run walks the plugin registered on g through the shared checklist. Build g
// with [Init]. It defines the tools gablorken, transferFunds, lookupOrder,
// runDiagnostics and fetchSwatch and agents named livetestAgent1 onward on g,
// so call it once per Genkit instance and keep those names free in the
// plugin's own subtests.
func Run(t *testing.T, g *genkit.Genkit, s Suite) {
	t.Helper()
	if s.Model == nil {
		t.Fatal("livetest: Suite.Model is required")
	}
	r := &runner{
		g:    g,
		s:    s,
		ctx:  t.Context(),
		caps: supportsOf(t, g, s.Model),
	}
	r.tools = defineFixtures(g)
	switch {
	case s.VisionModel != nil:
		if !supportsOf(t, g, s.VisionModel).Media {
			t.Fatalf("livetest: Suite.VisionModel %q does not claim media support", s.VisionModel.Name())
		}
		r.vision = s.VisionModel
	case r.caps.Media:
		r.vision = s.Model
	}
	if s.ReasoningModel != nil {
		supportsOf(t, g, s.ReasoningModel) // fails early on an unresolvable model
	}

	groups := []struct {
		name  string
		cases []liveCase
	}{
		{"generate", generateCases()},
		{"agent", agentCases()},
	}
	known := map[string]bool{}
	for _, grp := range groups {
		for _, c := range grp.cases {
			known[grp.name+"/"+c.name] = true
		}
	}
	var unknown []string
	for name := range s.Skip {
		if !known[name] {
			unknown = append(unknown, name)
		}
	}
	if len(unknown) > 0 {
		sort.Strings(unknown)
		t.Fatalf("livetest: Suite.Skip names no case: %q", unknown)
	}

	for _, grp := range groups {
		t.Run(grp.name, func(t *testing.T) {
			for _, c := range grp.cases {
				t.Run(c.name, func(t *testing.T) {
					if reason, ok := s.Skip[grp.name+"/"+c.name]; ok {
						t.Skip("provider gap: " + reason)
					}
					if reason := c.needs(r); reason != "" {
						t.Skip(reason)
					}
					c.run(t, r)
				})
			}
		})
	}
}

// always is the needs of a case every model runs.
func always(*runner) string { return "" }

// needAll combines needs: the case runs when every one of them allows it.
func needAll(needs ...func(*runner) string) func(*runner) string {
	return func(r *runner) string {
		for _, n := range needs {
			if reason := n(r); reason != "" {
				return reason
			}
		}
		return ""
	}
}

func needTools(r *runner) string {
	if !r.caps.Tools {
		return "model does not claim tool support"
	}
	return ""
}

func needToolChoice(r *runner) string {
	if !r.caps.ToolChoice {
		return "model does not claim tool choice support"
	}
	return ""
}

func needMultiturn(r *runner) string {
	if !r.caps.Multiturn {
		return "model does not claim multi-turn support"
	}
	return ""
}

func needSystemRole(r *runner) string {
	if !r.caps.SystemRole {
		return "model does not claim system role support"
	}
	return ""
}

func needVision(r *runner) string {
	if r.vision == nil {
		return "no model in the suite claims media support"
	}
	return ""
}

func needReasoning(r *runner) string {
	if r.s.ReasoningModel == nil {
		return "Suite.ReasoningModel is not set"
	}
	return ""
}

func needNonStreamReasoning(r *runner) string {
	if reason := needReasoning(r); reason != "" {
		return reason
	}
	if r.s.StreamOnlyReasoning {
		return "provider only reasons on streaming calls"
	}
	return ""
}

// needConstrained runs a case only on a model that claims to constrain its
// output to a schema.
func needConstrained(r *runner) string {
	switch r.caps.Constrained {
	case ai.ConstrainedSupportAll, ai.ConstrainedSupportNoTools:
		return ""
	}
	return "model does not claim constrained output"
}

// needConstrainedWithTools runs a case only on a model that claims to
// constrain output on a request that also carries tools.
func needConstrainedWithTools(r *runner) string {
	if r.caps.Constrained != ai.ConstrainedSupportAll {
		return "model does not claim constrained output alongside tools"
	}
	return ""
}

// supportsOf returns the capabilities the model behind m registered. The
// descriptor holds them as native Go values, so a JSON round trip turns them
// into the typed struct.
func supportsOf(t *testing.T, g *genkit.Genkit, m ai.ModelArg) ai.ModelSupports {
	t.Helper()
	model := genkit.LookupModel(g, m.Name())
	if model == nil {
		t.Fatalf("livetest: model %q does not resolve", m.Name())
	}
	action, ok := model.(api.Action)
	if !ok {
		t.Fatalf("livetest: model %q is a %T, not an action", m.Name(), model)
	}
	meta, _ := action.Desc().Metadata["model"].(map[string]any)
	b, err := json.Marshal(meta["supports"])
	if err != nil {
		t.Fatalf("livetest: model %q supports: %v", m.Name(), err)
	}
	var s ai.ModelSupports
	if err := json.Unmarshal(b, &s); err != nil {
		t.Fatalf("livetest: model %q supports: %v", m.Name(), err)
	}
	return s
}

// containsFold reports whether s contains substr, ignoring case.
func containsFold(s, substr string) bool {
	return strings.Contains(strings.ToLower(s), strings.ToLower(substr))
}

// containsAnyFold reports whether s contains any of substrs, ignoring case.
func containsAnyFold(s string, substrs ...string) bool {
	return slices.ContainsFunc(substrs, func(sub string) bool { return containsFold(s, sub) })
}
