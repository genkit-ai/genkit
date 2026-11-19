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

package main

import (
	"encoding/json"
	"testing"

	"github.com/anthropics/anthropic-sdk-go"
)

// TestConfigDecodeProbe pins what the Dev UI config editor's JSON turns into
// once the framework decodes it into anthropic.MessageNewParams. Fields that
// decode to nothing are the ones that vanish without an error (ANT-9, ANT-15);
// a server-side tool landing in OfTool is what makes ANT-10 misfire.
//
// The expectations record the current SDK behaviour, not desired behaviour:
// if an SDK upgrade fixes (or changes) a decode, this test fails and the
// matching audit finding needs revisiting.
//
// Run from the go/ directory: go test ./samples/dev-ui-qa -run TestConfigDecodeProbe -v
func TestConfigDecodeProbe(t *testing.T) {
	type decoded struct {
		enabled, disabled, adaptive bool
		system, tools               int
		ofTool                      bool
	}
	for _, tc := range []struct {
		in   string
		want decoded
	}{
		// Adaptive thinking decodes to nothing and does not error.
		{`{"thinking":{"type":"adaptive"}}`, decoded{}},
		{`{"thinking":{"type":"adaptiv"}}`, decoded{}},
		{`{"thinking":{"type":"disabled"}}`, decoded{disabled: true}},
		{`{"thinking":{"type":"enabled","budget_tokens":2048}}`, decoded{enabled: true}},
		// A string system prompt is dropped; only the block-array form lands.
		{`{"system":"be terse"}`, decoded{}},
		{`{"system":[{"type":"text","text":"be terse"}]}`, decoded{system: 1}},
		// A server tool lands in the custom-tool variant.
		{`{"tools":[{"type":"web_search_20260209","name":"web_search"}]}`, decoded{tools: 1, ofTool: true}},
	} {
		var p anthropic.MessageNewParams
		err := json.Unmarshal([]byte(tc.in), &p)
		if err != nil {
			t.Errorf("in=%s: unexpected decode error: %v", tc.in, err)
			continue
		}
		got := decoded{
			enabled:  p.Thinking.OfEnabled != nil,
			disabled: p.Thinking.OfDisabled != nil,
			adaptive: p.Thinking.OfAdaptive != nil,
			system:   len(p.System),
			tools:    len(p.Tools),
			ofTool:   len(p.Tools) > 0 && p.Tools[0].OfTool != nil,
		}
		if got != tc.want {
			t.Errorf("in=%s: decoded %+v, pinned %+v (SDK behaviour changed? revisit the audit finding)", tc.in, got, tc.want)
		}
		out, _ := json.Marshal(p)
		t.Logf("in=%s decoded=%+v remarshal=%s", tc.in, got, out)
	}
}
