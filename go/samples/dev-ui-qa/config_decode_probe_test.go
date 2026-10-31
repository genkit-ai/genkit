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

// TestConfigDecodeProbe shows what the Dev UI config editor's JSON turns into
// once the framework decodes it into anthropic.MessageNewParams. Fields that
// decode to nothing are the ones that vanish without an error (ANT-9, ANT-15);
// a server-side tool landing in OfTool is what makes ANT-10 misfire.
//
// Run: go test ./samples/dev-ui-qa -run TestConfigDecodeProbe -v
func TestConfigDecodeProbe(t *testing.T) {
	for _, in := range []string{
		`{"thinking":{"type":"adaptive"}}`,
		`{"thinking":{"type":"adaptiv"}}`,
		`{"thinking":{"type":"disabled"}}`,
		`{"thinking":{"type":"enabled","budget_tokens":2048}}`,
		`{"system":"be terse"}`,
		`{"system":[{"type":"text","text":"be terse"}]}`,
		`{"tools":[{"type":"web_search_20260209","name":"web_search"}]}`,
	} {
		var p anthropic.MessageNewParams
		err := json.Unmarshal([]byte(in), &p)
		out, _ := json.Marshal(p)
		t.Logf("in=%-52s err=%v thinking(en/dis/ad)=%v/%v/%v system=%d tools=%d ofTool=%v remarshal=%s",
			in, err,
			p.Thinking.OfEnabled != nil, p.Thinking.OfDisabled != nil, p.Thinking.OfAdaptive != nil,
			len(p.System), len(p.Tools), len(p.Tools) > 0 && p.Tools[0].OfTool != nil, string(out))
	}
}
