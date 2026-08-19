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

import "github.com/firebase/genkit/go/genkit"

// registerFlowAndToolCases covers the "Flows and tools" audit section:
// streaming chunk rendering, classified provider errors on an error flow
// (ccfe1093d overhaul), interrupt/resume from the UI, multi-turn code
// execution custom parts (GGA-4), and a tool name containing "/" (GGA-25).
// Also home for the trace cases of "Traces and logs": tool-loop spans,
// thinking parts, raw request/response visibility (GGA-49).
func registerFlowAndToolCases(g *genkit.Genkit) {
	// TODO: streaming flow, error flow, interrupt tool, slash-named tool,
	// code-execution flow.
	_ = g
}
