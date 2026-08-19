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
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/core"
	"github.com/firebase/genkit/go/core/logger"
	"github.com/firebase/genkit/go/genkit"
)

// registerFlowAndToolCases covers the "Flows and tools" audit section:
// streaming chunk rendering, classified provider errors on an error flow
// (ccfe1093d overhaul), interrupt/resume from the UI, multi-turn code
// execution custom parts (GGA-4), and a tool name containing "/" (GGA-25).
// Also home for the trace cases of "Traces and logs": tool-loop spans,
// thinking parts, raw request/response visibility (GGA-49).
func registerFlowAndToolCases(g *genkit.Genkit) {
	// Tier A: chunks must arrive in the UI one at a time, not as a single
	// batch, so each is delayed enough to defeat buffering.
	genkit.DefineStreamingFlow(g, "streamingCounter",
		func(ctx context.Context, count int, sendChunk core.StreamCallback[string]) (string, error) {
			if count <= 0 {
				count = 5
			}
			for i := 1; i <= count; i++ {
				if err := sendChunk(ctx, fmt.Sprintf("chunk %d of %d", i, count)); err != nil {
					return "", err
				}
				time.Sleep(400 * time.Millisecond)
			}
			return fmt.Sprintf("streamed %d chunks", count), nil
		},
	)

	// Tier A: exercises log streaming into the Dev UI (ccfe1093d) at each
	// level the console handler forwards.
	genkit.DefineFlow(g, "loggingFlow", func(ctx context.Context, input string) (string, error) {
		log := logger.FromContext(ctx)
		log.Info("loggingFlow info line", "input", input)
		log.Warn("loggingFlow warn line", "hint", "this should render as a warning")
		log.Error("loggingFlow error line", "code", "QA_TEST_ERROR")
		return "logged 3 lines at info/warn/error", nil
	})

	// Tier A: tools appear as standalone actions in the Dev UI; no flow
	// wraps this one on purpose.
	genkit.DefineTool(g, "shoutTool", "Uppercases the input and appends an exclamation mark",
		func(ctx *ai.ToolContext, input string) (string, error) {
			return strings.ToUpper(input) + "!", nil
		},
	)
}
