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
	"github.com/firebase/genkit/go/ai"
	aix "github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/internal/genkitbridge"
)

// DefineTool defines a tool with a simplified function signature, registers it
// as an action of type Tool, and returns an [aix.Tool].
//
// Unlike [genkit.DefineTool], the function receives a plain context.Context
// instead of [ai.ToolContext]. Use [tool.AttachParts] inside the function to
// return additional content parts alongside the output.
//
// For tools that don't need to be registered (e.g., dynamically created tools),
// use [aix.NewTool] instead.
//
// # Options
//
//   - [ai.WithInputSchema]: Provide a custom JSON schema instead of inferring from the type parameter
//   - [ai.WithInputSchemaName]: Reference a pre-registered schema by name
//
// Example:
//
//	type WeatherInput struct {
//		City string `json:"city" jsonschema:"description=city name"`
//	}
//
//	weatherTool := exp.DefineTool(g, "getWeather", "Fetches the weather for a given city",
//		func(ctx context.Context, input WeatherInput) (string, error) {
//			if input.City == "Paris" {
//				return "Sunny, 25°C", nil
//			}
//			return "Cloudy, 18°C", nil
//		},
//	)
//
//	resp, err := genkit.Generate(ctx, g,
//		ai.WithPrompt("What's the weather like in Paris?"),
//		ai.WithTools(weatherTool),
//	)
//	if err != nil {
//		log.Fatalf("Generate failed: %v", err)
//	}
//	fmt.Println(resp.Text())
func DefineTool[In, Out any](g *genkit.Genkit, name, description string, fn aix.ToolFunc[In, Out], opts ...ai.ToolOption) *aix.Tool[In, Out] {
	return aix.DefineTool(genkitbridge.RegistryOf(g), name, description, fn, opts...)
}

// DefineInterruptibleTool defines a tool that supports typed interrupt/resume,
// registers it as an action of type Tool, and returns an
// [aix.InterruptibleTool].
//
// The function receives a plain context.Context, the tool input, and a
// resumed parameter that is non-nil when the tool is being re-executed after
// an interrupt. Inside the function, call [tool.Interrupt] to pause execution
// and send data to the caller. The caller can inspect the interrupt with
// [tool.InterruptAs] and resume the tool with [tool.Resume] or the typed
// [aix.InterruptibleTool.Resume] method.
//
// The interrupt and resume payloads (the Resume type parameter and the value
// passed to [tool.Interrupt]) must each serialize to a JSON object, i.e. a
// struct or a map, since they travel as structured metadata on the tool
// request.
//
// For tools that don't need to be registered (e.g., dynamically created tools),
// use [aix.NewInterruptibleTool] instead.
//
// # Options
//
//   - [ai.WithInputSchema]: Provide a custom JSON schema instead of inferring from the type parameter
//   - [ai.WithInputSchemaName]: Reference a pre-registered schema by name
//
// Example:
//
//	type TransferInput struct {
//		ToAccount string  `json:"toAccount"`
//		Amount    float64 `json:"amount"`
//	}
//
//	type TransferOutput struct {
//		Status  string  `json:"status"`
//		Balance float64 `json:"balance"`
//	}
//
//	type TransferInterrupt struct {
//		Reason string  `json:"reason"`
//		Amount float64 `json:"amount"`
//	}
//
//	type Confirmation struct {
//		Approved bool `json:"approved"`
//	}
//
//	transferTool := exp.DefineInterruptibleTool(g, "transfer",
//		"Transfers money to another account.",
//		func(ctx context.Context, input TransferInput, confirm *Confirmation) (*TransferOutput, error) {
//			if confirm != nil && !confirm.Approved {
//				return &TransferOutput{Status: "cancelled"}, nil
//			}
//			if confirm == nil && input.Amount > 100 {
//				// Pause and ask the caller for confirmation.
//				return nil, tool.Interrupt(TransferInterrupt{
//					Reason: "large_amount",
//					Amount: input.Amount,
//				})
//			}
//			return &TransferOutput{Status: "completed", Balance: 50}, nil
//		},
//	)
//
//	// In a generate loop, handle the interrupt:
//	resp, _ := genkit.Generate(ctx, g,
//		ai.WithPrompt("Transfer $200 to Alice"),
//		ai.WithTools(transferTool),
//	)
//	if resp.FinishReason == ai.FinishReasonInterrupted {
//		for _, interrupt := range resp.Interrupts() {
//			// Ask the user for confirmation.
//			restart, _ := tool.Resume(interrupt, Confirmation{Approved: true})
//			resp, _ = genkit.Generate(ctx, g,
//				ai.WithMessages(resp.History()...),
//				ai.WithTools(transferTool),
//				ai.WithToolRestarts(restart),
//			)
//		}
//	}
func DefineInterruptibleTool[In, Out, Resume any](g *genkit.Genkit, name, description string, fn aix.InterruptibleToolFunc[In, Out, Resume], opts ...ai.ToolOption) *aix.InterruptibleTool[In, Out, Resume] {
	return aix.DefineInterruptibleTool(genkitbridge.RegistryOf(g), name, description, fn, opts...)
}
