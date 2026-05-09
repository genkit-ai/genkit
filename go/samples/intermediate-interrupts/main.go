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

// Tool-interrupts demonstrates the tool interrupts feature in Genkit.
// It shows how to pause generation for human-in-the-loop interactions
// and resume with user input using RestartWith and RespondWith.
package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"google.golang.org/genai"
)

// TransferInput is the input schema for the transferMoney tool.
type TransferInput struct {
	ToAccount string  `json:"toAccount" jsonschema:"description=destination account ID"`
	Amount    float64 `json:"amount" jsonschema:"description=amount in dollars (e.g. 50.00 for $50)"`
}

// TransferOutput is the output schema for the transferMoney tool.
type TransferOutput struct {
	Status     string  `json:"status"`
	Message    string  `json:"message,omitempty"`
	NewBalance float64 `json:"newBalance,omitempty"`
}

// TransferInterrupt is the typed interrupt metadata for transfer issues.
type TransferInterrupt struct {
	Reason    string  `json:"reason"` // "insufficient_balance" or "confirm_large"
	ToAccount string  `json:"toAccount"`
	Amount    float64 `json:"amount"`
	Balance   float64 `json:"balance,omitempty"`
}

var accountBalance = 150.00

func main() {
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))
	reader := bufio.NewReader(os.Stdin)

	// Define the transfer tool with interrupt logic
	transferMoney := genkit.DefineTool(g, "transferMoney",
		"Transfers money to another account. Use this when the user wants to send money.",
		func(ctx *ai.ToolContext, input TransferInput) (TransferOutput, error) {
			if input.Amount > accountBalance {
				if accountBalance <= 0 {
					return TransferOutput{"rejected", "Account balance is 0. Please add funds.", accountBalance}, nil
				}
				return TransferOutput{}, ai.InterruptWith(ctx, TransferInterrupt{
					"insufficient_balance", input.ToAccount, input.Amount, accountBalance,
				})
			}

			if !ctx.IsResumed() && input.Amount > 100 {
				return TransferOutput{}, ai.InterruptWith(ctx, TransferInterrupt{
					"confirm_large", input.ToAccount, input.Amount, accountBalance,
				})
			}

			accountBalance -= input.Amount
			message := fmt.Sprintf("Transferred $%.2f to %s", input.Amount, input.ToAccount)
			if orig, ok := ai.OriginalInputAs[TransferInput](ctx); ok {
				message = fmt.Sprintf("Transferred $%.2f to %s (adjusted from $%.2f due to insufficient balance)", input.Amount, input.ToAccount, orig.Amount)
			}

			return TransferOutput{"completed", message, accountBalance}, nil
		})

	// Define the payment agent flow
	paymentAgent := genkit.DefineFlow(g, "paymentAgent", func(ctx context.Context, request string) (string, error) {
		resp, err := genkit.Generate(ctx, g,
			ai.WithModel(googlegenai.ModelRef("googleai/gemini-2.5-flash", &genai.GenerateContentConfig{
				ThinkingConfig: &genai.ThinkingConfig{ThinkingBudget: genai.Ptr[int32](0)},
			})),
			ai.WithSystem("You are a helpful payment assistant. When the user wants to transfer money, use the transferMoney tool. Always confirm the result with the user."),
			ai.WithPrompt(request),
			ai.WithTools(transferMoney),
		)
		if err != nil {
			return "", err
		}

		for resp.FinishReason == ai.FinishReasonInterrupted {
			var restarts, responses []*ai.Part

			for _, interrupt := range resp.Interrupts() {
				meta, ok := ai.InterruptAs[TransferInterrupt](interrupt)
				if !ok {
					continue
				}

				switch meta.Reason {
				case "insufficient_balance":
					fmt.Printf("\n[Insufficient Balance] You requested $%.2f but only have $%.2f\n", meta.Amount, meta.Balance)
					fmt.Printf("Options: [1] Transfer $%.2f instead  [2] Cancel\n", meta.Balance)
					fmt.Print("Choice: ")

					if promptChoice(reader, 1, 2) == 1 {
						// RestartWith + WithNewInput: Retry with adjusted amount
						part, err := transferMoney.RestartWith(interrupt,
							ai.WithNewInput(TransferInput{meta.ToAccount, meta.Balance}))
						if err != nil {
							return "", fmt.Errorf("RestartWith: %w", err)
						}
						restarts = append(restarts, part)
					} else {
						// RespondWith: Provide cancelled output directly
						part, err := transferMoney.RespondWith(interrupt,
							TransferOutput{"cancelled", "Transfer cancelled by user.", accountBalance})
						if err != nil {
							return "", fmt.Errorf("RespondWith: %w", err)
						}
						responses = append(responses, part)
					}

				case "confirm_large":
					fmt.Printf("\n[Confirm Large Transfer] Send $%.2f to %s? (yes/no): ", meta.Amount, meta.ToAccount)

					if promptYesNo(reader) {
						// RestartWith: Re-execute the tool with approval
						part, err := transferMoney.RestartWith(interrupt)
						if err != nil {
							return "", fmt.Errorf("RestartWith: %w", err)
						}
						restarts = append(restarts, part)
					} else {
						// RespondWith: Provide cancelled output directly
						part, err := transferMoney.RespondWith(interrupt,
							TransferOutput{"cancelled", "Transfer cancelled by user.", accountBalance})
						if err != nil {
							return "", fmt.Errorf("RespondWith: %w", err)
						}
						responses = append(responses, part)
					}
				}
			}

			resp, err = genkit.Generate(ctx, g,
				ai.WithModel(googlegenai.ModelRef("googleai/gemini-2.5-flash", &genai.GenerateContentConfig{
					ThinkingConfig: &genai.ThinkingConfig{ThinkingBudget: genai.Ptr[int32](0)},
				})),
				ai.WithMessages(resp.History()...),
				ai.WithTools(transferMoney),
				ai.WithToolRestarts(restarts...),
				ai.WithToolResponses(responses...),
			)
			if err != nil {
				return "", err
			}
		}

		return resp.Text(), nil
	})

	fmt.Println("Payment Agent - Tool Interrupts Demo")
	fmt.Printf("Balance: $%.2f\n", accountBalance)
	fmt.Println("Type 'quit' to exit.")
	fmt.Println()

	for {
		fmt.Print("> ")
		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)

		if input == "quit" || input == "exit" {
			break
		}
		if input == "" {
			continue
		}

		result, err := paymentAgent.Run(ctx, input)
		if err != nil {
			fmt.Printf("Error: %v\n\n", err)
			continue
		}
		fmt.Printf("\n%s\n\n", result)
	}
}

// promptChoice reads a number choice from stdin.
func promptChoice(reader *bufio.Reader, min, max int) int {
	for {
		text, _ := reader.ReadString('\n')
		text = strings.TrimSpace(text)
		n, err := strconv.Atoi(text)
		if err == nil && n >= min && n <= max {
			return n
		}
		fmt.Printf("Please enter a number between %d and %d: ", min, max)
	}
}

// promptYesNo reads a yes/no response from stdin.
func promptYesNo(reader *bufio.Reader) bool {
	text, _ := reader.ReadString('\n')
	text = strings.ToLower(strings.TrimSpace(text))
	return text == "yes" || text == "y"
}
