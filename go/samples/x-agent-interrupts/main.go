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

// x-agent-interrupts demonstrates the experimental tool interrupts API
// using a prompt-backed agent (DefineAgent + FromPrompt). Unlike
// x-interrupts (which handles interrupts inside a flow), this sample
// separates concerns: the agent streams interrupts to the client, and the
// client handles user interaction and sends resume data back.
package main

import (
	"bufio"
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/ai/exp"
	"github.com/firebase/genkit/go/ai/exp/tool"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
)

// --- Tool schemas ---

type TransferInput struct {
	ToAccount string  `json:"toAccount" jsonschema:"description=destination account ID"`
	Amount    float64 `json:"amount" jsonschema:"description=amount in dollars (e.g. 50.00 for $50)"`
}

type TransferOutput struct {
	Status     string  `json:"status"`
	Message    string  `json:"message,omitempty"`
	NewBalance float64 `json:"newBalance,omitempty"`
}

type TransferInterrupt struct {
	Reason    string  `json:"reason"`
	ToAccount string  `json:"toAccount"`
	Amount    float64 `json:"amount"`
	Balance   float64 `json:"balance,omitempty"`
}

type Confirmation struct {
	Approved       bool     `json:"approved"`
	AdjustedAmount *float64 `json:"adjustedAmount,omitempty"`
}

var accountBalance = 150.00

func main() {
	ctx := context.Background()
	reader := bufio.NewReader(os.Stdin)
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))

	genkit.DefineInterruptibleTool(g, "transferMoney",
		"Transfers money to another account. Use this when the user wants to send money.",
		func(ctx context.Context, input TransferInput, confirm *Confirmation) (*TransferOutput, error) {
			if confirm != nil {
				if !confirm.Approved {
					return &TransferOutput{"cancelled", "Transfer cancelled by user.", accountBalance}, nil
				}
				if confirm.AdjustedAmount != nil {
					input.Amount = *confirm.AdjustedAmount
				}
			}

			if input.Amount > accountBalance {
				if accountBalance <= 0 {
					return &TransferOutput{"rejected", "Account balance is 0. Please add funds.", accountBalance}, nil
				}
				return nil, tool.Interrupt(TransferInterrupt{
					"insufficient_balance", input.ToAccount, input.Amount, accountBalance,
				})
			}

			if confirm == nil && input.Amount > 100 {
				return nil, tool.Interrupt(TransferInterrupt{
					"confirm_large", input.ToAccount, input.Amount, accountBalance,
				})
			}

			accountBalance -= input.Amount
			return &TransferOutput{
				"completed",
				fmt.Sprintf("Transferred $%.2f to %s.", input.Amount, input.ToAccount),
				accountBalance,
			}, nil
		})

	paymentAgent := genkit.DefineAgent[any](g, "paymentAgent", exp.FromPrompt())

	fmt.Println("Payment Agent (Prompt Agent + Interrupts)")
	fmt.Printf("Balance: $%.2f\n", accountBalance)
	fmt.Println("Type 'quit' to exit.")
	fmt.Println()

	conn, err := paymentAgent.StreamBidi(ctx)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	for {
		fmt.Print("> ")
		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)
		if input == "quit" || input == "exit" {
			conn.Close()
			break
		}
		if input == "" {
			continue
		}

		if err := conn.SendText(input); err != nil {
			fmt.Fprintf(os.Stderr, "Send error: %v\n", err)
			break
		}

		fmt.Println()
		var interrupts []*ai.Part
		for chunk, err := range conn.Receive() {
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error: %v\n", err)
				break
			}

			if chunk.ModelChunk != nil {
				fmt.Print(chunk.ModelChunk.Text())
				interrupts = append(interrupts, chunk.ModelChunk.Interrupts()...)
			}

			if chunk.TurnEnd != nil {
				if len(interrupts) == 0 {
					fmt.Println()
					fmt.Println()
					break
				}

				var restarts []*ai.Part
				for _, interrupt := range interrupts {
					if restart, err := handleInterrupt(reader, interrupt); err != nil {
						fmt.Fprintf(os.Stderr, "Error: %v\n", err)
					} else if restart != nil {
						restarts = append(restarts, restart)
					}
				}
				interrupts = nil

				if err := conn.SendResume(&exp.ToolResume{Restart: restarts}); err != nil {
					fmt.Fprintf(os.Stderr, "Error: %v\n", err)
					break
				}
			}
		}
	}

	conn.Output()
}

func handleInterrupt(reader *bufio.Reader, part *ai.Part) (*ai.Part, error) {
	meta, ok := tool.InterruptAs[TransferInterrupt](part)
	if !ok {
		return nil, nil
	}

	switch meta.Reason {
	case "insufficient_balance":
		fmt.Printf("\n[Insufficient Balance] You requested $%.2f but only have $%.2f\n", meta.Amount, meta.Balance)
		fmt.Printf("Options: [1] Transfer $%.2f instead  [2] Cancel\n", meta.Balance)
		fmt.Print("Choice: ")

		if promptChoice(reader, 1, 2) == 1 {
			return tool.Resume(part, Confirmation{
				Approved:       true,
				AdjustedAmount: &meta.Balance,
			})
		}
		return tool.Resume(part, Confirmation{Approved: false})

	case "confirm_large":
		fmt.Printf("\n[Confirm Large Transfer] Send $%.2f to %s? (yes/no): ", meta.Amount, meta.ToAccount)
		return tool.Resume(part, Confirmation{Approved: promptYesNo(reader)})

	default:
		fmt.Printf("\n[Unknown Interrupt] Reason: %q. Cancelling transaction.\n", meta.Reason)
		return tool.Resume(part, Confirmation{Approved: false})
	}
}

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

func promptYesNo(reader *bufio.Reader) bool {
	text, _ := reader.ReadString('\n')
	text = strings.ToLower(strings.TrimSpace(text))
	return text == "yes" || text == "y"
}
