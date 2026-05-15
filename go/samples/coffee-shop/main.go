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

// This program can be manually tested like so:
//
// In development mode (with the environment variable GENKIT_ENV="dev"):
// Start the server listening on port 3100:
//
//	go run . &
//
// Tell it to run a flow:
//
//	curl -d '{"key":"/flow/simpleGreeting/simpleGreeting", "input":{"start": {"input":{"customerName": "John Doe"}}}}' http://localhost:3100/api/runAction
//
// In production mode (GENKIT_ENV missing or set to "prod"):
// Start the server listening on port 3400:
//
//	go run . &
//
// Tell it to run a flow:
//
//  curl -d '{"customerName": "Stimpy"}' http://localhost:3400/simpleGreeting

package main

import (
	"context"
	"log"
	"net/http"

	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
	"github.com/firebase/genkit/go/plugins/server"
)

const simpleGreetingPromptTemplate = `
You're a barista at a nice coffee shop.
A regular customer named {{customerName}} enters.
Greet the customer in one sentence, and recommend a coffee drink.
`

const simpleStructuredGreetingPromptTemplate = `
You're a barista at a nice coffee shop.
A regular customer named {{customerName}} enters.
Greet the customer in one sentence.
Provide the name of the drink of the day, nothing else.
`

type simpleGreetingInput struct {
	CustomerName string `json:"customerName"`
}

type simpleGreetingOutput struct {
	CustomerName string `json:"customerName"`
	Greeting     string `json:"greeting,omitempty"`
	DrinkOfDay   string `json:"drinkOfDay"`
}

const greetingWithHistoryPromptTemplate = `
{{role "user"}}
Hi, my name is {{customerName}}. The time is {{currentTime}}. Who are you?

{{role "model"}}
I am Barb, a barista at this nice underwater-themed coffee shop called Krabby Kooffee.
I know pretty much everything there is to know about coffee,
and I can cheerfully recommend delicious coffee drinks to you based on whatever you like.

{{role "user"}}
Great. Last time I had {{previousOrder}}.
I want you to greet me in one sentence, and recommend a drink.
`

type customerTimeAndHistoryInput struct {
	CustomerName  string `json:"customerName"`
	CurrentTime   string `json:"currentTime"`
	PreviousOrder string `json:"previousOrder"`
}

type testAllCoffeeFlowsOutput struct {
	Pass    bool     `json:"pass"`
	Replies []string `json:"replies,omitempty"`
	Error   string   `json:"error,omitempty"`
}

func main() {
	ctx := context.Background()
	g := genkit.Init(ctx,
		genkit.WithDefaultModel("googleai/gemini-2.5-flash"),
		genkit.WithPlugins(&googlegenai.GoogleAI{}),
	)

	m := googlegenai.GoogleAIModel(g, "gemini-2.5-flash")
	simpleGreetingPrompt := genkit.DefinePrompt(g, "simpleGreeting2",
		ai.WithPrompt(simpleGreetingPromptTemplate),
		ai.WithModel(m),
		ai.WithInputType(simpleGreetingInput{}),
		ai.WithOutputFormat(ai.OutputFormatText),
	)

	simpleGreetingFlow := genkit.DefineStreamingFlow(g, "simpleGreeting", func(ctx context.Context, input *simpleGreetingInput, cb func(context.Context, string) error) (string, error) {
		var callback func(context.Context, *ai.ModelResponseChunk) error
		if cb != nil {
			callback = func(ctx context.Context, c *ai.ModelResponseChunk) error {
				return cb(ctx, c.Text())
			}
		}
		resp, err := simpleGreetingPrompt.Execute(ctx,
			ai.WithInput(input),
			ai.WithStreaming(callback),
		)
		if err != nil {
			return "", err
		}
		return resp.Text(), nil
	})

	greetingWithHistoryPrompt := genkit.DefinePrompt(g, "greetingWithHistory",
		ai.WithPrompt(greetingWithHistoryPromptTemplate),
		ai.WithModel(m),
		ai.WithInputType(customerTimeAndHistoryInput{}),
		ai.WithOutputFormat(ai.OutputFormatText),
	)

	greetingWithHistoryFlow := genkit.DefineFlow(g, "greetingWithHistory", func(ctx context.Context, input *customerTimeAndHistoryInput) (string, error) {
		resp, err := greetingWithHistoryPrompt.Execute(ctx,
			ai.WithInput(input),
		)
		if err != nil {
			return "", err
		}
		return resp.Text(), nil
	})

	simpleStructuredGreetingPrompt := genkit.DefinePrompt(g, "simpleStructuredGreeting",
		ai.WithPrompt(simpleStructuredGreetingPromptTemplate),
		ai.WithModel(m),
		ai.WithInputType(simpleGreetingInput{}),
		ai.WithOutputType(simpleGreetingOutput{}),
	)

	genkit.DefineStreamingFlow(g, "simpleStructuredGreeting", func(ctx context.Context, input *simpleGreetingInput, cb func(context.Context, string) error) (string, error) {
		var callback func(context.Context, *ai.ModelResponseChunk) error
		if cb != nil {
			callback = func(ctx context.Context, c *ai.ModelResponseChunk) error {
				return cb(ctx, c.Text())
			}
		}
		resp, err := simpleStructuredGreetingPrompt.Execute(ctx,
			ai.WithInput(input),
			ai.WithStreaming(callback),
		)
		if err != nil {
			return "", err
		}
		return resp.Text(), nil
	})

	genkit.DefineFlow(g, "testAllCoffeeFlows", func(ctx context.Context, _ struct{}) (*testAllCoffeeFlowsOutput, error) {
		test1, err := simpleGreetingFlow.Run(ctx, &simpleGreetingInput{
			CustomerName: "Sam",
		})
		if err != nil {
			out := &testAllCoffeeFlowsOutput{
				Pass:  false,
				Error: err.Error(),
			}
			return out, nil
		}
		test2, err := greetingWithHistoryFlow.Run(ctx, &customerTimeAndHistoryInput{
			CustomerName:  "Sam",
			CurrentTime:   "09:45am",
			PreviousOrder: "Caramel Macchiato",
		})
		if err != nil {
			out := &testAllCoffeeFlowsOutput{
				Pass:  false,
				Error: err.Error(),
			}
			return out, nil
		}
		out := &testAllCoffeeFlowsOutput{
			Pass: true,
			Replies: []string{
				test1,
				test2,
			},
		}
		return out, nil
	})

	mux := http.NewServeMux()
	for _, a := range genkit.ListFlows(g) {
		mux.HandleFunc("POST /"+a.Name(), genkit.Handler(a))
	}
	log.Fatal(server.Start(ctx, "127.0.0.1:8080", mux))
}
