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

package main

import (
	"github.com/firebase/genkit/go/ai"
	"github.com/firebase/genkit/go/genkit"
)

func setup01(g *genkit.Genkit, m ai.Model) error {
	genkit.DefinePrompt(g, "s01_vanillaPrompt",
		ai.WithPrompt(`
You are acting as a helpful AI assistant named "Walt" that can answer
questions about the food available on the menu at Walt's Burgers.
Customer says: {{question}}`),
		ai.WithModel(m),
		ai.WithInputType(menuQuestionInput{Question: "What's on the menu?"}),
	)

	genkit.DefinePrompt(g, "s01_staticMenuDotPrompt",
		ai.WithPrompt(`
You are acting as a helpful AI assistant named "Walt" that can answer
questions about the food available on the menu at Walt's Burgers.
Here is today's menu:

- The Regular Burger $12
The classic charbroiled to perfection with your choice of cheese

- The Fancy Burger $13
Classic burger topped with bacon & Blue Cheese

- The Bacon Burger $13
Bacon cheeseburger with your choice of cheese.

- Everything Burger $14
Heinz 57 sauce, American cheese, bacon, fried egg & crispy onion bits

- Chicken Breast Sandwich $12
Tender juicy chicken breast on a brioche roll.
Grilled, blackened, or fried

Our fresh 1/2 lb. beef patties are made using choice cut
brisket, short rib & sirloin. Served on a toasted
brioche roll with chips. Served with lettuce, tomato & pickles.
Onions upon request. Substitute veggie patty $2

Answer this customer's question, in a concise and helpful manner,
as long as it is about food.

Question:
{{question}} ?`),
		ai.WithModel(m),
		ai.WithInputType(menuQuestionInput{Question: "What's on the menu?"}),
		ai.WithOutputFormat(ai.OutputFormatText),
	)

	return nil
}
