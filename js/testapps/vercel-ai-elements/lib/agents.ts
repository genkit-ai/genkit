/**
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import { z } from "genkit";
import { InMemorySessionStore } from "genkit/beta";
import { ai } from "./genkit";

// ---------------------------------------------------------------------------
// Weather Agent — simple tool-calling agent
// ---------------------------------------------------------------------------

export const getWeather = ai.defineTool(
  {
    name: "getWeather",
    description: "Get the current weather for a given location.",
    inputSchema: z.object({ location: z.string() }),
    outputSchema: z.object({ weather: z.string(), temperature: z.string() }),
  },
  async (input) => {
    const weathers = ["Sunny", "Cloudy", "Rainy", "Snowy"];
    const weather = weathers[Math.floor(Math.random() * weathers.length)];
    const temp = `${Math.floor(Math.random() * 30 + 5)}°C`;
    return { weather: `${weather} in ${input.location}`, temperature: temp };
  }
);

export const weatherAgent = ai.defineAgent({
  name: "weatherAgent",
  system:
    "You are a friendly weather assistant. Use the getWeather tool to look up weather information when asked. Always include the temperature in your response.",
  tools: [getWeather],
  store: new InMemorySessionStore(),
});

// ---------------------------------------------------------------------------
// Banking Agent — demonstrates interrupt (human-in-the-loop)
// ---------------------------------------------------------------------------

export const userApproval = ai.defineInterrupt({
  name: "userApproval",
  description:
    "Ask the user for approval before proceeding with a sensitive action.",
  inputSchema: z.object({
    action: z.string().describe("The action to be approved"),
    details: z.string().describe("Details about the action"),
  }),
  outputSchema: z.object({
    approved: z.boolean().describe("Whether the user approved the action"),
    feedback: z.string().optional().describe("Optional feedback from the user"),
  }),
});

export const transferMoney = ai.defineTool(
  {
    name: "transferMoney",
    description: "Transfer money to a specified account.",
    inputSchema: z.object({
      amount: z.number(),
      toAccount: z.string(),
    }),
    outputSchema: z.object({
      success: z.boolean(),
      transactionId: z.string(),
    }),
  },
  async (_input) => {
    return { success: true, transactionId: `txn-${Date.now()}` };
  }
);

// A *restartable* tool. On its first call it interrupts to ask the user to
// double-check the recipient. If the user chooses "Restart" in the UI, the
// transport re-runs this tool with `resumed` set — at which point it fetches
// the live exchange rate and returns a real result instead of interrupting
// again. This demonstrates the interrupt → restart flow (as opposed to the
// interrupt → response flow handled by `userApproval`).
export const getExchangeRate = ai.defineTool(
  {
    name: "getExchangeRate",
    description:
      "Look up the current exchange rate for an international transfer.",
    inputSchema: z.object({
      fromCurrency: z.string(),
      toCurrency: z.string(),
    }),
    outputSchema: z.object({
      rate: z.number(),
      note: z.string().optional(),
    }),
  },
  async (input, { interrupt, resumed }) => {
    // First call: pause so the user can confirm before we fetch a live rate.
    if (!resumed) {
      interrupt();
    }
    // Restarted: produce a (mock) live rate. Any metadata the user attached
    // when restarting arrives as `resumed`.
    const rate = Number((0.8 + Math.random() * 0.4).toFixed(4));
    return {
      rate,
      note: `${input.fromCurrency}→${input.toCurrency} (confirmed by user)`,
    };
  }
);

export const bankingAgent = ai.defineAgent({
  name: "bankingAgent",
  system:
    "You are a helpful banking assistant. If the user wants to transfer money, ALWAYS use the userApproval interrupt to confirm the details before executing the transferMoney tool. For international transfers, use the getExchangeRate tool to look up the rate.",
  tools: [userApproval, transferMoney, getExchangeRate],
  store: new InMemorySessionStore(),
});
