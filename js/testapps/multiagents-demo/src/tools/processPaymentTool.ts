/**
 * Copyright 2025 Google LLC
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

import { z } from 'genkit';
import { ai } from '../config/genkit';

export const processPaymentTool = ai.defineTool(
  {
    name: 'processPayment',
    description: 'Process a payment for an order',
    inputSchema: z.object({
      orderId: z.string().describe('The order ID to process payment for'),
      amount: z.number().describe('The payment amount'),
      paymentMethod: z
        .string()
        .describe('Payment method (credit_card, debit_card, paypal)'),
    }),
  },
  async (input) => {
    // Mock payment processing
    return {
      success: true,
      transactionId: `TXN-${Date.now()}`,
      orderId: input.orderId,
      amount: input.amount,
      status: 'completed',
      message: 'Payment processed successfully',
    };
  }
);
