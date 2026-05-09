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

/**
 * Citation conversion utilities for Anthropic API responses.
 * Handles validation and conversion of citations from Anthropic's format to Genkit format.
 */

import { z } from 'genkit';
import type { AnthropicCitation } from '../../types.js';

/** Structural type for Anthropic citations (works with both stable and beta APIs). */
export interface AnthropicCitationInput {
  type: string;
  cited_text: string;
  // document_index is optional since web search citations don't have it
  document_index?: number;
  document_title?: string | null;
  file_id?: string | null;
  start_char_index?: number;
  end_char_index?: number;
  start_page_number?: number;
  end_page_number?: number;
  start_block_index?: number;
  end_block_index?: number;
}

// --- Citation validation schemas ---

/**
 * Base citation schema with common fields shared across all citation types.
 */
const baseCitationSchema = z.object({
  cited_text: z.string(),
  document_index: z.number(),
  document_title: z.string().nullable().optional(),
  file_id: z.string().nullable().optional(),
});

/**
 * Schema for character location citations (plain text documents).
 */
const charLocationCitationSchema = baseCitationSchema.extend({
  type: z.literal('char_location'),
  start_char_index: z.number(),
  end_char_index: z.number(),
});

/**
 * Schema for page location citations (PDF documents).
 */
const pageLocationCitationSchema = baseCitationSchema.extend({
  type: z.literal('page_location'),
  start_page_number: z.number(),
  end_page_number: z.number(),
});

/**
 * Schema for content block location citations (custom content documents).
 */
const contentBlockLocationCitationSchema = baseCitationSchema.extend({
  type: z.literal('content_block_location'),
  start_block_index: z.number(),
  end_block_index: z.number(),
});

/**
 * Discriminated union schema for all supported citation types.
 */
const citationSchema = z.discriminatedUnion('type', [
  charLocationCitationSchema,
  pageLocationCitationSchema,
  contentBlockLocationCitationSchema,
]);

/**
 * Converts Anthropic's citation format (snake_case) to genkit format (camelCase).
 * Only handles document-based citations (char_location, page_location, content_block_location).
 * Skips web search and other citation types that don't reference documents.
 */
export function fromAnthropicCitation(
  citation: AnthropicCitationInput
): AnthropicCitation | undefined {
  // Skip citations without document_index (e.g., web search results)
  if (citation.document_index === undefined) {
    return undefined;
  }

  // Validate and parse citation with Zod
  const result = citationSchema.safeParse(citation);
  if (!result.success) {
    // Invalid citation structure, skip it gracefully
    return undefined;
  }

  const parsed = result.data;

  // Convert validated citation to Genkit format
  switch (parsed.type) {
    case 'char_location':
      return {
        type: 'char_location',
        citedText: parsed.cited_text,
        documentIndex: parsed.document_index,
        documentTitle: parsed.document_title ?? undefined,
        fileId: parsed.file_id ?? undefined,
        startCharIndex: parsed.start_char_index,
        endCharIndex: parsed.end_char_index,
      };
    case 'page_location':
      return {
        type: 'page_location',
        citedText: parsed.cited_text,
        documentIndex: parsed.document_index,
        documentTitle: parsed.document_title ?? undefined,
        fileId: parsed.file_id ?? undefined,
        startPageNumber: parsed.start_page_number,
        endPageNumber: parsed.end_page_number,
      };
    case 'content_block_location':
      return {
        type: 'content_block_location',
        citedText: parsed.cited_text,
        documentIndex: parsed.document_index,
        documentTitle: parsed.document_title ?? undefined,
        fileId: parsed.file_id ?? undefined,
        startBlockIndex: parsed.start_block_index,
        endBlockIndex: parsed.end_block_index,
      };
  }
}
