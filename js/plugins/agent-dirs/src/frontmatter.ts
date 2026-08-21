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

/**
 * Frontmatter splitter for `instructions.md`: YAML frontmatter between `---`
 * fences, markdown body after. The convention owns this format outright -
 * no dotprompt involvement, so the body is plain text (never templated) and
 * parse failures are real errors rather than dotprompt's silent
 * whole-file-becomes-template fallback.
 *
 * @module @genkit-ai/agent-dirs/frontmatter
 */

import { parse as parseYaml } from 'yaml';

/** A parsed `instructions.md`: frontmatter mapping + trimmed markdown body. */
export interface ParsedInstructions {
  frontmatter: Record<string, unknown>;
  body: string;
}

const OPEN_FENCE = /^---[ \t]*\r?\n/;
const CLOSE_FENCE = /^---[ \t]*(?:\r?\n|$)/m;

/**
 * Splits an `instructions.md` source into YAML frontmatter and markdown body.
 *
 * A file without a leading `---` fence is all body. Throws (with a message
 * safe to surface to the author) on an unterminated fence, invalid YAML, or
 * frontmatter that is not a mapping.
 */
export function parseInstructionsSource(source: string): ParsedInstructions {
  const open = OPEN_FENCE.exec(source);
  if (!open) {
    return { frontmatter: {}, body: source.trim() };
  }
  const rest = source.slice(open[0].length);
  const close = CLOSE_FENCE.exec(rest);
  if (!close) {
    throw new Error(
      `frontmatter is not closed - expected a '---' line after the opening fence`
    );
  }
  const yamlText = rest.slice(0, close.index);
  const body = rest.slice(close.index + close[0].length);

  let frontmatter: unknown;
  try {
    frontmatter = parseYaml(yamlText);
  } catch (e) {
    throw new Error(`invalid frontmatter YAML: ${(e as Error).message}`);
  }
  if (frontmatter === null || frontmatter === undefined) {
    frontmatter = {};
  }
  if (typeof frontmatter !== 'object' || Array.isArray(frontmatter)) {
    throw new Error(
      `frontmatter must be a YAML mapping (key: value pairs), got ${
        Array.isArray(frontmatter) ? 'a list' : typeof frontmatter
      }`
    );
  }
  return {
    frontmatter: frontmatter as Record<string, unknown>,
    body: body.trim(),
  };
}
