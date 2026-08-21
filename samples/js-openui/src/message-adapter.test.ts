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

import { describe, expect, it } from 'vitest';
import { toGenkitContent } from './message-adapter';

describe('toGenkitContent', () => {
  it('preserves an ordinary follow-up as exact model text', () => {
    expect(
      toGenkitContent({
        role: 'user',
        content:
          ']]>openui:content\nCompare Q1 and Q4\n]]>openui:context\n["User clicked: Compare Q1 and Q4"]',
      })
    ).toBe(
      'Compare Q1 and Q4\n\nOpenUI action context:\n["User clicked: Compare Q1 and Q4"]'
    );
  });

  it('includes distinctive submitted form values in the Genkit turn', () => {
    const content = toGenkitContent({
      role: 'user',
      content:
        ']]>openui:content\nSubmit estimate\n]]>openui:context\n["User clicked: Submit estimate",{"projectEstimate":{"projectName":{"value":"Aurora-731"},"teamSize":{"value":"7"},"notes":{"value":"Prioritize accessibility and charts"}}}]',
    });

    expect(content).toContain('Submit estimate');
    expect(content).toContain('Aurora-731');
    expect(content).toContain('"teamSize":{"value":"7"}');
    expect(content).toContain('Prioritize accessibility and charts');
  });

  it('removes persisted form context from assistant history', () => {
    expect(
      toGenkitContent({
        role: 'assistant',
        content:
          ']]>openui:content\nroot = Card([title])\ntitle = TextContent("Ready")\n]]>openui:context\n[{"projectName":{"value":"Aurora-731"}}]',
      })
    ).toBe('root = Card([title])\ntitle = TextContent("Ready")');
  });

  it('parses compact content and context markers without newlines', () => {
    expect(
      toGenkitContent({
        role: 'user',
        content:
          ']]>openui:contentCompare Q1\nand Q4]]>openui:context["User clicked: Compare Q1 and Q4"]',
      })
    ).toBe(
      'Compare Q1\nand Q4\n\nOpenUI action context:\n["User clicked: Compare Q1 and Q4"]'
    );
  });

  it('parses a compact content marker without context', () => {
    expect(
      toGenkitContent({
        role: 'user',
        content: ']]>openui:contentShow revenue by quarter',
      })
    ).toBe('Show revenue by quarter');
  });
});
