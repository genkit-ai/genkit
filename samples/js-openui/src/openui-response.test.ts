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

import { createParser } from '@openuidev/react-lang';
import { openuiChatLibrary } from '@openuidev/react-ui';
import { describe, expect, it } from 'vitest';

const chartResponse = `root = Card([title, chart, followUps])
title = TextContent("Quarterly revenue", "large-heavy")
chart = BarChart(["Q1", "Q2", "Q3", "Q4"], [revenue], "grouped", "Quarter", "Revenue")
revenue = Series("Revenue", [120, 180, 150, 240])
followUps = FollowUpBlock([strongest, compare])
strongest = FollowUpItem("Explain why Q4 is strongest")
compare = FollowUpItem("Compare Q1 and Q4")`;

describe('OpenUI response contract', () => {
  it('parses a chart and follow-up response without errors', () => {
    const parser = createParser(openuiChatLibrary.toJSONSchema(), 'Card');
    const result = parser.parse(chartResponse);

    expect(result.meta.errors).toEqual([]);
    expect(result.meta.unresolved).toEqual([]);
  });
});
