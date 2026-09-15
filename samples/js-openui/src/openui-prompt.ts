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

import {
  openuiChatLibrary,
  openuiChatPromptOptions,
} from '@openuidev/react-ui';

const chartExample = `root = Card([title, chart, followUps])
title = TextContent("Quarterly revenue", "large-heavy")
chart = BarChart(["Q1", "Q2", "Q3", "Q4"], [revenue], "grouped", "Quarter", "Revenue")
revenue = Series("Revenue", [120, 180, 150, 240])
followUps = FollowUpBlock([strongest, compare])
strongest = FollowUpItem("Explain why Q4 is strongest")
compare = FollowUpItem("Compare Q1 and Q4")`;

const formExample = `root = Card([intro, estimateForm])
intro = TextContent("Tell me about the project and I will estimate it.")
estimateForm = Form("projectEstimate", formButtons, [projectControl, teamControl, notesControl])
projectControl = FormControl("Project name", Input("projectName", "Aurora-731", "text", { required: true }))
teamControl = FormControl("Team size", Input("teamSize", "7", "number", { required: true, numeric: true, min: 1 }))
notesControl = FormControl("Notes", TextArea("notes", "Priorities and constraints", 4, { required: true, minLength: 3 }))
formButtons = Buttons([submitButton])
submitButton = Button("Submit estimate", Action([@ToAssistant("Estimate this project from the submitted form values")]), "primary")`;

/**
 * The model prompt is compiled from the exact component library rendered by
 * the browser. This prevents the prompt and renderer schemas from drifting.
 */
export const openuiSystemPrompt = openuiChatLibrary.prompt({
  ...openuiChatPromptOptions,
  examples: [
    ...(openuiChatPromptOptions.examples ?? []),
    chartExample,
    formExample,
  ],
  additionalRules: [
    ...(openuiChatPromptOptions.additionalRules ?? []),
    'Return only OpenUI Lang. Do not use Markdown fences or explanatory text outside the program.',
    'Put root = Card(...) on the first line so the response can render while it streams.',
    'When the user asks for a chart, use the supplied labels and numeric values in a visible chart and end the Card with a FollowUpBlock containing exactly two relevant FollowUpItem controls.',
    'When the user asks for a form, use Form with named FormControl fields, validation rules, Buttons, and one primary Button with Action([@ToAssistant(...)]) so submitted form state reaches the next turn.',
    'When a user message contains OpenUI action context or form values, respond to those values directly and acknowledge the project name plus at least one other submitted field.',
  ],
});
