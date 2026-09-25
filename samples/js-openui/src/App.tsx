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
  fetchLLM,
  openAIAdapter,
  openAIMessageFormat,
} from '@openuidev/react-headless';
import {
  AgentInterface,
  createTheme,
  openuiChatLibrary,
  type ThemeProps,
} from '@openuidev/react-ui';

const llm = fetchLLM({
  url: '/api/chat',
  streamAdapter: openAIAdapter(),
  messageFormat: openAIMessageFormat,
});

const starters = [
  {
    displayText: 'Chart quarterly revenue',
    prompt:
      'Show quarterly revenue as a labeled bar chart: Q1 120, Q2 180, Q3 150, Q4 240. End with two relevant follow-up suggestions.',
  },
  {
    displayText: 'Compare two quarters',
    prompt:
      'Compare the strongest and weakest quarter, then add a FollowUpBlock with two next questions.',
  },
  {
    displayText: 'Create a project form',
    prompt:
      'Create a validated project estimate form with fields for project name, team size, and notes. Add a primary Submit button that sends the completed values to you.',
  },
];

const theme: ThemeProps = {
  mode: 'light',
  lightTheme: createTheme({
    background: 'oklch(0.98 0.004 255)',
    foreground: 'oklch(1 0 0)',
    interactiveAccentDefault: 'oklch(0.52 0.19 258)',
    interactiveAccentHover: 'oklch(0.46 0.2 258)',
    chatUserResponseBg: 'oklch(0.52 0.19 258)',
    chatUserResponseText: 'oklch(0.99 0 0)',
    textBrand: 'oklch(0.45 0.18 258)',
    borderAccent: 'oklch(0.52 0.19 258 / 0.14)',
    radiusM: '8px',
    fontBody: 'Inter, ui-sans-serif, system-ui, sans-serif',
  }),
};

export function App() {
  return (
    <AgentInterface
      agentName="Genkit + OpenUI"
      componentLibrary={openuiChatLibrary}
      llm={llm}
      scrollVariant="always"
      starters={starters}
      starterVariant="long"
      theme={theme}>
      <AgentInterface.Welcome
        title="What would you like to build?"
        description="Genkit streams each model turn; OpenUI renders the result as interactive UI."
        glowAnimation
      />
    </AgentInterface>
  );
}
