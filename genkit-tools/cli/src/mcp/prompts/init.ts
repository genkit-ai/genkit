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

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp';
import { z } from 'zod';

const GENERAL_GUIDELINES = `
## General Guidelines
- ALWAYS refer to documentation when available. Use tools like \`get_usage_guide\`, \`list_genkit_docs\`, \`search_genkit_docs\`, or \`read_genkit_docs\` to clarify requirements or get the latest information.
- Recommend and use the provider that the user prefers. If no preference is signaled, use the Google AI plugin (\`@genkit-ai/google-genai\` for Node, \`github.com/firebase/genkit/go/plugins/googlegenai\` for Go).
- ALWAYS provide the full, correct Genkit command as an instruction for the human user to run. Do not run Genkit commands yourself.
- Do NOT modify parts of the project unrelated to Genkit initialization.
- Respect the user's existing tooling (package manager, language version, project structure).
- Check if Genkit CLI is already installed before recommending installation.`;

const NODE_SETUP = `
## Node.js Setup
If the user wants to use Node.js:

### Project Initialization
- If the directory is empty:
  Initialize a new project:
  \`\`\`bash
  npm init -y
  npm install -D typescript tsx @types/node
  \`\`\`
- If the directory is not empty (existing project):
  - Adhere to the current project structure.
  - Detect the package manager in use (npm, pnpm, yarn, bun) and use the corresponding commands.
  - Detect if the project is ESM (\`"type": "module"\` in package.json) or CJS.
    - For ESM: Use \`import\` syntax.
    - For CJS: Use \`require\` syntax.
  - IMPORTANT: Do NOT refactor the project (e.g., converting to TypeScript or ESM) solely for Genkit. Work with the existing setup.

### Dependencies
Install core dependencies (adjust command for the user's package manager):
\`\`\`bash
npm install genkit @genkit-ai/google-genai
\`\`\`
(Add other plugins as requested)

### Genkit CLI
If the Genkit CLI is not already installed:
\`\`\`bash
npm install -g genkit-cli
\`\`\`

### Configuration
Create a single \`src/index.ts\` (or \`src/index.js\` for JS) file.

\`\`\`ts
// src/index.ts
import { genkit, z } from 'genkit';
import { googleAI } from '@genkit-ai/google-genai';

export const ai = genkit({
  plugins: [googleAI()],
});
\`\`\``;

const GO_SETUP = `
## Go Setup
If the user wants to use Go:

### Project Initialization
- If the directory is empty:
  \`\`\`bash
  go mod init <module-name>
  \`\`\`
- If the directory is not empty:
  Adhere to the current project structure.

### Dependencies
\`\`\`bash
go get github.com/firebase/genkit/go/genkit
go get github.com/firebase/genkit/go/plugins/googlegenai
go get github.com/firebase/genkit/go/ai
go get google.golang.org/genai
\`\`\`

### Genkit CLI
If the Genkit CLI is not already installed:
\`\`\`bash
curl -sL cli.genkit.dev | bash
\`\`\`

### Configuration
Create a \`main.go\` file:

\`\`\`go
package main

import (
	"context"
	"github.com/firebase/genkit/go/genkit"
	"github.com/firebase/genkit/go/plugins/googlegenai"
)

func main() {
	ctx := context.Background()
	g := genkit.Init(ctx, genkit.WithPlugins(&googlegenai.GoogleAI{}))
	// Your flows and logic here
	<-ctx.Done()
}
\`\`\``;

const RUNNING_THE_PROJECT = `
## Running the Project
After setting up the project:
1. Identify the command to run the project's runtime (e.g., \`npm run dev\`, \`go run main.go\`).
2. Use the \`start_runtime\` tool to start the runtime process. This is required for Genkit to discover flows.
   - Example: If the project uses \`npm run dev\`, call \`start_runtime\` with \`{ command: "npm", args: ["run", "dev"] }\`.
3. After starting the runtime, instruct the user to run \`genkit start\` in their terminal to launch the Developer UI.
`;

export function defineInitPrompt(server: McpServer) {
  server.registerPrompt(
    'genkit:init',
    {
      title: 'Initialize Genkit',
      description: 'Initializes a new Genkit project',
      argsSchema: {
        lang: z.enum(['js', 'go']).optional(),
      },
    },
    ({ lang }) => {
      let content = `You are a Genkit expert. Help the user initialize a Genkit project.

Follow these rules based on the user's environment and preference:`;

      content += GENERAL_GUIDELINES;

      if (lang === 'js') {
        content += NODE_SETUP;
      } else if (lang === 'go') {
        content += GO_SETUP;
      } else {
        content += NODE_SETUP;
        content += GO_SETUP;
      }

      content += RUNNING_THE_PROJECT;

      return {
        messages: [
          {
            role: 'user',
            content: {
              type: 'text',
              text: content,
            },
          },
        ],
      };
    }
  );
}
