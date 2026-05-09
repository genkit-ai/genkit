Follow these instructions to set up Genkit in the current codebase. These instructions are general-purpose and have not been written with specific codebase knowledge, so use your best judgement when following them.

0. Tell the user "I'm going to check out your workspace and set you up to use Genkit for GenAI workflows."
1. If the current workspace is empty or is a starter template, your goal will be to create a simple image generation flow that allows someone to generate an image based on a prompt and selectable style. If the current workspace is not empty, you will create a simple example flow to help get the user started.
2. Check to see if any Genkit provider plugin (such as `@genkit-ai/google-genai` or `@genkit-ai/oai-compat` or others, may start with `genkitx-*`) is installed. If not, install `@genkit-ai/google-genai`. If this is a Next.js app, install `@genkit-ai/next` as well.
3. Search the codebase for the exact string `genkit(` (remember to escape regexes properly) which would indicate that the user has already set up Genkit in the codebase. If found, no need to set it up again, tell the user "Genkit is already configured in this app." and exit this workflow.
4. Create an `ai` directory in the primary source directory of the project (this may be e.g. `src` but is project-dependent).
5. Create `{sourceDir}/ai/genkit.ts` and populate it using the example below. DO NOT add a `next` plugin to the file, ONLY add a model provider plugin to the plugins array:

```ts
import { genkit, z } from 'genkit';
import { googleAI } from '@genkit-ai/google-genai';

export const ai = genkit({
  plugins: [googleAI()],
  model: googleAI.model('gemini-2.5-flash'),
});

export { z };
```

6. Create `{sourceDir}/ai/tools` and `{sourceDir}/ai/flows` directories, but leave them empty for now.
7. Create `{sourceDir}/ai/index.ts` and populate it with the following (change the import to match import aliases in `tsconfig.json` as needed):

```ts
import './genkit.js';
// import each created flow, tool, etc. here for use in the Genkit Dev UI
```

8. Add a `genkit:ui` script to `package.json` that runs `npx genkit-cli start -- npx tsx --watch {sourceDir}/ai/index.ts` (or `pnpm dlx` or `yarn dlx` for those package managers). DO NOT try to run the script now.
9. Tell the user "Genkit is now configured and ready for use." as setup is now complete. Also remind them to set appropriate env variables (e.g. `GEMINI_API_KEY` for Google providers). Wait for the user to prompt further before creating any specific flows.
