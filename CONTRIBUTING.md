# How to contribute

We'd love to accept your patches and contributions to this project.

## Before you begin

### Sign our Contributor License Agreement

Contributions to this project must be accompanied by a
[Contributor License Agreement](https://cla.developers.google.com/about) (CLA).
You (or your employer) retain the copyright to your contribution; this simply
gives us permission to use and redistribute your contributions as part of the
project.

If you or your current employer have already signed the Google CLA (even if it
was for a different project), you probably don't need to do it again.

Visit <https://cla.developers.google.com/> to see your current agreements or to
sign a new one.

### Review our community guidelines

This project follows
[Google's Open Source Community Guidelines](https://opensource.google/conduct/).

## Contribution process

### Code reviews

All submissions, including submissions by project members, require review. We
use GitHub pull requests for this purpose. Consult
[GitHub Help](https://help.github.com/articles/about-pull-requests/) for more
information on using pull requests.

## Setup

**Note:** If you want to setup all runtime environments at the same time
(js/go/py), you may want to run the `bin/setup` script. Beware, this script may
break other setup you may already have in your project environment.

Genkit supports JavaScript, Go, and Python. Before contributing in any of these languages, complete these prerequisites:

1. Install Node.js 20 or later using [nvm](https://nodejs.org/en/download)

   > **Note:** Node.js v20 or greater is required. Earlier versions may not work properly.

2. Install the Genkit CLI globally:
   ```bash
   npm install -g genkit-cli
   ```

After completing these prerequisites, follow the language-specific setup instructions below.

## Go Guide

1. Install Go 1.24 or later
   Follow the [official Go installation guide](https://golang.org/doc/install).

2. Configure your AI model
   Most samples use Google's Gemini model. You'll need to generate an API key at [Google AI Studio](https://aistudio.google.com/app/apikey).

   Once you have your key, set it in your environment:

   ```bash
   export GOOGLE_GENAI_API_KEY=<your-api-key>
   ```

3. Run a sample application

   ```bash
   cd go/samples          # Navigate to samples directory
   cd <sample-name>       # Choose a sample to run
   go mod tidy            # Install Go dependencies
   genkit start -- go run .  # Start the Genkit server and run the application
   ```

   Once running, visit http://localhost:4000 to access the Developer UI.

4. Run tests
   ```bash
   cd <test-directory>    # Navigate to test directory
   go test .              # Run tests in current directory
   ```

## JS Guide

### Install dependencies

Run `corepack enable pnpm` to enable pnpm.

```
pnpm i
pnpm run setup
```

This will install all dependencies and build all packages.

### Build it

```
pnpm build
```

This will build all packages in this repository. This is recommended the first time you check out a fresh repo or pull new revisions.

### Pack it

Pack packages for testing/distribution.

Assuming you built everything previously....

```
pnpm pack:all
```

This will produce tarballs in the `dist` folder. Also `genkit-dist.zip` -- a zip of all the package tarballs.

### Link it

You will need the Genkit CLI to run test apps and the Developer UI (this is done for you with `pnpm run setup`):

```
pnpm link-genkit-cli
```

## Run it

### Run a flow

In the `js/testapps` folder you will find some test apps using Genkit. They might contain instructions for how to run them and what setup is necessary.

Here's one that requires no setup:

```
cd js/testapps/flow-sample1
genkit start -- tsx --watch src/index.ts
```

Point your browser to http://localhost:4000

### Run an evaluation

We'll be using the `pdfQA` flow for our example.

To start, let's make sure we have some context to pull from the vector store.

1. Start the Developer UI

```
cd js/testapps/evals
genkit start -- pnpm genkit:dev
```

2. Click on the `indexPdf` flow in the left nav.
3. Input paths to pdfs you want to index. There's one checked into the directory:

```
"./docs/cat-handbook.pdf"
```

4. Run an evaluation on a sample dataset checked into the testapp

```
genkit eval:flow pdfQA --input ./data/cat_adoption_questions.json
```

5. Go to the **Evaluations** tab in the Developer UI to view the evaluation results.

## Code it

FYI: `js` and `genkit-tools` are in two separate workspaces.

As you make changes you may want to build and test things by running test apps.
You can reduce the scope of what you're building by running a specific build command:

```
pnpm build:genkit
pnpm build:genkit-tools
```

But you can also go into specific package that you changed and run

```
pnpm build
```

If you are going to be coding for a while then do

```
pnpm build:watch
```

in the package that you're editing.

## Document it

If you are contributing to the core Genkit JS SDK (under the `/js` directory), please make sure that all exported members have a valid [JSDoc](https://www.typescriptlang.org/docs/handbook/jsdoc-supported-types.html) associated with them.

Use the following command to build and view the new API reference locally.

```
cd js && pnpm build && pnpm typedoc-html && open api-refs-js/index.html
```

## Submitting Changes

1.  **Create a Branch**: Create a new branch for your feature or bug fix.

    ```bash
    git checkout -b your-name/my-feature-branch
    ```

2.  **Format and Build**: Before submitting a pull request, ensure your changes are formatted and build cleanly:

    ```bash
    pnpm format
    pnpm build
    ```

3.  **Commit Your Changes**: Make your changes and commit them using [Conventional Commits](https://www.conventionalcommits.org/). Both commit messages and **Pull Request titles** must include a **type** and a **scope**:

    ```text
    <type>(<scope>): <description>
    ```

    Examples:

    - `feat(js/ai): add support for custom tool output handlers`
    - `fix(go/plugins/googlegenai): default ResponseModalities for image models`
    - `chore(global): update root formatting configuration`

    Common types include:

    - `feat`: A new feature
    - `fix`: A bug fix
    - `docs`: Documentation only changes
    - `style`: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc.)
    - `refactor`: A code change that neither fixes a bug nor adds a feature
    - `perf`: A code change that improves performance
    - `test`: Adding missing tests or correcting existing tests
    - `build`: Changes that affect the build system or external dependencies
    - `ci`: Changes to CI configuration files and scripts
    - `chore`: Changes to the build process or auxiliary tools and libraries that don't modify source or test files
    - `revert`: Reverts a previous commit

4.  **Open a Pull Request**: Submit your pull request to the `main` branch. Provide a clear description of the changes and link to any related issues.

## Pull Request Guidelines

- **Keep it Focused**: A pull request should ideally do one thing. If you have multiple unrelated changes, please split them into separate PRs.
- **Update Documentation**: If your changes introduce new features or change existing behavior, please update the relevant documentation and code comments as necessary.
- **Test Your Changes**: Ensure that your changes are well-tested and don't break existing functionality.
