# Manual Test for Genkit Plugins

To run this example:

1. Set the appropriate environment variables in a `.env` file in this directory.
2. Run `pnpm i` and `pnpm run setup` in the root of this repository.
3. In this app directory, run:
   ```bash
   npm run build
   npm run genkit:dev
   ```
4. In another terminal, execute:
   ```bash
   genkit flow:run jokeFlow "chicken"
   ```

> Note: Due to a known issue, testing of flows should be conducted via the CLI via genkit flow:run etc.
> Tools and other actions can still be inspected in the Developer UI started by `genkit start`
