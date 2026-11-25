# CommonJS Compatibility Test

Genkit packages are ESM-only. This app verifies that a CommonJS project
(no `"type": "module"`, `tsc` emitting `require()` calls) can still load every
Genkit package and plugin through Node's `require(esm)` support.

It deliberately uses the most legacy TypeScript setup we support: TypeScript
5.x with `"module": "commonjs"`, which resolves types with `node10` and ignores
`package.json` `exports`. Subpath types like `genkit/beta` resolve through each
package's `typesVersions` (generated from `exports` by
`pnpm types-versions` in `js/`).

It mirrors [`../esm`](../esm), which covers the same imports from an ESM
project. CommonJS testapps using `"module": "nodenext"` (e.g.
[`../basic-gemini`](../basic-gemini)) cover the recommended CommonJS setup.

## Requirements

- Node.js 22.12+ (unflagged `require(esm)`).

## Run the Test

```bash
pnpm test
```

This type-checks against the published `.d.ts` files, builds, and runs the
compiled CommonJS output. If any type or `require()` of a Genkit package fails,
the process exits with an error.
