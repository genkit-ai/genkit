# Evaluation in Genkit

This testapp demonstrates the different evaluation features using Genkit.

Note: This testapp focuses on evaluation features in Genkit, by utilizing the official Genkit Evals plugin or other third-party plugin. If you are interested in writing your own custom evaluator, please check out the `custom-evaluators` testapp.

## Setup and start the testapp

```posix-terminal
# Build the app
pnpm build

# Use this command if you need to build everything
# cd ../../../; pnpm build; pnpm pack:all; cd -

# Start the app

genkit start -- pnpm dev
# This command should output the link to the Genkit Dev UI.
```

The rest of the commands in this guide must be run in a separate terminal.

```posix-terminal
# Index "docs/cat-handbook.pdf" to start
# testing Genkit evaluation features. Please see
# src/setup.ts for more details.

genkit flow:run setup
```

## Evaluations

### Simple inference and evaluation

Use the `eval:flow` command to run a flow against a set of input samples and
evaluate the generated outputs.

```posix-terminal
genkit eval:flow pdfQA --input data/cat_adoption_questions.jsonl --evaluator=genkitEval/maliciousness
```

Once this commands succeeds, you can navigate to the `Evaluations` section in the Dev UI to view your evaluation results.

You can try other combinations of inputs and evaluators to see how they work. Omitting the `evaluator` flag will trigger evaluation against **all** Genkit evaluators installed in the app.

### Context for RAG evaluators

Evaluators such as faithfulness and answer relevancy also need the `context`
the answer was based on. `pdfQA` wraps its document lookup in a named step:

```ts
const docs = await ai.run('retrieve', () => pdfStore.query(query, { k: 3 }));
```

and [genkit-tools.conf.js](./genkit-tools.conf.js) tells the tooling to use the
text of that step's output documents as the eval context:

```js
module.exports = {
  evaluators: [
    {
      actionRef: '/flow/pdfQA',
      extractors: {
        context: (trace) =>
          Object.values(trace.spans)
            .filter((span) => span.displayName === 'retrieve')
            .flatMap((span) => JSON.parse(span.attributes['genkit:output'] ?? '[]'))
            .flatMap((doc) => doc.content.map((part) => part.text).filter(Boolean)),
      },
    },
  ],
};
```

Try it with:

```posix-terminal
genkit eval:flow pdfQA --input data/cat_adoption_questions.jsonl --evaluator=genkitEval/faithfulness
```

`input` and `output` default to the flow's input and output. Extractors can
also be a step name (`'retrieve'`) or a selector (`{ outputOf: 'retrieve' }`,
`{ inputOf: 'retrieve' }`).

### Evaluation in code

Please see the `src/eval-in-code.ts` file for an example on running evaluations from within code.

## Reference

For more details on using Genkit evaluations, please refer to the official [Genkit documentation](https://genkit.dev/docs/evaluation).
