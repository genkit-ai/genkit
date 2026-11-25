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

// Tells `genkit eval:flow` and Dev UI evaluations how to build eval data
// points from flow traces. Input and output default to the flow's input and
// output; `context` (required by e.g. faithfulness and answer relevancy) has
// to be pointed at the step that fetched the documents.
module.exports = {
  evaluators: [
    {
      actionRef: '/flow/pdfQA',
      extractors: {
        // Collects the text of the documents returned by the `retrieve` step
        // in src/pdf-rag.ts. `{ outputOf: 'retrieve' }` also works, but yields
        // whole DocumentData objects, which end up JSON-stringified in the
        // judge prompt.
        context: (trace) =>
          Object.values(trace.spans)
            .filter((span) => span.displayName === 'retrieve')
            .flatMap((span) =>
              JSON.parse(span.attributes['genkit:output'] ?? '[]')
            )
            .flatMap((doc) =>
              doc.content.map((part) => part.text).filter(Boolean)
            ),
      },
    },
  ],
};
