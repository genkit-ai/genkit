# Checks

**Checks** is an AI safety platform built by Google: [checks.google.com/ai-safety](https://checks.google.com/ai-safety).

This plugin provides evaluators and middleware for each Checks AI safety policy. Text is classified by calling the [Checks Guardrails API](https://console.cloud.google.com/marketplace/product/google/checks.googleapis.com?project=_).

> **Note**: The Guardrails API is currently in private preview, and you will need to request quota. To request quota, fill out this [Google form](https://docs.google.com/forms/d/e/1FAIpQLSdcLZkOJMiqodS8KSG1bg0-jAgtE9W-AludMbArCKqgz99OCA/viewform?usp=sf_link).
>
> Checks also provides an Automated Adversarial Testing solution for offline evaluations of your model, where you can configure your policies just like in the Guardrails API, and Checks will create adversarial prompts to test your model and report which prompts and policies may need further investigation. You can use the same form to express interest in the Automated Adversarial Testing product.

Currently, the supported policies include:

```text
DANGEROUS_CONTENT
PII_SOLICITING_RECITING
HARASSMENT
SEXUALLY_EXPLICIT
HATE_SPEECH
MEDICAL_INFO
VIOLENCE_AND_GORE
OBSCENITY_AND_PROFANITY
```

## Middleware

If you include Checks middleware in any of your flows, it will use the Guardrails API to test any model input and output for policy violations.

### How to Use

#### Example

```ts
// Import the checks middleware and metric types.
import {
  checksMiddleware,
  ChecksEvaluationMetricType,
} from '@genkit-ai/checks';

// Import any models you would like to use.
import { googleAI } from '@genkit-ai/google-genai';

export const ai = genkit({
  plugins: [
    googleAI(),
    // ... any other plugins you need.
  ],
});

// Simple example flow
export const poemFlow = ai.defineFlow(
  {
    name: 'menuSuggestionFlow',
  },
  async (topic) => {
    const { text } = await ai.generate({
      model: googleAI.model('gemini-2.5-flash'),
      prompt: `Write a poem on this topic: ${topic}`,
      // Add checks middleware to your generate calls.
      use: [
        checksMiddleware({
          authOptions: {
            // Project to charge quota to.
            // Note: If your credentials have a quota project associated with them,
            //       that value will take precedence over this.
            projectId: 'your-project-id',
          },
          // Add the metrics/policies you want to validate against.
          metrics: [
            // This will use the default threshold determined by Checks.
            ChecksEvaluationMetricType.DANGEROUS_CONTENT,
            // This is how you can override the default threshold.
            {
              type: ChecksEvaluationMetricType.VIOLENCE_AND_GORE,
              // If the content scores above 0.55, it fails and the response will be blocked.
              threshold: 0.55,
            },
          ],
        }),
      ],
    });
    return text;
  }
);
```

#### Test Your Flow

Start the UI:

```bash
cd your/project/directory
genkit ui:start
```

#### Run Your App

```bash
genkit start -- tsx --watch src/index.ts
```

#### Run the Flow

Open your app, most likely at `localhost:4000`. Select **Flows** in the left navigation menu. Select your flow and provide input. Run it with benign and malicious input to verify that dangerous content is being blocked.

#### Tune Thresholds

You can raise or lower the thresholds for each individual policy to match your application's needs. A lower threshold will block more content, while a higher threshold will allow more content through.

For example:

```ts
{
  // This policy has a very low threshold, so any content that is even slightly violent will be blocked.
  type: ChecksEvaluationMetricType.VIOLENCE_AND_GORE,
  threshold: 0.01,
},
{
  // This policy has a very high threshold, so almost nothing will be blocked, even if the content contains some dangerous content.
  type: ChecksEvaluationMetricType.DANGEROUS_CONTENT,
  threshold: 0.99,
}
```

## Evaluator

The Checks evaluators let you run offline safety tests of prompts.

### How to Use

#### Configure the Plugin

Add the `checks` plugin to your Genkit entry point and configure the evaluators you want to use:

```ts
import { checks, ChecksEvaluationMetricType } from '@genkit-ai/checks';

export const ai = genkit({
  plugins: [
    checks({
      // Project to charge quota to.
      // Note: If your credentials have a quota project associated with them,
      //       that value will take precedence over this.
      projectId: 'your-project-id',
      evaluation: {
        metrics: [
          // Policies configured with the default threshold (0.5).
          ChecksEvaluationMetricType.DANGEROUS_CONTENT,
          ChecksEvaluationMetricType.HARASSMENT,
          ChecksEvaluationMetricType.HATE_SPEECH,
          ChecksEvaluationMetricType.MEDICAL_INFO,
          ChecksEvaluationMetricType.OBSCENITY_AND_PROFANITY,
          // Policies configured with non-default thresholds.
          {
            type: ChecksEvaluationMetricType.PII_SOLICITING_RECITING,
            threshold: 0.6,
          },
          {
            type: ChecksEvaluationMetricType.SEXUALLY_EXPLICIT,
            threshold: 0.3,
          },
          {
            type: ChecksEvaluationMetricType.VIOLENCE_AND_GORE,
            threshold: 0.55,
          },
        ],
      },
    }),
  ],
});
```

#### Create a Test Dataset

Create a JSON file with the data you want to test. Add as many test cases as you want. `output` is the text that will be classified.

```json
[
  {
    "testCaseId": "test_case_id_1",
    "input": "The input to your model.",
    "output": "Example model output. This is what will be evaluated."
  }
]
```

#### Run the Evaluators

```bash
# Run the configured evaluators.
genkit eval:run test-dataset.json --evaluators=checks/guardrails
```

#### View the Results

Run `genkit start -- tsx --watch src/index.ts` and open the Genkit UI, usually at `localhost:4000`. Select the **Evaluate** tab.

# Genkit

The sources for this package are in the main [Genkit](https://github.com/genkit-ai/genkit) repository. Please file issues and pull requests against that repository.

Usage information and reference details can be found in the [official Genkit documentation](https://genkit.dev/docs/get-started/).
