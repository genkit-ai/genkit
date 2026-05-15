# Google Cloud Plugin for Genkit

The Google Cloud plugin provides integrations with Google Cloud Platform services for Genkit.

## Features

*   **Google Cloud Observability**: Exports telemetry (traces, metrics) and logs to Google Cloud's operations suite.
*   **Model Armor**: Middleware for sanitizing user prompts and model responses using Google Cloud Model Armor.

## Installation

```bash
npm install @genkit-ai/google-cloud
```

## Google Cloud Observability

The plugin allows you to export telemetry data to Google Cloud. This is useful for monitoring your Genkit flows and models in production.

To enable it, use `enableGoogleCloudTelemetry`:

```typescript
import { enableGoogleCloudTelemetry } from '@genkit-ai/google-cloud';

enableGoogleCloudTelemetry({
  // Optional configuration
  // projectId: 'your-project-id',
  // forceDevMode: false, // Set to true to enable export in dev environment
});
```

This will configure Genkit to send OpenTelemetry traces and metrics to Cloud Trace and Cloud Monitoring, and logs to Cloud Logging.

## Model Armor

[Google Cloud Model Armor](https://docs.cloud.google.com/model-armor/overview) helps you mitigate risks when using Large Language Models (LLMs) by providing a layer of protection that sanitizes both user prompts and model responses.

### Usage

You can use the `modelArmor` middleware in your generation requests:

```typescript
import { modelArmor } from '@genkit-ai/google-cloud/model-armor';
import { googleAI } from '@genkit-ai/google-genai';
import { genkit } from 'genkit';

const ai = genkit({
  plugins: [googleAI()],
});

const response = await ai.generate({
  model: googleAI.model('gemini-2.5-flash'),
  prompt: 'your prompt here',
  use: [
    modelArmor({
      templateName: 'projects/your-project/locations/your-location/templates/your-template',
      // Optional configuration
      filters: ['pi_and_jailbreak', 'malicious_uris'], // Specific filters to enforce
      strictSdpEnforcement: true, // Block if sensitive data is found even if masked
      protectionTarget: 'all', // 'all', 'userPrompt', or 'modelResponse'
      clientOptions: {
        apiEndpoint: 'modelarmor.us-central1.rep.googleapis.com',
      },
    }),
  ],
});
```

### Configuration Options

*   `templateName` (Required): The resource name of your Model Armor template (e.g., `projects/.../locations/.../templates/...`).
*   `filters` (Optional): A list of filters to enforce (e.g., `rai`, `pi_and_jailbreak`, `malicious_uris`, `csam`, `sdp`). If not specified, all filters enabled in the template are enforced.
*   `strictSdpEnforcement` (Optional): If `true`, blocks execution if Sensitive Data Protection (SDP) detects sensitive info, even if it was successfully de-identified. Defaults to `false`.
*   `protectionTarget` (Optional): specificies what to sanitize. Options: `'all'` (default), `'userPrompt'`, `'modelResponse'`.
*   `clientOptions` (Optional): Additional options for the underlying Model Armor client.

## Reference

Visit the [official Genkit documentation](https://genkit.dev/docs/get-started/) for more information.

The sources for this package are in the main [Genkit](https://github.com/genkit-ai/genkit) repo. Please file issues and pull requests against that repo.

License: Apache 2.0
