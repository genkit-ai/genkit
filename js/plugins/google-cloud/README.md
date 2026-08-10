# Google Cloud Plugin for Genkit

The Google Cloud plugin provides integrations with Google Cloud Platform services for Genkit.

## Features

*   **Google Cloud Observability**: Exports telemetry (traces, metrics) and logs to Google Cloud's operations suite.
*   **Model Armor**: Middleware for sanitizing user prompts and model responses using Google Cloud Model Armor.
*   **Firestore Session Store (Beta)**: Persists agent session snapshots in Firestore, sharded and scalable to arbitrarily long sessions.

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

## Firestore Session Store (Beta)

`FirestoreSessionStore` is a Firestore-backed `SessionStore` for persisting agent session snapshots. Unlike a naive single-document store, it persists each turn as an incremental JSON Patch diff anchored to periodic, sharded full-state checkpoints, so:

*   No single document approaches Firestore's [1 MiB limit](https://firebase.google.com/docs/firestore/quotas) (state is sharded across documents).
*   The number of documents read/written per turn is bounded by `checkpointInterval` rather than total session length, so it scales to arbitrarily long sessions (e.g. long-lived chatbots, coding agents).
*   Reconstruction uses only document-ID lookups inside a read-only transaction, so it needs no secondary indexes and is strongly consistent.

> If you are running on Firebase, the `@genkit-ai/firebase` package re-exports this store with Firebase app setup (a `firebaseApp` option). See its README.

### Usage

Import it from `@genkit-ai/google-cloud/beta` and pass it as the `store` when defining an agent:

```typescript
import { genkit } from 'genkit/beta';
import { FirestoreSessionStore } from '@genkit-ai/google-cloud/beta';

const ai = genkit({
  plugins: [
    // ...
  ],
});

const myAgent = ai.defineAgent({
  name: 'myAgent',
  system: 'You are a helpful assistant.',
  // Defaults to a new Firestore() instance using Application Default
  // Credentials; pass `db` to provide your own.
  store: new FirestoreSessionStore(),
});
```

### Options

*   `db`: An explicit Firestore instance. Defaults to a new `Firestore()` instance (which picks up Application Default Credentials and the `FIRESTORE_EMULATOR_HOST` environment variable).
*   `collection`: The collection where snapshot documents are stored. Defaults to `"genkit-sessions"`. Two companion collections are derived from it: `"<collection>-pointers"` (one pointer document per session) and `"<collection>-shards"` (the sharded checkpoint state).
*   `checkpointInterval`: Number of turns between full-state checkpoints. Defaults to `25`. Lower it (e.g. `10`) for small-state, read-heavy sessions; raise it (e.g. `50`-`100`) for large per-turn state retained for a long time.
*   `shardSize`: Maximum size in bytes of a single shard / diff document. Defaults to `512 KiB`. Any diff exceeding this is promoted to a sharded checkpoint so no document approaches the 1 MiB limit.

## Reference

Visit the [official Genkit documentation](https://genkit.dev/docs/js/get-started/) for more information.

The sources for this package are in the main [Genkit](https://github.com/genkit-ai/genkit) repo. Please file issues and pull requests against that repo.

License: Apache 2.0
