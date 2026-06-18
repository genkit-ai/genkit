# Firebase plugin for Genkit

The Firebase plugin integrates Genkit with Firebase and Google Cloud. It provides:

  * **Telemetry / Monitoring** - export traces, metrics, and logs to Google Cloud / Firebase for [Genkit monitoring](https://genkit.dev/docs/js/observability/getting-started/).
  * **Durable Streaming (Beta)** - persist flow stream state in Firestore or the Realtime Database so streams can be resumed.
  * **Session Store (Beta)** - persist agent session snapshots in Firestore, sharded and scalable to arbitrarily long sessions.

See also the official docs for [deploying Genkit with Firebase](https://genkit.dev/docs/js/deployment/firebase/).

## Installing the plugin

```bash
npm i --save @genkit-ai/firebase
```

## Telemetry / Monitoring

Call `enableFirebaseTelemetry()` to export Genkit traces, metrics, and logs to Google Cloud / Firebase:

```ts
import { genkit } from 'genkit';
import { enableFirebaseTelemetry } from '@genkit-ai/firebase';

enableFirebaseTelemetry();

const ai = genkit({
  plugins: [
    // ...
  ],
});
```

See the [monitoring documentation](https://genkit.dev/docs/js/observability/getting-started/) for details.

## Durable Streaming (Beta)

This plugin provides two `StreamManager` implementations for durable streaming:

*   `FirestoreStreamManager`: Persists stream state in Google Firestore.
*   `RtdbStreamManager`: Persists stream state in the Firebase Realtime Database.

You can use these with `expressHandler` or `appRoute` to make your flow streams durable.

### Usage

To use a stream manager, import it from `@genkit-ai/firebase/beta` and provide it to your flow handler:

```ts
import { expressHandler } from '@genkit-ai/express';
import { FirestoreStreamManager } from '@genkit-ai/firebase/beta';
import express from 'express';
import { initializeApp } from 'firebase-admin/app';
import { getFirestore } from 'firebase-admin/firestore';

// ... define your flow: myFlow

const fApp = initializeApp();
const firestore = new FirestoreStreamManager({
  firebaseApp: fApp,
  db: getFirestore(fApp),
  collection: 'streams',
});

const app = express();
app.use(express.json());

app.post('/myDurableFlow', expressHandler(myFlow, { streamManager: firestore }));

app.listen(8080);
```

Similarly, for the Realtime Database:

```ts
import { RtdbStreamManager } from '@genkit-ai/firebase/beta';

const rtdb = new RtdbStreamManager({
  firebaseApp: fApp,
  refPrefix: 'streams',
});

app.post('/myDurableRtdbFlow', expressHandler(myFlow, { streamManager: rtdb }));
```

### Limitations

*   **Firestore**: The entire stream history (chunks and final result) is stored in a single document. Firestore has a strict [1MB limitation on document size](https://firebase.google.com/docs/firestore/quotas). If your stream output exceeds this limit, the flow will fail.
*   **Realtime Database**: While RTDB does not have the same 1MB limit, storing very large streams may impact performance or hit other quotas.

## Session Store (Beta)

`FirestoreSessionStore` is a Firestore-backed `SessionStore` for persisting agent session snapshots. It is a thin Firebase wrapper around the core implementation in [`@genkit-ai/google-cloud`](https://www.npmjs.com/package/@genkit-ai/google-cloud), adding a `firebaseApp` option for deriving the Firestore instance from a Firebase app. Unlike a naive single-document store, it persists each turn as an incremental JSON Patch diff anchored to periodic, sharded full-state checkpoints, so:

*   No single document approaches Firestore's [1 MiB limit](https://firebase.google.com/docs/firestore/quotas) (state is sharded across documents).
*   The number of documents read/written per turn is bounded by `checkpointInterval` rather than total session length, so it scales to arbitrarily long sessions (e.g. long-lived chatbots, coding agents).
*   Reconstruction uses only document-ID lookups inside a read-only transaction, so it needs no secondary indexes and is strongly consistent.

### Usage

Import it from `@genkit-ai/firebase/beta` and pass it as the `store` when defining an agent:

```ts
import { genkit } from 'genkit/beta';
import { FirestoreSessionStore } from '@genkit-ai/firebase/beta';
import { initializeApp } from 'firebase-admin/app';

const fApp = initializeApp();

const ai = genkit({
  plugins: [
    // ...
  ],
});

const myAgent = ai.defineAgent({
  name: 'myAgent',
  model: 'googleai/gemini-2.5-flash',
  system: 'You are a helpful assistant.',
  store: new FirestoreSessionStore({ firebaseApp: fApp }),
});
```

### Options

`FirestoreSessionStore` accepts the following options:

*   `firebaseApp`: A Firebase app to derive the Firestore instance from.
*   `db`: An explicit Firestore instance. Takes precedence over `firebaseApp`.
*   `collection`: The collection where snapshot documents are stored. Defaults to `"genkit-sessions"`. Two companion collections are derived from it: `"<collection>-pointers"` (one pointer document per session) and `"<collection>-shards"` (the sharded checkpoint state).
*   `checkpointInterval`: Number of turns between full-state checkpoints. Defaults to `25`. Lower it (e.g. `10`) for small-state, read-heavy sessions; raise it (e.g. `50`-`100`) for large per-turn state retained for a long time.
*   `shardSize`: Maximum size in bytes of a single shard / diff document. Defaults to `512 KiB`. Any diff exceeding this is promoted to a sharded checkpoint so no document approaches the 1 MiB limit.

---

The sources for this package are in the main [Genkit](https://github.com/genkit-ai/genkit) repo. Please file issues and pull requests against that repo.

Usage information and reference details can be found in [Genkit documentation](https://genkit.dev/).

License: Apache 2.0
