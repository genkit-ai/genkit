# Firebase plugin for Genkit

See the official documentation for more:

- [Deploying Genkit with Firebase](https://genkit.dev/docs/deployment/firebase/)
- [Genkit monitoring with Firebase](https://genkit.dev/docs/observability/getting-started/)

## Installing the plugin

```bash
npm i --save @genkit-ai/firebase
```

## Using the plugin

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

The sources for this package are in the main [Genkit](https://github.com/genkit-ai/genkit) repo. Please file issues and pull requests against that repo.

Usage information and reference details can be found in [Genkit documentation](https://genkit.dev/).

License: Apache 2.0
