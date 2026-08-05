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

import {
  FirestoreSessionStore as GcpFirestoreSessionStore,
  type FirestoreSessionStoreOptions as GcpFirestoreSessionStoreOptions,
} from '@genkit-ai/google-cloud/beta';
import { App } from 'firebase-admin/app';
import { getFirestore } from 'firebase-admin/firestore';

/**
 * Options for {@link FirestoreSessionStore}.
 *
 * Extends the core {@link GcpFirestoreSessionStoreOptions} from
 * `@genkit-ai/google-cloud` with Firebase-specific app setup.
 */
export interface FirestoreSessionStoreOptions
  extends GcpFirestoreSessionStoreOptions {
  /** A Firebase app to derive the Firestore instance from. */
  firebaseApp?: App;
}

/**
 * A Firestore-backed `SessionStore` for persisting agent session snapshots.
 *
 * This is a thin Firebase wrapper around the core implementation in
 * `@genkit-ai/google-cloud`; it adds the `firebaseApp` option (deriving the
 * Firestore instance from a Firebase app) and otherwise behaves identically.
 * See {@link GcpFirestoreSessionStore} for the storage layout and tuning
 * options (`collection`, `checkpointInterval`, `shardSize`).
 */
export class FirestoreSessionStore<
  S = unknown,
> extends GcpFirestoreSessionStore<S> {
  constructor(opts?: FirestoreSessionStoreOptions) {
    super({
      ...opts,
      db:
        opts?.db ??
        (opts?.firebaseApp ? getFirestore(opts.firebaseApp) : getFirestore()),
    });
  }
}
