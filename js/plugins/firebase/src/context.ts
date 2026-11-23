/**
 * Copyright 2024 Google LLC
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
  getAppCheck,
  type DecodedAppCheckToken,
} from 'firebase-admin/app-check';
import { getAuth, type DecodedIdToken } from 'firebase-admin/auth';
// @ts-ignore - `firebase` is an optional peer dep, don't error if it's missing
import type {
  FirebaseApp,
  FirebaseOptions,
  FirebaseServerApp,
} from 'firebase/app';
import { UserFacingError } from 'genkit';
import type { ContextProvider, RequestData } from 'genkit/context';
import { initializeAppIfNecessary } from './helpers.js';

/**
 * Debug features that can be enabled to simplify testing.
 * These features are in a JSON object for FIREBASE_DEBUG_FEATURES and only take
 * effect if FIREBASE_DEBUG_MODE=true.
 *
 * Do not set these variables in production.
 */
export interface DebugFeatures {
  skipTokenVerification?: boolean;
}

let cachedDebugSkipTokenVerification: boolean | undefined;

export function setDebugSkipTokenVerification(skip: boolean) {
  cachedDebugSkipTokenVerification = skip;
}

function debugSkipTokenVerification(): boolean {
  if (cachedDebugSkipTokenVerification !== undefined) {
    return cachedDebugSkipTokenVerification;
  }
  if (!process.env.FIREBASE_DEBUG_MODE) {
    return false;
  }
  if (!process.env.FIREBASE_DEBUG_FEATURES) {
    return false;
  }
  const features = JSON.parse(
    process.env.FIREBASE_DEBUG_FEATURES
  ) as DebugFeatures;
  cachedDebugSkipTokenVerification = features.skipTokenVerification ?? false;
  return cachedDebugSkipTokenVerification;
}

/**
 * The type of data that will be added to an Action's context when using the fireabse middleware.
 * You can safely cast Action's context to a Firebase Context to help type checking and code complete.
 */
export interface FirebaseContext {
  /**
   * Information about the authorized user.
   * This comes from the Authentication header, which is a JWT bearer token.
   * Will be omitted if auth is not defined or the key is invalid. To reject requests in these cases
   * set signedIn in a declarative policy or check in a policy callback.
   */
  auth?: {
    uid: string;
    token: DecodedIdToken;
    rawToken: string;
  };

  /**
   * Information about the AppCheck token for a request.
   * This comes form the X-Firebase-AppCheck header and is included in the firebase-functions
   * client libraries (which can be used for Genkit requests irrespective of whether they're hosted
   * on Firebase).
   * Will be omitted if AppCheck tokens are invalid. To reject requests in these cases,
   * set enforceAppCheck in a declaritve policy or check in a policy callback.
   */
  app?: {
    appId: string;
    token: DecodedAppCheckToken;
    alreadyConsumed?: boolean;
    rawToken: string;
  };

  /**
   * An unverified token for a Firebase Instance ID.
   */
  instanceIdToken?: string;

  /**
   * A FirebaseServerApp with the same Auth and App Check credentials as the request.
   */
  firebaseApp?: FirebaseServerApp;
}

export interface FirebaseContextProvider<I = any> extends ContextProvider<
  FirebaseContext,
  I
> {
  (request: RequestData<I>): Promise<FirebaseContext>;
}

/**
 * Helper methods that provide most common needs for an authorization policy.
 */
export interface DeclarativePolicy {
  /**
   * Requires the user to be signed in or not.
   * Implicitly part of hasClaims.
   */
  signedIn?: boolean;

  /**
   * Requires the user's email to be verified.
   * Requires the user to be signed in.
   */
  emailVerified?: boolean;

  /**
   * Clam or Claims that must be present in the request.
   * Can be a singel claim name or array of claim names to merely test the presence
   * of a clam or can be an object of claim names and values that must be present.
   * Requires the user to be signed in.
   */
  hasClaim?: string | string[] | Record<string, string>;

  /**
   * Whether appCheck must be enforced
   */
  enforceAppCheck?: boolean;

  /**
   * Whether app check enforcement includes consuming tokens.
   * Consuming tokens adds more security at the cost of performance.
   */
  consumeAppCheckToken?: boolean;

  /**
   * Either a FirebaseApp or the options used to initialize one. When provided,
   * `context.firebaseApp` will be populated as a FirebaseServerApp with the current
   * request's auth and app check credentials allowing you to perform actions using
   * Firebase Client SDKs authenticated as the requesting user.
   *
   * You must have the `firebase` dependency in your `package.json` to use this option.
   */
  serverAppConfig?: FirebaseApp | FirebaseOptions;
}

/**
 * Calling firebaseContext() without any parameters merely parses firebase context data.
 * It does not do any validation on the data found. To do automatic validation,
 * pass either an options object or function for freeform validation.
 */
export function firebaseContext<I = any>(): FirebaseContextProvider<I>;

/**
 * Calling firebaseContext() with a declarative policy both parses and enforces context.
 * Honors the same environment variables that Cloud Functions for Firebase does to
 * mock token validation in preproduction environmets.
 */
export function firebaseContext<I = any>(
  policy: DeclarativePolicy
): FirebaseContextProvider<I>;

/**
 * Calling firebaseContext() with a policy callback parses context but delegates enforcement.
 * To control the message sent to a user, throw UserFacingError.
 * For security reasons, other error types will be returned as a 500 "internal error".
 */
export function firebaseContext<I = any>(
  policy: (context: FirebaseContext, input: I) => void | Promise<void>
): FirebaseContextProvider<I>;

export function firebaseContext<I = any>(
  policy?:
    | DeclarativePolicy
    | ((context: FirebaseContext, input: I) => void | Promise<void>)
): FirebaseContextProvider<I> {
  return async (request: RequestData): Promise<FirebaseContext> => {
    initializeAppIfNecessary();
    let auth: FirebaseContext['auth'];

    const authIdToken = extractBearerToken(request.headers['authorization']);
    const appCheckToken = request.headers['x-firebase-appcheck'];

    if ('authorization' in request.headers) {
      auth = await verifyAuthToken(authIdToken);
    }
    let app: FirebaseContext['app'];
    if ('x-firebase-appcheck' in request.headers) {
      const consumeAppCheckToken =
        typeof policy === 'object' && policy['consumeAppCheckToken'];
      app = await verifyAppCheckToken(
        appCheckToken,
        consumeAppCheckToken ?? false
      );
    }
    let instanceIdToken: FirebaseContext['instanceIdToken'];
    if ('firebase-instance-id-token' in request.headers) {
      instanceIdToken = request.headers['firebase-instance-id-token'];
    }
    const context: FirebaseContext = {};

    if (typeof policy === 'object' && policy.serverAppConfig) {
      // we dynamically import here to keep `firebase` an optional peer dep
      const { initializeServerApp } = await import('firebase/app');
      context.firebaseApp = initializeServerApp(policy.serverAppConfig, {
        appCheckToken,
        authIdToken,
        releaseOnDeref: context,
      });
    }

    if (auth) {
      context.auth = auth;
    }
    if (app) {
      context.app = app;
    }
    if (instanceIdToken) {
      context.instanceIdToken = instanceIdToken;
    }
    if (typeof policy === 'function') {
      await policy(context, request.input);
    } else if (typeof policy === 'object') {
      enforceDelcarativePolicy(policy, context);
    }
    return context;
  };
}

function verifyHasClaims(claims: string[], token: DecodedIdToken) {
  for (const claim of claims) {
    if (!token[claim] || token[claim] === 'false') {
      if (claim == 'email_verified') {
        throw new UserFacingError(
          'PERMISSION_DENIED',
          'Email must be verified'
        );
      }
      if (claim === 'admin') {
        throw new UserFacingError('PERMISSION_DENIED', 'Must be an admin');
      }
      throw new UserFacingError(
        'PERMISSION_DENIED',
        `${claim} claim is required`
      );
    }
  }
}

function enforceDelcarativePolicy(
  policy: DeclarativePolicy,
  context: FirebaseContext
) {
  if (
    (policy.signedIn || policy.hasClaim || policy.emailVerified) &&
    !context.auth
  ) {
    throw new UserFacingError('UNAUTHENTICATED', 'Auth is required');
  }
  if (policy.hasClaim) {
    if (typeof policy.hasClaim === 'string') {
      verifyHasClaims([policy.hasClaim], context.auth!.token);
    } else if (Array.isArray(policy.hasClaim)) {
      verifyHasClaims(policy.hasClaim, context.auth!.token);
    } else if (typeof policy.hasClaim === 'object') {
      for (const [claim, value] of Object.entries(policy.hasClaim)) {
        if (context.auth!.token[claim] !== value) {
          throw new UserFacingError(
            'PERMISSION_DENIED',
            `Claim ${claim} must be ${value}`
          );
        }
      }
    } else {
      // Not a user facing error so this turns into a log + 500 internal to the user.
      throw Error(`Invalid type ${typeof policy.hasClaim} for hasClaim`);
    }
  }
  if (policy.emailVerified) {
    verifyHasClaims(['email_verified'], context.auth!.token);
  }
  if (policy.enforceAppCheck && !context.app) {
    throw new UserFacingError(
      'PERMISSION_DENIED',
      `AppCheck token is required`
    );
  }
}

function extractBearerToken(authHeader: string): string | undefined {
  return /[bB]earer (.*)/.exec(authHeader)?.[1];
}

async function verifyAuthToken(
  token?: string
): Promise<FirebaseContext['auth']> {
  if (!token) {
    return undefined;
  }
  if (debugSkipTokenVerification()) {
    const decoded = unsafeDecodeToken(token) as DecodedIdToken;
    return {
      uid: decoded['sub'],
      token: decoded,
      rawToken: token,
    };
  }
  try {
    const decoded = await getAuth().verifyIdToken(token);
    return {
      uid: decoded['sub'],
      token: decoded,
      rawToken: token,
    };
  } catch (err) {
    console.error(`Error decoding auth token: ${err}`);
    throw new UserFacingError('PERMISSION_DENIED', 'Invalid auth token');
  }
}

async function verifyAppCheckToken(
  token: string,
  consumeAppCheckToken: boolean
): Promise<FirebaseContext['app']> {
  if (debugSkipTokenVerification()) {
    const decoded = unsafeDecodeToken(token) as DecodedAppCheckToken;
    return {
      appId: decoded['sub'],
      token: decoded,
      alreadyConsumed: false,
      rawToken: token,
    };
  }
  try {
    return {
      ...(await getAppCheck().verifyToken(token, {
        consume: consumeAppCheckToken,
      })),
      rawToken: token,
    };
  } catch (err) {
    console.error(`Got error verifying AppCheck token: ${err}`);
    throw new UserFacingError('PERMISSION_DENIED', 'Invalid AppCheck token');
  }
}

export function fakeToken(claims: Record<string, string>): string {
  return `fake.${Buffer.from(JSON.stringify(claims), 'utf-8').toString('base64')}.fake`;
}

const TOKEN_REGEX = /[a-zA-Z0-9_=-]+\.[a-zA-Z0-9_=-]+\.[a-zA-Z0-9_=-]+/;
function unsafeDecodeToken(token: string): Record<string, unknown> {
  if (!TOKEN_REGEX.test(token)) {
    throw new UserFacingError(
      'PERMISSION_DENIED',
      'Invalid fake token. Use the fakeToken() method to create a valid fake token'
    );
  }
  try {
    return JSON.parse(Buffer.from(token.split('.')[1], 'base64').toString());
  } catch (err) {
    throw new UserFacingError(
      'PERMISSION_DENIED',
      'Invalid fake token. Use the fakeToken() method to create a valid fake token'
    );
  }
}
