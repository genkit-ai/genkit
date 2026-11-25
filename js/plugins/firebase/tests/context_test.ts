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

import type { RequestData } from 'genkit/context';
import { describe, expect, it } from 'vitest';
import {
  fakeToken,
  firebaseContext,
  setDebugSkipTokenVerification,
  type FirebaseContext,
} from '../src/context.js';

function request(headers: Record<string, string> = {}): RequestData {
  return {
    method: 'POST',
    headers,
    input: undefined,
  };
}
describe('firebaseAuth', () => {
  setDebugSkipTokenVerification(true);

  describe('no policy', () => {
    it('handles noop', async () => {
      const context = await firebaseContext()(request());
      expect(context).toEqual({});
    });

    it('handles all headers', async () => {
      const context = await firebaseContext()(
        request({
          authorization: `bearer ${fakeToken({ sub: 'user' })}`,
          'x-firebase-appcheck': fakeToken({ sub: 'appId' }),
          'firebase-instance-id-token': 'token',
        })
      );
      expect(context).toEqual({
        auth: {
          uid: 'user',
          token: { sub: 'user' },
          rawToken: fakeToken({ sub: 'user' }),
        },
        app: {
          appId: 'appId',
          token: { sub: 'appId' },
          alreadyConsumed: false,
          rawToken: fakeToken({ sub: 'appId' }),
        },
        instanceIdToken: 'token',
      });
    });
  });

  describe('declaritive policies', () => {
    it('handles signedIn', async () => {
      await expect(
        firebaseContext({ signedIn: true })(request())
      ).rejects.toThrow();
      expect(await firebaseContext({ signedIn: false })(request())).toEqual({});
      expect(
        await firebaseContext({ signedIn: false })(
          request({ authorization: `bearer ${fakeToken({ sub: 'user' })}` })
        )
      ).toEqual({
        auth: {
          uid: 'user',
          token: {
            sub: 'user',
          },
          rawToken: fakeToken({ sub: 'user' }),
        },
      });
    });

    it('handles emailVerified', async () => {
      await expect(
        firebaseContext({ emailVerified: true })(request())
      ).rejects.toThrow();
      expect(
        await firebaseContext({ emailVerified: false })(request())
      ).toEqual({});
      await expect(
        firebaseContext({ emailVerified: true })(
          request({ authorization: `bearer ${fakeToken({ sub: 'user' })}` })
        )
      ).rejects.toThrow();
      await expect(
        firebaseContext({ emailVerified: true })(
          request({
            authorization: `bearer ${fakeToken({ sub: 'user', email: 'user@google.com', email_verified: 'false' })}`,
          })
        )
      ).rejects.toThrow();
      expect(
        await firebaseContext({ emailVerified: true })(
          request({
            authorization: `bearer ${fakeToken({ sub: 'user', email: 'user@google.com', email_verified: 'true' })}`,
          })
        )
      ).toEqual({
        auth: {
          uid: 'user',
          token: {
            sub: 'user',
            email: 'user@google.com',
            email_verified: 'true',
          },
          rawToken: fakeToken({
            sub: 'user',
            email: 'user@google.com',
            email_verified: 'true',
          }),
        },
      });
    });

    it('enforces hasClaim (string)', async () => {
      expect(
        await firebaseContext({ hasClaim: 'email' })(
          request({
            authorization: `bearer ${fakeToken({ sub: 'user', email: 'user@google.com' })}`,
          })
        )
      ).toEqual({
        auth: {
          uid: 'user',
          token: {
            sub: 'user',
            email: 'user@google.com',
          },
          rawToken: fakeToken({ sub: 'user', email: 'user@google.com' }),
        },
      });
      await expect(
        firebaseContext({ hasClaim: 'admin' })(
          request({
            authorization: `bearer ${fakeToken({
              sub: 'user',
            })}`,
          })
        )
      ).rejects.toThrow();
      await expect(
        firebaseContext({ hasClaim: 'admin' })(
          request({
            authorization: `bearer ${fakeToken({
              sub: 'user',
              admin: 'false',
            })}`,
          })
        )
      ).rejects.toThrow();
    });

    it('handles hasClaim(string[])', async () => {
      await expect(
        firebaseContext({ hasClaim: ['email', 'admin'] })(
          request({
            authorization: `bearer ${fakeToken({
              sub: 'user',
              admin: 'true',
            })}`,
          })
        )
      ).rejects.toThrow();
      await expect(
        firebaseContext({ hasClaim: ['email', 'admin'] })(
          request({
            authorization: `bearer ${fakeToken({
              sub: 'user',
              admin: 'false',
              email: 'user@google.com',
            })}`,
          })
        )
      ).rejects.toThrow();
      expect(
        await firebaseContext({ hasClaim: ['email', 'admin'] })(
          request({
            authorization: `bearer ${fakeToken({
              sub: 'user',
              admin: 'true',
              email: 'user@google.com',
            })}`,
          })
        )
      ).toEqual({
        auth: {
          uid: 'user',
          token: {
            sub: 'user',
            admin: 'true',
            email: 'user@google.com',
          },
          rawToken: fakeToken({
            sub: 'user',
            admin: 'true',
            email: 'user@google.com',
          }),
        },
      });
    });

    it('handles hasClaim(Record<string, boolean>)', async () => {
      await expect(
        firebaseContext({
          hasClaim: {
            admin: 'true',
            humor: 'dad',
          },
        })(
          request({
            authorization: `bearer ${fakeToken({
              uid: 'user',
              admin: 'true',
            })}`,
          })
        )
      ).rejects.toThrow();
      await expect(
        firebaseContext({
          hasClaim: {
            admin: 'true',
            humor: 'dad',
          },
        })(
          request({
            authorization: `bearer ${fakeToken({
              uid: 'user',
              admin: 'true',
              humor: 'programming',
            })}`,
          })
        )
      ).rejects.toThrow();
      expect(
        await firebaseContext({
          hasClaim: {
            admin: 'true',
            humor: 'dad',
          },
        })(
          request({
            authorization: `bearer ${fakeToken({
              sub: 'user',
              admin: 'true',
              humor: 'dad',
            })}`,
          })
        )
      ).toEqual({
        auth: {
          uid: 'user',
          token: {
            sub: 'user',
            admin: 'true',
            humor: 'dad',
          },
          rawToken: fakeToken({
            sub: 'user',
            admin: 'true',
            humor: 'dad',
          }),
        },
      });
    });

    it('handles enforceAppCheck', async () => {
      expect(
        await firebaseContext({ enforceAppCheck: true })(
          request({
            'x-firebase-appcheck': fakeToken({ sub: 'appId' }),
          })
        )
      ).toEqual({
        app: {
          appId: 'appId',
          token: {
            sub: 'appId',
          },
          alreadyConsumed: false,
          rawToken: fakeToken({ sub: 'appId' }),
        },
      });
      await expect(
        firebaseContext({ enforceAppCheck: true })(request())
      ).rejects.toThrow();
    });
  });

  describe('policy functions', () => {
    it('passes context', () => {
      firebaseContext((context: FirebaseContext) => {
        expect(context).toEqual({
          auth: {
            uid: 'user',
            token: {
              sub: 'user',
            },
            rawToken: fakeToken({ sub: 'user' }),
          },
          app: {
            appId: 'app',
            token: {
              sub: 'app',
            },
            alreadyConsumed: false,
            rawToken: fakeToken({ sub: 'app' }),
          },
          instanceIdToken: 'iid',
        });
      })(
        request({
          authorization: `bearer ${fakeToken({ sub: 'user' })}`,
          'x-firebase-appcheck': fakeToken({ sub: 'app' }),
          'firebase-instance-id-token': 'iid',
        })
      );
      firebaseContext((context: FirebaseContext) => {
        expect(context).toEqual({});
      })(request());
    });
  });
});
