# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Resolves how the reflection API runs, from the environment.

The reflection API used to be reachable only under ``GENKIT_ENV=dev``, which
also turned on a pile of unrelated development behavior. These settings give it
its own switches so it can run in production-ish settings (a container that
publishes the reflection port) without the rest.
"""

import hmac
import os
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

__all__ = [
    'DEFAULT_REFLECTION_HOST',
    'DEFAULT_REFLECTION_PORT',
    'REFLECTION_AUTH_ERROR_CODE',
    'REFLECTION_SECRET_ENV',
    'REFLECTION_SECRET_HEADER',
    'ReflectionConfig',
    'is_loopback_host',
    'resolve_reflection_config',
    'secrets_equal',
]

#: Header carrying the reflection secret on v1 requests.
REFLECTION_SECRET_HEADER = 'x-genkit-reflection-secret'

#: Environment variable holding the reflection secret.
REFLECTION_SECRET_ENV = 'GENKIT_REFLECTION_SECRET_TOKEN'

#: Interface used when none is configured.
DEFAULT_REFLECTION_HOST = '127.0.0.1'

#: First port tried when none is pinned.
DEFAULT_REFLECTION_PORT = 3100

#: JSON-RPC code the CLI returns when a v2 register fails auth. Terminal: the
#: secret will not change, so a runtime that sees it must stop reconnecting.
REFLECTION_AUTH_ERROR_CODE = -32001


@dataclass(frozen=True)
class ReflectionConfig:
    """Resolved reflection API configuration.

    ``mode`` is ``disabled`` when explicitly switched off (honoured everywhere),
    ``off`` when nothing asked for a server, ``v1`` when listening, and ``v2``
    when dialing out to the CLI.
    """

    mode: Literal['disabled', 'off', 'v1', 'v2']
    v2_url: str | None = None
    host: str = DEFAULT_REFLECTION_HOST
    port: int = DEFAULT_REFLECTION_PORT
    #: Whether port must be bound exactly, with no probing.
    pinned: bool = False
    secret: str | None = None

    @property
    def enabled(self) -> bool:
        """Whether a server should run at all."""
        return self.mode in ('v1', 'v2')


def _parse_port(raw: str | None) -> int | None:
    """Parse GENKIT_REFLECTION_PORT, raising on anything invalid.

    A typo in a deployment config should fail loudly rather than quietly fall
    back to probing some other port.
    """
    if not raw:
        return None
    try:
        port = int(raw)
    except ValueError:
        port = -1
    if port < 0 or port > 65535 or raw.strip() != raw:
        raise ValueError(f'GENKIT_REFLECTION_PORT must be an integer between 0 and 65535, got {raw!r}')
    return port


def resolve_reflection_config(
    env: dict[str, str] | None = None,
    port: int | None = None,
) -> ReflectionConfig:
    """Decide how the reflection API should run.

    First match wins:

    1. ``GENKIT_REFLECTION_DISABLED == 'true'`` turns everything off.
    2. ``GENKIT_REFLECTION_V2_SERVER`` dials out instead of listening.
    3. ``GENKIT_REFLECTION_PORT`` or ``GENKIT_REFLECTION_HOST`` starts v1.
    4. ``GENKIT_ENV == 'dev'`` starts v1 with defaults.
    5. Otherwise off.

    Setting a host or port is itself the on-switch, so there is no way to
    configure a server and then wonder why it never started. The environment
    beats ``port`` on purpose: whoever set the variable is typically the
    supervisor that already published that port.

    Args:
        env: Environment to read. Defaults to ``os.environ``.
        port: Programmatic probe start, used only when the environment pins no
            port.

    Returns:
        The resolved configuration.

    Raises:
        ValueError: If ``GENKIT_REFLECTION_PORT`` is not a valid port.
    """
    environ = dict(os.environ) if env is None else env
    if environ.get('GENKIT_REFLECTION_DISABLED') == 'true':
        return ReflectionConfig(mode='disabled')

    secret = environ.get(REFLECTION_SECRET_ENV) or None
    v2_url = environ.get('GENKIT_REFLECTION_V2_SERVER')
    if v2_url:
        return ReflectionConfig(mode='v2', v2_url=v2_url, secret=secret)

    env_port = _parse_port(environ.get('GENKIT_REFLECTION_PORT'))
    host = environ.get('GENKIT_REFLECTION_HOST')
    if env_port is None and not host and environ.get('GENKIT_ENV') != 'dev':
        return ReflectionConfig(mode='off')

    return ReflectionConfig(
        mode='v1',
        host=host or DEFAULT_REFLECTION_HOST,
        port=env_port if env_port is not None else (port or DEFAULT_REFLECTION_PORT),
        pinned=env_port is not None,
        secret=secret,
    )


def is_loopback_host(host: str) -> bool:
    """Whether a host is unreachable from other machines."""
    return host in ('localhost', '::1', '[::1]') or host.startswith('127.')


def secrets_equal(a: str, b: str) -> bool:
    """Compare secrets in constant time.

    Hashing first gives equal-length inputs without leaking the expected length.
    """
    return hmac.compare_digest(sha256(a.encode()).digest(), sha256(b.encode()).digest())
