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
    'REFLECTION_PORT_AUTO',
    'REFLECTION_SECRET_ENV',
    'REFLECTION_SECRET_HEADER',
    'ReflectionConfig',
    'advertised_reflection_host',
    'is_loopback_host',
    'reflection_enabled',
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

#: Programmatic port meaning "let the OS pick". Code cannot use 0 for this (it
#: reads as unset); the environment spells the same thing
#: ``GENKIT_REFLECTION_PORT=0``.
REFLECTION_PORT_AUTO = -1

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
    # ASCII digits only: int() would also accept '+7', '1_000', ' 7' and
    # non-ASCII digits, all of which are more likely typos than intent.
    port = int(raw) if raw.isascii() and raw.isdigit() else -1
    if port < 0 or port > 65535:
        raise ValueError(f'GENKIT_REFLECTION_PORT must be an integer between 0 and 65535, got {raw!r}')
    return port


def _resolve_port(env_port: int | None, port: int | None) -> tuple[int, bool]:
    """Resolve ``(port, pinned)``. A chosen port is exact; only an unchosen one probes.

    Raises:
        ValueError: If ``port`` is not ``None``, ``0``, ``-1`` or 1..65535.
    """
    if env_port is not None:
        return env_port, True
    if not port:
        return DEFAULT_REFLECTION_PORT, False
    if port == REFLECTION_PORT_AUTO:
        return 0, True
    if port < 1 or port > 65535:
        raise ValueError(f'reflection port must be -1 (OS-assigned) or an integer between 1 and 65535, got {port}')
    return port, True


def resolve_reflection_config(
    env: dict[str, str] | None = None,
    port: int | None = None,
    host: str | None = None,
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
        port: Programmatic port, used only when the environment pins none.
            Bound exactly; ``None`` or ``0`` means unset (probe from 3100) and
            ``REFLECTION_PORT_AUTO`` (-1) lets the OS pick, the same as
            ``GENKIT_REFLECTION_PORT=0``. It does not turn the server on, and
            is validated even when unused so a bad value fails early.
        host: Programmatic interface, used only when ``GENKIT_REFLECTION_HOST``
            is unset. Unlike the variable, it does not turn the server on.

    Returns:
        The resolved configuration.

    Raises:
        ValueError: If ``GENKIT_REFLECTION_PORT`` or ``port`` is not a valid port.
    """
    environ = dict(os.environ) if env is None else env
    if environ.get('GENKIT_REFLECTION_DISABLED') == 'true':
        return ReflectionConfig(mode='disabled')

    secret = environ.get(REFLECTION_SECRET_ENV) or None
    v2_url = environ.get('GENKIT_REFLECTION_V2_SERVER')
    if v2_url:
        return ReflectionConfig(mode='v2', v2_url=v2_url, secret=secret)

    env_port = _parse_port(environ.get('GENKIT_REFLECTION_PORT'))
    resolved_port, pinned = _resolve_port(env_port, port)
    env_host = environ.get('GENKIT_REFLECTION_HOST')
    if env_port is None and not env_host and environ.get('GENKIT_ENV') != 'dev':
        return ReflectionConfig(mode='off')

    return ReflectionConfig(
        mode='v1',
        host=env_host or host or DEFAULT_REFLECTION_HOST,
        port=resolved_port,
        pinned=pinned,
        secret=secret,
    )


def reflection_enabled() -> bool:
    """Whether the environment turns the reflection API on, without raising.

    For telemetry setup, which runs at import time. An invalid
    ``GENKIT_REFLECTION_PORT`` counts as off here so ``import genkit`` still
    works; ``Genkit()`` raises the real error when it resolves the config.
    """
    try:
        return resolve_reflection_config().enabled
    except ValueError:
        return False


def advertised_reflection_host(host: str) -> str:
    """Host to advertise in the runtime discovery file for a server bound to ``host``.

    A wildcard bind is reachable on loopback, and ``0.0.0.0`` is not a valid
    destination everywhere, so it is advertised as ``127.0.0.1``. IPv6
    literals are bracketed for use in a URL.
    """
    if host in ('0.0.0.0', '::', '[::]'):  # noqa: S104 - comparing, not binding
        return DEFAULT_REFLECTION_HOST
    return f'[{host}]' if ':' in host and not host.startswith('[') else host


def is_loopback_host(host: str) -> bool:
    """Whether a host is unreachable from other machines."""
    return host in ('localhost', '::1', '[::1]') or host.startswith('127.')


def secrets_equal(a: str, b: str) -> bool:
    """Compare secrets in constant time.

    Hashing first gives equal-length inputs without leaking the expected length.
    """
    return hmac.compare_digest(sha256(a.encode()).digest(), sha256(b.encode()).digest())
