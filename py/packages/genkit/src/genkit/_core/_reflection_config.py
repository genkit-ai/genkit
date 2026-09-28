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
its own switches, so it can run without the rest of the dev behavior.
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

    ``mode`` is ``disabled`` when explicitly switched off with
    ``GENKIT_REFLECTION_ENABLED=false`` (honoured everywhere), ``off`` when
    nothing turned it on, ``v1`` when listening, and ``v2`` when dialing out to
    the CLI.
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


def _parse_enabled(raw: str | None) -> bool | None:
    """Parse GENKIT_REFLECTION_ENABLED: ``true``, ``false``, or unset (empty is unset).

    Anything else raises, for the same reason as ``_parse_port``.
    """
    if not raw:
        return None
    if raw in ('true', 'false'):
        return raw == 'true'
    raise ValueError(f'GENKIT_REFLECTION_ENABLED must be "true" or "false", got {raw!r}')


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

    Whether it runs:

    - ``GENKIT_REFLECTION_ENABLED=false`` turns it off, even under dev.
    - ``GENKIT_REFLECTION_ENABLED=true`` turns it on in any environment.
    - Unset, it runs only under ``GENKIT_ENV=dev``, as it always has.

    How it runs, once on: ``GENKIT_REFLECTION_V2_SERVER`` dials out; otherwise
    the v1 server listens on ``GENKIT_REFLECTION_HOST``/``GENKIT_REFLECTION_PORT``.
    Those are settings, not on-switches: a stray value in a production env does
    not expose the API, and is not even parsed while reflection is off.

    The environment port beats ``port`` on purpose: whoever set the variable is
    typically the supervisor that already published that port.

    Args:
        env: Environment to read. Defaults to ``os.environ``.
        port: Programmatic port, used only when the environment pins none.
            Bound exactly; ``None`` or ``0`` means unset (probe from 3100) and
            ``REFLECTION_PORT_AUTO`` (-1) lets the OS pick, the same as
            ``GENKIT_REFLECTION_PORT=0``. It does not turn the server on, and
            is validated even when unused so a bad value fails early.
        host: Programmatic interface, used only when ``GENKIT_REFLECTION_HOST``
            is unset. It does not turn the server on.

    Returns:
        The resolved configuration.

    Raises:
        ValueError: If ``GENKIT_REFLECTION_ENABLED`` is not ``true``/``false``,
            or ``GENKIT_REFLECTION_PORT`` or ``port`` is not a valid port.
    """
    environ = dict(os.environ) if env is None else env
    # Validated before the on/off check so a bad value fails early.
    option_port, option_pinned = _resolve_port(None, port)
    enabled = _parse_enabled(environ.get('GENKIT_REFLECTION_ENABLED'))
    if enabled is False:
        return ReflectionConfig(mode='disabled')
    if enabled is None and environ.get('GENKIT_ENV') != 'dev':
        return ReflectionConfig(mode='off')

    secret = environ.get(REFLECTION_SECRET_ENV) or None
    v2_url = environ.get('GENKIT_REFLECTION_V2_SERVER')
    if v2_url:
        return ReflectionConfig(mode='v2', v2_url=v2_url, secret=secret)

    env_port = _parse_port(environ.get('GENKIT_REFLECTION_PORT'))
    resolved_port, pinned = (env_port, True) if env_port is not None else (option_port, option_pinned)
    return ReflectionConfig(
        mode='v1',
        host=environ.get('GENKIT_REFLECTION_HOST') or host or DEFAULT_REFLECTION_HOST,
        port=resolved_port,
        pinned=pinned,
        secret=secret,
    )


def reflection_enabled() -> bool:
    """Whether the environment turns the reflection API on, without raising.

    For telemetry setup, which runs at import time. An invalid setting counts
    as off here so ``import genkit`` still works; ``Genkit()`` raises the real
    error when it resolves the config.
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
