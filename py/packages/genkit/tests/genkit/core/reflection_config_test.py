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

"""Tests for reflection API configuration resolution."""

import pytest

from genkit._core._reflection_config import (
    DEFAULT_REFLECTION_HOST,
    DEFAULT_REFLECTION_PORT,
    REFLECTION_PORT_AUTO,
    advertised_reflection_host,
    is_loopback_host,
    reflection_enabled,
    resolve_reflection_config,
    secrets_equal,
)

#: Wildcard bind address, as test data rather than an actual bind.
ALL_INTERFACES = '0.0.0.0'  # noqa: S104


def test_off_when_nothing_is_set() -> None:
    assert resolve_reflection_config({}).mode == 'off'
    assert not resolve_reflection_config({}).enabled


def test_disabled_beats_everything() -> None:
    config = resolve_reflection_config({
        'GENKIT_REFLECTION_DISABLED': 'true',
        'GENKIT_ENV': 'dev',
        'GENKIT_REFLECTION_PORT': '3100',
        'GENKIT_REFLECTION_V2_SERVER': 'ws://127.0.0.1:3200',
    })
    assert config.mode == 'disabled'
    assert not config.enabled


@pytest.mark.parametrize('value', ['1', 'yes', 'on', 'TRUE'])
def test_only_exact_true_disables(value: str) -> None:
    config = resolve_reflection_config({'GENKIT_REFLECTION_DISABLED': value, 'GENKIT_ENV': 'dev'})
    assert config.mode == 'v1'


def test_v2_beats_a_configured_v1_port() -> None:
    config = resolve_reflection_config({
        'GENKIT_REFLECTION_V2_SERVER': 'ws://127.0.0.1:3200',
        'GENKIT_REFLECTION_PORT': '3100',
        'GENKIT_REFLECTION_SECRET_TOKEN': 's3cret',
    })
    assert config.mode == 'v2'
    assert config.v2_url == 'ws://127.0.0.1:3200'
    assert config.secret == 's3cret'


def test_host_alone_turns_it_on() -> None:
    config = resolve_reflection_config({'GENKIT_REFLECTION_HOST': ALL_INTERFACES})
    assert (config.mode, config.host, config.port, config.pinned) == (
        'v1',
        ALL_INTERFACES,
        DEFAULT_REFLECTION_PORT,
        False,
    )


def test_port_alone_turns_it_on_and_pins() -> None:
    config = resolve_reflection_config({'GENKIT_REFLECTION_PORT': '4200'})
    assert (config.mode, config.host, config.port, config.pinned) == ('v1', DEFAULT_REFLECTION_HOST, 4200, True)


def test_dev_probes_from_3100() -> None:
    config = resolve_reflection_config({'GENKIT_ENV': 'dev'})
    assert (config.mode, config.host, config.port, config.pinned) == (
        'v1',
        DEFAULT_REFLECTION_HOST,
        DEFAULT_REFLECTION_PORT,
        False,
    )


def test_env_port_beats_the_programmatic_one() -> None:
    config = resolve_reflection_config({'GENKIT_REFLECTION_PORT': '4200'}, port=9999)
    assert (config.port, config.pinned) == (4200, True)


def test_programmatic_port_is_pinned() -> None:
    config = resolve_reflection_config({'GENKIT_ENV': 'dev'}, port=9999)
    assert (config.port, config.pinned) == (9999, True)


@pytest.mark.parametrize('port', [None, 0])
def test_unset_programmatic_port_probes_from_3100(port: int | None) -> None:
    config = resolve_reflection_config({'GENKIT_ENV': 'dev'}, port=port)
    assert (config.port, config.pinned) == (DEFAULT_REFLECTION_PORT, False)


def test_programmatic_auto_matches_env_zero() -> None:
    auto = resolve_reflection_config({'GENKIT_ENV': 'dev'}, port=REFLECTION_PORT_AUTO)
    env_zero = resolve_reflection_config({'GENKIT_REFLECTION_PORT': '0'})
    assert (auto.port, auto.pinned) == (env_zero.port, env_zero.pinned) == (0, True)


def test_programmatic_port_does_not_turn_it_on() -> None:
    assert resolve_reflection_config({}, port=9999).mode == 'off'


@pytest.mark.parametrize('port', [-2, 70000])
def test_invalid_programmatic_port_raises_even_when_off(port: int) -> None:
    with pytest.raises(ValueError, match='reflection port'):
        resolve_reflection_config({}, port=port)


@pytest.mark.parametrize(
    ('host', 'expected'),
    [
        (ALL_INTERFACES, '127.0.0.1'),
        ('::', '127.0.0.1'),
        ('127.0.0.1', '127.0.0.1'),
        ('192.168.1.5', '192.168.1.5'),
        ('::1', '[::1]'),
        ('[::1]', '[::1]'),
    ],
)
def test_advertised_reflection_host(host: str, expected: str) -> None:
    assert advertised_reflection_host(host) == expected


def test_programmatic_host_is_used_when_env_host_is_unset() -> None:
    config = resolve_reflection_config({'GENKIT_ENV': 'dev'}, host=ALL_INTERFACES)
    assert config.host == ALL_INTERFACES


def test_env_host_beats_the_programmatic_one() -> None:
    config = resolve_reflection_config({'GENKIT_REFLECTION_HOST': '127.0.0.2'}, host=ALL_INTERFACES)
    assert config.host == '127.0.0.2'


def test_programmatic_host_does_not_turn_it_on() -> None:
    assert resolve_reflection_config({}, host=ALL_INTERFACES).mode == 'off'


def test_reflection_enabled_does_not_raise_on_invalid_port(monkeypatch: pytest.MonkeyPatch) -> None:
    # Telemetry setup calls this at import time; Genkit() raises the real error.
    monkeypatch.setenv('GENKIT_REFLECTION_PORT', 'abc')
    assert reflection_enabled() is False


def test_reflection_enabled_follows_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('GENKIT_ENV', raising=False)
    monkeypatch.delenv('GENKIT_REFLECTION_DISABLED', raising=False)
    monkeypatch.delenv('GENKIT_REFLECTION_V2_SERVER', raising=False)
    monkeypatch.delenv('GENKIT_REFLECTION_HOST', raising=False)
    monkeypatch.setenv('GENKIT_REFLECTION_PORT', '4200')
    assert reflection_enabled() is True


def test_port_zero_is_valid_and_pinned() -> None:
    config = resolve_reflection_config({'GENKIT_REFLECTION_PORT': '0'})
    assert (config.port, config.pinned) == (0, True)


@pytest.mark.parametrize('value', ['abc', '-1', '70000', '3100.5', ' 3100', '+7', '0x10', '1e3', '1_000', '\u0663'])
def test_invalid_port_raises_rather_than_falling_back(value: str) -> None:
    with pytest.raises(ValueError, match='GENKIT_REFLECTION_PORT'):
        resolve_reflection_config({'GENKIT_REFLECTION_PORT': value})


@pytest.mark.parametrize('host', ['127.0.0.1', '127.1.2.3', 'localhost', '::1', '[::1]'])
def test_loopback_hosts(host: str) -> None:
    assert is_loopback_host(host)


@pytest.mark.parametrize('host', [ALL_INTERFACES, '192.168.1.5', '10.0.0.1', 'example.com'])
def test_routable_hosts(host: str) -> None:
    assert not is_loopback_host(host)


def test_secrets_equal() -> None:
    assert secrets_equal('abc', 'abc')
    assert not secrets_equal('abc', 'abd')
    assert not secrets_equal('abc', 'much-longer-secret')
