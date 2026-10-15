#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Shipped middleware configs use the same keys as the JS and Go middleware.

A .prompt file written for JS (`use: [{name: retry, config: {maxRetries: 0}}]`)
has to configure the Python middleware the same way, not run on defaults.
"""

import pytest
from genkit_middleware import Fallback, Filesystem, Retry, Skills, ToolApproval
from genkit_middleware._artifacts import Artifacts
from genkit_middleware._fallback import FallbackModelEntry

from genkit import GenkitError
from genkit._core._middleware import GenerateMiddleware
from genkit.middleware import BaseMiddleware
from genkit.plugin_api import GenkitConfig

_SHIPPED: list[type[BaseMiddleware]] = [Artifacts, Fallback, Filesystem, Retry, Skills, ToolApproval]


@pytest.mark.parametrize('middleware', _SHIPPED, ids=lambda c: c.__name__)
def test_shipped_config_is_genkit_config(middleware: type[BaseMiddleware]) -> None:
    """Each config takes camelCase and snake_case keys and rejects unknown ones."""
    assert issubclass(middleware.Config, GenkitConfig)


def test_fallback_model_entry_is_genkit_config() -> None:
    """The nested entry under `models` follows the same key rules as its parent."""
    assert issubclass(FallbackModelEntry, GenkitConfig)


def test_retry_takes_the_js_prompt_keys() -> None:
    """`maxRetries: 0` from a .prompt file turns retry off instead of keeping the default 3."""
    retry = GenerateMiddleware(cls=Retry, name='retry').instantiate({'maxRetries': 0, 'initialDelayMs': 10})

    assert isinstance(retry, Retry)
    assert retry.config.max_retries == 0
    assert retry.config.initial_delay_ms == 10


def test_retry_schema_matches_js_keys() -> None:
    """The Dev UI form for retry shows the keys the JS RetryOptions schema uses."""
    schema = GenerateMiddleware(cls=Retry, name='retry').config_schema
    assert schema is not None

    assert set(schema['properties']) == {
        'maxRetries',
        'statuses',
        'initialDelayMs',
        'maxDelayMs',
        'backoffFactor',
        'noJitter',
    }


def test_retry_misspelled_prompt_key_raises() -> None:
    """A typo in a .prompt `config` names the accepted keys."""
    with pytest.raises(GenkitError, match='Middleware "retry" got unknown config key\\(s\\) maxRetry;'):
        GenerateMiddleware(cls=Retry, name='retry').instantiate({'maxRetry': 0})
