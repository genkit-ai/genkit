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

"""Tests for Django handler decorator validation."""

import pytest
from genkit_django.handler import genkit_django_handler

from genkit._core._error import GenkitError


class TestGenkitDjangoHandlerValidation:
    """Tests that genkit_django_handler rejects non-flow inputs."""

    def test_rejects_plain_function(self) -> None:
        """The decorator must reject arguments that are not Flow."""

        class FakeGenkit:
            pass

        handler = genkit_django_handler(FakeGenkit())  # type: ignore[arg-type]
        with pytest.raises(GenkitError, match='must apply @genkit_django_handler on a @flow'):
            handler(lambda: None)  # type: ignore[arg-type]

    def test_rejects_string(self) -> None:
        """Test Rejects string."""

        class FakeGenkit:
            pass

        handler = genkit_django_handler(FakeGenkit())  # type: ignore[arg-type]
        with pytest.raises(GenkitError, match='must apply @genkit_django_handler on a @flow'):
            handler('not a flow')  # type: ignore[arg-type]

    def test_rejects_none(self) -> None:
        """Test Rejects none."""

        class FakeGenkit:
            pass

        handler = genkit_django_handler(FakeGenkit())  # type: ignore[arg-type]
        with pytest.raises(GenkitError, match='must apply @genkit_django_handler on a @flow'):
            handler(None)  # type: ignore[arg-type]


class TestDjangoHandlerImports:
    """Tests that module-level exports are correct."""

    def test_genkit_django_handler_is_callable(self) -> None:
        """Test Genkit django handler is callable."""
        assert callable(genkit_django_handler)

    def test_handler_accepts_context_provider(self) -> None:
        """genkit_django_handler can be called with optional context_provider."""

        class FakeGenkit:
            pass

        handler = genkit_django_handler(FakeGenkit(), context_provider=None)  # type: ignore[arg-type]
        assert callable(handler)
