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

"""Tests for Flask handler decorator validation."""

import pytest

from genkit.core.error import GenkitError
from genkit.plugins.flask.handler import genkit_flask_handler


class TestGenkitFlaskHandlerValidation:
    """Tests that genkit_flask_handler rejects non-flow inputs."""

    def test_rejects_plain_function(self) -> None:
        """The decorator must reject arguments that are not FlowWrapper."""

        class FakeGenkit:
            pass

        handler = genkit_flask_handler(FakeGenkit())  # type: ignore[arg-type]
        with pytest.raises(GenkitError, match='must apply @genkit_flask_handler on a @flow'):
            handler(lambda: None)  # type: ignore[arg-type]

    def test_rejects_string(self) -> None:
        """Test Rejects string."""

        class FakeGenkit:
            pass

        handler = genkit_flask_handler(FakeGenkit())  # type: ignore[arg-type]
        with pytest.raises(GenkitError, match='must apply @genkit_flask_handler on a @flow'):
            handler('not a flow')  # type: ignore[arg-type]

    def test_rejects_none(self) -> None:
        """Test Rejects none."""

        class FakeGenkit:
            pass

        handler = genkit_flask_handler(FakeGenkit())  # type: ignore[arg-type]
        with pytest.raises(GenkitError, match='must apply @genkit_flask_handler on a @flow'):
            handler(None)  # type: ignore[arg-type]


class TestFlaskHandlerImports:
    """Tests that module-level exports are correct."""

    def test_genkit_flask_handler_is_callable(self) -> None:
        """Test Genkit flask handler is callable."""
        assert callable(genkit_flask_handler)

    def test_handler_accepts_context_provider(self) -> None:
        """genkit_flask_handler can be called with optional context_provider."""

        class FakeGenkit:
            pass

        handler = genkit_flask_handler(FakeGenkit(), context_provider=None)  # type: ignore[arg-type]
        assert callable(handler)

    def test_flask_route_return_alias_exists(self) -> None:
        """Test Flask route return alias exists."""
        from genkit.plugins.flask.handler import FlaskRouteReturn

        assert FlaskRouteReturn is not None
