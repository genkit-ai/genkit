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

"""Tests for Flask plugin module exports and integration types."""

from genkit.core.context import RequestData


class TestFlaskModuleExports:
    """Tests for Flask plugin module-level exports."""

    def test_handler_module_importable(self) -> None:
        """Test Handler module importable."""
        from genkit.plugins.flask import handler

        assert hasattr(handler, 'genkit_flask_handler')

    def test_flask_route_return_type_alias(self) -> None:
        """Test Flask route return type alias."""
        from genkit.plugins.flask.handler import FlaskRouteReturn

        assert FlaskRouteReturn is not None

    def test_genkit_flask_handler_signature(self) -> None:
        """Test Genkit flask handler signature."""
        import inspect

        from genkit.plugins.flask.handler import genkit_flask_handler

        sig = inspect.signature(genkit_flask_handler)
        params = list(sig.parameters.keys())
        assert 'ai' in params
        assert 'context_provider' in params


class TestRequestDataBase:
    """Tests for the RequestData base class used by _FlaskRequestData."""

    def test_request_data_is_importable(self) -> None:
        """Test Request data is importable."""
        assert RequestData is not None

    def test_request_data_is_a_class(self) -> None:
        """Test Request data is a class."""
        assert isinstance(RequestData, type)

    def test_request_data_has_init(self) -> None:
        """Test Request data has init."""
        assert hasattr(RequestData, '__init__')
