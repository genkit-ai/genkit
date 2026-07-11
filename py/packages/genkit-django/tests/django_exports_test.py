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

"""Tests for Django plugin module exports and integration types."""

from genkit_django.handler import RequestData


class TestDjangoModuleExports:
    """Tests for Django plugin module-level exports."""

    def test_handler_module_importable(self) -> None:
        """Test Handler module importable."""
        from genkit_django import handler

        assert hasattr(handler, 'genkit_django_handler')

    def test_genkit_django_handler_signature(self) -> None:
        """Test Genkit django handler signature."""
        import inspect

        from genkit_django.handler import genkit_django_handler

        sig = inspect.signature(genkit_django_handler)
        params = list(sig.parameters.keys())
        assert 'ai' in params
        assert 'context_provider' in params

    def test_package_name(self) -> None:
        """Package exports its fully qualified module name."""
        from genkit_django import package_name

        assert package_name() == 'genkit_django'


class TestRequestDataBase:
    """Tests for the RequestData base class used by _DjangoRequestData."""

    def test_request_data_is_importable(self) -> None:
        """Test Request data is importable."""
        assert RequestData is not None

    def test_request_data_is_a_class(self) -> None:
        """Test Request data is a class."""
        assert isinstance(RequestData, type)

    def test_request_data_has_init(self) -> None:
        """Test Request data has init."""
        assert hasattr(RequestData, '__init__')
