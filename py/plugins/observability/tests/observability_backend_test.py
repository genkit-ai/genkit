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

"""Tests for observability Backend enum and environment variable mappings."""

from genkit.plugins.observability import Backend, package_name


class TestBackendEnum:
    """Tests for Backend StrEnum values."""

    def test_sentry_value(self) -> None:
        """Test Sentry value."""
        assert Backend.SENTRY == 'sentry'

    def test_honeycomb_value(self) -> None:
        """Test Honeycomb value."""
        assert Backend.HONEYCOMB == 'honeycomb'

    def test_datadog_value(self) -> None:
        """Test Datadog value."""
        assert Backend.DATADOG == 'datadog'

    def test_grafana_value(self) -> None:
        """Test Grafana value."""
        assert Backend.GRAFANA == 'grafana'

    def test_axiom_value(self) -> None:
        """Test Axiom value."""
        assert Backend.AXIOM == 'axiom'

    def test_custom_value(self) -> None:
        """Test Custom value."""
        assert Backend.CUSTOM == 'custom'

    def test_total_backends(self) -> None:
        """Test Total backends."""
        assert len(Backend) == 6

    def test_all_values_are_lowercase(self) -> None:
        """Test All values are lowercase."""
        for member in Backend:
            assert member.value == member.value.lower()

    def test_is_string(self) -> None:
        """Test Is string."""
        for member in Backend:
            assert isinstance(member.value, str)


class TestPackageName:
    """Tests for the package_name function."""

    def test_returns_string(self) -> None:
        """Test Returns string."""
        name = package_name()
        assert isinstance(name, str)

    def test_contains_observability(self) -> None:
        """Test Contains observability."""
        name = package_name()
        assert 'observability' in name

    def test_not_empty(self) -> None:
        """Test Not empty."""
        name = package_name()
        assert len(name) > 0


class TestModuleExports:
    """Tests for module-level exports."""

    def test_backend_importable(self) -> None:
        """Test Backend importable."""
        from genkit.plugins.observability import Backend

        assert Backend is not None

    def test_configure_telemetry_importable(self) -> None:
        """Test Configure telemetry importable."""
        from genkit.plugins.observability import configure_telemetry

        assert callable(configure_telemetry)

    def test_all_exports(self) -> None:
        """Test All exports."""
        from genkit.plugins import observability

        assert 'Backend' in observability.__all__
        assert 'configure_telemetry' in observability.__all__
        assert 'package_name' in observability.__all__
