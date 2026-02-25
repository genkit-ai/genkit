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

"""Tests for conform.util_test_model pure functions."""

from __future__ import annotations

from pathlib import Path

import yaml
from conform.util_test_model import SpecTestCounts, count_spec_tests


class TestCountSpecTests:
    """Tests for count_spec_tests."""

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Missing file returns zero counts."""
        result = count_spec_tests(tmp_path / 'missing.yaml')
        assert result == SpecTestCounts(supports=0, custom=0, total=0)

    def test_single_suite_supports_only(self, tmp_path: Path) -> None:
        """Suite with only supports capabilities."""
        spec = tmp_path / 'spec.yaml'
        spec.write_text(
            yaml.dump([
                {
                    'model': 'test/model-a',
                    'supports': ['tool-request', 'multiturn', 'system-role'],
                },
            ])
        )
        result = count_spec_tests(spec)
        assert result.supports == 3
        assert result.custom == 0
        assert result.total == 3

    def test_single_suite_custom_tests_only(self, tmp_path: Path) -> None:
        """Suite with only custom tests."""
        spec = tmp_path / 'spec.yaml'
        spec.write_text(
            yaml.dump([
                {
                    'model': 'test/model-a',
                    'tests': [
                        {'name': 'custom1', 'input': {'messages': []}},
                        {'name': 'custom2', 'input': {'messages': []}},
                    ],
                },
            ])
        )
        result = count_spec_tests(spec)
        assert result.supports == 0
        assert result.custom == 2
        assert result.total == 2

    def test_mixed_supports_and_custom(self, tmp_path: Path) -> None:
        """Suite with both supports and custom tests."""
        spec = tmp_path / 'spec.yaml'
        spec.write_text(
            yaml.dump([
                {
                    'model': 'test/model-a',
                    'supports': ['tool-request', 'multiturn'],
                    'tests': [
                        {'name': 'custom1', 'input': {'messages': []}},
                    ],
                },
            ])
        )
        result = count_spec_tests(spec)
        assert result.supports == 2
        assert result.custom == 1
        assert result.total == 3

    def test_multiple_suites(self, tmp_path: Path) -> None:
        """Multiple suites aggregate counts."""
        spec = tmp_path / 'spec.yaml'
        spec.write_text(
            yaml.dump([
                {
                    'model': 'test/model-a',
                    'supports': ['tool-request', 'multiturn'],
                },
                {
                    'model': 'test/model-b',
                    'supports': ['system-role'],
                    'tests': [
                        {'name': 'c1', 'input': {'messages': []}},
                    ],
                },
            ])
        )
        result = count_spec_tests(spec)
        assert result.supports == 3
        assert result.custom == 1
        assert result.total == 4

    def test_unknown_capability_not_counted(self, tmp_path: Path) -> None:
        """Unknown capabilities are excluded from the count."""
        spec = tmp_path / 'spec.yaml'
        spec.write_text(
            yaml.dump([
                {
                    'model': 'test/model-a',
                    'supports': ['tool-request', 'nonexistent-capability'],
                },
            ])
        )
        result = count_spec_tests(spec)
        # Only tool-request is a real TEST_CASES key.
        assert result.supports == 1
        assert result.total == 1

    def test_single_dict_spec(self, tmp_path: Path) -> None:
        """Spec file with a single dict (not a list) should still work."""
        spec = tmp_path / 'spec.yaml'
        spec.write_text(
            yaml.dump({
                'model': 'test/model-a',
                'supports': ['tool-request'],
            })
        )
        result = count_spec_tests(spec)
        assert result.supports == 1
        assert result.total == 1

    def test_empty_supports_with_tests_uses_empty(self, tmp_path: Path) -> None:
        """When supports is absent but tests is present, supports defaults to []."""
        spec = tmp_path / 'spec.yaml'
        spec.write_text(
            yaml.dump([
                {
                    'model': 'test/model-a',
                    'tests': [
                        {'name': 'c1', 'input': {'messages': []}},
                    ],
                },
            ])
        )
        result = count_spec_tests(spec)
        assert result.supports == 0
        assert result.custom == 1
