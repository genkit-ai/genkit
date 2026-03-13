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

"""Tests for conform.display pure functions."""

from __future__ import annotations

from conform.display import (
    build_detail_text,
    build_progress_bar,
    build_progress_table,
    build_summary_table,
    elapsed_str,
    status_emoji,
    status_style,
)
from conform.types import PluginResult, Status


class TestElapsedStr:
    """Tests for elapsed_str formatting."""

    def test_zero(self) -> None:
        """Zero seconds returns a dash."""
        assert elapsed_str(0) == '—'

    def test_negative(self) -> None:
        """Negative seconds returns a dash."""
        assert elapsed_str(-1.0) == '—'

    def test_seconds_only(self) -> None:
        """Seconds-only formatting."""
        assert elapsed_str(5.0) == '5.0s'

    def test_minutes_and_seconds(self) -> None:
        """Minutes and seconds formatting."""
        assert elapsed_str(125.3) == '2m 5.3s'

    def test_fractional_seconds(self) -> None:
        """Fractional seconds formatting."""
        assert elapsed_str(0.5) == '0.5s'

    def test_exactly_one_minute(self) -> None:
        """Exactly one minute formatting."""
        assert elapsed_str(60.0) == '1m 0.0s'


class TestStatusHelpers:
    """Tests for status_emoji and status_style."""

    def test_emoji_for_all_statuses(self) -> None:
        """Every Status has a non-empty emoji."""
        for s in Status:
            result = status_emoji(s)
            assert isinstance(result, str)
            assert len(result) > 0

    def test_style_for_all_statuses(self) -> None:
        """Every Status has a non-empty style."""
        for s in Status:
            result = status_style(s)
            assert isinstance(result, str)
            assert len(result) > 0


class TestBuildProgressBar:
    """Tests for build_progress_bar."""

    def test_zero_total_returns_dash(self) -> None:
        """Zero total shows a dash."""
        text = build_progress_bar(0, 0, 0)
        assert text.plain == '—'

    def test_negative_total_returns_dash(self) -> None:
        """Negative total shows a dash."""
        text = build_progress_bar(0, 0, -1)
        assert text.plain == '—'

    def test_all_passed(self) -> None:
        """All passed shows full green bar."""
        text = build_progress_bar(10, 0, 10, bar_width=10)
        assert '10/10' in text.plain
        # All blocks should be filled (no ░).
        assert '░' not in text.plain

    def test_all_failed(self) -> None:
        """All failed shows full red bar."""
        text = build_progress_bar(0, 10, 10, bar_width=10)
        assert '0/10' in text.plain
        assert '░' not in text.plain

    def test_partial_progress(self) -> None:
        """Partial progress shows mixed bar."""
        text = build_progress_bar(5, 0, 10, bar_width=10)
        assert '5/10' in text.plain
        # Should have both █ and ░.
        assert '█' in text.plain
        assert '░' in text.plain

    def test_mixed_pass_fail(self) -> None:
        """Mixed pass/fail counts."""
        text = build_progress_bar(3, 2, 10, bar_width=10)
        assert '3/10' in text.plain

    def test_no_progress_yet(self) -> None:
        """No progress shows all remaining blocks."""
        text = build_progress_bar(0, 0, 10, bar_width=10)
        assert '0/10' in text.plain
        # All remaining.
        assert '█' not in text.plain

    def test_custom_bar_width(self) -> None:
        """Custom bar width is respected."""
        text = build_progress_bar(5, 0, 5, bar_width=20)
        # 20 filled blocks + space + count.
        assert '█' * 20 in text.plain

    def test_bar_width_no_overflow_on_rounding(self) -> None:
        """Regression: rounding passed and failed independently can overflow.

        With passed=3, failed=5, total=8, bar_width=12:
        round(3/8*12)=5, round(5/8*12)=8 → 5+8=13 > 12 without clamping.
        """
        for bar_width in (10, 12, 20):
            for total in range(1, 20):
                for passed in range(total + 1):
                    for failed in range(total - passed + 1):
                        text = build_progress_bar(passed, failed, total, bar_width=bar_width)
                        # Extract only the block characters (█ and ░).
                        blocks = [c for c in text.plain if c in ('█', '░')]
                        assert len(blocks) == bar_width, (
                            f'bar overflow: passed={passed}, failed={failed}, '
                            f'total={total}, bar_width={bar_width}, '
                            f'got {len(blocks)} blocks'
                        )


class TestBuildDetailText:
    """Tests for build_detail_text."""

    def test_supports_and_custom(self) -> None:
        """Shows supports and custom counts."""
        r = PluginResult(plugin='test', tests_supports=9, tests_custom=2)
        assert build_detail_text(r) == '9 std + 2 custom'

    def test_supports_only(self) -> None:
        """Shows supports with zero custom."""
        r = PluginResult(plugin='test', tests_supports=5, tests_custom=0)
        assert build_detail_text(r) == '5 std + 0 custom'

    def test_no_counts(self) -> None:
        """Empty result returns empty string."""
        r = PluginResult(plugin='test')
        assert build_detail_text(r) == ''

    def test_skipped_with_missing_env(self) -> None:
        """Skipped status shows missing env vars."""
        r = PluginResult(
            plugin='test',
            status=Status.SKIPPED,
            missing_env_vars=['API_KEY', 'SECRET'],
            tests_supports=5,
        )
        assert 'missing: API_KEY, SECRET' in build_detail_text(r)
        assert '5 std + 0 custom' in build_detail_text(r)

    def test_failed_with_error(self) -> None:
        """Failed status shows error message."""
        r = PluginResult(
            plugin='test',
            status=Status.FAILED,
            error_message='3 test(s) failed',
            tests_supports=8,
        )
        result = build_detail_text(r)
        assert '8 std + 0 custom' in result
        assert '3 test(s) failed' in result

    def test_error_truncation(self) -> None:
        """Long error messages are truncated."""
        r = PluginResult(
            plugin='test',
            status=Status.ERROR,
            error_message='x' * 200,
        )
        result = build_detail_text(r, max_error_len=50)
        assert len(result) <= 50

    def test_passed_no_error_shown(self) -> None:
        """Passed status does not show error message."""
        r = PluginResult(
            plugin='test',
            status=Status.PASSED,
            error_message='should not appear',
            tests_supports=5,
        )
        assert 'should not appear' not in build_detail_text(r)


class TestTableBuilders:
    """Smoke tests for build_progress_table and build_summary_table."""

    def _make_results(self) -> dict[str, PluginResult]:
        return {
            'plugin-a': PluginResult(
                plugin='plugin-a',
                runtime='python',
                status=Status.PASSED,
                tests_total=10,
                tests_supports=8,
                tests_custom=2,
                tests_passed=10,
                tests_failed=0,
                elapsed_s=5.2,
            ),
            'plugin-b': PluginResult(
                plugin='plugin-b',
                runtime='python',
                status=Status.FAILED,
                tests_total=6,
                tests_supports=6,
                tests_passed=4,
                tests_failed=2,
                elapsed_s=3.1,
                error_message='2 test(s) failed',
            ),
            'plugin-c': PluginResult(
                plugin='plugin-c',
                runtime='python',
                status=Status.SKIPPED,
                missing_env_vars=['API_KEY'],
            ),
        }

    def test_progress_table_returns_table(self) -> None:
        """Progress table has expected row count."""
        table = build_progress_table(self._make_results())
        assert table is not None
        assert table.row_count == 3

    def test_summary_table_returns_table(self) -> None:
        """Summary table has expected row count."""
        table = build_summary_table(self._make_results())
        assert table is not None
        assert table.row_count == 3

    def test_empty_results(self) -> None:
        """Empty results produce zero rows."""
        table = build_progress_table({})
        assert table.row_count == 0

    def test_pending_result(self) -> None:
        """Pending result produces one row."""
        results = {
            'x': PluginResult(
                plugin='x',
                runtime='python',
                status=Status.PENDING,
                tests_total=5,
            ),
        }
        table = build_progress_table(results)
        assert table.row_count == 1
