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

"""Cloud Logging I/O for tool spans — including the current tool.v2 subtype."""

from unittest.mock import MagicMock, patch

from genkit_google_cloud.telemetry.action import ActionTelemetry


def _span(*, subtype: str, name: str = 'weather', extra: dict[str, str] | None = None) -> MagicMock:
    span = MagicMock()
    span.attributes = {
        'genkit:name': name,
        'genkit:metadata:subtype': subtype,
        'genkit:path': f'/{subtype}/{name}',
        'genkit:input': '{"city":"Austin"}',
        **(extra or {}),
    }
    return span


def test_tick_logs_tool_v2_input() -> None:
    tel = ActionTelemetry()
    with patch.object(tel, '_write_log') as write:
        tel.tick(_span(subtype='tool.v2'), log_input_and_output=True, project_id='p')
    write.assert_called()
    assert write.call_args.args[1] == 'Input'


def test_tick_skips_flow_spans() -> None:
    tel = ActionTelemetry()
    with patch.object(tel, '_write_log') as write:
        tel.tick(_span(subtype='flow', name='ask'), log_input_and_output=True, project_id='p')
    write.assert_not_called()
