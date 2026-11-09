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

"""Packaging contract: Firestore is an extra, and import stays warning-free."""

import subprocess
import sys
import textwrap


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [sys.executable, '-c', textwrap.dedent(code)], capture_output=True, text=True, timeout=120
    )


def test_telemetry_imports_without_firestore() -> None:
    """`import genkit_google_cloud` never touches google-cloud-firestore."""
    proc = _run("""
        import sys
        sys.modules['google.cloud.firestore'] = None
        from genkit_google_cloud import enable_google_cloud_telemetry
        print('ok')
    """)

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == 'ok'


def test_firestore_session_store_without_extra_names_the_install_command() -> None:
    """Importing FirestoreSessionStore without the extra raises one line naming the fix."""
    proc = _run("""
        import sys
        sys.modules['google.cloud.firestore'] = None
        try:
            from genkit_google_cloud.exp import FirestoreSessionStore
        except ModuleNotFoundError as e:
            print(type(e).__name__, e.name, str(e), sep='|')
    """)

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == (
        'ModuleNotFoundError|google.cloud.firestore|'
        "FirestoreSessionStore needs google-cloud-firestore: uv add 'genkit-google-cloud[firestore]'"
    )


def test_import_is_silent_when_upstream_deprecates_cloud_trace_exporter() -> None:
    """opentelemetry-exporter-gcp-trace>=1.15 deprecates the base class; importing the plugin stays silent."""
    proc = _run("""
        import warnings
        import typing_extensions
        from opentelemetry.exporter import cloud_trace

        # Same decorator and message upstream applies in 1.15.0.
        cloud_trace.CloudTraceSpanExporter = typing_extensions.deprecated(
            'CloudTraceSpanExporter is deprecated. See migration guide at https://example.invalid'
        )(cloud_trace.CloudTraceSpanExporter)
        warnings.filterwarnings('error', message='CloudTraceSpanExporter is deprecated')

        import genkit_google_cloud
        from genkit_google_cloud._telemetry._trace_exporter import GenkitGCPExporter
        assert GenkitGCPExporter.__mro__[1] is cloud_trace.CloudTraceSpanExporter
        print('ok')
    """)

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == 'ok'
