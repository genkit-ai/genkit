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

"""Native conformance executors for all runtimes.

This package contains the executor implementations:

- ``conformance_native.py``  — Python (JSONL-over-stdio subprocess)
- ``conformance_native.go``  — Go (JSONL-over-stdio subprocess)
- ``conformance_native.ts``  — TypeScript (JSONL-over-stdio subprocess)
- ``in_process_runner.py``   — Python (in-process via ``ai.generate()``)
"""
