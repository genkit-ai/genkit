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

"""genkit-flask must pull in Flask's async extra on its own.

``genkit_flask_handler`` returns an ``async def`` view, which Flask runs only
when ``asgiref`` is installed. The workspace venv gets ``asgiref`` from
genkit-django, so this checks the declared metadata rather than the venv.
"""

import re
from importlib.metadata import requires


def test_declares_flask_async_extra() -> None:
    """``genkit-flask`` requires ``flask[async]``, not bare ``flask``."""
    reqs = [r for r in requires('genkit-flask') or [] if re.match(r'flask\b', r, re.IGNORECASE)]
    assert len(reqs) == 1, reqs
    assert re.match(r'flask\s*\[\s*async\s*\]', reqs[0], re.IGNORECASE), reqs[0]
