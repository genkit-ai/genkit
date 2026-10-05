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

"""Experimental FastAPI serving for Genkit agents.

Agents are still changing, so serving them lives here rather than on the
``genkit_fastapi`` root. Apps that only serve flows never load the agent types.

```python
from genkit_fastapi.exp import serve_agent

app.include_router(serve_agent(weather_agent), prefix='/api')
```
"""

from genkit_fastapi.exp._agent import serve_agent

__all__ = [
    'serve_agent',
]
