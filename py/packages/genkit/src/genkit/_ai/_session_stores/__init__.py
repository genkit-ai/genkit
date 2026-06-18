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

"""Session store variants package."""

from genkit._ai._session_stores.latest_state import (
    LatestRecord,
    LatestStateStore,
    InMemoryLatestStateStore,
    FileLatestStateStore,
)
from genkit._ai._session_stores.linear import (
    TurnRecord,
    LinearSessionStore,
    InMemoryLinearSessionStore,
    FileLinearSessionStore,
)
from genkit._ai._session_stores.branching import (
    BranchRecord,
    BranchingSessionStore,
    InMemoryBranchingSessionStore,
    FileBranchingSessionStore,
)

__all__ = [
    'LatestRecord',
    'LatestStateStore',
    'InMemoryLatestStateStore',
    'FileLatestStateStore',
    'TurnRecord',
    'LinearSessionStore',
    'InMemoryLinearSessionStore',
    'FileLinearSessionStore',
    'BranchRecord',
    'BranchingSessionStore',
    'InMemoryBranchingSessionStore',
    'FileBranchingSessionStore',
]
