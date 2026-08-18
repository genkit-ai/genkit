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

"""Turn google-genai SDK config ValidationErrors into Genkit errors."""

from pydantic import ValidationError

from genkit import GenkitError


def sdk_config_error(*, action_name: str, error: ValidationError) -> GenkitError:
    """Name the action and the field the SDK rejected.

    Family schemas keep leftover keys so a newly supported field can ride
    through. The google-genai request types reject unknowns, so this is
    the boundary where we tell the caller which key did not make it.
    """
    loc = ()
    errors = error.errors()
    if errors:
        loc = errors[0].get('loc') or ()
    key = str(loc[0]) if loc else 'config'
    return GenkitError(
        status='INVALID_ARGUMENT',
        message=f'{action_name}: invalid config field {key}',
        cause=error,
    )
