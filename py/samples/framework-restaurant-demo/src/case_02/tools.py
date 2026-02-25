# Copyright 2025 Google LLC
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


"""Tools for case 02."""

import json
import os
import pathlib

from menu_ai import ai
from menu_schemas import MenuToolOutputSchema

menu_json_path = os.path.join(pathlib.Path(__file__).parent, '..', '..', 'data', 'menu.json')
with pathlib.Path(menu_json_path).open() as f:
    menu_data = json.load(f)


@ai.tool(name='todaysMenu')
def todays_menu(input: object | None = None) -> MenuToolOutputSchema:
    """Use this tool to retrieve all the items on today's menu.

    Args:
        input: Optional input.

    Returns:
        The menu items.
    """
    return MenuToolOutputSchema(
        menu_data=menu_data,
    )
