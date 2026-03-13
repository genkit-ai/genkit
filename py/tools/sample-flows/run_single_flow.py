#!/usr/bin/env python3
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

# pyrefly: ignore-file
# flake8: noqa: ASYNC240

"""Helper script to run a single Genkit flow in isolation.

This script is called by review_sample_flows.py to execute each flow in a
separate subprocess, avoiding event loop conflicts and enabling proper
async execution.

Usage:
    python run_single_flow.py <sample_dir> <flow_name> [--input <json_string>]

Output:
    JSON object with 'success', 'result', and 'error' fields
"""

import argparse
import asyncio
import importlib.util
import json
import logging
import sys
import traceback
from pathlib import Path
from typing import Any

from genkit.core.action import ActionKind
from genkit.types import Media


def format_output(
    output: Any,  # noqa: ANN401
    visited: set[int] | None = None,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Format flow output for serialization.

    Args:
        output: The flow output to format
        visited: Set of object IDs already visited to prevent infinite recursion

    Returns:
        Serializable representation
    """
    if visited is None:
        visited = set()

    # Track visited objects to prevent infinite recursion
    # Only track mutable/container types that could be involved in cycles
    if isinstance(output, (dict, list)) or hasattr(output, 'model_dump'):
        if id(output) in visited:
            return '(Recursive Reference)'
        visited.add(id(output))

    # Handle None
    if output is None:
        return None

    # Handle Media objects
    if isinstance(output, Media):
        return {
            'type': 'Media',
            'url': '(Media data not shown)',
            'content_type': output.content_type,
        }

    # Handle Pydantic models
    if hasattr(output, 'model_dump'):
        try:
            return format_output(output.model_dump(), visited)
        except Exception:  # noqa: S110 - intentional fallback if model_dump fails
            pass

    # Handle dicts
    if isinstance(output, dict):
        return {k: format_output(v, visited) for k, v in output.items()}

    # Handle lists
    if isinstance(output, list):
        return [format_output(v, visited) for v in output]

    # Default: convert to string for non-serializable objects (except basics)
    if isinstance(output, (str, int, float, bool, type(None))):
        return output

    return str(output)


async def run_flow(sample_dir: str, flow_name: str, input_data: Any) -> dict[str, Any]:  # noqa: ANN401 - intentional use of Any for arbitrary flow outputs
    """Run a single flow and return result.

    Args:
        sample_dir: Path to sample directory
        flow_name: Name of flow to run
        input_data: Input data for the flow

    Returns:
        Dict with 'success', 'result', 'error' fields
    """
    result: dict[str, Any] = {
        'success': False,
        'result': None,
        'error': None,
    }

    try:
        # Import the sample module
        sample_path = Path(sample_dir).resolve()
        main_py = sample_path / 'src' / 'main.py'
        if not main_py.exists():
            main_py = sample_path / 'main.py'

            result['error'] = f'No main.py found in {sample_path}'
            return result

        # Add the py/ root directory to sys.path so 'samples.shared' imports work
        # sample_path is .../py/samples/sample-name
        sys.path.insert(0, str(sample_path.parent.parent))

        # Add the sample's src/ directory to sys.path for relative imports
        # (e.g., 'from case_01 import prompts' in framework-restaurant-demo)
        if main_py.parent.name == 'src':
            # Add the sample root so 'from src import ...' works
            sys.path.insert(0, str(sample_path))
            module_name = 'src.main'
        else:
            sys.path.insert(0, str(main_py.parent))
            module_name = 'sample_main'

        # Clear existing modules with the same name if they exist.
        # This happens in a monorepo because multiple samples might use 'src.main'
        # as their entry point, or have a 'src' package.
        for m in list(sys.modules.keys()):
            if m == module_name or m.startswith(module_name + '.') or m == 'src' or m.startswith('src.'):
                sys.modules.pop(m, None)

        # Load the module
        try:
            if module_name == 'src.main':
                # Use import_module for package-structured samples to correctly handle relative imports
                module = importlib.import_module(module_name)
            else:
                # Fallback to spec-based loading for simple samples
                spec = importlib.util.spec_from_file_location(module_name, main_py)
                if not spec or not spec.loader:
                    result['error'] = 'Failed to load sample module'
                    return result

                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
        except Exception as e:
            result['error'] = f'Failed to import sample: {e}\n{traceback.format_exc()}'
            return result

        # Find the Genkit instance
        ai_instance = None
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if hasattr(attr, '__class__') and attr.__class__.__name__ == 'Genkit':
                ai_instance = attr
                break

        if not ai_instance:
            result['error'] = 'No Genkit instance found in sample'
            return result

        # Manually load prompts from the sample's prompts directory if it exists.
        # This is needed because the current working directory isn't the sample dir.
        prompts_dir = sample_path / 'prompts'
        if prompts_dir.exists() and prompts_dir.is_dir():
            from genkit.blocks.prompt import load_prompt_folder

            load_prompt_folder(ai_instance.registry, prompts_dir)

        # Get the flow action from registry
        try:
            registry = ai_instance.registry
            actions_map = await registry.resolve_actions_by_kind(ActionKind.FLOW)

            if flow_name not in actions_map:
                result['error'] = f"Flow '{flow_name}' not found in registry"
                return result

            flow_action = actions_map[flow_name]
        except Exception as e:
            result['error'] = f"Failed to retrieve flow '{flow_name}': {e}"
            return result

        # Run the flow - use arun() in async context
        try:
            # Convert dict input to Pydantic model if an input schema is defined
            validated_input = input_data
            if isinstance(input_data, dict) and hasattr(flow_action, 'input_type') and flow_action.input_type:
                try:
                    # Use the Action's Pydantic TypeAdapter to validate and convert the input
                    validated_input = flow_action.input_type.validate_python(input_data)
                except Exception:  # noqa: S110 - intentional fallback if validation fails
                    # If validation fails, we try with the original dict
                    pass

            # Always use arun() since we're in an async context
            flow_result = await flow_action.arun(validated_input)

            # Extract response
            response_obj = flow_result.response

            # Format output
            formatted_output = format_output(response_obj)

            result['success'] = True
            result['result'] = formatted_output

        except Exception as e:
            # pyrefly: ignore[unbound-name] - traceback is imported at top of file
            result['error'] = f'Flow execution failed: {e}\n{traceback.format_exc()}'

    except Exception as e:
        result['error'] = f'Unexpected error: {e}'

    return result


def main() -> None:
    """Run a single flow and output JSON result."""
    parser = argparse.ArgumentParser(description='Run a single Genkit flow.')
    parser.add_argument('sample_dir', type=str, help='Path to sample directory')
    parser.add_argument('flow_name', type=str, help='Name of flow to run')
    parser.add_argument('--input', type=str, default='null', help='JSON string of input data')
    args = parser.parse_args()

    # Suppress verbose logging
    logging.basicConfig(level=logging.ERROR)
    logging.getLogger('genkit').setLevel(logging.ERROR)
    logging.getLogger('google').setLevel(logging.ERROR)
    logging.getLogger('asyncio').setLevel(logging.ERROR)
    logging.getLogger('httpx').setLevel(logging.ERROR)
    logging.getLogger('httpcore').setLevel(logging.ERROR)

    # Parse input
    try:
        input_data = json.loads(args.input)
    except json.JSONDecodeError:
        return

    # Run flow in async context
    # We do NOT redirect stdout so that logs/prints from the flow are visible
    try:
        result = asyncio.run(run_flow(args.sample_dir, args.flow_name, input_data))
    except Exception as e:
        result = {
            'success': False,
            'result': None,
            'error': f'Subprocess execution failed: {e}\n{traceback.format_exc()}',
        }

    # Print result with markers so the caller can extract it from stdout
    print('\n---JSON_RESULT_START---')  # noqa: T201
    print(json.dumps(result))  # noqa: T201
    print('---JSON_RESULT_END---')  # noqa: T201


if __name__ == '__main__':
    main()
