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

# pyrefly: ignore-file

"""Tool to review and test all Genkit flows in a sample's main.py.

Usage:
    python review_sample_flows.py <path_to_sample_directory>

Example:
    python review_sample_flows.py samples/provider-google-genai-hello
"""

import argparse
import asyncio
import importlib.util
import json
import logging
import platform
import re
import subprocess  # noqa: S404
import sys
import time
import traceback
import warnings
from pathlib import Path
from typing import Any

from genkit.core.action import ActionKind
from genkit.types import Media

logging.getLogger().setLevel(logging.ERROR)
logging.getLogger('asyncio').setLevel(logging.ERROR)
logging.getLogger('httpx').setLevel(logging.ERROR)
logging.getLogger('httpcore').setLevel(logging.ERROR)

warnings.filterwarnings('ignore')


def open_file(path: str) -> None:
    """Open a file with the default system application."""
    try:
        if platform.system() == 'Darwin':  # macOS
            subprocess.run(['open', path], check=False)  # noqa: S603, S607
        elif platform.system() == 'Linux':
            subprocess.run(['xdg-open', path], check=False)  # noqa: S603, S607
        elif platform.system() == 'Windows':
            subprocess.run(['start', path], shell=True, check=False)  # noqa: S602, S607
    except Exception:  # noqa: S110
        pass


def write_report(
    path: str,
    action_count: int,
    successful_flows: list[str],
    failed_flows: list[str],
    detail_lines: list[str],
    sample_name: str,
) -> None:
    """Write the test report to a file."""
    report_lines = []
    report_lines.append(f'Flow Review Report for {sample_name}')
    report_lines.append('=' * 60)
    report_lines.append('')

    report_lines.append('SUMMARY')
    report_lines.append('=' * 60)
    report_lines.append(f'Total Flows: {action_count}')
    report_lines.append(f'Successful: {len(successful_flows)}')
    report_lines.append(f'Failed: {len(failed_flows)}')
    report_lines.append('')

    if failed_flows:
        report_lines.append('Failed Flows:')
        for flow in failed_flows:
            report_lines.append(f'  ✗ {flow}')
        report_lines.append('')

    if successful_flows:
        report_lines.append('Successful Flows:')
        for flow in successful_flows:
            report_lines.append(f'  ✓ {flow}')
        report_lines.append('')

    report_lines.append('=' * 60)
    report_lines.append('')
    report_lines.append('DETAILED RESULTS')
    report_lines.append('=' * 60)
    report_lines.append('')

    # Append detailed results
    report_lines.extend(detail_lines)

    # Write report
    with open(path, 'w') as f:
        f.write('\n'.join(report_lines))


def main() -> None:  # noqa: ASYNC240, ASYNC230 - test script, blocking I/O acceptable
    """Test all flows in a Genkit sample and generate a report."""
    parser = argparse.ArgumentParser(description='Test all flows in a Genkit sample.')
    parser.add_argument('sample_dir', type=str, help='Path to the sample directory')
    parser.add_argument('--output', type=str, default='flow_review_results.txt', help='Output report file')
    args = parser.parse_args()

    # Suppress verbose logging from genkit framework to avoid printing full data URLs
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger('genkit').setLevel(logging.WARNING)
    logging.getLogger('google').setLevel(logging.WARNING)

    sample_path = Path(args.sample_dir).resolve()
    if not sample_path.exists():
        sys.exit(1)

    # Assume the main entry point is at src/main.py or main.py
    is_package_structured = False
    main_py_path = sample_path / 'src' / 'main.py'
    if not main_py_path.exists():
        main_py_path = sample_path / 'main.py'
        if not main_py_path.exists():
            sys.exit(1)
    else:
        is_package_structured = True

    # Add the source directory to sys.path so imports work
    if is_package_structured:
        # For src/main.py, we want top-level imports to be relative to the sample dir
        # and relative imports inside src to work via 'src.main' package context.
        sys.path.insert(0, str(sample_path))
        module_name = 'src.main'
    else:
        src_dir = main_py_path.parent
        sys.path.insert(0, str(src_dir))
        module_name = 'sample_main'

    # Add the py/ root directory to sys.path so 'samples.shared' imports work
    # sample_path is .../py/samples/sample-name
    # sample_path.parent is .../py/samples
    # sample_path.parent.parent is .../py
    sys.path.insert(0, str(sample_path.parent.parent))

    # Clear existing modules with the same name if they exist.
    # This happens in a monorepo because multiple samples might use 'src.main'
    # as their entry point, or have a 'src' package.
    for m in list(sys.modules.keys()):
        if m == module_name or m.startswith(module_name + '.') or m == 'src' or m.startswith('src.'):
            sys.modules.pop(m, None)

    # Import the module
    try:
        if is_package_structured:
            # Use import_module for package-structured samples to correctly handle relative imports
            module = importlib.import_module(module_name)
        else:
            # Fallback to spec-based loading for simple samples
            spec = importlib.util.spec_from_file_location(module_name, main_py_path)
            if spec is None or spec.loader is None:
                sys.exit(1)
            # Add explicit assertions for type narrowing if needed by some checkers
            assert spec is not None
            assert spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
    except Exception:
        traceback.print_exc()
        sys.exit(1)

    # Find the Genkit instance
    ai_instance = None
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        # Check if it looks like a Genkit instance (has registry)
        if hasattr(attr, 'registry') and hasattr(attr, 'generate'):
            ai_instance = attr
            break

    if ai_instance:
        # Manually load prompts from the sample's prompts directory if it exists.
        # This is needed because the current working directory isn't the sample dir.
        prompts_dir = sample_path / 'prompts'
        if prompts_dir.exists() and prompts_dir.is_dir():
            from genkit.blocks.prompt import load_prompt_folder

            load_prompt_folder(ai_instance.registry, prompts_dir)

    if ai_instance is None:
        sys.exit(1)  # pyrefly: ignore[unbound-name] - sys is imported at top of file

    assert ai_instance is not None  # Type narrowing for ai_instance.registry

    # List all flows
    registry = ai_instance.registry
    actions_map = asyncio.run(registry.resolve_actions_by_kind(ActionKind.FLOW))

    # Track results for summary
    successful_flows = []
    failed_flows = []

    # We'll add the summary after testing all flows
    class LiveLogger(list):
        def append(self, item: Any) -> None:  # noqa: ANN401 - override requires Any
            super().append(item)
            # Print to stdout for immediate feedback
            print(item)  # noqa: T201

    detail_lines = LiveLogger()

    try:
        for flow_name, flow_action in actions_map.items():
            detail_lines.append(f'\nFlow: {flow_name}')
            detail_lines.append('-' * 30)

            try:
                input_data = generate_input(flow_action)
                detail_lines.append(f'Generated Input: {input_data}')

                # Run flow in subprocess to avoid event loop conflicts
                # Get path to helper script
                script_dir = Path(__file__).parent
                helper_script = script_dir / 'run_single_flow.py'

                # Prepare subprocess command
                cmd = [
                    'uv',
                    'run',
                    str(helper_script),
                    str(sample_path),
                    flow_name,
                    '--input',
                    json.dumps(input_data) if input_data is not None else 'null',
                ]

                # Run subprocess
                process = subprocess.Popen(  # noqa: S603 - cmd is constructed internally from trusted script paths
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,  # Line buffered
                    cwd=sample_path.parent.parent,  # Run from py/ directory
                )

                # Stream output
                stdout_lines = []
                if process.stdout:
                    for line in process.stdout:
                        stdout_lines.append(line)
                        # Suppress debug logs as requested, but stream everything else for "live logging"
                        # We strip ANSI codes before checking per report/pic requirement
                        clean_line = re.sub(r'\x1b\[[0-9;]*[mGKF]', '', line).lower()
                        # Level tags like [debug], [debug   ], etc.
                        is_debug = '[debug' in clean_line
                        # List of noisy patterns to filter from test output.
                        noise_patterns = [
                            'userwarning:',
                            'shadows an attribute',
                            'class outputconfig',
                            'class promptinputconfig',
                            'class promptoutputconfig',
                            '---json_result_',
                            '{"success":',
                            '[info',
                        ]
                        is_noise = any(p in clean_line for p in noise_patterns)

                        if not is_debug and not is_noise:
                            # Also filter out empty lines or just markers
                            if line.strip():
                                print(f'  {line}', end='')  # noqa: T201

                process.wait(timeout=120)

                # Reconstruct stdout for parsing
                stdout = ''.join(stdout_lines)

                try:
                    if '---JSON_RESULT_START---' in stdout and '---JSON_RESULT_END---' in stdout:
                        json_str = stdout.split('---JSON_RESULT_START---')[1].split('---JSON_RESULT_END---')[0].strip()
                        result_data = json.loads(json_str)
                    else:
                        # Fallback try to parse the whole thing if markers missing (unlikely for success case)
                        result_data = json.loads(stdout)
                except (json.JSONDecodeError, IndexError):
                    detail_lines.append('Status: FAILED')
                    detail_lines.append('Error: Failed to parse subprocess output')

                    # Print raw output for debugging since parsing failed
                    detail_lines.append('Raw Output:')
                    detail_lines.append(stdout)

                    failed_flows.append(flow_name)
                    continue

                if result_data.get('success'):
                    detail_lines.append('Status: SUCCESS')

                    # Format the result
                    flow_result = result_data.get('result')
                    formatted_output = format_output(flow_result, max_length=500)
                    detail_lines.append(f'Output: {formatted_output}')

                    successful_flows.append(flow_name)
                else:
                    detail_lines.append('Status: FAILED')
                    error_msg = result_data.get('error', 'Unknown error')
                    detail_lines.append(f'Error: {error_msg}')
                    failed_flows.append(flow_name)

            except subprocess.TimeoutExpired:
                detail_lines.append('Status: FAILED')
                detail_lines.append('Error: Flow execution timed out (120s)')
                failed_flows.append(flow_name)
            except Exception as e:
                detail_lines.append('Status: FAILED')
                detail_lines.append(f'Error: Subprocess failed: {e}')
                failed_flows.append(flow_name)

                # Add traceback for debugging
                tb_lines = traceback.format_exc().split('\n')
                detail_lines.append('Traceback:')
                for line in tb_lines:
                    detail_lines.append(f'  {line}')

                failed_flows.append(flow_name)

        detail_lines.append('')

        # Add a small delay between tests to avoid rate limiting
        time.sleep(10)

    except KeyboardInterrupt:
        pass
    finally:
        write_report(
            args.output,
            len(actions_map),
            successful_flows,
            failed_flows,
            detail_lines,
            sample_path.name,
        )
        open_file(args.output)


def format_output(output: Any, max_length: int = 500) -> str:  # noqa: ANN401 - intentional use of Any for arbitrary flow outputs
    """Format flow output for human-readable display.

    Args:
        output: The flow output to format
        max_length: Maximum length before truncation

    Returns:
        Formatted string representation
    """
    # Handle None
    if output is None:
        return 'None'

    # Handle Media objects
    if isinstance(output, Media):
        if output.url and len(output.url) > max_length:
            truncated_url = f'{output.url[:100]}...{output.url[-50:]}'
            return f"Media(url='{truncated_url}' [{len(output.url)} chars], content_type='{output.content_type}')"
        return f"Media(url='{output.url}', content_type='{output.content_type}')"

    # Handle Pydantic models
    if hasattr(output, 'model_dump'):
        try:
            data = output.model_dump()
            json_str = json.dumps(data, indent=2)
            if len(json_str) > max_length:
                return f'{json_str[:max_length]}... [truncated, {len(json_str)} total chars]'
            return json_str
        except Exception:  # noqa: S110 - intentional fallback if model_dump fails
            pass

    # Handle dicts
    if isinstance(output, dict):
        try:
            json_str = json.dumps(output, indent=2)
            if len(json_str) > max_length:
                return f'{json_str[:max_length]}... [truncated, {len(json_str)} total chars]'
            return json_str
        except Exception:  # noqa: S110 - intentional fallback if json.dumps fails
            pass

    # Handle lists
    if isinstance(output, list):
        try:
            json_str = json.dumps(output, indent=2)
            if len(json_str) > max_length:
                return f'{json_str[:max_length]}... [truncated, {len(json_str)} total chars]'
            return json_str
        except Exception:  # noqa: S110 - intentional fallback if json.dumps fails
            pass

    # Default: convert to string
    output_str = str(output)
    if len(output_str) > max_length:
        return f'{output_str[:max_length]}... [truncated, {len(output_str)} total chars]'
    return output_str


def generate_input(flow_action: Any) -> Any:  # noqa: ANN401 - intentional use of Any for arbitrary flow inputs
    """Generates heuristic input for a flow based on its schema."""
    schema = flow_action.input_schema
    if not schema:
        return None

    # Generate dict from schema
    input_dict = generate_from_json_schema(schema)

    # If the flow has a Pydantic model for input, instantiate it
    # The schema has a 'title' field that matches the model class name
    if isinstance(input_dict, dict) and 'title' in schema:
        # Try to get the actual model class from the flow's metadata
        # For now, just return the dict - Genkit should handle conversion
        pass

    return input_dict


def generate_from_json_schema(schema: dict[str, Any]) -> Any:  # noqa: ANN401 - intentional use of Any for arbitrary flow inputs
    """Simplistic JSON schema input generator."""
    type_ = schema.get('type')

    if 'default' in schema:
        return schema['default']

    if type_ == 'string':
        return 'test_string'
    elif type_ == 'integer':
        return 42
    elif type_ == 'number':
        return 3.14
    elif type_ == 'boolean':
        return True
    elif type_ == 'object':
        properties = schema.get('properties', {})
        result = {}
        for prop_name, prop_schema in properties.items():
            result[prop_name] = generate_from_json_schema(prop_schema)
        return result
    elif type_ == 'array':
        items_schema = schema.get('items', {})
        return [generate_from_json_schema(items_schema)]

    return None


if __name__ == '__main__':
    main()
