#!/usr/bin/env python3
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tool to generate or verify TypedDict configuration classes (*ConfigDict) from Pydantic config schemas for core plugins."""

import ast
from pathlib import Path
from typing import Any


def ast_to_type_str(node: ast.AST | None) -> str:
    """Convert an AST annotation node back to a readable Python type string."""
    if node is None:
        return 'Any'
    return ast.unparse(node)


def strip_optional(type_str: str) -> str:
    """Strip top-level '| None' or 'Optional[...]' from a type string for TypedDict ergonomics."""
    type_str = type_str.strip()
    if type_str.endswith('| None'):
        return type_str[:-6].strip()
    if type_str.startswith('Optional[') and type_str.endswith(']'):
        return type_str[9:-1].strip()
    return type_str


def generate_typedict_from_ast(class_node: ast.ClassDef, typedict_name: str) -> str:
    """Generate a TypedDict definition string from a Pydantic ClassDef AST node."""
    lines = [
        f'class {typedict_name}(CommonModelConfigDict, total=False):',
        f'    """Typed dictionary configuration generated from {class_node.name} for IDE autocomplete support."""',
        '',
    ]
    fields_found = 0
    for item in class_node.body:
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            field_name = item.target.id
            if field_name.startswith('_') or field_name in ('model_config',):
                continue
            raw_type = ast_to_type_str(item.annotation)
            # Check if it's Annotated[T, ...]
            if raw_type.startswith('Annotated['):
                # Extract first argument inside Annotated[...]
                inner = raw_type[10:-1].split(',')[0].strip()
                raw_type = inner
            clean_type = strip_optional(raw_type)
            lines.append(f'    {field_name}: {clean_type}')
            fields_found += 1

    if fields_found == 0:
        lines.append('    pass')
    return '\n'.join(lines)


import sys


def parse_and_generate(file_path: Path, class_name: str, typedict_name: str) -> str:
    """Parse a Python file and generate a TypedDict from the specified class name."""
    tree = ast.parse(file_path.read_text(encoding='utf-8'), filename=str(file_path))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return generate_typedict_from_ast(node, typedict_name)
    raise ValueError(f'Class {class_name} not found in {file_path}')


def main() -> None:
    if len(sys.argv) == 4:
        file_path = Path(sys.argv[1])
        class_name = sys.argv[2]
        typedict_name = sys.argv[3]
        print(parse_and_generate(file_path, class_name, typedict_name))
    else:
        print('Usage: generate_config_dicts.py <path_to_file.py> <PydanticClassName> <TargetTypedDictName>')
        print('Example: generate_config_dicts.py py/plugins/compat-oai/src/genkit/plugins/compat_oai/typing.py OpenAIConfig OpenAIConfigDict')


if __name__ == '__main__':
    main()
