"""YAML configuration loader with PyYAML primary backend and lightweight stdlib fallback."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


def _parse_scalar(raw: str) -> Any:
    """Parse a YAML scalar value when PyYAML is not installed."""
    val = raw.strip()
    if not val or val in ("null", "~"):
        return None
    if val in ("true", "True"):
        return True
    if val in ("false", "False"):
        return False
    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
        return val[1:-1]
    if val.startswith("[") and val.endswith("]"):
        inner = val[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part) for part in inner.split(",")]
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val


def _strip_inline_comment(line: str) -> str:
    in_single = False
    in_double = False
    for idx, ch in enumerate(line):
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            return line[:idx]
    return line


def _fallback_parse_yaml(text: str) -> dict[str, Any]:
    """Parse block-style YAML (nested mappings, sequences of mappings, inline lists)."""
    cleaned_lines: list[tuple[int, str]] = []
    for raw_line in text.splitlines():
        stripped_comment = _strip_inline_comment(raw_line).rstrip()
        if not stripped_comment.strip():
            continue
        indent = len(stripped_comment) - len(stripped_comment.lstrip(" "))
        cleaned_lines.append((indent, stripped_comment.strip()))

    def parse_block(index: int, min_indent: int) -> tuple[Any, int]:
        if index >= len(cleaned_lines):
            return {}, index
        first_indent, first_content = cleaned_lines[index]
        if first_indent < min_indent:
            return {}, index

        if first_content.startswith("- "):
            items: list[Any] = []
            while index < len(cleaned_lines):
                indent, content = cleaned_lines[index]
                if indent < first_indent or not content.startswith("- "):
                    break
                if indent > first_indent:
                    break
                rest = content[2:].strip()
                index += 1
                if ":" in rest and not (rest.startswith('"') or rest.startswith("'")):
                    key, val_part = rest.split(":", 1)
                    item_dict: dict[str, Any] = {}
                    val_part = val_part.strip()
                    if val_part:
                        item_dict[key.strip()] = _parse_scalar(val_part)
                    else:
                        nested, index = parse_block(index, indent + 2)
                        item_dict[key.strip()] = nested
                    if index < len(cleaned_lines) and cleaned_lines[index][0] > indent:
                        extra, index = parse_block(index, indent + 1)
                        if isinstance(extra, dict):
                            item_dict.update(extra)
                    items.append(item_dict)
                else:
                    items.append(_parse_scalar(rest))
            return items, index

        mapping: dict[str, Any] = {}
        while index < len(cleaned_lines):
            indent, content = cleaned_lines[index]
            if indent < first_indent:
                break
            if indent > first_indent or content.startswith("- "):
                break
            key, val_part = content.split(":", 1)
            key = key.strip()
            val_part = val_part.strip()
            index += 1
            if val_part:
                mapping[key] = _parse_scalar(val_part)
            else:
                nested, index = parse_block(index, indent + 1)
                mapping[key] = nested
        return mapping, index

    parsed, _ = parse_block(0, 0)
    return parsed if isinstance(parsed, dict) else {}


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file into a Python dictionary."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    raw_text = config_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore[import-untyped]

        data = yaml.safe_load(raw_text)
        return data if isinstance(data, dict) else {}
    except ImportError:
        return _fallback_parse_yaml(raw_text)
