#!/usr/bin/env python3
"""Validate public Markdown links, JSON prompt files, and Python syntax."""

from __future__ import annotations

import json
import py_compile
import re
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", ".venv", "__pycache__"}
LOCAL_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
EXPECTED_PROMPT_COUNTS = {
    "behaviour_prompts.json": 169,
    "christomorphic_geometry_probe_suite_v1.json": 89,
}


def repository_files(pattern: str):
    for path in ROOT.rglob(pattern):
        if not SKIP_PARTS.intersection(path.parts):
            yield path


def validate_markdown_links(errors: list[str]) -> int:
    checked = 0
    for markdown_path in repository_files("*.md"):
        text = markdown_path.read_text(encoding="utf-8")
        for match in LOCAL_LINK.finditer(text):
            raw_target = match.group(1).strip()
            if raw_target.startswith("<") and raw_target.endswith(">"):
                raw_target = raw_target[1:-1]

            target = raw_target.split("#", 1)[0]
            if not target or "://" in target or target.startswith(("mailto:", "#")):
                continue

            target = unquote(target)
            resolved = (markdown_path.parent / target).resolve()
            checked += 1
            if not resolved.exists():
                relative_source = markdown_path.relative_to(ROOT)
                errors.append(f"{relative_source}: missing local link target {raw_target!r}")
    return checked


def prompt_entries(data):
    if isinstance(data, dict) and isinstance(data.get("prompts"), list):
        return data["prompts"]
    if isinstance(data, list):
        return data
    return None


def validate_json(errors: list[str]) -> tuple[int, int]:
    file_count = 0
    prompt_count = 0
    for json_path in repository_files("*.json"):
        file_count += 1
        relative_path = json_path.relative_to(ROOT)
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            errors.append(f"{relative_path}: invalid JSON: {exc}")
            continue

        if json_path.parent.name != "eval":
            continue

        entries = prompt_entries(data)
        if entries is None:
            errors.append(f"{relative_path}: expected a list or an object with a prompts list")
            continue

        expected = EXPECTED_PROMPT_COUNTS.get(json_path.name)
        if expected is not None and len(entries) != expected:
            errors.append(
                f"{relative_path}: expected {expected} prompts, found {len(entries)}"
            )

        seen_ids: set[str] = set()
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict) or not isinstance(entry.get("prompt"), str):
                errors.append(f"{relative_path}: entry {index} has no string prompt")
                continue
            prompt_count += 1
            prompt_id = entry.get("id", entry.get("prompt_id"))
            if prompt_id is None:
                continue
            if not isinstance(prompt_id, str) or not prompt_id.strip():
                errors.append(f"{relative_path}: entry {index} has an invalid prompt id")
            elif prompt_id in seen_ids:
                errors.append(f"{relative_path}: duplicate prompt id {prompt_id!r}")
            else:
                seen_ids.add(prompt_id)
    return file_count, prompt_count


def validate_python(errors: list[str]) -> int:
    checked = 0
    for python_path in repository_files("*.py"):
        checked += 1
        try:
            py_compile.compile(str(python_path), doraise=True)
        except py_compile.PyCompileError as exc:
            errors.append(f"{python_path.relative_to(ROOT)}: {exc.msg}")
    return checked


def main() -> int:
    errors: list[str] = []
    link_count = validate_markdown_links(errors)
    json_count, prompt_count = validate_json(errors)
    python_count = validate_python(errors)

    if errors:
        print("Repository validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "Repository validation passed: "
        f"{link_count} local links, {json_count} JSON files, "
        f"{prompt_count} prompts, {python_count} Python files."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
